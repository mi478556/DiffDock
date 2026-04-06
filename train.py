import copy
import functools
import math
import os
import shutil
from functools import partial

import wandb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml
import time
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, t_to_sigma_individual
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, inference_epoch_fix
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage


def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset2):

    loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                      tor_weight=args.tor_weight, no_torsion=args.no_torsion, backbone_weight=args.backbone_loss_weight,
                      sidechain_weight=args.sidechain_loss_weight)

    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_val_secondary_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0

    freeze_params = 0
    scheduler_mode = args.inference_earlystop_goal if args.val_inference_freq is not None else 'min'
    if args.scheduler == 'layer_linear_warmup':
        freeze_params = args.warmup_dur * (args.num_conv_layers + 2) - 1
        print("Freezing some parameters until epoch {}".format(freeze_params))

    # ensure run_dir exists (run_dir comes from main_function and points to args.log_dir/args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Optional per-epoch logging (CSV + TensorBoard). Enabled via --enable_logging to avoid
    # injecting IO for users who don't want it. When enabled we append a CSV row each epoch
    # and write scalars to TensorBoard under run_dir/tensorboard.
    csv_writer = None
    csv_file = None
    tb_writer = None
    if hasattr(args, 'enable_logging') and args.enable_logging:
        try:
            import csv as _csv
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
            csv_path = os.path.join(run_dir, 'training_log.csv')
            csv_header = ['epoch', 'train_loss', 'tr_loss', 'rot_loss', 'tor_loss', 'rank_loss',
                          'val_loss', 'val_tr', 'val_rot', 'val_tor', 'val_rank_loss',
                          'valinf_min_rmsds_lt2', 'valinf_min_rmsds_lt5', 'lr', 'timestamp']
            csv_exists = os.path.exists(csv_path)
            csv_file = open(csv_path, 'a', newline='')
            csv_writer = _csv.writer(csv_file)
            if not csv_exists:
                csv_writer.writerow(csv_header)
                csv_file.flush()
            try:
                tb_writer = _SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
            except Exception:
                tb_writer = None
        except Exception as e:
            print('Warning: failed to initialize logging:', e)

    print("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0: print("Run name: ", args.run_name)

        if args.scheduler == 'layer_linear_warmup' and (epoch+1) % args.warmup_dur == 0:
            step = (epoch+1) // args.warmup_dur
            if step < args.num_conv_layers + 2:
                print("New unfreezing step")
                optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=step, scheduler_mode=scheduler_mode)
            elif step == args.num_conv_layers + 2:
                print("Unfreezing all parameters")
                optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=step, scheduler_mode=scheduler_mode)
                ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
        elif args.scheduler == 'linear_warmup' and epoch == args.warmup_dur:
            print("Moving to plateu scheduler")
            optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=1, scheduler_mode=scheduler_mode,
                                                               optimizer=optimizer)

        logs = {}
        loss_fn = functools.partial(
            loss_function,
            rank_weight=args.rank_weight,
            rank_k=args.rank_k,
            rank_sigma=args.rank_sigma,
            rank_alpha_tr=args.rank_alpha_tr,
            rank_alpha_rot=args.rank_alpha_rot,
        )
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            t_to_sigma,
            loss_fn,
            ema_weights if epoch > freeze_params else None,
            grad_accum_steps=args.grad_accum_steps,
        )
        print("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   rank {:.4f}   sc {:.4f}  lr {:.4f}"
              .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                      train_losses['tor_loss'], train_losses['rank_loss'], train_losses['sidechain_loss'], optimizer.param_groups[0]['lr']))

        if epoch > freeze_params:
            ema_weights.store(model.parameters())
            if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
        val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, args.test_sigma_intervals)
        print("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   rank {:.4f}   sc {:.4f}"
              .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss'], val_losses['rank_loss'], val_losses['sidechain_loss']))

        if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
            inf_dataset = [val_loader.dataset.get(i) for i in range(min(args.num_inference_complexes, val_loader.dataset.__len__()))]
            inf_metrics = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
            print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
            logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        if args.double_val and args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
            inf_dataset = [val_dataset2.get(i) for i in range(min(args.num_inference_complexes, val_dataset2.__len__()))]
            inf_metrics2 = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
            print("Epoch {}: Val inference on second validation rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics2['rmsds_lt2'], inf_metrics2['rmsds_lt5'], inf_metrics2['min_rmsds_lt2'], inf_metrics2['min_rmsds_lt5']))
            logs.update({'valinf2_' + k: v for k, v in inf_metrics2.items()}, step=epoch + 1)
            logs.update({'valinfcomb_' + k: (v + inf_metrics[k])/2 for k, v in inf_metrics2.items()}, step=epoch + 1)

        if args.train_inference_freq != None and (epoch + 1) % args.train_inference_freq == 0:
            inf_dataset = [train_loader.dataset.get(i) for i in range(min(min(args.num_inference_complexes, 300), train_loader.dataset.__len__()))]
            inf_metrics = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
            print("Epoch {}: Train inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
            logs.update({'traininf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        if epoch > freeze_params:
            if not args.use_ema: ema_weights.copy_to(model.parameters())
            # model may be wrapped in a DataParallel-like module (model.module) or not; be robust to either
            ema_state_dict = copy.deepcopy(getattr(model, 'module', model).state_dict())
            ema_weights.restore(model.parameters())

        if args.wandb:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch + 1)

        # Be robust whether the model is wrapped (has attribute 'module') or not.
        state_dict = getattr(model, 'module', model).state_dict()
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))

        if args.inference_secondary_metric is not None and args.inference_secondary_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_secondary_metric] <= best_val_secondary_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_secondary_metric] >= best_val_secondary_value):
            best_val_secondary_value = logs[args.inference_secondary_metric]
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_secondary_epoch_model.pt'))

        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

        if args.save_model_freq is not None and (epoch + 1) % args.save_model_freq == 0:
            shutil.copyfile(os.path.join(run_dir, 'best_model.pt'),
                            os.path.join(run_dir, f'epoch{epoch+1}_best_model.pt'))

        if scheduler:
            if epoch < freeze_params or (args.scheduler == 'linear_warmup' and epoch < args.warmup_dur):
                scheduler.step()
            elif args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

        # Optional per-epoch logging: write CSV row and TensorBoard scalars when enabled
        if csv_writer is not None:
            try:
                current_lr = optimizer.param_groups[0]['lr'] if optimizer is not None else None
                valinf_min_rmsds_lt2 = logs.get('valinf_min_rmsds_lt2', None)
                valinf_min_rmsds_lt5 = logs.get('valinf_min_rmsds_lt5', None)

                csv_writer.writerow([
                    epoch,
                    train_losses.get('loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('tr_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('rot_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('tor_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_loss') if isinstance(train_losses, dict) else None,
                    val_losses.get('loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('tr_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('rot_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('tor_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_loss') if isinstance(val_losses, dict) else None,
                    valinf_min_rmsds_lt2,
                    valinf_min_rmsds_lt5,
                    current_lr,
                    time.time()
                ])
                csv_file.flush()

                if tb_writer is not None:
                    if isinstance(train_losses, dict) and train_losses.get('loss') is not None:
                        tb_writer.add_scalar('train/loss', train_losses.get('loss'), epoch)
                    if isinstance(val_losses, dict) and val_losses.get('loss') is not None:
                        tb_writer.add_scalar('val/loss', val_losses.get('loss'), epoch)
                    if valinf_min_rmsds_lt2 is not None:
                        tb_writer.add_scalar('val/inference_min_rmsds_lt2', valinf_min_rmsds_lt2, epoch)
                    if current_lr is not None:
                        tb_writer.add_scalar('train/lr', current_lr, epoch)
                    if isinstance(train_losses, dict):
                        for k in ['tr_loss', 'rot_loss', 'tor_loss', 'rank_loss']:
                            if train_losses.get(k) is not None:
                                tb_writer.add_scalar(f'train/{k}', train_losses.get(k), epoch)
                    if isinstance(val_losses, dict):
                        for k in ['tr_loss', 'rot_loss', 'tor_loss', 'rank_loss']:
                            if val_losses.get(k) is not None:
                                tb_writer.add_scalar(f'val/{k}', val_losses.get(k), epoch)
                    tb_writer.flush()
            except Exception as e:
                print('Warning: failed to write epoch logs', e)

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


def main_function():
    args = parse_train_args()
    # Load config from file if provided. Support argparse FileType or a string path.
    if getattr(args, 'config', None):
        # args.config may be an open file (argparse.FileType) or a path string
        if hasattr(args.config, 'read'):
            config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
            config_name = getattr(args.config, 'name', None)
        else:
            # treat as path-like
            with open(str(args.config), 'r') as cf:
                config_dict = yaml.load(cf, Loader=yaml.FullLoader)
            config_name = str(args.config)
        arg_dict = args.__dict__
        for key, value in (config_dict or {}).items():
            if isinstance(value, list) and key in arg_dict and isinstance(arg_dict[key], list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = config_name
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if args.wandb:
        wandb.init(
            entity='',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )

    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader, val_dataset2 = construct_loader(args, t_to_sigma, device)
    
    model = get_model(args, device, t_to_sigma=t_to_sigma)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)

    if args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/{args.restart_ckpt}.pt', map_location=torch.device('cpu'))
            if args.restart_lr is not None: dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(dict['optimizer'])
            getattr(model, 'module', model).load_state_dict(dict['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(dict['ema_weights'], device=device)
            print("Restarting from epoch", dict['epoch'])
        except Exception as e:
            print("Exception", e)
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            getattr(model, 'module', model).load_state_dict(dict, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")
    elif args.pretrain_dir:
        dict = torch.load(f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt', map_location=torch.device('cpu'))
        getattr(model, 'module', model).load_state_dict(dict, strict=True)
        print("Using pretrained model", f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt')

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    if args.wandb:
        wandb.log({'numel': numel})

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset2)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
