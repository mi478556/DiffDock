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


def _nested_tensors_are_finite(obj):
    if torch.is_tensor(obj):
        return (not obj.is_floating_point()) or torch.isfinite(obj).all().item()
    if isinstance(obj, dict):
        return all(_nested_tensors_are_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_nested_tensors_are_finite(v) for v in obj)
    return True


def _save_finite_state_dict(state_dict, path, label):
    if not _nested_tensors_are_finite(state_dict):
        print(f"Skipping {label} checkpoint save because model state contains non-finite values")
        return False
    torch.save(state_dict, path)
    return True


def _clamp_optional(value, low=None, high=None):
    if low is not None:
        value = max(value, low)
    if high is not None:
        value = min(value, high)
    return value


def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset2):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_val_secondary_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    current_rank_oracle_rot_weight = float(getattr(args, 'rank_oracle_rot_weight', 0.0))
    rank_oracle_rot_ema_rank_loss = getattr(args, '_rank_oracle_rot_ema_rank_loss', None)
    rank_oracle_rot_ema_oracle_loss = getattr(args, '_rank_oracle_rot_ema_oracle_loss', None)
    restart_epoch_offset = int(getattr(args, '_restart_epoch_offset', 0))

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
            csv_header = [
                'epoch',
                'train_score_loss', 'train_total_loss', 'train_tr_loss', 'train_rot_loss', 'train_tor_loss',
                'train_rank_loss', 'train_rank_contribution', 'train_rank_gate_mean',
                'train_rank_teacher_loss', 'train_rank_teacher_contribution', 'train_rank_teacher_tr_loss',
                'train_rank_teacher_rot_loss', 'train_rank_teacher_active_mean',
                'train_rank_teacher_tr_cos', 'train_rank_teacher_rot_cos',
                'train_rank_oracle_rot_loss', 'train_rank_oracle_rot_contribution',
                'train_rank_oracle_rot_active_mean', 'train_rank_oracle_rot_cos',
                'train_rank_oracle_rot_delta', 'train_rank_oracle_rot_energy_drop',
                'val_score_loss', 'val_total_loss', 'val_tr_loss', 'val_rot_loss', 'val_tor_loss',
                'val_rank_loss', 'val_rank_contribution', 'val_rank_gate_mean',
                'val_rank_teacher_loss', 'val_rank_teacher_contribution', 'val_rank_teacher_tr_loss',
                'val_rank_teacher_rot_loss', 'val_rank_teacher_active_mean',
                'val_rank_teacher_tr_cos', 'val_rank_teacher_rot_cos',
                'val_rank_oracle_rot_loss', 'val_rank_oracle_rot_contribution',
                'val_rank_oracle_rot_active_mean', 'val_rank_oracle_rot_cos',
                'val_rank_oracle_rot_delta', 'val_rank_oracle_rot_energy_drop',
                'valinf_rmsds_lt2', 'valinf_rmsds_lt5', 'valinf_min_rmsds_lt2', 'valinf_min_rmsds_lt5',
                'valinf2_rmsds_lt2', 'valinf2_rmsds_lt5', 'valinf2_min_rmsds_lt2', 'valinf2_min_rmsds_lt5',
                'valinfcomb_rmsds_lt2', 'valinfcomb_rmsds_lt5',
                'valinfcomb_min_rmsds_lt2', 'valinfcomb_min_rmsds_lt5',
                'rank_oracle_rot_weight_effective',
                'rank_oracle_rot_dynamic_target_weight',
                'rank_oracle_rot_dynamic_ema_rank_loss',
                'rank_oracle_rot_dynamic_ema_oracle_loss',
                'lr', 'rank_weight', 'rank_teacher_weight', 'rank_oracle_rot_weight', 'timestamp',
            ]
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
        epoch_num = epoch + restart_epoch_offset
        if epoch_num % 5 == 0: print("Run name: ", args.run_name)
        epoch_rank_oracle_rot_weight = current_rank_oracle_rot_weight

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
        train_loss_fn = functools.partial(
            loss_function,
            tr_weight=1.0,
            rot_weight=1.0,
            tor_weight=1.0,
            no_torsion=args.no_torsion,
            rank_weight=args.rank_weight,
            rank_mode=args.rank_mode,
            rank_k=args.rank_k,
            rank_sigma=args.rank_sigma,
            rank_alpha_tr=args.rank_alpha_tr,
            rank_alpha_rot=args.rank_alpha_rot,
            rank_ensemble_samples=args.rank_ensemble_samples,
            rank_ensemble_tr_std=args.rank_ensemble_tr_std,
            rank_ensemble_rot_std=args.rank_ensemble_rot_std,
            rank_sigma_gate_cutoff=args.rank_sigma_gate_cutoff,
            rank_gate_type=args.rank_gate_type,
            rank_soft_gate_cutoff=args.rank_soft_gate_cutoff,
            rank_soft_gate_temp=args.rank_soft_gate_temp,
            rank_prune_eps=args.rank_prune_eps,
            rank_prune_sigma_cutoff=args.rank_prune_sigma_cutoff,
            rank_teacher_weight=args.rank_teacher_weight,
            rank_teacher_tr_weight=args.rank_teacher_tr_weight,
            rank_teacher_rot_weight=args.rank_teacher_rot_weight,
            rank_teacher_min_tr_norm=args.rank_teacher_min_tr_norm,
            rank_teacher_min_rot_norm=args.rank_teacher_min_rot_norm,
            rank_teacher_use_rot_sign_flip=args.rank_teacher_use_rot_sign_flip,
            rank_teacher_mode=getattr(args, 'rank_teacher_mode', None),
            rank_oracle_rot_weight=epoch_rank_oracle_rot_weight,
            rank_oracle_rot_mode=getattr(args, 'rank_oracle_rot_mode', None),
            rank_oracle_rot_probe_eps=args.rank_oracle_rot_probe_eps,
            rank_oracle_rot_sigma_min=args.rank_oracle_rot_sigma_min,
            rank_oracle_rot_sigma_max=args.rank_oracle_rot_sigma_max,
            rank_oracle_rot_min_delta=args.rank_oracle_rot_min_delta,
            rank_oracle_rot_min_cos=args.rank_oracle_rot_min_cos,
            rank_oracle_rot_min_energy_drop=args.rank_oracle_rot_min_energy_drop,
            rank_oracle_rot_require_energy_drop=args.rank_oracle_rot_require_energy_drop,
            rank_teacher_pred_norm_eps=args.rank_teacher_pred_norm_eps,
        )
        val_rank_weight = args.rank_weight if getattr(args, 'val_rank_weight', None) is None else args.val_rank_weight
        val_rank_teacher_weight = (
            args.rank_teacher_weight
            if getattr(args, 'val_rank_teacher_weight', None) is None
            else args.val_rank_teacher_weight
        )
        val_rank_oracle_rot_weight = (
            epoch_rank_oracle_rot_weight
            if getattr(args, 'val_rank_oracle_rot_weight', None) is None
            else args.val_rank_oracle_rot_weight
        )
        val_loss_fn = functools.partial(
            loss_function,
            tr_weight=1.0,
            rot_weight=1.0,
            tor_weight=1.0,
            no_torsion=args.no_torsion,
            rank_weight=val_rank_weight,
            rank_mode=args.rank_mode,
            rank_k=args.rank_k,
            rank_sigma=args.rank_sigma,
            rank_alpha_tr=args.rank_alpha_tr,
            rank_alpha_rot=args.rank_alpha_rot,
            rank_ensemble_samples=args.rank_ensemble_samples,
            rank_ensemble_tr_std=args.rank_ensemble_tr_std,
            rank_ensemble_rot_std=args.rank_ensemble_rot_std,
            rank_sigma_gate_cutoff=args.rank_sigma_gate_cutoff,
            rank_gate_type=args.rank_gate_type,
            rank_soft_gate_cutoff=args.rank_soft_gate_cutoff,
            rank_soft_gate_temp=args.rank_soft_gate_temp,
            rank_prune_eps=args.rank_prune_eps,
            rank_prune_sigma_cutoff=args.rank_prune_sigma_cutoff,
            rank_teacher_weight=val_rank_teacher_weight,
            rank_teacher_tr_weight=args.rank_teacher_tr_weight,
            rank_teacher_rot_weight=args.rank_teacher_rot_weight,
            rank_teacher_min_tr_norm=args.rank_teacher_min_tr_norm,
            rank_teacher_min_rot_norm=args.rank_teacher_min_rot_norm,
            rank_teacher_use_rot_sign_flip=args.rank_teacher_use_rot_sign_flip,
            rank_teacher_mode=getattr(args, 'rank_teacher_mode', None),
            rank_oracle_rot_weight=val_rank_oracle_rot_weight,
            rank_oracle_rot_mode=getattr(args, 'rank_oracle_rot_mode', None),
            rank_oracle_rot_probe_eps=args.rank_oracle_rot_probe_eps,
            rank_oracle_rot_sigma_min=args.rank_oracle_rot_sigma_min,
            rank_oracle_rot_sigma_max=args.rank_oracle_rot_sigma_max,
            rank_oracle_rot_min_delta=args.rank_oracle_rot_min_delta,
            rank_oracle_rot_min_cos=args.rank_oracle_rot_min_cos,
            rank_oracle_rot_min_energy_drop=args.rank_oracle_rot_min_energy_drop,
            rank_oracle_rot_require_energy_drop=args.rank_oracle_rot_require_energy_drop,
            rank_teacher_pred_norm_eps=args.rank_teacher_pred_norm_eps,
        )
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            t_to_sigma,
            train_loss_fn,
            ema_weights if epoch > freeze_params else None,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
        )
        print("Epoch {}: Training score_loss {:.4f}  total_loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   rank {:.4f}   rank_gate {:.4f}  lr {:.4f}"
              .format(epoch_num, train_losses['score_loss'], train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                      train_losses['tor_loss'], train_losses['rank_loss'], train_losses.get('rank_gate_mean', 1.0), optimizer.param_groups[0]['lr']))
        print("Epoch {}: Training teacher {:.4f}  teacher_tr {:.4f}  teacher_rot {:.4f}  teacher_active {:.4f}  teacher_tr_cos {:.4f}  teacher_rot_cos {:.4f}"
              .format(epoch_num, train_losses.get('rank_teacher_loss', 0.0), train_losses.get('rank_teacher_tr_loss', 0.0),
                      train_losses.get('rank_teacher_rot_loss', 0.0), train_losses.get('rank_teacher_active_mean', 0.0),
                      train_losses.get('rank_teacher_tr_cos', 0.0), train_losses.get('rank_teacher_rot_cos', 0.0)))
        print("Epoch {}: Training oracle_rot {:.4f}  active {:.4f}  target_cos {:.4f}  delta {:.4f}  energy_drop {:.4f}"
              .format(epoch_num, train_losses.get('rank_oracle_rot_loss', 0.0), train_losses.get('rank_oracle_rot_active_mean', 0.0),
                      train_losses.get('rank_oracle_rot_cos', 0.0), train_losses.get('rank_oracle_rot_delta', 0.0),
                      train_losses.get('rank_oracle_rot_energy_drop', 0.0)))

        dynamic_target_weight = None
        if getattr(args, 'rank_oracle_rot_dynamic_weight', False):
            ema_beta = float(getattr(args, 'rank_oracle_rot_dynamic_ema_beta', 0.9))
            ema_beta = min(max(ema_beta, 0.0), 0.999999)
            current_rank_loss = float(train_losses.get('rank_loss', 0.0) or 0.0)
            current_oracle_loss = float(train_losses.get('rank_oracle_rot_loss', 0.0) or 0.0)
            if rank_oracle_rot_ema_rank_loss is None:
                rank_oracle_rot_ema_rank_loss = current_rank_loss
            else:
                rank_oracle_rot_ema_rank_loss = (
                    ema_beta * rank_oracle_rot_ema_rank_loss + (1.0 - ema_beta) * current_rank_loss
                )
            if rank_oracle_rot_ema_oracle_loss is None:
                rank_oracle_rot_ema_oracle_loss = current_oracle_loss
            else:
                rank_oracle_rot_ema_oracle_loss = (
                    ema_beta * rank_oracle_rot_ema_oracle_loss + (1.0 - ema_beta) * current_oracle_loss
                )

            oracle_eps = max(float(getattr(args, 'rank_oracle_rot_dynamic_eps', 1e-6)), 1e-12)
            dynamic_target_weight = (
                float(args.rank_oracle_rot_dynamic_target_ratio) * rank_oracle_rot_ema_rank_loss /
                max(rank_oracle_rot_ema_oracle_loss, oracle_eps)
            )
            dynamic_target_weight = _clamp_optional(
                dynamic_target_weight,
                getattr(args, 'rank_oracle_rot_dynamic_weight_min', None),
                getattr(args, 'rank_oracle_rot_dynamic_weight_max', None),
            )
            max_rel_change = getattr(args, 'rank_oracle_rot_dynamic_max_rel_change', None)
            if max_rel_change is not None and current_rank_oracle_rot_weight > 0.0:
                max_rel_change = max(float(max_rel_change), 0.0)
                dynamic_target_weight = _clamp_optional(
                    dynamic_target_weight,
                    current_rank_oracle_rot_weight * (1.0 - max_rel_change),
                    current_rank_oracle_rot_weight * (1.0 + max_rel_change),
                )

            print(
                "Epoch {}: Dynamic oracle weight {:.4f} -> {:.4f}  target_ratio {:.4f}  ema_rank {:.4f}  ema_oracle {:.4f}".format(
                    epoch_num,
                    epoch_rank_oracle_rot_weight,
                    dynamic_target_weight,
                    float(args.rank_oracle_rot_dynamic_target_ratio),
                    rank_oracle_rot_ema_rank_loss,
                    rank_oracle_rot_ema_oracle_loss,
                )
            )
            current_rank_oracle_rot_weight = dynamic_target_weight

        if epoch > freeze_params:
            ema_weights.store(model.parameters())
            if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
        val_losses = test_epoch(model, val_loader, device, t_to_sigma, val_loss_fn, args.test_sigma_intervals)
        print("Epoch {}: Validation score_loss {:.4f}  total_loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   rank {:.4f}   rank_gate {:.4f}"
              .format(epoch_num, val_losses['score_loss'], val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss'], val_losses['rank_loss'], val_losses.get('rank_gate_mean', 1.0)))
        print("Epoch {}: Validation teacher {:.4f}  teacher_tr {:.4f}  teacher_rot {:.4f}  teacher_active {:.4f}  teacher_tr_cos {:.4f}  teacher_rot_cos {:.4f}"
              .format(epoch_num, val_losses.get('rank_teacher_loss', 0.0), val_losses.get('rank_teacher_tr_loss', 0.0),
                      val_losses.get('rank_teacher_rot_loss', 0.0), val_losses.get('rank_teacher_active_mean', 0.0),
                      val_losses.get('rank_teacher_tr_cos', 0.0), val_losses.get('rank_teacher_rot_cos', 0.0)))
        print("Epoch {}: Validation oracle_rot {:.4f}  active {:.4f}  target_cos {:.4f}  delta {:.4f}  energy_drop {:.4f}"
              .format(epoch_num, val_losses.get('rank_oracle_rot_loss', 0.0), val_losses.get('rank_oracle_rot_active_mean', 0.0),
                      val_losses.get('rank_oracle_rot_cos', 0.0), val_losses.get('rank_oracle_rot_delta', 0.0),
                      val_losses.get('rank_oracle_rot_energy_drop', 0.0)))
        val_selection_loss = val_losses.get('score_loss', val_losses['loss'])

        if args.val_inference_freq != None and (epoch_num + 1) % args.val_inference_freq == 0:
            inf_dataset = [val_loader.dataset.get(i) for i in range(min(args.num_inference_complexes, val_loader.dataset.__len__()))]
            inf_metrics = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
            print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch_num, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
            logs.update({'valinf_' + k: v for k, v in inf_metrics.items()})
            logs['step'] = epoch_num + 1

        if args.double_val and args.val_inference_freq != None and (epoch_num + 1) % args.val_inference_freq == 0:
            inf_dataset = [val_dataset2.get(i) for i in range(min(args.num_inference_complexes, val_dataset2.__len__()))]
            inf_metrics2 = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
            print("Epoch {}: Val inference on second validation rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch_num, inf_metrics2['rmsds_lt2'], inf_metrics2['rmsds_lt5'], inf_metrics2['min_rmsds_lt2'], inf_metrics2['min_rmsds_lt5']))
            logs.update({'valinf2_' + k: v for k, v in inf_metrics2.items()})
            logs.update({'valinfcomb_' + k: (v + inf_metrics[k])/2 for k, v in inf_metrics2.items()})
            logs['step'] = epoch_num + 1

        if args.train_inference_freq != None and (epoch_num + 1) % args.train_inference_freq == 0:
            inf_dataset = [train_loader.dataset.get(i) for i in range(min(min(args.num_inference_complexes, 300), train_loader.dataset.__len__()))]
            inf_metrics = inference_epoch_fix(model, inf_dataset, device, t_to_sigma, args)
            print("Epoch {}: Train inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch_num, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
            logs.update({'traininf_' + k: v for k, v in inf_metrics.items()})
            logs['step'] = epoch_num + 1

        if epoch > freeze_params:
            if not args.use_ema: ema_weights.copy_to(model.parameters())
            # model may be wrapped in a DataParallel-like module (model.module) or not; be robust to either
            ema_state_dict = copy.deepcopy(getattr(model, 'module', model).state_dict())
            ema_weights.restore(model.parameters())

        if args.wandb:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch_num + 1)

        # Be robust whether the model is wrapped (has attribute 'module') or not.
        # Deep-copy before saving so a later bad optimizer step cannot poison a
        # previously selected best checkpoint through live tensor references.
        state_dict = copy.deepcopy(getattr(model, 'module', model).state_dict())
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            saved_best_inference = _save_finite_state_dict(
                state_dict,
                os.path.join(run_dir, 'best_inference_epoch_model.pt'),
                'best inference',
            )
            if saved_best_inference:
                best_val_inference_value = logs[args.inference_earlystop_metric]
                best_val_inference_epoch = epoch_num
            if saved_best_inference and epoch > freeze_params:
                _save_finite_state_dict(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'), 'best EMA inference')

        if args.inference_secondary_metric is not None and args.inference_secondary_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_secondary_metric] <= best_val_secondary_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_secondary_metric] >= best_val_secondary_value):
            if epoch > freeze_params:
                saved_secondary = _save_finite_state_dict(
                    ema_state_dict,
                    os.path.join(run_dir, 'best_ema_secondary_epoch_model.pt'),
                    'best EMA secondary',
                )
                if saved_secondary:
                    best_val_secondary_value = logs[args.inference_secondary_metric]

        if val_selection_loss <= best_val_loss:
            saved_best_validation = _save_finite_state_dict(
                state_dict,
                os.path.join(run_dir, 'best_model.pt'),
                'best validation',
            )
            if saved_best_validation:
                best_val_loss = val_selection_loss
                best_epoch = epoch_num
            if saved_best_validation and epoch > freeze_params:
                _save_finite_state_dict(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'), 'best EMA validation')

        if args.save_model_freq is not None and (epoch_num + 1) % args.save_model_freq == 0:
            shutil.copyfile(os.path.join(run_dir, 'best_model.pt'),
                            os.path.join(run_dir, f'epoch{epoch_num+1}_best_model.pt'))

        if scheduler:
            if epoch < freeze_params or (args.scheduler == 'linear_warmup' and epoch < args.warmup_dur):
                scheduler.step()
            elif args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_selection_loss)

        last_checkpoint = {
            'epoch': epoch_num,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
            'rank_oracle_rot_weight': current_rank_oracle_rot_weight,
            'rank_oracle_rot_ema_rank_loss': rank_oracle_rot_ema_rank_loss,
            'rank_oracle_rot_ema_oracle_loss': rank_oracle_rot_ema_oracle_loss,
        }
        if _nested_tensors_are_finite(last_checkpoint):
            torch.save(last_checkpoint, os.path.join(run_dir, 'last_model.pt'))
        else:
            print("Skipping last_model checkpoint save because checkpoint state contains non-finite values")

        # Optional per-epoch logging: write CSV row and TensorBoard scalars when enabled
        if csv_writer is not None:
            try:
                current_lr = optimizer.param_groups[0]['lr'] if optimizer is not None else None
                train_rank_contribution = (
                    args.rank_weight * train_losses.get('rank_loss')
                    if isinstance(train_losses, dict) and train_losses.get('rank_loss') is not None else None
                )
                val_rank_contribution = (
                    val_rank_weight * val_losses.get('rank_loss')
                    if isinstance(val_losses, dict) and val_losses.get('rank_loss') is not None else None
                )
                train_rank_teacher_contribution = (
                    args.rank_teacher_weight * train_losses.get('rank_teacher_loss')
                    if isinstance(train_losses, dict) and train_losses.get('rank_teacher_loss') is not None else None
                )
                val_rank_teacher_contribution = (
                    val_rank_teacher_weight * val_losses.get('rank_teacher_loss')
                    if isinstance(val_losses, dict) and val_losses.get('rank_teacher_loss') is not None else None
                )
                train_rank_oracle_rot_contribution = (
                    epoch_rank_oracle_rot_weight * train_losses.get('rank_oracle_rot_loss')
                    if isinstance(train_losses, dict) and train_losses.get('rank_oracle_rot_loss') is not None else None
                )
                val_rank_oracle_rot_contribution = (
                    val_rank_oracle_rot_weight * val_losses.get('rank_oracle_rot_loss')
                    if isinstance(val_losses, dict) and val_losses.get('rank_oracle_rot_loss') is not None else None
                )

                csv_writer.writerow([
                    epoch_num,
                    train_losses.get('score_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('tr_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('rot_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('tor_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_loss') if isinstance(train_losses, dict) else None,
                    train_rank_contribution,
                    train_losses.get('rank_gate_mean') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_teacher_loss') if isinstance(train_losses, dict) else None,
                    train_rank_teacher_contribution,
                    train_losses.get('rank_teacher_tr_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_teacher_rot_loss') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_teacher_active_mean') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_teacher_tr_cos') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_teacher_rot_cos') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_oracle_rot_loss') if isinstance(train_losses, dict) else None,
                    train_rank_oracle_rot_contribution,
                    train_losses.get('rank_oracle_rot_active_mean') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_oracle_rot_cos') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_oracle_rot_delta') if isinstance(train_losses, dict) else None,
                    train_losses.get('rank_oracle_rot_energy_drop') if isinstance(train_losses, dict) else None,
                    val_losses.get('score_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('tr_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('rot_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('tor_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_loss') if isinstance(val_losses, dict) else None,
                    val_rank_contribution,
                    val_losses.get('rank_gate_mean') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_teacher_loss') if isinstance(val_losses, dict) else None,
                    val_rank_teacher_contribution,
                    val_losses.get('rank_teacher_tr_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_teacher_rot_loss') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_teacher_active_mean') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_teacher_tr_cos') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_teacher_rot_cos') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_oracle_rot_loss') if isinstance(val_losses, dict) else None,
                    val_rank_oracle_rot_contribution,
                    val_losses.get('rank_oracle_rot_active_mean') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_oracle_rot_cos') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_oracle_rot_delta') if isinstance(val_losses, dict) else None,
                    val_losses.get('rank_oracle_rot_energy_drop') if isinstance(val_losses, dict) else None,
                    logs.get('valinf_rmsds_lt2'),
                    logs.get('valinf_rmsds_lt5'),
                    logs.get('valinf_min_rmsds_lt2'),
                    logs.get('valinf_min_rmsds_lt5'),
                    logs.get('valinf2_rmsds_lt2'),
                    logs.get('valinf2_rmsds_lt5'),
                    logs.get('valinf2_min_rmsds_lt2'),
                    logs.get('valinf2_min_rmsds_lt5'),
                    logs.get('valinfcomb_rmsds_lt2'),
                    logs.get('valinfcomb_rmsds_lt5'),
                    logs.get('valinfcomb_min_rmsds_lt2'),
                    logs.get('valinfcomb_min_rmsds_lt5'),
                    epoch_rank_oracle_rot_weight,
                    dynamic_target_weight,
                    rank_oracle_rot_ema_rank_loss,
                    rank_oracle_rot_ema_oracle_loss,
                    current_lr,
                    args.rank_weight,
                    args.rank_teacher_weight,
                    epoch_rank_oracle_rot_weight,
                    time.time()
                ])
                csv_file.flush()

                if tb_writer is not None:
                    if isinstance(train_losses, dict) and train_losses.get('loss') is not None:
                        tb_writer.add_scalar('train/total_loss', train_losses.get('loss'), epoch_num)
                    if isinstance(train_losses, dict) and train_losses.get('score_loss') is not None:
                        tb_writer.add_scalar('train/score_loss', train_losses.get('score_loss'), epoch_num)
                    if isinstance(val_losses, dict) and val_losses.get('loss') is not None:
                        tb_writer.add_scalar('val/total_loss', val_losses.get('loss'), epoch_num)
                    if isinstance(val_losses, dict) and val_losses.get('score_loss') is not None:
                        tb_writer.add_scalar('val/score_loss', val_losses.get('score_loss'), epoch_num)
                    if train_rank_contribution is not None:
                        tb_writer.add_scalar('train/rank_contribution', train_rank_contribution, epoch_num)
                    if val_rank_contribution is not None:
                        tb_writer.add_scalar('val/rank_contribution', val_rank_contribution, epoch_num)
                    if train_rank_teacher_contribution is not None:
                        tb_writer.add_scalar('train/rank_teacher_contribution', train_rank_teacher_contribution, epoch_num)
                    if val_rank_teacher_contribution is not None:
                        tb_writer.add_scalar('val/rank_teacher_contribution', val_rank_teacher_contribution, epoch_num)
                    if train_rank_oracle_rot_contribution is not None:
                        tb_writer.add_scalar('train/rank_oracle_rot_contribution', train_rank_oracle_rot_contribution, epoch_num)
                    if val_rank_oracle_rot_contribution is not None:
                        tb_writer.add_scalar('val/rank_oracle_rot_contribution', val_rank_oracle_rot_contribution, epoch_num)
                    for metric_key, metric_value in logs.items():
                        if metric_key.startswith(('valinf_', 'valinf2_', 'valinfcomb_', 'traininf_')):
                            tb_writer.add_scalar(metric_key.replace('_', '/', 1), metric_value, epoch_num)
                    if current_lr is not None:
                        tb_writer.add_scalar('train/lr', current_lr, epoch_num)
                    tb_writer.add_scalar('train/rank_weight', args.rank_weight, epoch_num)
                    tb_writer.add_scalar('train/rank_teacher_weight', args.rank_teacher_weight, epoch_num)
                    tb_writer.add_scalar('train/rank_oracle_rot_weight', epoch_rank_oracle_rot_weight, epoch_num)
                    if dynamic_target_weight is not None:
                        tb_writer.add_scalar('train/rank_oracle_rot_dynamic_target_weight', dynamic_target_weight, epoch_num)
                    if rank_oracle_rot_ema_rank_loss is not None:
                        tb_writer.add_scalar('train/rank_oracle_rot_dynamic_ema_rank_loss', rank_oracle_rot_ema_rank_loss, epoch_num)
                    if rank_oracle_rot_ema_oracle_loss is not None:
                        tb_writer.add_scalar('train/rank_oracle_rot_dynamic_ema_oracle_loss', rank_oracle_rot_ema_oracle_loss, epoch_num)
                    if isinstance(train_losses, dict) and train_losses.get('rank_gate_mean') is not None:
                        tb_writer.add_scalar('train/rank_gate_mean', train_losses.get('rank_gate_mean'), epoch_num)
                    if isinstance(val_losses, dict) and val_losses.get('rank_gate_mean') is not None:
                        tb_writer.add_scalar('val/rank_gate_mean', val_losses.get('rank_gate_mean'), epoch_num)
                    if isinstance(train_losses, dict):
                        for k in ['tr_loss', 'rot_loss', 'tor_loss', 'rank_loss', 'rank_teacher_loss',
                                  'rank_teacher_tr_loss', 'rank_teacher_rot_loss', 'rank_teacher_active_mean',
                                  'rank_teacher_tr_cos', 'rank_teacher_rot_cos',
                                  'rank_oracle_rot_loss', 'rank_oracle_rot_active_mean', 'rank_oracle_rot_cos',
                                  'rank_oracle_rot_delta', 'rank_oracle_rot_energy_drop']:
                            if train_losses.get(k) is not None:
                                tb_writer.add_scalar(f'train/{k}', train_losses.get(k), epoch_num)
                    if isinstance(val_losses, dict):
                        for k in ['tr_loss', 'rot_loss', 'tor_loss', 'rank_loss', 'rank_teacher_loss',
                                  'rank_teacher_tr_loss', 'rank_teacher_rot_loss', 'rank_teacher_active_mean',
                                  'rank_teacher_tr_cos', 'rank_teacher_rot_cos',
                                  'rank_oracle_rot_loss', 'rank_oracle_rot_active_mean', 'rank_oracle_rot_cos',
                                  'rank_oracle_rot_delta', 'rank_oracle_rot_energy_drop']:
                            if val_losses.get(k) is not None:
                                tb_writer.add_scalar(f'val/{k}', val_losses.get(k), epoch_num)
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
            if 'rank_oracle_rot_weight' in dict:
                args.rank_oracle_rot_weight = dict['rank_oracle_rot_weight']
            args._rank_oracle_rot_ema_rank_loss = dict.get('rank_oracle_rot_ema_rank_loss')
            args._rank_oracle_rot_ema_oracle_loss = dict.get('rank_oracle_rot_ema_oracle_loss')
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
