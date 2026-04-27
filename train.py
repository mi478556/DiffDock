import copy
import functools
import math
import os
import shutil
from functools import partial

import wandb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml
import time
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, t_to_sigma_individual
from datasets.loader import construct_loader, construct_datasets, construct_loader_from_datasets
from datasets.shared_memory import share_dataset_tree_
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, inference_epoch_fix
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage
from torch.utils.data import Sampler


def _is_main_process(args):
    return int(getattr(args, 'rank', 0)) == 0


def _all_reduce_metric_dict(metrics, device):
    if not dist.is_initialized():
        return metrics
    reduced = {}
    for k, v in metrics.items():
        if v is None:
            reduced[k] = v
            continue
        tensor = torch.tensor(float(v), device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[k] = float(tensor.item() / dist.get_world_size())
    return reduced


def _all_reduce_inference_counts(counts, device):
    if not dist.is_initialized():
        return counts
    reduced = {}
    for k, v in counts.items():
        tensor = torch.tensor(float(v), device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[k] = float(tensor.item())
    return reduced


def _counts_to_inference_metrics(counts):
    rmsds_total = max(float(counts.get('rmsds_total', 0.0)), 1.0)
    min_rmsds_total = max(float(counts.get('min_rmsds_total', 0.0)), 1.0)
    return {
        'rmsds_lt2': 100.0 * float(counts.get('rmsds_lt2_count', 0.0)) / rmsds_total,
        'rmsds_lt5': 100.0 * float(counts.get('rmsds_lt5_count', 0.0)) / rmsds_total,
        'min_rmsds_lt2': 100.0 * float(counts.get('min_rmsds_lt2_count', 0.0)) / min_rmsds_total,
        'min_rmsds_lt5': 100.0 * float(counts.get('min_rmsds_lt5_count', 0.0)) / min_rmsds_total,
    }


def _distributed_inference_indices(total_size, rank, local_batches):
    local_batches = [int(x) for x in local_batches]
    global_batch_size = int(sum(local_batches))
    offset = int(sum(local_batches[:int(rank)]))
    local_batch_size = int(local_batches[int(rank)])
    indices = []
    for start in range(0, int(total_size), global_batch_size):
        chunk = list(range(start, min(start + global_batch_size, int(total_size))))
        indices.extend(chunk[offset:offset + local_batch_size])
    return indices


def _broadcast_scalar(value, device, src=0):
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(float(value), device=device)
    dist.broadcast(tensor, src=src)
    return float(tensor.item())


def _broadcast_optional_scalar(value, device, src=0):
    if not dist.is_initialized():
        return value
    has_value = torch.tensor(0 if value is None else 1, device=device, dtype=torch.int64)
    dist.broadcast(has_value, src=src)
    if int(has_value.item()) == 0:
        return None
    return _broadcast_scalar(0.0 if value is None else value, device, src=src)


def _parse_gpu_list(spec):
    if not spec:
        raise ValueError("--shared_ddp_gpus is required when --use_shared_ddp is set")
    return [int(x.strip()) for x in spec.split(',') if x.strip()]


def _parse_local_batches(spec, global_batch_size, world_size):
    if spec:
        local_batches = [int(x.strip()) for x in spec.split(',') if x.strip()]
        if len(local_batches) != world_size:
            raise ValueError(f"--shared_ddp_local_batches must provide exactly {world_size} entries")
        if sum(local_batches) != int(global_batch_size):
            raise ValueError(
                f"--shared_ddp_local_batches sums to {sum(local_batches)} but global batch_size is {global_batch_size}"
            )
        return local_batches

    base = int(global_batch_size) // int(world_size)
    remainder = int(global_batch_size) % int(world_size)
    return [base + (1 if i < remainder else 0) for i in range(world_size)]


class UnevenDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, local_batch_sizes, shuffle=True, drop_last=False, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.local_batch_sizes = [int(x) for x in local_batch_sizes]
        self.local_batch_size = int(self.local_batch_sizes[rank])
        self.global_batch_size = int(sum(self.local_batch_sizes))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _num_steps(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.global_batch_size
        return math.ceil(n / self.global_batch_size)

    def __len__(self):
        return self._num_steps() * self.local_batch_size

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        num_steps = self._num_steps()
        total_size = num_steps * self.global_batch_size
        if self.drop_last:
            indices = indices[:total_size]
        elif total_size > len(indices):
            indices = indices + indices[: total_size - len(indices)]

        offset = sum(self.local_batch_sizes[:self.rank])
        rank_indices = []
        for step_idx in range(num_steps):
            chunk = indices[step_idx * self.global_batch_size:(step_idx + 1) * self.global_batch_size]
            rank_indices.extend(chunk[offset:offset + self.local_batch_size])
        return iter(rank_indices)


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


def _load_training_state(args, model, optimizer, ema_weights, device):
    restart_epoch_offset = 0
    if args.restart_dir:
        try:
            checkpoint = torch.load(f'{args.restart_dir}/{args.restart_ckpt}.pt', map_location=torch.device('cpu'))
            if args.restart_lr is not None:
                checkpoint['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(checkpoint['optimizer'])
            getattr(model, 'module', model).load_state_dict(checkpoint['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(checkpoint['ema_weights'], device=device)
            if 'rank_oracle_rot_weight' in checkpoint:
                args.rank_oracle_rot_weight = checkpoint['rank_oracle_rot_weight']
            args._rank_oracle_rot_ema_rank_loss = checkpoint.get('rank_oracle_rot_ema_rank_loss')
            args._rank_oracle_rot_ema_oracle_loss = checkpoint.get('rank_oracle_rot_ema_oracle_loss')
            restart_epoch_offset = int(checkpoint['epoch']) + 1
            print("Restarting from epoch", checkpoint['epoch'])
        except Exception as e:
            print("Exception", e)
            checkpoint = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            getattr(model, 'module', model).load_state_dict(checkpoint, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")
    elif args.pretrain_dir:
        checkpoint = torch.load(f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt', map_location=torch.device('cpu'))
        getattr(model, 'module', model).load_state_dict(checkpoint, strict=True)
        print("Using pretrained model", f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt')
    args._restart_epoch_offset = restart_epoch_offset


def train_shared_ddp(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset, val_dataset2, device):
    rank = int(args.rank)
    is_main = rank == 0
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_val_secondary_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    current_rank_oracle_rot_weight = float(getattr(args, 'rank_oracle_rot_weight', 0.0))
    rank_oracle_rot_ema_rank_loss = getattr(args, '_rank_oracle_rot_ema_rank_loss', None)
    rank_oracle_rot_ema_oracle_loss = getattr(args, '_rank_oracle_rot_ema_oracle_loss', None)
    restart_epoch_offset = int(getattr(args, '_restart_epoch_offset', 0))

    os.makedirs(run_dir, exist_ok=True) if is_main else None
    if is_main:
        save_yaml_file(os.path.join(run_dir, 'model_parameters.yml'), args.__dict__)
        print("Starting shared-memory DDP training...")

    freeze_params = 0
    scheduler_mode = args.inference_earlystop_goal if args.val_inference_freq is not None else 'min'
    if args.scheduler == 'layer_linear_warmup':
        freeze_params = args.warmup_dur * (args.num_conv_layers + 2) - 1

    for epoch in range(args.n_epochs):
        epoch_num = epoch + restart_epoch_offset
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch_num)
        epoch_rank_oracle_rot_weight = current_rank_oracle_rot_weight
        logs = {}

        train_loss_fn = functools.partial(
            loss_function,
            tr_weight=1.0, rot_weight=1.0, tor_weight=1.0,
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
            rank_teacher_pred_norm_eps=args.rank_teacher_pred_norm_eps,
        )
        train_losses = train_epoch(
            model, train_loader, optimizer, device, t_to_sigma, train_loss_fn,
            ema_weights if epoch > freeze_params else None,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            ddp_loss_scale=float(getattr(args, '_shared_ddp_loss_scale', 1.0)),
        )
        train_losses = _all_reduce_metric_dict(train_losses, device)

        if is_main:
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
            logs.update({'train_' + k: v for k, v in train_losses.items()})

        dynamic_target_weight = None
        if getattr(args, 'rank_oracle_rot_dynamic_weight', False):
            ema_beta = float(getattr(args, 'rank_oracle_rot_dynamic_ema_beta', 0.9))
            ema_beta = min(max(ema_beta, 0.0), 0.999999)
            current_rank_loss = float(train_losses.get('rank_loss', 0.0) or 0.0)
            current_oracle_loss = float(train_losses.get('rank_oracle_rot_loss', 0.0) or 0.0)
            rank_oracle_rot_ema_rank_loss = current_rank_loss if rank_oracle_rot_ema_rank_loss is None else (ema_beta * rank_oracle_rot_ema_rank_loss + (1.0 - ema_beta) * current_rank_loss)
            rank_oracle_rot_ema_oracle_loss = current_oracle_loss if rank_oracle_rot_ema_oracle_loss is None else (ema_beta * rank_oracle_rot_ema_oracle_loss + (1.0 - ema_beta) * current_oracle_loss)
            oracle_eps = max(float(getattr(args, 'rank_oracle_rot_dynamic_eps', 1e-6)), 1e-12)
            dynamic_target_weight = float(args.rank_oracle_rot_dynamic_target_ratio) * rank_oracle_rot_ema_rank_loss / max(rank_oracle_rot_ema_oracle_loss, oracle_eps)
            dynamic_target_weight = _clamp_optional(dynamic_target_weight, getattr(args, 'rank_oracle_rot_dynamic_weight_min', None), getattr(args, 'rank_oracle_rot_dynamic_weight_max', None))
            max_rel_change = getattr(args, 'rank_oracle_rot_dynamic_max_rel_change', None)
            if max_rel_change is not None and current_rank_oracle_rot_weight > 0.0:
                max_rel_change = max(float(max_rel_change), 0.0)
                dynamic_target_weight = _clamp_optional(dynamic_target_weight, current_rank_oracle_rot_weight * (1.0 - max_rel_change), current_rank_oracle_rot_weight * (1.0 + max_rel_change))
            current_rank_oracle_rot_weight = dynamic_target_weight
            if is_main:
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

        if epoch > freeze_params:
            ema_weights.store(model.parameters())
            if args.use_ema:
                ema_weights.copy_to(model.parameters())

        if is_main:
            val_rank_weight = args.rank_weight if getattr(args, 'val_rank_weight', None) is None else args.val_rank_weight
            val_rank_teacher_weight = args.rank_teacher_weight if getattr(args, 'val_rank_teacher_weight', None) is None else args.val_rank_teacher_weight
            val_rank_oracle_rot_weight = epoch_rank_oracle_rot_weight if getattr(args, 'val_rank_oracle_rot_weight', None) is None else args.val_rank_oracle_rot_weight
            val_loss_fn = functools.partial(
                loss_function,
                tr_weight=1.0, rot_weight=1.0, tor_weight=1.0,
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
                rank_teacher_pred_norm_eps=args.rank_teacher_pred_norm_eps,
            )
            val_losses = test_epoch(model.module, val_loader, device, t_to_sigma, val_loss_fn, args.test_sigma_intervals)
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            val_selection_loss = val_losses.get('score_loss', val_losses['loss'])
            print("Epoch {}: Validation score_loss {:.4f}  total_loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   rank {:.4f}   rank_gate {:.4f}"
                  .format(epoch_num, val_losses['score_loss'], val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'],
                          val_losses['tor_loss'], val_losses['rank_loss'], val_losses.get('rank_gate_mean', 1.0)))
            print("Epoch {}: Validation teacher {:.4f}  teacher_tr {:.4f}  teacher_rot {:.4f}  teacher_active {:.4f}  teacher_tr_cos {:.4f}  teacher_rot_cos {:.4f}"
                  .format(epoch_num, val_losses.get('rank_teacher_loss', 0.0), val_losses.get('rank_teacher_tr_loss', 0.0),
                          val_losses.get('rank_teacher_rot_loss', 0.0), val_losses.get('rank_teacher_active_mean', 0.0),
                          val_losses.get('rank_teacher_tr_cos', 0.0), val_losses.get('rank_teacher_rot_cos', 0.0)))
            print("Epoch {}: Validation oracle_rot {:.4f}  active {:.4f}  target_cos {:.4f}  delta {:.4f}  energy_drop {:.4f}"
                  .format(epoch_num, val_losses.get('rank_oracle_rot_loss', 0.0), val_losses.get('rank_oracle_rot_active_mean', 0.0),
                          val_losses.get('rank_oracle_rot_cos', 0.0), val_losses.get('rank_oracle_rot_delta', 0.0),
                          val_losses.get('rank_oracle_rot_energy_drop', 0.0)))

        run_val_inference = args.val_inference_freq is not None and (epoch_num + 1) % args.val_inference_freq == 0
        if is_main and run_val_inference:
            print("Epoch {}: Running shared-DDP validation inference".format(epoch_num))
        if run_val_inference:
            total_inf = min(args.num_inference_complexes, val_dataset.__len__())
            local_indices = _distributed_inference_indices(total_inf, rank, args._shared_ddp_local_batches)
            inf_dataset = [val_dataset.get(i) for i in local_indices]
            inf_counts = inference_epoch_fix(
                model.module,
                inf_dataset,
                device,
                t_to_sigma,
                args,
                return_counts=True,
                show_progress=is_main,
            )
            inf_counts = _all_reduce_inference_counts(inf_counts, device)
            inf_metrics = _counts_to_inference_metrics(inf_counts)
            if is_main:
                print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                      .format(epoch_num, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
                logs.update({'valinf_' + k: v for k, v in inf_metrics.items()})
                logs['step'] = epoch_num + 1

            if is_main and args.double_val and val_dataset2 is not None:
                print("Epoch {}: Running shared-DDP second validation inference".format(epoch_num))
            if args.double_val and val_dataset2 is not None:
                total_inf2 = min(args.num_inference_complexes, val_dataset2.__len__())
                local_indices2 = _distributed_inference_indices(total_inf2, rank, args._shared_ddp_local_batches)
                inf_dataset2 = [val_dataset2.get(i) for i in local_indices2]
                inf_counts2 = inference_epoch_fix(
                    model.module,
                    inf_dataset2,
                    device,
                    t_to_sigma,
                    args,
                    return_counts=True,
                    show_progress=is_main,
                )
                inf_counts2 = _all_reduce_inference_counts(inf_counts2, device)
                inf_metrics2 = _counts_to_inference_metrics(inf_counts2)
            if is_main and args.double_val and val_dataset2 is not None:
                print("Epoch {}: Val inference on second validation rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                      .format(epoch_num, inf_metrics2['rmsds_lt2'], inf_metrics2['rmsds_lt5'], inf_metrics2['min_rmsds_lt2'], inf_metrics2['min_rmsds_lt5']))
                logs.update({'valinf2_' + k: v for k, v in inf_metrics2.items()})
                logs.update({'valinfcomb_' + k: (v + inf_metrics[k]) / 2 for k, v in inf_metrics2.items()})
                logs['step'] = epoch_num + 1

        if is_main:
            if args.inference_earlystop_metric in logs.keys() and \
                    (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                     args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
                state_dict = copy.deepcopy(model.module.state_dict())
                saved_best_inference = _save_finite_state_dict(
                    state_dict,
                    os.path.join(run_dir, 'best_inference_epoch_model.pt'),
                    'best inference',
                )
                if saved_best_inference:
                    best_val_inference_value = logs[args.inference_earlystop_metric]
                    best_val_inference_epoch = epoch_num
                if saved_best_inference and epoch > freeze_params:
                    ema_state_dict = copy.deepcopy(model.module.state_dict())
                    _save_finite_state_dict(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'), 'best EMA inference')

            if args.inference_secondary_metric is not None and args.inference_secondary_metric in logs.keys() and \
                    (args.inference_earlystop_goal == 'min' and logs[args.inference_secondary_metric] <= best_val_secondary_value or
                     args.inference_earlystop_goal == 'max' and logs[args.inference_secondary_metric] >= best_val_secondary_value):
                if epoch > freeze_params:
                    ema_state_dict = copy.deepcopy(model.module.state_dict())
                    saved_secondary = _save_finite_state_dict(
                        ema_state_dict,
                        os.path.join(run_dir, 'best_ema_secondary_epoch_model.pt'),
                        'best EMA secondary',
                    )
                    if saved_secondary:
                        best_val_secondary_value = logs[args.inference_secondary_metric]

            if val_selection_loss <= best_val_loss:
                state_dict = copy.deepcopy(model.module.state_dict())
                saved_best_validation = _save_finite_state_dict(state_dict, os.path.join(run_dir, 'best_model.pt'), 'best validation')
                if saved_best_validation:
                    best_val_loss = val_selection_loss
                    best_epoch = epoch_num
                if saved_best_validation and epoch > freeze_params:
                    ema_state_dict = copy.deepcopy(model.module.state_dict())
                    _save_finite_state_dict(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'), 'best EMA validation')
            last_checkpoint = {
                'epoch': epoch_num,
                'model': copy.deepcopy(model.module.state_dict()),
                'optimizer': optimizer.state_dict(),
                'ema_weights': ema_weights.state_dict(),
                'rank_oracle_rot_weight': current_rank_oracle_rot_weight,
                'rank_oracle_rot_ema_rank_loss': rank_oracle_rot_ema_rank_loss,
                'rank_oracle_rot_ema_oracle_loss': rank_oracle_rot_ema_oracle_loss,
            }
            if _nested_tensors_are_finite(last_checkpoint):
                torch.save(last_checkpoint, os.path.join(run_dir, 'last_model.pt'))
        else:
            val_selection_loss = 0.0

        val_selection_loss = _broadcast_scalar(val_selection_loss, device)
        current_rank_oracle_rot_weight = _broadcast_scalar(current_rank_oracle_rot_weight, device)
        rank_oracle_rot_ema_rank_loss = _broadcast_optional_scalar(rank_oracle_rot_ema_rank_loss, device)
        rank_oracle_rot_ema_oracle_loss = _broadcast_optional_scalar(rank_oracle_rot_ema_oracle_loss, device)

        if epoch > freeze_params and not args.use_ema:
            ema_weights.copy_to(model.parameters())
        if epoch > freeze_params:
            ema_weights.restore(model.parameters())

        if scheduler:
            if epoch < freeze_params or (args.scheduler == 'linear_warmup' and epoch < args.warmup_dur):
                scheduler.step()
            elif args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_selection_loss)
        dist.barrier()

    if is_main:
        print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
        print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


def _shared_ddp_worker(rank, args, gpu_ids, datasets, local_batches):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(args.shared_ddp_port)
    torch.cuda.set_device(gpu_ids[rank])
    dist.init_process_group('nccl', rank=rank, world_size=len(gpu_ids))
    device = torch.device(f'cuda:{gpu_ids[rank]}')
    args.rank = rank
    args.world_size = len(gpu_ids)
    args.device = device
    args.parallel = 1
    args._shared_ddp_local_batches = [int(x) for x in local_batches]
    args._shared_ddp_local_batch_size = int(local_batches[rank])
    args._shared_ddp_global_batch_size = int(sum(local_batches))
    args._shared_ddp_loss_scale = (
        float(len(gpu_ids)) * float(args._shared_ddp_local_batch_size) / float(args._shared_ddp_global_batch_size)
    )
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_dataset, val_dataset, val_dataset2 = datasets
    train_sampler = UnevenDistributedSampler(
        train_dataset,
        num_replicas=len(gpu_ids),
        rank=rank,
        local_batch_sizes=local_batches,
        shuffle=True,
        drop_last=args.dataloader_drop_last,
    )
    train_loader, _ = construct_loader_from_datasets(
        args,
        device,
        train_dataset,
        val_dataset,
        train_sampler=train_sampler,
        batch_size=args._shared_ddp_local_batch_size,
    )
    if rank == 0:
        _, val_loader = construct_loader_from_datasets(args, device, train_dataset, val_dataset)
    else:
        val_loader = None

    model = get_model(args, device, t_to_sigma=t_to_sigma)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
    _load_training_state(args, model, optimizer, ema_weights, device)
    model = DDP(
        model.to(device),
        device_ids=[gpu_ids[rank]],
        output_device=gpu_ids[rank],
        broadcast_buffers=True,
        find_unused_parameters=True,
    )

    run_dir = os.path.join(args.log_dir, args.run_name)
    train_shared_ddp(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset, val_dataset2, device)
    dist.destroy_process_group()


def launch_shared_ddp(args):
    if args.num_dataloader_workers != 0:
        print("Overriding num_dataloader_workers to 0 for shared-memory DDP")
        args.num_dataloader_workers = 0
    gpu_ids = _parse_gpu_list(args.shared_ddp_gpus)
    local_batches = _parse_local_batches(args.shared_ddp_local_batches, args.batch_size, len(gpu_ids))
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_dataset, val_dataset, val_dataset2 = construct_datasets(args, t_to_sigma)
    share_dataset_tree_(train_dataset)
    share_dataset_tree_(val_dataset)
    if val_dataset2 is not None and val_dataset2 is not val_dataset:
        share_dataset_tree_(val_dataset2)
    datasets = (train_dataset, val_dataset, val_dataset2)
    print(f"Shared-memory DDP local train batches: {local_batches} (global batch_size={args.batch_size})")
    mp.start_processes(
        _shared_ddp_worker,
        args=(args, gpu_ids, datasets, local_batches),
        nprocs=len(gpu_ids),
        start_method='spawn',
        join=True,
    )


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

    if getattr(args, 'use_shared_ddp', False):
        launch_shared_ddp(args)
        return

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
