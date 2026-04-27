import copy
import os
import numpy as np
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import torch

from confidence.dataset import ListDataset
from utils import so3, torus
from utils.molecules_utils import get_symmetry_rmsd
from utils.sampling import randomize_position, sampling
from utils.diffusion_utils import get_t_schedule


# utils/training.py  — replace only loss_function with this version
import numpy as np
import torch
from utils import so3, torus
from utils.rank_regularizer import (
    detached_rank_se3_teacher,
    low_rank_contact_loss,
    target_conditioned_rank_oracle_rot_teacher,
)




def _bf16_forward_context(device):
    """Use bf16 for the model forward on CUDA while keeping loss math in fp32."""
    return torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda')


def _as_fp32_prediction(pred):
    return pred.float() if isinstance(pred, torch.Tensor) else pred


def _uses_pyg_dataparallel(model):
    return hasattr(model, 'module')


def _batch_names(data):
    if isinstance(data, (list, tuple)):
        names = []
        for idx, graph in enumerate(data):
            name = getattr(graph, 'name', None)
            if isinstance(name, str):
                names.append(name)
            elif name is None:
                names.append(f'graph_{idx}')
            else:
                try:
                    names.append(str(name))
                except Exception:
                    names.append(f'graph_{idx}')
        return names

    names = getattr(data, 'name', None)
    if names is None:
        return ['unknown']
    if isinstance(names, str):
        return [names]
    try:
        return [str(n) for n in list(names)]
    except Exception:
        return [str(names)]


def _cosine_teacher_loss(pred, teacher, gate, min_teacher_norm, contribution_mask=None, pred_norm_eps=1e-3):
    pred = pred.float()
    teacher = teacher.detach().float()

    pred_norm = torch.linalg.vector_norm(pred, dim=-1)
    teacher_norm = torch.linalg.vector_norm(teacher, dim=-1)
    denom = pred_norm.clamp_min(float(pred_norm_eps)) * teacher_norm.clamp_min(1e-12)

    cosine = (pred * teacher).sum(dim=-1) / denom
    valid = teacher_norm > float(min_teacher_norm)

    if contribution_mask is None:
        contribution_mask = torch.ones_like(valid, dtype=torch.bool)
    else:
        contribution_mask = contribution_mask.bool()

    effective = valid & contribution_mask

    per_graph = torch.where(effective, 1.0 - cosine, torch.zeros_like(cosine))
    weighted = gate.to(dtype=per_graph.dtype) * per_graph

    return weighted, valid.to(dtype=per_graph.dtype), effective.to(dtype=per_graph.dtype), cosine.detach()


def _bad_loss_tensor(x):
    return torch.is_tensor(x) and not torch.isfinite(x).all()


def loss_function(
    tr_pred, rot_pred, tor_pred, sidechain_pred, data, t_to_sigma, device,
    tr_weight: float = 1.0, rot_weight: float = 1.0, tor_weight: float = 1.0,
    apply_mean: bool = True, no_torsion: bool = False,
    # new knobs for the low-rank regularizer:
    rank_weight: float = 0.0,        # set >0 to turn it on
    rank_mode: str = 'single',
    rank_k: int = 8,
    rank_sigma: float = 2.0,
    rank_alpha_tr: float = 0.25,
    rank_alpha_rot: float = 0.25,
    rank_ensemble_samples: int = 4,
    rank_ensemble_tr_std: float = 0.5,
    rank_ensemble_rot_std: float = 0.15,
    rank_sigma_gate_cutoff: float = 2.0,
    rank_gate_type: str = 'hard',
    rank_soft_gate_cutoff: float = 2.0,
    rank_soft_gate_temp: float = 0.5,
    rank_prune_eps: float = 0.02,
    rank_prune_sigma_cutoff: float = None,
    rank_teacher_weight: float = 0.0,
    rank_teacher_tr_weight: float = 1.0,
    rank_teacher_rot_weight: float = 1.0,
    rank_teacher_min_tr_norm: float = 1e-6,
    rank_teacher_min_rot_norm: float = 1e-6,
    rank_teacher_use_rot_sign_flip: bool = False,
    rank_teacher_mode: str = None,
    rank_oracle_rot_weight: float = 0.0,
    rank_oracle_rot_mode: str = None,
    rank_oracle_rot_probe_eps: float = 0.05,
    rank_oracle_rot_sigma_min: float = 2.0,
    rank_oracle_rot_sigma_max: float = 3.0,
    rank_oracle_rot_min_delta: float = 0.05,
    rank_oracle_rot_min_cos: float = 0.0,
    rank_oracle_rot_min_energy_drop: float = 0.0,
    rank_teacher_pred_norm_eps: float = 1e-3,
):
    # Gather complex times for the batch (support both list and Batched inputs)
    if isinstance(data, list):
        complex_t_tr = torch.cat([d.complex_t['tr'] for d in data])
        complex_t_rot = torch.cat([d.complex_t['rot'] for d in data])
        complex_t_tor = torch.cat([d.complex_t['tor'] for d in data])
    else:
        complex_t_tr = data.complex_t['tr']
        complex_t_rot = data.complex_t['rot']
        complex_t_tor = data.complex_t['tor']

    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(complex_t_tr, complex_t_rot, complex_t_tor)

    # Ensure sigmas/live tensors live on the same device as model outputs
    pred_device = tr_pred.device if isinstance(tr_pred, torch.Tensor) else device
    if isinstance(tr_sigma, torch.Tensor):
        tr_sigma = tr_sigma.to(pred_device)
    if isinstance(rot_sigma, torch.Tensor):
        rot_sigma = rot_sigma.to(pred_device)
    if isinstance(tor_sigma, torch.Tensor):
        tor_sigma = tor_sigma.to(pred_device)

    mean_dims = (0, 1) if apply_mean else 1

    # translation
    if isinstance(data, list):
        tr_score = torch.cat([d.tr_score for d in data], dim=0).to(pred_device)
    else:
        tr_score = data.tr_score.to(pred_device)
    tr_sigma_vec = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred - tr_score) ** 2 * tr_sigma_vec ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma_vec ** 2).mean(dim=mean_dims).detach()

    # rotation
    if isinstance(data, list):
        rot_score = torch.cat([d.rot_score for d in data], dim=0).to(pred_device)
    else:
        rot_score = data.rot_score.to(pred_device)
    rot_score_norm = so3.score_norm(rot_sigma).unsqueeze(-1).to(pred_device)
    rot_loss = (((rot_pred - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion
    if not no_torsion:
        if isinstance(data, list):
            edge_tor_arr = np.concatenate([d.tor_sigma_edge for d in data])
            edge_tor_sigma = torch.from_numpy(edge_tor_arr).to(pred_device)
            tor_score = torch.cat([d.tor_score for d in data], dim=0).to(pred_device)
        else:
            # When using a Batched object, some fields (like tor_sigma_edge) may still be lists
            if isinstance(data.tor_sigma_edge, list):
                edge_tor_arr = np.concatenate(data.tor_sigma_edge)
                edge_tor_sigma = torch.from_numpy(edge_tor_arr).to(pred_device)
            else:
                edge_tor_sigma = torch.from_numpy(data.tor_sigma_edge).to(pred_device)
            tor_score = data.tor_score.to(pred_device)

        tor_score_norm2 = torus.score_norm(edge_tor_sigma).float().to(pred_device)
        tor_loss = ((tor_pred - tor_score) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float, device=pred_device)
            tor_base_loss = tor_base_loss.mean() * torch.ones(1, dtype=torch.float, device=pred_device)
        else:
            if isinstance(data, list):
                index = torch.cat([torch.ones(int(d['ligand'].edge_mask.sum())) * i for i, d in enumerate(data)]).long().to(pred_device)
                num_graphs = len(data)
            else:
                index = data['ligand'].batch[data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
                num_graphs = data.num_graphs
            t_l = torch.zeros(num_graphs, device=pred_device)
            t_b_l = torch.zeros(num_graphs, device=pred_device)
            c = torch.zeros(num_graphs, device=pred_device)
            c.index_add_(0, index, torch.ones(tor_loss.shape, device=pred_device))
            c = c + 1e-4
            t_l.index_add_(0, index, tor_loss.to(pred_device))
            t_b_l.index_add_(0, index, tor_base_loss.to(pred_device))
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        tor_loss = torch.zeros(1, dtype=torch.float, device=pred_device)
        tor_base_loss = torch.zeros(1, dtype=torch.float, device=pred_device) if apply_mean else torch.zeros(len(rot_loss), dtype=torch.float, device=pred_device)

    # Base score-matching loss. For apples-to-apples comparisons with the
    # original run, train.py passes tr/rot/tor weights as 1.0.
    score_loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight
    loss = score_loss

    rank_loss = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_gate_mean = torch.ones(1, dtype=torch.float, device=pred_device)
    rank_teacher_loss = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_teacher_tr_loss = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_teacher_rot_loss = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_teacher_active_mean = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_teacher_tr_cos = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_teacher_rot_cos = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_oracle_rot_loss = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_oracle_rot_active_mean = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_oracle_rot_cos = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_oracle_rot_delta = torch.zeros(1, dtype=torch.float, device=pred_device)
    rank_oracle_rot_energy_drop = torch.zeros(1, dtype=torch.float, device=pred_device)

    # add our low-rank contact loss, computed from a one-step denoised pose
    if rank_weight > 0.0 or rank_teacher_weight > 0.0 or rank_oracle_rot_weight > 0.0:
        if not isinstance(tr_sigma, torch.Tensor):
            raise ValueError("rank loss requires tr_sigma")
        sigma = tr_sigma.to(pred_device).reshape(-1)
        if sigma.numel() == 1:
            if isinstance(data, list):
                num_graphs = len(data)
            else:
                num_graphs = int(data.num_graphs)
            sigma = sigma.expand(num_graphs)

        rank_gate_type = str(rank_gate_type).lower()
        gate_soft = torch.sigmoid(
            (float(rank_soft_gate_cutoff) - sigma) / float(rank_soft_gate_temp)
        ).to(dtype=tr_pred.dtype)
        rank_gate_mean = gate_soft.mean() if apply_mean else gate_soft
        teacher_active_graph_mask = None
        if rank_prune_sigma_cutoff is not None:
            teacher_active_graph_mask = sigma <= float(rank_prune_sigma_cutoff)
        else:
            teacher_active_graph_mask = gate_soft > float(rank_prune_eps)

        if rank_weight > 0.0 and rank_gate_type == 'hard':
            # Historical hard-gate objective: prune inactive graphs and reduce
            # over active graphs inside the regularizer.
            sigma_cutoff = float(rank_sigma_gate_cutoff)
            rank_loss = low_rank_contact_loss(
                data=data,
                tr_pred=tr_pred.to(pred_device),
                rot_pred=rot_pred.to(pred_device),
                tr_sigma=tr_sigma.to(pred_device),
                sigma_cutoff=sigma_cutoff,
                rank_mode=str(rank_mode),
                rank_k=int(rank_k),
                gaussian_sigma=float(rank_sigma),
                alpha_tr=float(rank_alpha_tr),
                alpha_rot=float(rank_alpha_rot),
                use_receptor_atoms=True,
                ensemble_samples=int(rank_ensemble_samples),
                ensemble_translation_std=float(rank_ensemble_tr_std),
                ensemble_rotation_std=float(rank_ensemble_rot_std),
                return_per_graph=not apply_mean,
            )
            if not apply_mean and sigma.numel() == 1 and isinstance(rank_loss, torch.Tensor) and rank_loss.numel() > 1:
                sigma = sigma.expand_as(rank_loss)
            active = (sigma <= sigma_cutoff).to(torch.float32)
            rank_gate_mean = active.mean() if apply_mean else active
        elif rank_weight > 0.0 and rank_gate_type in ('soft', 'soft_prune'):
            # Soft objective: always apply the sigmoid gate outside the
            # regularizer and keep the full-batch mean reduction.
            active_graph_mask = None
            if rank_gate_type == 'soft_prune':
                active_graph_mask = teacher_active_graph_mask

            per_graph_rank_loss = low_rank_contact_loss(
                data=data,
                tr_pred=tr_pred.to(pred_device),
                rot_pred=rot_pred.to(pred_device),
                tr_sigma=tr_sigma.to(pred_device),
                sigma_cutoff=None,
                active_graph_mask=active_graph_mask,
                rank_mode=str(rank_mode),
                rank_k=int(rank_k),
                gaussian_sigma=float(rank_sigma),
                alpha_tr=float(rank_alpha_tr),
                alpha_rot=float(rank_alpha_rot),
                use_receptor_atoms=True,
                ensemble_samples=int(rank_ensemble_samples),
                ensemble_translation_std=float(rank_ensemble_tr_std),
                ensemble_rotation_std=float(rank_ensemble_rot_std),
                return_per_graph=True,
            )
            rank_loss_per_graph = gate_soft * per_graph_rank_loss
            rank_loss = rank_loss_per_graph.mean() if apply_mean else rank_loss_per_graph
            rank_gate_mean = gate_soft.mean() if apply_mean else gate_soft
        else:
            if rank_weight > 0.0:
                raise ValueError(f"Unknown rank_gate_type: {rank_gate_type}")

        if rank_weight > 0.0:
            loss = loss + rank_weight * rank_loss

        if rank_teacher_weight > 0.0:
            # choose teacher rank mode from explicit teacher knob, default to 'single'
            teacher_mode = rank_teacher_mode if rank_teacher_mode is not None else 'single'
            teacher_tr, teacher_rot, _ = detached_rank_se3_teacher(
                data=data,
                tr_pred=tr_pred.to(pred_device),
                rot_pred=rot_pred.to(pred_device),
                tr_sigma=tr_sigma.to(pred_device),
                active_graph_mask=teacher_active_graph_mask,
                rank_mode=teacher_mode,
                rank_k=int(rank_k),
                gaussian_sigma=float(rank_sigma),
                alpha_tr=float(rank_alpha_tr),
                alpha_rot=float(rank_alpha_rot),
                use_receptor_atoms=True,
                ensemble_samples=int(rank_ensemble_samples),
                ensemble_translation_std=float(rank_ensemble_tr_std),
                ensemble_rotation_std=float(rank_ensemble_rot_std),
            )
            if bool(rank_teacher_use_rot_sign_flip):
                teacher_rot = -teacher_rot
            tr_teacher_per_graph, tr_teacher_valid, tr_teacher_effective, tr_cos = _cosine_teacher_loss(
                tr_pred.to(pred_device),
                teacher_tr,
                gate_soft,
                rank_teacher_min_tr_norm,
                contribution_mask=teacher_active_graph_mask,
                pred_norm_eps=rank_teacher_pred_norm_eps,
            )
            rot_teacher_per_graph, rot_teacher_valid, rot_teacher_effective, rot_cos = _cosine_teacher_loss(
                rot_pred.to(pred_device),
                teacher_rot,
                gate_soft,
                rank_teacher_min_rot_norm,
                contribution_mask=teacher_active_graph_mask,
                pred_norm_eps=rank_teacher_pred_norm_eps,
            )
            rank_teacher_tr_loss = tr_teacher_per_graph.mean() if apply_mean else tr_teacher_per_graph
            rank_teacher_rot_loss = rot_teacher_per_graph.mean() if apply_mean else rot_teacher_per_graph
            rank_teacher_loss = (
                float(rank_teacher_tr_weight) * rank_teacher_tr_loss
                + float(rank_teacher_rot_weight) * rank_teacher_rot_loss
            )
            # active graphs where the teacher both passes the norm threshold
            # and is inside the prune/gate-based contribution mask
            teacher_active = torch.maximum(tr_teacher_effective, rot_teacher_effective)
            rank_teacher_active_mean = teacher_active.mean() if apply_mean else teacher_active

            # Mask logged cosines to reflect only contributing graphs
            if apply_mean:
                if tr_teacher_effective.sum().item() > 0:
                    rank_teacher_tr_cos = tr_cos[tr_teacher_effective.bool()].mean()
                else:
                    rank_teacher_tr_cos = torch.zeros(1, dtype=torch.float, device=pred_device)

                if rot_teacher_effective.sum().item() > 0:
                    rank_teacher_rot_cos = rot_cos[rot_teacher_effective.bool()].mean()
                else:
                    rank_teacher_rot_cos = torch.zeros(1, dtype=torch.float, device=pred_device)
            else:
                rank_teacher_tr_cos = torch.where(tr_teacher_effective.bool(), tr_cos, torch.zeros_like(tr_cos))
                rank_teacher_rot_cos = torch.where(rot_teacher_effective.bool(), rot_cos, torch.zeros_like(rot_cos))
            loss = loss + float(rank_teacher_weight) * rank_teacher_loss

        if rank_oracle_rot_weight > 0.0:
            oracle_mode = rank_oracle_rot_mode if rank_oracle_rot_mode is not None else str(rank_mode)
            oracle_rot, oracle_active, oracle_cos, oracle_delta, oracle_energy_drop = target_conditioned_rank_oracle_rot_teacher(
                data=data,
                tr_pred=tr_pred.to(pred_device),
                rot_pred=rot_pred.to(pred_device),
                rot_score=rot_score.to(pred_device),
                tr_sigma=tr_sigma.to(pred_device),
                active_graph_mask=teacher_active_graph_mask,
                rank_mode=oracle_mode,
                rank_k=int(rank_k),
                gaussian_sigma=float(rank_sigma),
                alpha_tr=float(rank_alpha_tr),
                alpha_rot=float(rank_alpha_rot),
                probe_eps=float(rank_oracle_rot_probe_eps),
                sigma_min=float(rank_oracle_rot_sigma_min),
                sigma_max=float(rank_oracle_rot_sigma_max),
                min_delta=float(rank_oracle_rot_min_delta),
                min_cos=float(rank_oracle_rot_min_cos),
                min_energy_drop=float(rank_oracle_rot_min_energy_drop),
                use_receptor_atoms=True,
                ensemble_samples=int(rank_ensemble_samples),
                ensemble_translation_std=float(rank_ensemble_tr_std),
                ensemble_rotation_std=float(rank_ensemble_rot_std),
            )
            oracle_per_graph, _, oracle_effective, oracle_pred_cos = _cosine_teacher_loss(
                rot_pred.to(pred_device),
                oracle_rot,
                gate_soft,
                rank_teacher_min_rot_norm,
                contribution_mask=oracle_active,
                pred_norm_eps=rank_teacher_pred_norm_eps,
            )
            rank_oracle_rot_loss = oracle_per_graph.mean() if apply_mean else oracle_per_graph
            rank_oracle_rot_active_mean = oracle_effective.mean() if apply_mean else oracle_effective
            if apply_mean:
                if oracle_effective.sum().item() > 0:
                    active_bool = oracle_effective.bool()
                    rank_oracle_rot_cos = oracle_cos[active_bool].mean()
                    rank_oracle_rot_delta = oracle_delta[active_bool].mean()
                    rank_oracle_rot_energy_drop = oracle_energy_drop[active_bool].mean()
                else:
                    rank_oracle_rot_cos = torch.zeros(1, dtype=torch.float, device=pred_device)
                    rank_oracle_rot_delta = torch.zeros(1, dtype=torch.float, device=pred_device)
                    rank_oracle_rot_energy_drop = torch.zeros(1, dtype=torch.float, device=pred_device)
            else:
                active_bool = oracle_effective.bool()
                rank_oracle_rot_cos = torch.where(active_bool, oracle_cos, torch.zeros_like(oracle_cos))
                rank_oracle_rot_delta = torch.where(active_bool, oracle_delta, torch.zeros_like(oracle_delta))
                rank_oracle_rot_energy_drop = torch.where(active_bool, oracle_energy_drop, torch.zeros_like(oracle_energy_drop))
            loss = loss + float(rank_oracle_rot_weight) * rank_oracle_rot_loss

    if not apply_mean:
        if rank_loss.numel() == 1 and score_loss.numel() > 1:
            rank_loss = rank_loss.expand_as(score_loss)
        if rank_gate_mean.numel() == 1 and score_loss.numel() > 1:
            rank_gate_mean = rank_gate_mean.expand_as(score_loss)
        if rank_teacher_loss.numel() == 1 and score_loss.numel() > 1:
            rank_teacher_loss = rank_teacher_loss.expand_as(score_loss)
        if rank_teacher_tr_loss.numel() == 1 and score_loss.numel() > 1:
            rank_teacher_tr_loss = rank_teacher_tr_loss.expand_as(score_loss)
        if rank_teacher_rot_loss.numel() == 1 and score_loss.numel() > 1:
            rank_teacher_rot_loss = rank_teacher_rot_loss.expand_as(score_loss)
        if rank_teacher_active_mean.numel() == 1 and score_loss.numel() > 1:
            rank_teacher_active_mean = rank_teacher_active_mean.expand_as(score_loss)
        if rank_teacher_tr_cos.numel() == 1 and score_loss.numel() > 1:
            rank_teacher_tr_cos = rank_teacher_tr_cos.expand_as(score_loss)
        if rank_teacher_rot_cos.numel() == 1 and score_loss.numel() > 1:
            rank_teacher_rot_cos = rank_teacher_rot_cos.expand_as(score_loss)
        if rank_oracle_rot_loss.numel() == 1 and score_loss.numel() > 1:
            rank_oracle_rot_loss = rank_oracle_rot_loss.expand_as(score_loss)
        if rank_oracle_rot_active_mean.numel() == 1 and score_loss.numel() > 1:
            rank_oracle_rot_active_mean = rank_oracle_rot_active_mean.expand_as(score_loss)
        if rank_oracle_rot_cos.numel() == 1 and score_loss.numel() > 1:
            rank_oracle_rot_cos = rank_oracle_rot_cos.expand_as(score_loss)
        if rank_oracle_rot_delta.numel() == 1 and score_loss.numel() > 1:
            rank_oracle_rot_delta = rank_oracle_rot_delta.expand_as(score_loss)
        if rank_oracle_rot_energy_drop.numel() == 1 and score_loss.numel() > 1:
            rank_oracle_rot_energy_drop = rank_oracle_rot_energy_drop.expand_as(score_loss)

    return (
        loss,
        score_loss.detach(),
        tr_loss.detach(),
        rot_loss.detach(),
        tor_loss.detach(),
        rank_loss.detach(),
        rank_gate_mean.detach(),
        rank_teacher_loss.detach(),
        rank_teacher_tr_loss.detach(),
        rank_teacher_rot_loss.detach(),
        rank_teacher_active_mean.detach(),
        rank_teacher_tr_cos.detach(),
        rank_teacher_rot_cos.detach(),
        rank_oracle_rot_loss.detach(),
        rank_oracle_rot_active_mean.detach(),
        rank_oracle_rot_cos.detach(),
        rank_oracle_rot_delta.detach(),
        rank_oracle_rot_energy_drop.detach(),
        tr_base_loss,
        rot_base_loss,
        tor_base_loss,
    )


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum().cpu() if self.unpooled_metrics else v.cpu()
        else:
            for type_idx, v in enumerate(vals):
                # Ensure tensors used for indexing/accumulation are on CPU to avoid device mismatch
                v_cpu = v.cpu()
                self.count[type_idx].index_add_(0, interval_idx[type_idx].cpu(), torch.ones(len(v_cpu)))
                if not torch.allclose(v_cpu, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx].cpu(), v_cpu)

    def summary(self):
        if self.intervals == 1:
            if self.count == 0:
                return {k: 0.0 for k in self.types}
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    cnt = self.count[type_idx][i]
                    if cnt == 0:
                        out['int' + str(i) + '_' + k] = 0.0
                    else:
                        out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / cnt).item()
            return out


def _has_nonfinite_grad(parameters):
    for p in parameters:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def _has_nonfinite_param(parameters):
    for p in parameters:
        if p.is_floating_point() and not torch.isfinite(p).all():
            return True
    return False


def _dist_max_int(value, device):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return int(value)
    tensor = torch.tensor(int(value), device=device, dtype=torch.int64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights, grad_accum_steps=1, max_grad_norm=None, ddp_loss_scale=1.0):
    model.train()
    meter = AverageMeter(['loss', 'score_loss', 'tr_loss', 'rot_loss', 'tor_loss', 'rank_loss', 'rank_gate_mean',
                          'rank_teacher_loss', 'rank_teacher_tr_loss', 'rank_teacher_rot_loss',
                          'rank_teacher_active_mean', 'rank_teacher_tr_cos', 'rank_teacher_rot_cos',
                          'rank_oracle_rot_loss', 'rank_oracle_rot_active_mean', 'rank_oracle_rot_cos',
                          'rank_oracle_rot_delta', 'rank_oracle_rot_energy_drop',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss'])
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, total=len(loader))
    postfix_interval = 10
    for batch_idx, data in enumerate(progress):
        local_skip = 0
        skip_reason = None
        loss_tuple = None
        loss = None
        score_loss_for_display = None

        if isinstance(data, list):
            single_batch = len(data) == 1
        else:
            single_batch = getattr(data, 'num_graphs', 1) == 1

        if single_batch:
            has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
            if has_bn:
                local_skip = 1
                skip_reason = "Skipping batch of size 1 since otherwise batchnorm would not work."

        if isinstance(data, list):
            if not _uses_pyg_dataparallel(model):
                data = Batch.from_data_list(data)
                data = data.to(device) if device.type == 'cuda' else data
        else:
            data = data.to(device) if device.type == 'cuda' else data

        try:
            if local_skip == 0:
                with _bf16_forward_context(device):
                    tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
                loss_tuple = loss_fn(
                    _as_fp32_prediction(tr_pred),
                    _as_fp32_prediction(rot_pred),
                    _as_fp32_prediction(tor_pred),
                    sidechain_pred,
                    data=data,
                    t_to_sigma=t_to_sigma,
                    device=device,
                )
                if loss_tuple is None:
                    local_skip = 1
                    skip_reason = "None loss tuple, skipping"
                else:
                    loss = loss_tuple[0]
                    if _bad_loss_tensor(loss):
                        names = _batch_names(data)
                        local_skip = 1
                        skip_reason = f"Bad loss, skipping batch with complexes {names}"
        except RuntimeError as e:
            if 'out of memory' in str(e):
                local_skip = 1
                skip_reason = '| WARNING: ran out of memory, skipping batch'
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
            elif 'Input mismatch' in str(e):
                local_skip = 1
                skip_reason = '| WARNING: weird torch_cluster error, skipping batch'
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
            else:
                print(e)
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                continue

        global_skip = _dist_max_int(local_skip, device)
        if global_skip:
            if skip_reason is not None:
                print(skip_reason)
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            continue

        score_loss_for_display = loss_tuple[1].detach()
        scaled_loss = (loss * float(ddp_loss_scale)) / grad_accum_steps
        scaled_loss.backward()
        accum_count += 1

        if accum_count == grad_accum_steps:
            local_step_skip = 0
            step_skip_reason = None
            if max_grad_norm is not None and float(max_grad_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                if not torch.isfinite(grad_norm):
                    names = _batch_names(data)
                    local_step_skip = 1
                    step_skip_reason = f"Bad gradient norm, skipping optimizer step with complexes {names}"
            elif _has_nonfinite_grad(model.parameters()):
                names = _batch_names(data)
                local_step_skip = 1
                step_skip_reason = f"Bad gradients, skipping optimizer step with complexes {names}"

            global_step_skip = _dist_max_int(local_step_skip, device)
            if global_step_skip:
                if step_skip_reason is not None:
                    print(step_skip_reason)
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                continue

            optimizer.step()
            if _has_nonfinite_param(model.parameters()):
                names = _batch_names(data)
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Stopping training: optimizer step produced non-finite model parameters after complexes {names}"
                )
            optimizer.zero_grad(set_to_none=True)
            if ema_weights is not None:
                ema_weights.update(model.parameters())
            accum_count = 0

        meter.add([loss.detach().cpu(), *loss_tuple[1:]])
        if not score_loss_for_display.dim() == 0:
            score_loss_for_display = score_loss_for_display.mean()
        if batch_idx % postfix_interval == 0 or batch_idx + 1 == len(loader):
            progress.set_postfix(score_loss=f'{score_loss_for_display.item():.4f}')

    if accum_count > 0:
        local_step_skip = 0
        if max_grad_norm is not None and float(max_grad_norm) > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            if not torch.isfinite(grad_norm):
                local_step_skip = 1
        elif _has_nonfinite_grad(model.parameters()):
            local_step_skip = 1

        global_step_skip = _dist_max_int(local_step_skip, device)
        if global_step_skip:
            optimizer.zero_grad(set_to_none=True)
            return meter.summary()

        optimizer.step()
        if _has_nonfinite_param(model.parameters()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Stopping training: final optimizer step produced non-finite model parameters")
        optimizer.zero_grad(set_to_none=True)
        if ema_weights is not None:
            ema_weights.update(model.parameters())

    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'score_loss', 'tr_loss', 'rot_loss', 'tor_loss', 'rank_loss', 'rank_gate_mean',
                          'rank_teacher_loss', 'rank_teacher_tr_loss', 'rank_teacher_rot_loss',
                          'rank_teacher_active_mean', 'rank_teacher_tr_cos', 'rank_teacher_rot_cos',
                          'rank_oracle_rot_loss', 'rank_oracle_rot_active_mean', 'rank_oracle_rot_cos',
                          'rank_oracle_rot_delta', 'rank_oracle_rot_energy_drop',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'score_loss', 'tr_loss', 'rot_loss', 'tor_loss', 'rank_loss', 'rank_gate_mean',
             'rank_teacher_loss', 'rank_teacher_tr_loss', 'rank_teacher_rot_loss',
             'rank_teacher_active_mean', 'rank_teacher_tr_cos', 'rank_teacher_rot_cos',
             'rank_oracle_rot_loss', 'rank_oracle_rot_active_mean', 'rank_oracle_rot_cos',
             'rank_oracle_rot_delta', 'rank_oracle_rot_energy_drop',
             'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
            unpooled_metrics=True, intervals=10)

    progress = tqdm(loader, total=len(loader))
    postfix_interval = 10
    for batch_idx, data in enumerate(progress):
        try:
            if isinstance(data, list):
                if not _uses_pyg_dataparallel(model):
                    data = Batch.from_data_list(data)
                    data = data.to(device) if device.type == 'cuda' else data
            else:
                data = data.to(device) if device.type == 'cuda' else data
            with torch.no_grad():
                with _bf16_forward_context(device):
                    tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(
                _as_fp32_prediction(tr_pred),
                _as_fp32_prediction(rot_pred),
                _as_fp32_prediction(tor_pred),
                sidechain_pred,
                data=data,
                t_to_sigma=t_to_sigma,
                apply_mean=False,
                device=device,
            )
            if loss_tuple is None: continue
            if any(_bad_loss_tensor(v) for v in loss_tuple):
                names = _batch_names(data)
                print("Bad validation loss, skipping batch with complexes", names)
                continue
            meter.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]])
            score_loss_for_display = loss_tuple[1].detach()
            if not score_loss_for_display.dim() == 0:
                score_loss_for_display = score_loss_for_display.mean()
            if batch_idx % postfix_interval == 0 or batch_idx + 1 == len(loader):
                progress.set_postfix(score_loss=f'{score_loss_for_display.item():.4f}')

            if test_sigma_intervals > 0:
                complex_t_tr = data.complex_t['tr']
                complex_t_rot = data.complex_t['rot']
                complex_t_tor = data.complex_t['tor']
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]],
                    [sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr,
                     sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tr, sigma_index_tr, sigma_index_rot,
                     sigma_index_rot, sigma_index_tr, sigma_index_rot, sigma_index_rot, sigma_index_tr,
                     sigma_index_tr, sigma_index_rot, sigma_index_tor])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    raise RuntimeError("DDP rank hit OOM during validation; aborting to avoid desynchronization") from e
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                print(e)
                continue

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch_fix(model, complex_graphs, device, t_to_sigma, args, return_counts=False, show_progress=True):
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds, min_rmsds = [], []

    for orig_complex_graph in tqdm(loader, disable=not show_progress):
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.inference_samples)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                # pass the underlying model whether or not it's wrapped in DataParallel
                predictions_list, confidences = sampling(data_list=data_list, model=getattr(model, 'module', model),
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args,
                                                         t_schedule=t_schedule)
            except Exception as e:
                failed_convergence_counter += 1
                if failed_convergence_counter > 5:
                    print('failed 5 times - skipping the complex')
                    break
                print("Exception while running inference on complex:", e)
        if failed_convergence_counter > 5:
            rmsds.extend([100] * args.inference_samples)
            min_rmsds.append(100)
            continue

        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph[
                                                         'ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
        # if len(orig_complex_graph['ligand'].orig_pos.shape) == 3:
        #     orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        if len(orig_complex_graph['ligand'].orig_pos.shape) == 2:
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[None, :, :]
        try:
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[:, filterHs] - orig_complex_graph.original_center.cpu().numpy()
        except Exception as e:
            print("problem with orig_pos which is of shape:", orig_complex_graph['ligand'].orig_pos.shape, e)
            continue
        mol = RemoveAllHs(orig_complex_graph.mol[0])
        complex_rmsds = []
        for i in range(len(orig_ligand_pos)):
            try:
                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])
            except Exception as e:
                print("Using non corrected RMSD because of the error:", e)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
            complex_rmsds.append(rmsd)
        complex_rmsds = np.asarray(complex_rmsds)
        rmsd = np.min(complex_rmsds, axis=0)
        
        rmsds.extend([r for r in rmsd])
        min_rmsds.append(rmsd.min(axis=0))

    rmsds = np.array(rmsds)
    min_rmsds = np.array(min_rmsds)
    if return_counts:
        return {
            'rmsds_lt2_count': int((rmsds < 2).sum()),
            'rmsds_lt5_count': int((rmsds < 5).sum()),
            'rmsds_total': int(len(rmsds)),
            'min_rmsds_lt2_count': int((min_rmsds < 2).sum()),
            'min_rmsds_lt5_count': int((min_rmsds < 5).sum()),
            'min_rmsds_total': int(len(min_rmsds)),
        }
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
              'min_rmsds_lt2': (100 * (min_rmsds < 2).sum() / len(min_rmsds)),
              'min_rmsds_lt5': (100 * (min_rmsds < 5).sum() / len(min_rmsds)),}
    return losses
