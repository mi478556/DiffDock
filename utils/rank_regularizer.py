# utils/rank_regularizer.py
# Low-rank contact regularizer for DiffDock training.
# Works with both DataParallel (CUDA) list batches and single PyG Batch on CPU.

from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_scatter import scatter_max, scatter_mean

@torch.jit.script
def _expmap_so3(w: Tensor) -> Tensor:
    """
    Rodrigues' formula.
    w: [B, 3] axis-angle vector, magnitude is angle in radians.
    returns R: [B, 3, 3]
    """
    B = w.shape[0]
    theta = torch.linalg.norm(w, dim=1).unsqueeze(1).unsqueeze(2)  # [B,1,1]
    w_hat = torch.zeros(B, 3, 3, device=w.device, dtype=w.dtype)
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    w_hat[:, 0, 1] = -wz
    w_hat[:, 0, 2] =  wy
    w_hat[:, 1, 0] =  wz
    w_hat[:, 1, 2] = -wx
    w_hat[:, 2, 0] = -wy
    w_hat[:, 2, 1] =  wx

    I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, -1, -1)
    theta2 = (theta * theta).clamp_min(1e-12)
    A = torch.where(theta > 1e-8, torch.sin(theta) / theta, 1 - theta2 / 6 + theta2 * theta2 / 120)
    Bc = torch.where(theta > 1e-8, (1 - torch.cos(theta)) / theta2, 0.5 - theta2 / 24 + theta2 * theta2 / 720)
    return I + A * w_hat + Bc * (w_hat @ w_hat)

def _stack_graph_field(data, key0: str, key1: str) -> Tensor:
    """
    Returns concatenated tensor across graphs for fields like data['ligand'].pos
    Works for CUDA DataParallel lists and for single PyG Batch.
    """
    if isinstance(data, (list, tuple)):  # DataParallel data_list
        return torch.cat([d[key0].__dict__[key1] for d in data], dim=0)
    else:
        return data[key0].__dict__[key1]

def _make_batch_index(data, key0: str, n_graphs: Optional[int] = None) -> Tensor:
    """
    Returns per-node batch index for key0 graph.
    If running with DataParallel, rebuild from sizes.
    """
    if isinstance(data, (list, tuple)):
        bs = []
        for i, d in enumerate(data):
            n = d[key0].pos.size(0)
            bs.append(torch.full((n,), i, device=d[key0].pos.device, dtype=torch.long))
        return torch.cat(bs, dim=0)
    else:
        return data[key0].batch

def _num_graphs(data) -> int:
    if isinstance(data, (list, tuple)):
        return len(data)
    else:
        return int(data.num_graphs)

def _heavy_atom_mask(mol_x: Tensor) -> Tensor:
    """
    Heuristic heavy-atom mask used by DiffDock inference:
    treat first categorical channel as element index where 0 means hydrogen.
    If that is not available, fall back to all True.
    """
    try:
        return mol_x[:, 0] != 0
    except Exception:
        return torch.ones(mol_x.size(0), dtype=torch.bool, device=mol_x.device)

def _per_graph_center(positions: Tensor, batch_idx: Tensor, num_graphs: int) -> Tensor:
    return scatter_mean(positions, batch_idx, dim=0, dim_size=num_graphs)

def _apply_one_step_se3(
    lig_pos: Tensor,
    lig_batch: Tensor,
    tr_vec: Tensor,       # [B,3]
    rot_vec: Tensor,      # [B,3]
    step_tr: Tensor,      # [B] translation step scale
    step_rot: Tensor      # [B] rotation step scale
) -> Tensor:
    """
    One tiny reverse step that depends on model outputs,
    so the contact loss backpropagates into the score network.
    """
    B = tr_vec.size(0)
    centers = _per_graph_center(lig_pos, lig_batch, B)              # [B,3]
    x0 = lig_pos - centers[lig_batch]                               # center each complex
    R = _expmap_so3((step_rot.unsqueeze(1) * rot_vec).contiguous()) # [B,3,3]
    # rotate each point by its graph rotation
    # pack into a padded tensor per graph for efficient bmm
    # build index map
    out = torch.empty_like(lig_pos)
    for b in range(B):
        mask = lig_batch == b
        if mask.any():
            xb = x0[mask].unsqueeze(0)                              # [1, Nb, 3]
            Rb = R[b].T.unsqueeze(0)                                # [1,3,3], transpose for right-multiply
            xb_rot = torch.matmul(xb, Rb)                           # [1,Nb,3]
            tb = (step_tr[b] * tr_vec[b]).view(1, 1, 3)             # [1,1,3]
            out[mask] = xb_rot.squeeze(0) + tb.squeeze(0) + centers[b].view(1, 3)
    return out

def _frobenius_tail_energy_from_svals(svals: Tensor, k: int) -> Tensor:
    """
    Sum of squares of singular values past rank k, divided by total entries count
    to keep scale comparable across variable sizes.
    svals sorted descending by torch.linalg.svdvals
    """
    if svals.numel() <= k:
        return svals.new_tensor(0.0)
    tail = svals[k:]
    return torch.sum(tail * tail)

def low_rank_contact_loss(
    data,
    tr_pred: Tensor,                 # [B,3]
    rot_pred: Tensor,                # [B,3]
    tr_sigma: Optional[Tensor] = None,   # [B,1] from loss_function
    rank_k: int = 8,
    gaussian_sigma: float = 2.0,
    alpha_tr: float = 0.25,
    alpha_rot: float = 0.25,
    use_receptor_atoms: bool = True
) -> Tensor:
    """
    Build a smooth ligand–receptor contact matrix from a one-step denoised pose,
    and penalize the Frobenius norm of the residual after best rank-k approximation.

    Returns a scalar tensor.
    """
    device = tr_pred.device
    B = _num_graphs(data)

    lig_pos_t = _stack_graph_field(data, 'ligand', 'pos')           # [NL,3]
    lig_batch = _make_batch_index(data, 'ligand')                   # [NL]
    lig_x = _stack_graph_field(data, 'ligand', 'x')                 # categorical features
    lig_mask_heavy = _heavy_atom_mask(lig_x)

    # receptor positions: prefer atom graph if present
    try:
        rec_pos = _stack_graph_field(data, 'atom' if use_receptor_atoms else 'receptor', 'pos')
        rec_batch = _make_batch_index(data, 'atom' if use_receptor_atoms else 'receptor')
    except Exception:
        rec_pos = _stack_graph_field(data, 'receptor', 'pos')
        rec_batch = _make_batch_index(data, 'receptor')

    # step sizes, make them gentle and scale by current σ_tr if provided
    if tr_sigma is not None:
        # tr_sigma shape [B,1], map to [B]
        step_tr = alpha_tr * tr_sigma.squeeze(-1).to(device).clamp_min(1e-3)
    else:
        step_tr = torch.full((B,), alpha_tr, device=device)
    step_rot = torch.full((B,), alpha_rot, device=device)

    # one tiny reverse SE3 step to get a denoised pose that depends on params
    lig_pos_hat = _apply_one_step_se3(lig_pos_t, lig_batch, tr_pred, rot_pred, step_tr, step_rot)

    # per-graph contact low-rank penalty
    sigma2 = float(gaussian_sigma) ** 2
    total = lig_pos_hat.new_tensor(0.0)
    norm = 0.0

    for b in range(B):
        lmask = lig_batch == b
        rmask = rec_batch == b
        if not lmask.any() or not rmask.any():
            continue
        L = lig_pos_hat[lmask & lig_mask_heavy]    # heavy ligand atoms [nL,3]
        R = rec_pos[rmask]                         # receptor atoms or residues [nR,3]
        if L.size(0) == 0 or R.size(0) == 0:
            continue

        # pairwise distances and Gaussian contact kernel
        D = torch.cdist(L, R, compute_mode='donot_use_mm_for_euclid_dist')  # [nL,nR]
        M = torch.exp(-(D * D) / sigma2).clamp_min(1e-6)                    # positive smooth contacts

        # low-rank residual via singular values tail energy
        s = torch.linalg.svdvals(M)     # descending
        tail_energy = _frobenius_tail_energy_from_svals(s, rank_k)  # scalar
        total = total + tail_energy / M.numel()
        norm += 1.0

    if norm == 0.0:
        return total  # zero

    return total / norm


# utils/rank_regularizer.py  — inference guidance helpers


def _stack(data, key0: str, key1: str) -> Tensor:
    if isinstance(data, (list, tuple)):
        return torch.cat([d[key0].__dict__[key1] for d in data], dim=0)
    return data[key0].__dict__[key1]

def _batch_index(data, key0: str) -> Tensor:
    if isinstance(data, (list, tuple)):
        idxs = []
        for i, d in enumerate(data):
            n = d[key0].pos.size(0)
            idxs.append(torch.full((n,), i, device=d[key0].pos.device, dtype=torch.long))
        return torch.cat(idxs, dim=0)
    return data[key0].batch

def _heavy_mask_from_x(x: Tensor) -> Tensor:
    # Treat channel zero as element index, zero for H, same convention many loaders use
    try:
        return x[:, 0] != 0
    except Exception:
        return torch.ones(x.size(0), dtype=torch.bool, device=x.device)

def low_rank_contact_energy_from_data(
    data,
    rank_k: int = 8,
    gaussian_sigma: float = 2.0,
    use_receptor_atoms: bool = False
) -> Tensor:
    """
    Builds a smooth ligand receptor contact matrix per complex and returns
    the average tail energy past rank k. Fully differentiable w.r.t ligand coords.
    """
    device = _stack(data, 'ligand', 'pos').device
    lig_pos = _stack(data, 'ligand', 'pos')
    rec_pos = _stack(data, 'atom' if use_receptor_atoms else 'receptor', 'pos')
    lig_batch = _batch_index(data, 'ligand')
    rec_batch = _batch_index(data, 'atom' if use_receptor_atoms else 'receptor')

    # heavy atom mask if available
    try:
        lig_x = _stack(data, 'ligand', 'x')
        lig_heavy = _heavy_mask_from_x(lig_x)
    except Exception:
        lig_heavy = torch.ones(lig_pos.size(0), dtype=torch.bool, device=device)

    B = int(lig_batch.max().item()) + 1 if lig_batch.numel() > 0 else 0
    total = lig_pos.new_tensor(0.0)
    count = 0.0
    sigma2 = float(gaussian_sigma) ** 2

    for b in range(B):
        lm = (lig_batch == b) & lig_heavy
        rm = rec_batch == b
        if not lm.any() or not rm.any():
            continue
        L = lig_pos[lm]   # [nL, 3]
        R = rec_pos[rm]   # [nR, 3]
        if L.size(0) == 0 or R.size(0) == 0:
            continue

        D = torch.cdist(L, R, compute_mode='donot_use_mm_for_euclid_dist')
        M = torch.exp(-(D * D) / sigma2).clamp_min(1e-6)
        s = torch.linalg.svdvals(M)
        if s.numel() > rank_k:
            tail = s[rank_k:]
            total = total + (tail * tail).sum() / M.numel()
            count += 1.0

    return total / count if count > 0 else total

@torch.no_grad()
def _per_graph_norm(grad: Tensor, batch: Tensor) -> Tensor:
    """Max norm per graph, broadcast to nodes, avoids tiny graphs dominating."""
    gnorm = grad.norm(dim=1)
    if batch.numel() == 0:
        return grad
    gmax, _ = scatter_max(gnorm, batch, dim=0, dim_size=int(batch.max().item()) + 1)
    gmax = gmax.clamp_min(1e-6)
    return grad / gmax[batch].unsqueeze(1)

def low_rank_guidance_step_inplace(
    data,
    step_size: float = 0.05,
    rank_k: int = 8,
    gaussian_sigma: float = 2.0,
    use_receptor_atoms: bool = False,
    mask_heavy_only: bool = True,
    trust_radius: float = 0.5
) -> None:
    """
    One guidance step that reduces the low rank contact energy.
    Operates directly on data['ligand'].pos.
    """
    device = _stack(data, 'ligand', 'pos').device
    lig_pos = _stack(data, 'ligand', 'pos')
    lig_batch = _batch_index(data, 'ligand')

    # set up gradient
    lig_pos_detached = lig_pos.detach().requires_grad_(True)

    # temporarily swap into data for energy call
    if isinstance(data, (list, tuple)):
        # DataParallel case
        start = 0
        for d in data:
            n = d['ligand'].pos.size(0)
            d['ligand'].pos = lig_pos_detached[start:start + n]
            start += n
    else:
        data['ligand'].pos = lig_pos_detached

    energy = low_rank_contact_energy_from_data(
        data, rank_k=rank_k, gaussian_sigma=gaussian_sigma, use_receptor_atoms=use_receptor_atoms
    )
    grad = torch.autograd.grad(energy, lig_pos_detached, create_graph=False, retain_graph=False)[0]

    # optional heavy atom mask
    if mask_heavy_only:
        try:
            lig_x = _stack(data, 'ligand', 'x')
            heavy = _heavy_mask_from_x(lig_x)
            grad = grad * heavy.float().unsqueeze(1)
        except Exception:
            pass

    # normalize per graph and clamp
    grad_n = _per_graph_norm(grad, lig_batch)
    step_vec = -step_size * grad_n
    step_norm = step_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    step_vec = step_vec * torch.clamp(trust_radius / step_norm, max=1.0)

    # write back
    new_pos = lig_pos + step_vec
    if isinstance(data, (list, tuple)):
        start = 0
        for d in data:
            n = d['ligand'].pos.size(0)
            d['ligand'].pos = new_pos[start:start + n].detach()
            start += n
    else:
        data['ligand'].pos = new_pos.detach()
