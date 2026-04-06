import copy
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
from utils.rank_regularizer import low_rank_contact_loss

def loss_function(
    tr_pred, rot_pred, tor_pred, sidechain_pred, data, t_to_sigma, device,
    tr_weight: float = 1.0, rot_weight: float = 1.0, tor_weight: float = 1.0,
    apply_mean: bool = True, no_torsion: bool = False,
    # new knobs for the low-rank regularizer:
    rank_weight: float = 0.0,        # set >0 to turn it on
    rank_k: int = 8,
    rank_sigma: float = 2.0,
    rank_alpha_tr: float = 0.25,
    rank_alpha_rot: float = 0.25,
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
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1).to(pred_device)
    rot_loss = (((rot_pred - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion
    if not no_torsion:
        if isinstance(data, list):
            edge_tor_arr = np.concatenate([d.tor_sigma_edge for d in data])
            edge_tor_sigma = torch.from_numpy(edge_tor_arr)
            tor_score = torch.cat([d.tor_score for d in data], dim=0).to(pred_device)
        else:
            # When using a Batched object, some fields (like tor_sigma_edge) may still be lists
            if isinstance(data.tor_sigma_edge, list):
                edge_tor_arr = np.concatenate(data.tor_sigma_edge)
                edge_tor_sigma = torch.from_numpy(edge_tor_arr)
            else:
                edge_tor_sigma = torch.from_numpy(data.tor_sigma_edge)
            tor_score = data.tor_score.to(pred_device)

        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float().to(pred_device)
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

    # stock DiffDock weighted loss
    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight

    rank_loss = torch.zeros(1, dtype=torch.float, device=pred_device)

    # add our low-rank contact loss, computed from a one-step denoised pose
    if rank_weight > 0.0:
        rank_loss = low_rank_contact_loss(
            data=data,
            tr_pred=tr_pred.to(pred_device), rot_pred=rot_pred.to(pred_device),
            tr_sigma=tr_sigma.to(pred_device) if isinstance(tr_sigma, torch.Tensor) else None,
            rank_k=int(rank_k),
            gaussian_sigma=float(rank_sigma),
            alpha_tr=float(rank_alpha_tr),
            alpha_rot=float(rank_alpha_rot),
            use_receptor_atoms=True
        )
        loss = loss + rank_weight * rank_loss

    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), rank_loss.detach(), tr_base_loss, rot_base_loss, tor_base_loss


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


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights, grad_accum_steps=1):
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'rank_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'])
    accum_count = 0
    optimizer.zero_grad()

    for data in tqdm(loader, total=len(loader)):
        # determine if this is a single-example batch (support list or Batch)
        if isinstance(data, list):
            single_batch = len(data) == 1
        else:
            single_batch = getattr(data, 'num_graphs', 1) == 1

        if single_batch:
            # only skip if the model actually contains BatchNorm modules
            has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
            if has_bn:
                print("Skipping batch of size 1 since otherwise batchnorm would not work.")
                continue
        # If loader yields a list (DataListLoader), convert to a Batch so model.forward receives the expected object
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        # move the whole batch to device (keep as a Batch), so model.forward receives the expected object
        data = data.to(device) if device.type == 'cuda' else data
        try:
            tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, device=device)
            if loss_tuple is None:
                print("None loss tuple, skipping")
                continue
            loss = loss_tuple[0]

            if torch.any(torch.isnan(loss)):
                names = data.name if device.type == 'cpu' else [d.name for d in data]
                print("Nan loss, skipping batch with complexes", names)
                continue
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            accum_count += 1

            if accum_count == grad_accum_steps:
                optimizer.step()
                optimizer.zero_grad()
                if ema_weights is not None:
                    ema_weights.update(model.parameters())
                accum_count = 0

            meter.add([loss.detach().cpu(), *loss_tuple[1:]])
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_count = 0
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_count = 0
                continue
            else:
                #raise e
                print(e)
                optimizer.zero_grad()
                accum_count = 0
                continue

    if accum_count > 0:
        optimizer.step()
        optimizer.zero_grad()
        if ema_weights is not None:
            ema_weights.update(model.parameters())
            
    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'rank_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'rank_loss', 'backbone_loss', 'sidechain_loss',
             'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            # If loader yields a list (DataListLoader), convert to a Batch so model.forward receives the expected object
            if isinstance(data, list):
                data = Batch.from_data_list(data)
            # move the whole batch to device (keep as a Batch), so model.forward receives tensors on the same device
            data = data.to(device) if device.type == 'cuda' else data
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            if loss_tuple is None: continue
            meter.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor = [torch.cat([data[i].complex_t[noise_type] for i in range(len(data))]) for
                                                              noise_type in ['tr', 'rot', 'tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr, sigma_index_tr,
                     sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr])

        except RuntimeError as e:
            if 'out of memory' in str(e):
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


def inference_epoch_fix(model, complex_graphs, device, t_to_sigma, args):
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds, min_rmsds = [], []

    for orig_complex_graph in tqdm(loader):
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
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
              'min_rmsds_lt2': (100 * (min_rmsds < 2).sum() / len(min_rmsds)),
              'min_rmsds_lt5': (100 * (min_rmsds < 5).sum() / len(min_rmsds)),}
    return losses
