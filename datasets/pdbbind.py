import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                                                                                 
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing as mp
import signal

                                                                                 
try:
    mp.set_start_method("fork", force=False)
except RuntimeError:
    pass

import pickle
import random
import copy
import gc
import sys
from collections import defaultdict, Counter
import threading
import time


class PreprocessHealthError(RuntimeError):

    pass


def build_gold_standard_cache_path(
    *,
    cache_path,
    dataset,
    split_path,
    limit_complexes,
    max_lig_size,
    remove_hs,
    receptor_radius,
    c_alpha_max_neighbors,
    chain_cutoff,
    all_atoms,
    atom_radius,
    atom_max_neighbors,
    matching,
    num_conformers,
    esm_embeddings_path,
    keep_local_structures=False,
    protein_path_list=None,
    ligand_descriptions=None,
    protein_file="protein_processed",
    fixed_knn_radius_graph=True,
    knn_only_graph=False,
    include_miscellaneous_atoms=False,
    use_old_wrong_embedding_order=False,
    matching_tries=1,
):
    import os
    import binascii

    return os.path.join(
        cache_path,
        f"{dataset}3_limit{limit_complexes}"
        f"_INDEX{os.path.splitext(os.path.basename(split_path))[0]}"
        f"_maxLigSize{max_lig_size}_H{int(not remove_hs)}"
        f"_recRad{receptor_radius}_recMax{c_alpha_max_neighbors}"
        f"_chainCutoff{chain_cutoff if chain_cutoff is None else int(chain_cutoff)}"
        + ("" if not all_atoms else f"_atomRad{atom_radius}_atomMax{atom_max_neighbors}")
        + ("" if not matching or num_conformers == 1 else f"_confs{num_conformers}")
        + ("" if esm_embeddings_path is None else "_esmEmbeddings")
        + "_full"
        + ("" if not keep_local_structures else "_keptLocalStruct")
        + (
            ""
            if protein_path_list is None or ligand_descriptions is None
            else str(binascii.crc32("".join(ligand_descriptions + protein_path_list).encode()))
        )
        + ("" if protein_file == "protein_processed" else "_" + protein_file)
        + (
            ""
            if not fixed_knn_radius_graph
            else ("_fixedKNN" if not knn_only_graph else "_fixedKNNonly")
        )
        + ("" if not include_miscellaneous_atoms else "_miscAtoms")
        + ("" if not use_old_wrong_embedding_order else "_chainOrd")
        + ("" if matching_tries == 1 else f"_tries{matching_tries}")
    )

import numpy as np
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import RemoveAllHs

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform

from datasets.process_mols import (
    read_molecule,
    get_lig_graph_with_matching,
    moad_extract_receptor_structure,
)
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt, crop_beyond
from utils import so3, torus
from utils.torsion import get_transformation_mask


_WORKER_CFG = {}


def _worker_init(cfg):
    global _WORKER_CFG
    _WORKER_CFG = cfg

                                                                    
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _rdkit_mol_to_bytes(mol):
                                                                                       
    return pickle.dumps(mol, protocol=pickle.HIGHEST_PROTOCOL)


def _rdkit_mol_from_bytes(b):
    return pickle.loads(b)


class _AlarmTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _AlarmTimeout()


def _process_one_complex(task):
    name, lm = task
    cfg = _WORKER_CFG

                                                                            
    if lm is not None:
        try:
            lm = [torch.from_numpy(x) for x in lm]
        except Exception:
                                                                                   
            pass

                                    
    try:
        if cfg.get("esm_embeddings_enabled", False) and lm is None:
            return False, "skip_missing_esm_embeddings"
    except Exception:
        pass

    timeout_s = cfg.get("per_complex_timeout_s", None)
                                                                    
    if os.name != "posix":
        timeout_s = None
    if timeout_s and timeout_s > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(timeout_s))

    try:
        complex_dir = os.path.join(cfg["pdbbind_dir"], name)
        if not os.path.exists(complex_dir):
            return False, "skip_missing_dir"

        lig = None
        try:
            lig = read_molecule(
                os.path.join(complex_dir, f"{name}_{cfg['ligand_file']}.sdf"),
                remove_hs=False,
                sanitize=True,
            )
        except Exception:
            lig = None

        if lig is None:
            try:
                lig = read_molecule(
                    os.path.join(complex_dir, f"{name}_{cfg['ligand_file']}.mol2"),
                    remove_hs=False,
                    sanitize=True,
                )
            except Exception:
                lig = None

        if lig is None:
            return False, "skip_ligand_unreadable"

        if cfg["max_lig_size"] is not None and lig.GetNumHeavyAtoms() > cfg["max_lig_size"]:
            return False, "skip_ligand_too_large"

        cg = HeteroData()
        cg.name = name

        try:
            get_lig_graph_with_matching(
                lig,
                cg,
                cfg["popsize"],
                cfg["maxiter"],
                cfg["matching"],
                cfg["keep_original"],
                cfg["num_conformers"],
                remove_hs=cfg["remove_hs"],
                tries=cfg["matching_tries"],
            )
        except Exception:
            return False, "skip_lig_graph_fail"

        try:
            moad_extract_receptor_structure(
                path=os.path.join(complex_dir, f"{name}_{cfg['protein_file']}.pdb"),
                complex_graph=cg,
                neighbor_cutoff=cfg["receptor_radius"],
                max_neighbors=cfg["c_alpha_max_neighbors"],
                lm_embeddings=lm,
                knn_only_graph=cfg["knn_only_graph"],
                all_atoms=cfg["all_atoms"],
                atom_cutoff=cfg["atom_radius"],
                atom_max_neighbors=cfg["atom_max_neighbors"],
            )
        except Exception:
            return False, "skip_receptor_fail"

        try:
            center = torch.mean(cg["receptor"].pos, dim=0, keepdim=True)
            cg["receptor"].pos -= center
            if cfg["all_atoms"] and "atom" in cg:
                cg["atom"].pos -= center

            if isinstance(cg["ligand"].pos, list):
                for p in cg["ligand"].pos:
                    p -= center
            else:
                cg["ligand"].pos -= center

            cg.original_center = center
            cg.receptor_name = name
        except Exception:
            return False, "skip_postprocess_fail"

        cg_bytes = pickle.dumps(cg, protocol=pickle.HIGHEST_PROTOCOL)
        lig_bytes = _rdkit_mol_to_bytes(lig)

        return True, (cg_bytes, lig_bytes)

    except _AlarmTimeout:
        return False, "skip_timeout"

    finally:
        if timeout_s and timeout_s > 0:
            signal.alarm(0)


class PreprocessHealthMonitor:
    def __init__(
        self,
        min_attempts=500,
        max_skip_rate=0.10,
        max_skips_abs=2000,
        reason_rate_caps=None,
        check_every=200,
        early_catastrophic_min_attempts=500,
        early_catastrophic_min_skips=300,
        early_catastrophic_rate=0.98,
        report_path=None,
    ):
        self.min_attempts = int(min_attempts)
        self.max_skip_rate = float(max_skip_rate)
        self.max_skips_abs = int(max_skips_abs)
        self.check_every = int(check_every)
        self.reason_rate_caps = reason_rate_caps or {}

        self.early_catastrophic_min_attempts = int(early_catastrophic_min_attempts)
        self.early_catastrophic_min_skips = int(early_catastrophic_min_skips)
        self.early_catastrophic_rate = float(early_catastrophic_rate)

        self.report_path = report_path

        self.attempted = 0
        self.ok = 0
        self.skips = Counter()

    def record_ok(self):
        self.attempted += 1
        self.ok += 1

    def record_skip(self, reason):
        self.attempted += 1
        self.skips[reason] += 1

    def should_check(self):
        return (
            self.attempted >= self.min_attempts
            and (self.attempted % self.check_every) == 0
        )

    def check_or_exit(self, context=""):
        total_skips = sum(self.skips.values())
        if self.attempted != self.ok + total_skips:
            self._abort(
                f"Internal accounting invariant violated: attempted={self.attempted}, ok={self.ok}, skipped={total_skips}",
                context,
            )

        skip_rate = total_skips / max(1, self.attempted)

        if (
            self.attempted >= self.early_catastrophic_min_attempts
            and total_skips >= self.early_catastrophic_min_skips
            and skip_rate > self.early_catastrophic_rate
        ):
            self._abort("Early catastrophic skip rate", context)

        if total_skips > self.max_skips_abs:
            self._abort("Too many skipped complexes", context)

        if skip_rate > self.max_skip_rate:
            self._abort(f"Skip rate too high ({skip_rate:.3f})", context)

        for reason, cap in self.reason_rate_caps.items():
            r = self.skips.get(reason, 0) / max(1, self.attempted)
            if r > cap:
                self._abort(f"Skip reason '{reason}' too frequent ({r:.3f})", context)

    def _abort(self, headline, context):
        total_skips = sum(self.skips.values())
        skip_rate = total_skips / max(1, self.attempted)

        lines = [
            "",
            "PDBBind preprocessing failed health check",
            f"Context: {context}" if context else "",
            headline,
            f"Attempted: {self.attempted}",
            f"OK: {self.ok}",
            f"Skipped: {total_skips}",
            f"Skip rate: {skip_rate:.3f}",
            "Top skip reasons:",
        ]

        for k, v in self.skips.most_common(10):
            denom = self.attempted if self.attempted else 1
            lines.append(f"  - {k}: {v} ({v / denom:.3f})")

        report = "\n".join(lines)
        print(report, flush=True)

        if self.report_path:
            try:
                with open(self.report_path, "w") as f:
                    f.write(report)
            except Exception:
                pass

                                                                              
        raise PreprocessHealthError(report)


class PreprocessingWatchdog:


    def __init__(self, *, interval_seconds=30, name="preprocess", out=sys.stdout):
        self.interval = float(interval_seconds)
        self.name = name
        self.out = out

        self._stop_event = threading.Event()
        self._thread = None

                                                      
        self._state = {
            "batch_idx": None,
            "total_batches": None,
            "attempted": 0,
            "ok": 0,
            "skipped": 0,
            "current_item": None,
        }

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-watchdog", daemon=True)
        self._thread.start()

    def stop(self, timeout=None):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def update(self, *, batch_idx=None, total_batches=None, attempted=None, ok=None, skipped=None, current_item=None):
        if batch_idx is not None:
            self._state["batch_idx"] = batch_idx
        if total_batches is not None:
            self._state["total_batches"] = total_batches
        if attempted is not None:
            self._state["attempted"] = attempted
        if ok is not None:
            self._state["ok"] = ok
        if skipped is not None:
            self._state["skipped"] = skipped
        if current_item is not None:
            self._state["current_item"] = current_item

    def _run(self):
                       
        if self._stop_event.wait(self.interval):
            return
        while not self._stop_event.is_set():
            self._emit()
            if self._stop_event.wait(self.interval):
                return

    def _emit(self):
        s = self._state
        batch_info = ""
        if s["batch_idx"] is not None and s["total_batches"] is not None:
            batch_info = f"batch {s['batch_idx']}/{s['total_batches']}"

        item_info = ""
        if s["current_item"] is not None:
            item_info = f"current={s['current_item']}"

        msg = (
            f"[watchdog:{self.name}] "
            f"{batch_info} "
            f"attempted={s['attempted']} "
            f"ok={s['ok']} "
            f"skipped={s['skipped']} "
            f"{item_info}"
        ).strip()

        try:
            print(msg, file=self.out, flush=True)
        except Exception:
            pass


def _ensure_ligand_torsion_metadata_compatible(data):
    """
    Make ligand.edge_mask and ligand.mask_rotate compatible with diffusion_utils.modify_conformer,
    which indexes mask_rotate[0] for non-numpy inputs.
    """
    lig = data["ligand"]

    mr = getattr(lig, "mask_rotate", None)
    em = getattr(lig, "edge_mask", None)

    if em is not None and not torch.is_tensor(em):
        try:
            em = torch.as_tensor(em)
            lig.edge_mask = em
        except Exception:
            em = None

    mr_mat = None

    if isinstance(mr, (list, tuple)):
        if len(mr) >= 1:
            mr0 = mr[0]
            if isinstance(mr0, np.ndarray):
                mr_mat = torch.from_numpy(mr0)
            elif torch.is_tensor(mr0):
                mr_mat = mr0
    elif isinstance(mr, np.ndarray):
        mr_mat = torch.from_numpy(mr)
    elif torch.is_tensor(mr):
        mr_mat = mr

    if torch.is_tensor(mr_mat) and mr_mat.dim() == 3 and mr_mat.shape[0] == 1:
        mr_mat = mr_mat[0]

    bad = False
    if em is None or not torch.is_tensor(em) or em.dim() != 1:
        bad = True
    if mr_mat is None or (torch.is_tensor(mr_mat) and mr_mat.dim() != 2):
        bad = True

    if bad:
        edge_mask_new, mask_rotate_new = get_transformation_mask(data)
        lig.edge_mask = torch.as_tensor(edge_mask_new).bool()
        mr_mat = mask_rotate_new
        if torch.is_tensor(mr_mat) and mr_mat.dim() == 3 and mr_mat.shape[0] == 1:
            mr_mat = mr_mat[0]

    lig._mask_rotate = [mr_mat]


class NoiseTransform(BaseTransform):
    def __init__(
        self,
        t_to_sigma,
        no_torsion,
        all_atom,
        alpha=1,
        beta=1,
        include_miscellaneous_atoms=False,
        crop_beyond_cutoff=None,
        time_independent=False,
        rmsd_cutoff=0,
        minimum_t=0,
        sampling_mixing_coeff=0,
    ):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.minimum_t = minimum_t
        self.mixing_coeff = sampling_mixing_coeff
        self.alpha = alpha
        self.beta = beta
        self.crop_beyond_cutoff = crop_beyond_cutoff
        self.rmsd_cutoff = rmsd_cutoff
        self.time_independent = time_independent

    def __call__(self, data):
        t_tr, t_rot, t_tor, t = self.get_time()
        return self.apply_noise(data, t_tr, t_rot, t_tor, t)

    def get_time(self):
        if self.time_independent:
            t = np.random.beta(self.alpha, self.beta)
            return t, t, t, t

        if self.mixing_coeff == 0:
            t = np.random.beta(self.alpha, self.beta)
            t = self.minimum_t + t * (1 - self.minimum_t)
        else:
            choice = np.random.binomial(1, self.mixing_coeff)
            t1 = np.random.beta(self.alpha, self.beta) * self.minimum_t
            t2 = self.minimum_t + np.random.beta(self.alpha, self.beta) * (1 - self.minimum_t)
            t = choice * t1 + (1 - choice) * t2

        return t, t, t, t

    def apply_noise(
        self, data, t_tr, t_rot, t_tor, t,
        tr_update=None, rot_update=None, torsion_updates=None
    ):
        if not torch.is_tensor(data["ligand"].pos):
            data = copy.deepcopy(data)
            data["ligand"].pos = random.choice(data["ligand"].pos)

        if self.time_independent:
            orig_complex_graph = copy.deepcopy(data)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)

        set_time(
            data,
            0 if self.time_independent else t,
            0 if self.time_independent else t_tr,
            0 if self.time_independent else t_rot,
            0 if self.time_independent else t_tor,
            1,
            self.all_atom,
            device=None,
            include_miscellaneous_atoms=self.include_miscellaneous_atoms,
        )

        tr_update = tr_update if tr_update is not None else torch.normal(0, tr_sigma, size=(1, 3))
        rot_update = rot_update if rot_update is not None else so3.sample_vec(eps=rot_sigma)

        if not self.no_torsion:
            _ensure_ligand_torsion_metadata_compatible(data)

            n_edges = int(data["ligand"].edge_mask.sum())
            torsion_updates = (
                torsion_updates if torsion_updates is not None
                else np.random.normal(0.0, tor_sigma, size=n_edges)
            )
        else:
            torsion_updates = None


        # Temporarily expose mask_rotate for modify_conformer only
        lig = data["ligand"]
        orig_mask_rotate = getattr(lig, "mask_rotate", None)
        lig.mask_rotate = lig._mask_rotate

        modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)

        # Clean up to keep PyG schema stable
        if orig_mask_rotate is None:
            delattr(lig, "mask_rotate")
        else:
            lig.mask_rotate = orig_mask_rotate

        if self.time_independent:
            # compute orig_pos as a local numpy array only
            orig_pos = None
            if self.no_torsion:
                orig_pos = (
                    orig_complex_graph["ligand"].pos.cpu().numpy()
                    + orig_complex_graph.original_center.cpu().numpy()
                )

            filterHs = torch.not_equal(data["ligand"].x[:, 0], 0).cpu().numpy()

            # orig_pos must exist if you're about to use it
            if orig_pos is None:
                raise RuntimeError("time_independent path requires orig_pos but it was not computed")

            ligand_pos = data["ligand"].pos.cpu().numpy()[filterHs]
            orig_pos = orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy()

            rmsd = np.sqrt(((ligand_pos - orig_pos) ** 2).sum(axis=1).mean())

            data.y = torch.tensor(rmsd < self.rmsd_cutoff).float().unsqueeze(0)
            data.atom_y = data.y
            return data

        data.tr_score = -tr_update / (tr_sigma ** 2)
        data.rot_score = torch.from_numpy(
            so3.score_vec(vec=rot_update, eps=rot_sigma)
        ).float().unsqueeze(0)

        if data["ligand"].pos.shape[0] == 1:
            data.rot_score = data.rot_score * 0

        if not self.no_torsion:
            data.tor_score = torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
            data.tor_sigma_edge = np.ones(int(data["ligand"].edge_mask.sum())) * tor_sigma
        else:
            data.tor_score = None
            data.tor_sigma_edge = None

        if self.crop_beyond_cutoff is not None:
            crop_beyond(data, tr_sigma * 3 + self.crop_beyond_cutoff, self.all_atom)

        set_time(
            data,
            t, t_tr, t_rot, t_tor,
            1,
            self.all_atom,
            device=None,
            include_miscellaneous_atoms=self.include_miscellaneous_atoms,
        )

        return data


class PDBBind(Dataset):

    def __init__(
        self,
        root=None,
        transform=None,
        cache_path="data/cache",
        split_path="data/",
        limit_complexes=0,
        chain_cutoff=10,
        receptor_radius=30,
        c_alpha_max_neighbors=None,
        popsize=15,
        maxiter=15,
        matching=True,
        keep_original=False,
        max_lig_size=None,
        remove_hs=False,
        num_conformers=1,
        all_atoms=False,
        atom_radius=5,
        atom_max_neighbors=None,
        esm_embeddings_path=None,
        require_ligand=False,
        include_miscellaneous_atoms=False,
        protein_file="protein_processed",
        ligand_file="ligand",
        knn_only_graph=False,
        matching_tries=1,
        dataset="PDBBind",
        batch_size=1000,
        enable_health_check=False,
        **kwargs,
    ):
                              
        if root is None:
            root = "data/PDBBind_processed"

        super(PDBBind, self).__init__(root, transform)

                                                   
        self.pdbbind_dir = root
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.chain_cutoff = chain_cutoff
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.popsize = popsize
        self.maxiter = maxiter
        self.matching = matching
        self.keep_original = keep_original
        self.max_lig_size = max_lig_size
        self.remove_hs = remove_hs
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius = atom_radius
        self.atom_max_neighbors = atom_max_neighbors
        self.esm_embeddings_path = esm_embeddings_path
        self.require_ligand = require_ligand
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.protein_file = protein_file
        self.ligand_file = ligand_file
        self.knn_only_graph = knn_only_graph
        self.matching_tries = matching_tries
        self.dataset = dataset
        self.batch_size = int(batch_size)
                                                                       
                                                                        
        self.enable_health_check = bool(enable_health_check)
        if os.environ.get("PDBBIND_DISABLE_HEALTH_CHECK", "0") in ("1", "true", "True"):
            self.enable_health_check = False

                                               
        self.fixed_knn_radius_graph = True

                                                                                             
        self.full_cache_path = build_gold_standard_cache_path(
            cache_path=cache_path,
            dataset=dataset,
            split_path=self.split_path,
            limit_complexes=self.limit_complexes,
            max_lig_size=self.max_lig_size,
            remove_hs=self.remove_hs,
            receptor_radius=self.receptor_radius,
            c_alpha_max_neighbors=self.c_alpha_max_neighbors,
            chain_cutoff=self.chain_cutoff,
            all_atoms=self.all_atoms,
            atom_radius=self.atom_radius,
            atom_max_neighbors=self.atom_max_neighbors,
            matching=self.matching,
            num_conformers=self.num_conformers,
            esm_embeddings_path=self.esm_embeddings_path,
            keep_local_structures=False,
            protein_path_list=None,
            ligand_descriptions=None,
            protein_file=self.protein_file,
            fixed_knn_radius_graph=self.fixed_knn_radius_graph,
            knn_only_graph=self.knn_only_graph,
            include_miscellaneous_atoms=self.include_miscellaneous_atoms,
            use_old_wrong_embedding_order=False,
            matching_tries=self.matching_tries,
        )
        os.makedirs(self.full_cache_path, exist_ok=True)

                                           
        if not self._cache_complete():
            try:
                self.preprocessing()
            except PreprocessHealthError:
                                                                               
                raise
                                                                        
            names = read_strings_from_txt(self.split_path)
            if self.limit_complexes:
                names = names[: self.limit_complexes]
            total_batches = (len(names) + self.batch_size - 1) // self.batch_size
            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), "wb") as f:
                pickle.dump({"num_batches": total_batches, "format": "batched"}, f)

        self.complex_graphs, self.rdkit_ligands = self._load_cache()


    def preprocessing(self):
        names = read_strings_from_txt(self.split_path)
        if self.limit_complexes:
            names = names[: self.limit_complexes]

                                                   
        if mp.get_start_method(allow_none=True) != "fork":
            raise RuntimeError("PDBBind preprocessing requires Linux fork start method.")

        lm_embeddings_all = None
        lm_indices = None
        if self.esm_embeddings_path is not None:
            id_to_emb = torch.load(self.esm_embeddings_path)
            lm_embeddings_all = defaultdict(list)
            lm_indices = defaultdict(list)
            for k, v in id_to_emb.items():
                base = k.split("_chain_")[0]
                idx = int(k.split("_chain_")[1])
                lm_embeddings_all[base].append(v)
                lm_indices[base].append(idx)

        def lm_for_name(name):
            if lm_embeddings_all is None or name not in lm_embeddings_all:
                return None
            pairs = list(zip(lm_indices[name], lm_embeddings_all[name]))
            pairs.sort(key=lambda x: x[0])
                                                                                  
            return [emb.cpu().numpy() for _, emb in pairs]

        monitor = PreprocessHealthMonitor(
            reason_rate_caps={
                "skip_missing_dir": 0.01,
                "skip_ligand_unreadable": 0.05,
                "skip_receptor_fail": 0.03,
                "skip_postprocess_fail": 0.02,
                "skip_timeout": 0.05,
                "skip_lig_graph_fail": 0.05,
                "skip_ligand_too_large": 0.05,
                "skip_missing_esm_embeddings": 0.90,
                "skip_bad_esm_embedding": 0.05,
            },
            report_path=os.path.join(self.full_cache_path, "preprocess_health_report.txt"),
        )

        total_batches = (len(names) + self.batch_size - 1) // self.batch_size

                                                                              
        meta_path = os.path.join(self.full_cache_path, "heterographs.pkl")
        try:
            if not os.path.exists(meta_path):
                with open(meta_path, "wb") as f:
                    pickle.dump({"num_batches": total_batches, "format": "batched"}, f)
        except Exception:
            pass

        for batch_idx in range(total_batches):
            graphs, ligands = [], []

            batch_names = names[
                batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
            ]

                                                                                   
            graphs_path = os.path.join(self.full_cache_path, f"heterographs{batch_idx}.pkl")
            ligs_path = os.path.join(self.full_cache_path, f"rdkit_ligands{batch_idx}.pkl")
            if os.path.exists(graphs_path) and os.path.exists(ligs_path):
                                                                                      
                continue

                                                                                   
            watchdog = PreprocessingWatchdog(interval_seconds=30, name="pdbbind_preprocess")
            watchdog.start()
            watchdog.update(batch_idx=batch_idx, total_batches=total_batches)

            try:
                env_workers = int(os.environ.get("PDBBIND_PREPROC_WORKERS", "32"))
                                                                         
                num_workers = min(env_workers, max(1, mp.cpu_count() // 2))
                num_workers = max(1, num_workers)

                cfg = {
                    "pdbbind_dir": self.pdbbind_dir,
                    "ligand_file": self.ligand_file,
                    "protein_file": self.protein_file,
                    "max_lig_size": self.max_lig_size,
                    "popsize": self.popsize,
                    "maxiter": self.maxiter,
                    "matching": self.matching,
                    "keep_original": self.keep_original,
                    "num_conformers": self.num_conformers,
                    "remove_hs": self.remove_hs,
                    "matching_tries": self.matching_tries,
                    "receptor_radius": self.receptor_radius,
                    "c_alpha_max_neighbors": self.c_alpha_max_neighbors,
                    "knn_only_graph": self.knn_only_graph,
                    "all_atoms": self.all_atoms,
                    "atom_radius": self.atom_radius,
                    "atom_max_neighbors": self.atom_max_neighbors,
                    "per_complex_timeout_s": int(os.environ.get("PDBBIND_PER_COMPLEX_TIMEOUT_S", "0")) or None,
                    "esm_embeddings_enabled": self.esm_embeddings_path is not None,
                }

                ctx = mp.get_context("fork")
                tasks = [(name, lm_for_name(name)) for name in batch_names]

                with ctx.Pool(
                    processes=num_workers,
                    initializer=_worker_init,
                    initargs=(cfg,),
                    maxtasksperchild=1,
                ) as pool:

                    it = pool.imap_unordered(_process_one_complex, tasks, chunksize=1)

                    with tqdm(total=len(tasks), desc=f"preprocessing batch {batch_idx}/{total_batches}") as pbar:

                        for ok, payload in it:
                            if ok:
                                cg_bytes, lig_bytes = payload
                                graphs.append(pickle.loads(cg_bytes))
                                ligands.append(_rdkit_mol_from_bytes(lig_bytes))
                                monitor.record_ok()

                                if len(graphs) % 50 == 0:
                                    gc.collect()
                            else:
                                monitor.record_skip(payload)

                                                                
                            try:
                                watchdog.update(
                                    attempted=monitor.attempted,
                                    ok=monitor.ok,
                                    skipped=sum(monitor.skips.values()),
                                )
                            except Exception:
                                pass

                            pbar.update(1)

                            if len(graphs) >= self.batch_size:
                                break

                            if monitor.should_check() and self.enable_health_check:
                                monitor.check_or_exit(context=f"batch={batch_idx}")

                if monitor.attempted >= monitor.min_attempts and self.enable_health_check:
                    monitor.check_or_exit(context=f"end_of_batch={batch_idx}")
            finally:
                try:
                    watchdog.stop()
                except Exception:
                    pass

            with open(os.path.join(self.full_cache_path, f"heterographs{batch_idx}.pkl"), "wb") as f:
                pickle.dump(graphs, f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{batch_idx}.pkl"), "wb") as f:
                pickle.dump(ligands, f)

            gc.collect()

    def _cache_complete(self):
        meta = os.path.join(self.full_cache_path, "heterographs.pkl")
        if not os.path.exists(meta):
            return False
        info = pickle.load(open(meta, "rb"))
        if isinstance(info, list):
            return True
        for i in range(info["num_batches"]):
            if not os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                return False
            if not os.path.exists(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl")):
                return False
        return True

    def _load_cache(self):
        graphs_all, ligs_all = [], []
        info = pickle.load(open(os.path.join(self.full_cache_path, "heterographs.pkl"), "rb"))

        if isinstance(info, list):
            return info, []

        for i in range(info["num_batches"]):
            graphs_all.extend(
                pickle.load(open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), "rb"))
            )
            ligs_all.extend(
                pickle.load(open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), "rb"), )
            )
        return graphs_all, ligs_all

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        graph = copy.deepcopy(self.complex_graphs[idx])
        if self.require_ligand:
            graph.mol = RemoveAllHs(copy.deepcopy(self.rdkit_ligands[idx]))

                                                            
        DROP_KEYS = {
            "coords",
            "seq",
            "sequence",
            "mask",
            "rmsd_matching",
            "cluster",
            "orig_seq",
            "to_keep",
            "chain_ids",
        }

        def _scrub_store(store):
                                                            
            try:
                for k in list(store.keys()):
                    if k in DROP_KEYS:
                        store.pop(k, None)
            except Exception:
                pass

                                                                
            for k in DROP_KEYS:
                if hasattr(store, k):
                    try:
                        delattr(store, k)
                    except Exception:
                        pass

                               
        _scrub_store(graph)

                           
        for ntype in getattr(graph, "node_types", []):
            try:
                _scrub_store(graph[ntype])
            except Exception:
                pass

                                       
        for etype in getattr(graph, "edge_types", []):
            try:
                _scrub_store(graph[etype])
            except Exception:
                pass

        return graph