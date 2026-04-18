import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                                                                              
import pickle
import multiprocessing as mp
import time
import queue
import threading
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass
import random
import copy
from torch_geometric.data import Batch
import itertools

import numpy as np
import torch
import json
from prody import confProDy
from rdkit import Chem
from rdkit.Chem import RemoveHs
from rdkit.Chem.rdchem import Mol as RDMol
from functools import partial
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import subgraph
from tqdm import tqdm
import sys
confProDy(verbosity='none')
from datasets.process_mols import get_lig_graph_with_matching, moad_extract_receptor_structure
from utils.utils import read_strings_from_txt


DEBUG_MAX_RECEPTOR_BATCHES = int(os.environ.get("MOAD_DEBUG_MAX_RECEPTOR_BATCHES", "0"))
DEBUG_MAX_LIGAND_BATCHES = int(os.environ.get("MOAD_DEBUG_MAX_LIGAND_BATCHES", "0"))


def _debug_batch_limit_reached(i, limit):
    return limit > 0 and i >= limit

                              
_RECEPTOR_CFG = None
_RECEPTOR_MOAD_DIR = None


def _scrub_graph_schema(g: HeteroData):


    DROP_KEYS = {
                                         
        "random_coords",
        "rmsd_matching",

                          
        "seq",
        "sequence",
        "orig_seq",
        "orig_sequence",

                         
        "mask",
        "to_keep",

                                  
        "cluster",

                           
        "coords",
    }

                      
    def _scrub_store(store):
                            
        for k in list(store.keys()):
            if k in DROP_KEYS:
                try:
                    store.pop(k)
                except Exception:
                    pass

                                
        for k in DROP_KEYS:
            if hasattr(store, k):
                try:
                    delattr(store, k)
                except Exception:
                    pass

                               
    try:
        _scrub_store(g)
    except Exception:
        pass

                           
    try:
        for ntype in getattr(g, "node_types", []):
            try:
                _scrub_store(g[ntype])
            except Exception:
                pass
    except Exception:
        pass

                           
    try:
        for etype in getattr(g, "edge_types", []):
            try:
                _scrub_store(g[etype])
            except Exception:
                pass
    except Exception:
        pass

    return g


def _normalize_numeric_to_tensors(g: HeteroData) -> HeteroData:


    def _to_tensor(v):
                          
        if torch.is_tensor(v):
            return v
                                     
        if isinstance(v, np.ndarray):
            try:
                return torch.from_numpy(v)
            except Exception:
                return v
                                       
        if isinstance(v, (list, tuple)) and len(v) > 0:
                                                                     
                                                                                                      
            if isinstance(v[0], np.ndarray):
                try:
                    return [torch.from_numpy(x) for x in v]
                except Exception:
                    return v
                             
            if isinstance(v[0], (int, float, np.integer, np.floating)):
                try:
                    return torch.tensor(v)
                except Exception:
                    return v
        return v

                                                 
    stores = [g]
    try:
        stores += [g[n] for n in getattr(g, "node_types", [])]
    except Exception:
        pass
    try:
        stores += [g[e] for e in getattr(g, "edge_types", [])]
    except Exception:
        pass

    for store in stores:
        try:
            for k in list(store.keys()):
                store[k] = _to_tensor(store[k])
        except Exception:
            pass

    return g


def _strip_non_tensor_fields_keep_required(g: HeteroData) -> HeteroData:


    SAFE_NON_TENSOR = {
        "name",
        "receptor_name",
    }

                                                                                   
    SAFE_LIST_OF_TENSORS = {
        "pos",
        "orig_pos",
    }

    stores = [g]
    try:
        stores += [g[n] for n in getattr(g, "node_types", [])]
    except Exception:
        pass
    try:
        stores += [g[e] for e in getattr(g, "edge_types", [])]
    except Exception:
        pass

    for store in stores:
        try:
            for k in list(store.keys()):
                if k in SAFE_NON_TENSOR:
                    continue
                v = store.get(k, None)

                                                                          
                try:
                    if isinstance(v, RDMol):
                        store.pop(k, None)
                        continue
                except Exception:
                    pass

                                                          
                if torch.is_tensor(v):
                    continue

                if k in SAFE_LIST_OF_TENSORS and isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                    continue

                store.pop(k, None)
        except Exception:
            pass

    return g


class _AlarmTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _AlarmTimeout()


def _receptor_worker_init():
                                                                                      
                                                                                   
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                             
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _build_receptor_from_name(args):

    import pickle
    import torch
    import signal

                                                
    try:
        name, cfg, moad_dir, emb = args
    except Exception:
        return ("skip", "bad_args")

    rec_path = os.path.join(moad_dir, "pdb_protein", name + "_protein.pdb")
    if not os.path.exists(rec_path):
        return ("skip", "missing_file")

                                                               
    timeout_s = int(cfg.get("per_receptor_timeout_s", 0) or 0)
    if os.name == "posix" and timeout_s > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_s)

    try:
        from torch_geometric.data import HeteroData
        from datasets.process_mols import moad_extract_receptor_structure

        cg = HeteroData()
        cg["receptor_name"] = name
        try:
            cg.receptor_name = name
        except Exception:
            pass

        try:
                                                                               
                                                                                  
            lm_embeddings = None
            if emb is not None:
                rec_emb_path = os.path.join(emb, f"{name}.pt")
                if os.path.exists(rec_emb_path):
                    try:
                        lm_embeddings = torch.load(rec_emb_path, map_location="cpu")
                    except Exception:
                        return ("skip", "lm_load_fail")
                else:
                    return ("skip", "missing_lm_embeddings")

            moad_extract_receptor_structure(
                path=rec_path,
                complex_graph=cg,
                neighbor_cutoff=cfg["receptor_radius"],
                max_neighbors=cfg["c_alpha_max_neighbors"],
                lm_embeddings=lm_embeddings,
                knn_only_graph=cfg["knn_only_graph"],
                all_atoms=cfg["all_atoms"],
                atom_cutoff=cfg["atom_radius"],
                atom_max_neighbors=cfg["atom_max_neighbors"],
            )
        except Exception:
            return ("skip", "receptor_extract_fail")


        try:
            if "receptor" not in cg.node_types:
                return ("skip", "no_receptor")
            rec = cg["receptor"]
            center = torch.mean(rec.pos, dim=0, keepdim=True)
            rec.pos -= center
            if cfg.get("all_atoms") and ("atom" in cg.node_types):
                cg["atom"].pos -= center
            cg.original_center = center
        except Exception:
            return ("skip", "centering_fail")

                                                                                             
        if not hasattr(cg, "original_center"):
            return ("skip", "missing_original_center")

        cg = _scrub_graph_schema(cg)
        return ("ok", pickle.dumps(cg, protocol=pickle.HIGHEST_PROTOCOL))

    except _AlarmTimeout:
        return ("skip", "timeout")

    except Exception:
        return ("skip", "worker_crash")

    finally:
        if os.name == "posix" and timeout_s > 0:
            signal.alarm(0)

class MOAD(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split='train', limit_complexes=0, chain_cutoff=None,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, esm_embeddings_sequences_path=None, require_ligand=False,
                 include_miscellaneous_atoms=False, keep_local_structures=False,
                 min_ligand_size=0, knn_only_graph=False, matching_tries=1, multiplicity=1,
                 max_receptor_size=None, remove_promiscuous_targets=None, unroll_clusters=False, remove_pdbbind=False,
                 enforce_timesplit=False, no_randomness=False, single_cluster_name=None, total_dataset_size=None, skip_matching=False):

        super(MOAD, self).__init__(root, transform)
        self.moad_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split = split
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.require_ligand = require_ligand
        self.esm_embeddings_path = esm_embeddings_path
        self.esm_embeddings_sequences_path = esm_embeddings_sequences_path
        self.keep_local_structures = keep_local_structures
        self.knn_only_graph = knn_only_graph
        self.matching_tries = matching_tries
        self.all_atoms = all_atoms
        self.multiplicity = multiplicity
        self.chain_cutoff = chain_cutoff
        self.no_randomness = no_randomness
        self.total_dataset_size = total_dataset_size
        self.skip_matching = skip_matching

        self.prot_cache_path = os.path.join(cache_path, f'MOAD12_limit{self.limit_complexes}_INDEX{self.split}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                            + (''if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + ('' if not self.include_miscellaneous_atoms else '_miscAtoms')
                                            + ('' if not self.knn_only_graph else '_knnOnly'))

        self.lig_cache_path = os.path.join(cache_path, f'MOAD12_limit{self.limit_complexes}_INDEX{self.split}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                            + ('' if not matching else f'_matching')
                                            + ('' if not skip_matching else f'skip')
                                            + (''if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if self.matching_tries == 1 else f'_tries{matching_tries}'))

        if DEBUG_MAX_RECEPTOR_BATCHES > 0 or DEBUG_MAX_LIGAND_BATCHES > 0:
            print(
                f"[DEBUG MODE] receptor_batches={DEBUG_MAX_RECEPTOR_BATCHES}, "
                f"ligand_batches={DEBUG_MAX_LIGAND_BATCHES}"
            )

        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.single_cluster_name = single_cluster_name
        if split == 'train':
            split = 'PDBBind'

        with open("./data/splits/MOAD_generalisation_splits.pkl", "rb") as f:
            self.split_clusters = pickle.load(f)[split]

        clustes_path = os.path.join(self.moad_dir, "new_cluster_to_ligands.pkl")
        with open(clustes_path, "rb") as f:
            self.cluster_to_ligands = pickle.load(f)
                                                                                                                    

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        if not self.check_all_receptors():
            os.makedirs(self.prot_cache_path, exist_ok=True)
            self.preprocessing_receptors()

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        if not os.path.exists(os.path.join(self.lig_cache_path, "ligands.pkl")):
            os.makedirs(self.lig_cache_path, exist_ok=True)
            self.preprocessing_ligands()

        print('loading ligands from memory: ', os.path.join(self.lig_cache_path, "ligands.pkl"))
        with open(os.path.join(self.lig_cache_path, "ligands.pkl"), 'rb') as f:
            ligs = pickle.load(f)

        if isinstance(ligs, list):
            self.ligands = {g.name: g for g in ligs}
        else:
            self.ligands = ligs

                                                   
        if self.require_ligand:
            rdkit_path = os.path.join(self.lig_cache_path, "rdkit_ligands.pkl")
            if os.path.exists(rdkit_path):
                with open(rdkit_path, 'rb') as rf:
                    mols = pickle.load(rf)
                try:
                    self.rdkit_ligands = {lig.name: mol for lig, mol in zip(self.ligands.values(), mols)}
                except Exception:
                                                         
                    self.rdkit_ligands = {}

                                                                                             
        receptors_names = set([k[:6] for k in self.ligands.keys()])
        self.collect_receptors(
            receptors_to_keep=receptors_names,
            max_receptor_size=max_receptor_size,
            remove_promiscuous_targets=remove_promiscuous_targets,
        )

        tot_before = len(self.ligands)
        self.ligands = {k: v for k, v in self.ligands.items() if k[:6] in self.receptors}
        print('removed', tot_before - len(self.ligands), 'ligands with no receptor out of', tot_before)

        if remove_pdbbind:
            complexes_pdbbind = read_strings_from_txt('data/splits/timesplit_no_lig_overlap_train') + read_strings_from_txt('data/splits/timesplit_no_lig_overlap_val')
            with open('data/BindingMOAD_2020_ab_processed_biounit/ecod_t_group_binding_site_assignment_dict_major_domain.pkl', 'rb') as f:
                pdbbind_to_cluster = pickle.load(f)
            clusters_pdbbind = set([pdbbind_to_cluster[c] for c in complexes_pdbbind])
            self.split_clusters = [c for c in self.split_clusters if c not in clusters_pdbbind]
            self.cluster_to_ligands = {k: v for k, v in self.cluster_to_ligands.items() if k not in clusters_pdbbind}
            ligand_accepted = []
            for c, ligands in self.cluster_to_ligands.items():
                ligand_accepted += ligands
            ligand_accepted = set(ligand_accepted)
            tot_before = len(self.ligands)
            self.ligands = {k: v for k, v in self.ligands.items() if k in ligand_accepted}
            print('removed', tot_before - len(self.ligands), 'ligands in overlap with PDBBind out of', tot_before)

        if enforce_timesplit:
            with open("data/splits/pdbids_2019", "r") as f:
                lines = f.readlines()
            pdbids_from2019 = []
            for i in range(6, len(lines), 4):
                pdbids_from2019.append(lines[i][18:22])

            pdbids_from2019 = set(pdbids_from2019)
            len_before = len(self.ligands)
            self.ligands = {k: v for k, v in self.ligands.items() if k[:4].upper() not in pdbids_from2019}
            print('removed', len_before - len(self.ligands), 'ligands from 2019 out of', len_before)

        if unroll_clusters:
            rec_keys = set([k[:6] for k in self.ligands.keys()])
            self.cluster_to_ligands = {k:[k2 for k2 in self.ligands.keys() if k2[:6] == k] for k in rec_keys}
            self.split_clusters = list(rec_keys)
        else:
            for c in self.cluster_to_ligands.keys():
                 self.cluster_to_ligands[c] = [v for v in self.cluster_to_ligands[c] if v in self.ligands]
            self.split_clusters = [c for c in self.split_clusters if len(self.cluster_to_ligands[c])>0]

        if os.environ.get("MOAD_DEBUG_STATS", "") == "1":
            print_statistics(self)
        list_names = [name for cluster in self.split_clusters for name in self.cluster_to_ligands[cluster]]
        with open(os.path.join(self.prot_cache_path, f'moad_{self.split}_names.txt'), 'w') as f:
            f.write('\n'.join(list_names))

    def len(self):
        return len(self.split_clusters) * self.multiplicity if self.total_dataset_size is None else self.total_dataset_size

    def get_by_name(self, ligand_name, cluster):
        assert ligand_name in self.ligands
        assert ligand_name[:6] in self.receptors
        ligand_graph = copy.deepcopy(self.ligands[ligand_name])
        complex_graph = copy.deepcopy(self.receptors[ligand_name[:6]])

        if False and self.keep_original and hasattr(ligand_graph['ligand'], 'orig_pos'):
            lig_path = os.path.join(self.moad_dir, 'pdb_superligand', ligand_name + '.pdb')
            lig = Chem.MolFromPDBFile(lig_path)
            formula = np.asarray([atom.GetSymbol() for atom in lig.GetAtoms()])

                                                                                   
            for ligand_comp in self.cluster_to_ligands[cluster]:
                if ligand_comp == ligand_name or ligand_comp[:6] != ligand_name[:6]:
                    continue

                lig_path_comp = os.path.join(self.moad_dir, 'pdb_superligand', ligand_comp + '.pdb')
                if not os.path.exists(lig_path_comp):
                    continue

                lig_comp = Chem.MolFromPDBFile(lig_path_comp)
                formula_comp = np.asarray([atom.GetSymbol() for atom in lig_comp.GetAtoms()])

                if formula.shape == formula_comp.shape and np.all(formula == formula_comp) and hasattr(
                        self.ligands[ligand_comp], 'orig_pos'):
                    print(f'Found complex {ligand_comp} to have the same complex/ligand pair, adding it into orig_pos')
                                                              
                    if not isinstance(ligand_graph['ligand'].orig_pos, list):
                        ligand_graph['ligand'].orig_pos = [ligand_graph['ligand'].orig_pos]
                    ligand_graph['ligand'].orig_pos.append(self.ligands[ligand_comp].orig_pos)

        for type in ligand_graph.node_types + ligand_graph.edge_types:
            for key, value in ligand_graph[type].items():
                complex_graph[type][key] = value
        complex_graph.name = ligand_graph.name
        if isinstance(complex_graph['ligand'].pos, list):
            for i in range(len(complex_graph['ligand'].pos)):
                complex_graph['ligand'].pos[i] -= complex_graph.original_center
        else:
            complex_graph['ligand'].pos -= complex_graph.original_center
                                                                           
                                                                                                 
        if False and self.require_ligand and self.split != "val":
            complex_graph.mol = copy.deepcopy(self.rdkit_ligands[ligand_name])

        if self.chain_cutoff:
            try:
                distances = torch.norm(
                    (torch.from_numpy(complex_graph['ligand'].orig_pos[0]) - complex_graph.original_center)
                    .unsqueeze(1) - complex_graph['receptor'].pos.unsqueeze(0),
                    dim=2
                )
                distances = distances.min(dim=0)[0]
            except Exception:
                return self.get(random.randint(0, self.len()))

            if torch.min(distances) >= self.chain_cutoff:
                return self.get(random.randint(0, self.len()))

            within_cutoff = distances < self.chain_cutoff
            chains_within_cutoff = torch.zeros(torch.max(complex_graph['receptor'].chain_ids) + 1)
            chains_within_cutoff.index_add_(0, complex_graph['receptor'].chain_ids, within_cutoff.float())
            chains_within_cutoff_bool = chains_within_cutoff > 0
            residues_to_keep = chains_within_cutoff_bool[complex_graph['receptor'].chain_ids]

            if self.all_atoms:
                atom_to_res_mapping = complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index[1]
                atoms_to_keep = residues_to_keep[atom_to_res_mapping]
                rec_remapper = (torch.cumsum(residues_to_keep.long(), dim=0) - 1)
                atom_to_res_new_mapping = rec_remapper[atom_to_res_mapping][atoms_to_keep]
                atom_res_edge_index = torch.stack([torch.arange(len(atom_to_res_new_mapping)), atom_to_res_new_mapping])

                complex_graph['atom'].x = complex_graph['atom'].x[atoms_to_keep]
                complex_graph['atom'].pos = complex_graph['atom'].pos[atoms_to_keep]
                complex_graph['atom', 'atom_contact', 'atom'].edge_index =\
                    subgraph(atoms_to_keep, complex_graph['atom', 'atom_contact', 'atom'].edge_index,
                             relabel_nodes=True)[0]
                complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index

            complex_graph['receptor'].pos = complex_graph['receptor'].pos[residues_to_keep]
            complex_graph['receptor'].x = complex_graph['receptor'].x[residues_to_keep]
            complex_graph['receptor'].side_chain_vecs = complex_graph['receptor'].side_chain_vecs[residues_to_keep]
            complex_graph['receptor', 'rec_contact', 'receptor'].edge_index =\
                subgraph(residues_to_keep, complex_graph['receptor', 'rec_contact', 'receptor'].edge_index,
                         relabel_nodes=True)[0]

                                                                                       
            try:
                extra_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
                complex_graph['receptor'].pos -= extra_center
                if isinstance(complex_graph['ligand'].pos, list):
                    for j in range(len(complex_graph['ligand'].pos)):
                        complex_graph['ligand'].pos[j] -= extra_center
                else:
                    complex_graph['ligand'].pos -= extra_center
                                                                              
                complex_graph.original_center += extra_center
            except Exception:
                pass

        try:
            if 'chain_ids' in complex_graph['receptor']:
                complex_graph['receptor'].pop('chain_ids')
        except Exception:
            pass

        complex_graph = _scrub_graph_schema(complex_graph)
        return complex_graph

    def get(self, idx):
        assert hasattr(self, "receptors"), "Receptors not loaded"
        assert hasattr(self, "ligands"), "Ligands not loaded"

        max_tries = 50

        for _ in range(max_tries):
                                                                             
                                                                                            
            if self.total_dataset_size is not None:
                idx = random.randint(0, len(self.split_clusters) - 1)

            idx = idx % len(self.split_clusters)
            cluster = self.split_clusters[idx]

            if self.no_randomness:
                ligand_name = sorted(self.cluster_to_ligands[cluster])[0]
            else:
                ligand_name = random.choice(self.cluster_to_ligands[cluster])

            g = self.get_by_name(ligand_name, cluster)

            return g

        raise RuntimeError("MOAD.get failed to sample a valid graph with embeddings enabled")

    def get_all_complexes(self):
        complexes = {}
        for cluster in self.split_clusters:
            for ligand_name in self.cluster_to_ligands[cluster]:
                complexes[ligand_name] = self.get_by_name(ligand_name, cluster)
        return complexes



    def preprocessing_receptors(self):
        print(f'Processing receptors from [{self.split}] and saving it to [{self.prot_cache_path}]')
        self._preprocessing_running = True

        complex_names_all = sorted([l for c in self.split_clusters for l in self.cluster_to_ligands[c]])
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]

        receptor_names_all = [l[:6] for l in complex_names_all]
        receptor_names_all = sorted(list(dict.fromkeys(receptor_names_all)))
        print(f'Loading {len(receptor_names_all)} receptors.')

                                                                                   
        lm_embeddings_dir = self.esm_embeddings_path
        if lm_embeddings_dir is not None:
            assert os.path.isdir(lm_embeddings_dir), f"Expected directory, got {lm_embeddings_dir}"
                                                                                  
            emb_dim = None
            try:
                for fn in os.listdir(lm_embeddings_dir):
                    if fn.endswith('.meta.json'):
                        with open(os.path.join(lm_embeddings_dir, fn), 'r') as mf:
                            meta = json.load(mf)
                        if meta.get('esm_dim') is not None:
                            emb_dim = int(meta.get('esm_dim'))
                            break
            except Exception:
                emb_dim = None
        else:
            lm_embeddings_dir = None
            emb_dim = None
                                                  
        self.esm_dim = emb_dim

                                                                                                            
        n_batches = (len(receptor_names_all) + 999) // 1000
        list_indices = list(range(n_batches))
        if DEBUG_MAX_RECEPTOR_BATCHES <= 0:
            random.shuffle(list_indices)
        else:
            list_indices = list_indices[:DEBUG_MAX_RECEPTOR_BATCHES]
        for i in list_indices:
            batch_path = os.path.join(self.prot_cache_path, f"receptors_batch_{i}.pkl")
            if os.path.exists(batch_path):
                continue
            receptor_names = receptor_names_all[1000*i:1000*(i+1)]
            receptor_graphs = []
            attempted = 0
            skipped = 0
            skipped_by_reason = {}

            stop_event = threading.Event()
            t = None

            def _watchdog(stop_event):
                while not stop_event.is_set():
                    tqdm.write(f"[watchdog_MOAD] batch {i}, processed {len(receptor_graphs)}")
                    stop_event.wait(30)

            try:
                                                                                                   
                cfg = {
                    "receptor_radius": self.receptor_radius,
                    "c_alpha_max_neighbors": self.c_alpha_max_neighbors,
                    "knn_only_graph": self.knn_only_graph,
                    "all_atoms": self.all_atoms,
                    "atom_radius": self.atom_radius,
                    "atom_max_neighbors": self.atom_max_neighbors,
                                                             
                    "per_receptor_timeout_s": int(os.environ.get("MOAD_PER_RECEPTOR_TIMEOUT_S", "0") or 0),
                                                                            
                    "esm_dim": (int(emb_dim) if emb_dim is not None else None),
                }

                num_workers_pool = int(os.environ.get("MOAD_RECEPTOR_WORKERS", "28"))
                num_workers_pool = max(1, min(num_workers_pool, max(1, mp.cpu_count() - 2)))

                                                                          
                prev_threads = None
                prev_interop = None
                try:
                    prev_threads = torch.get_num_threads()
                    prev_interop = torch.get_num_interop_threads()
                except Exception:
                    pass
                try:
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

                attempted = 0
                ok_count = 0
                skipped = 0
                skipped_by_reason = {}
                receptor_graphs = []

                                                           
                method = mp.get_start_method(allow_none=True)
                if method is not None and method != "fork":
                    raise RuntimeError(
                        f"Multiprocessing start method is '{method}'. This code expects 'fork' for worker behavior. "
                        "Set multiprocessing start method to 'fork' before creating the dataset (mp.set_start_method('fork')) "
                        "or run with num_workers=0 to disable multiprocessing."
                    )

                assert lm_embeddings_dir is None or isinstance(lm_embeddings_dir, str)

                ctx = mp.get_context("fork")
                with ctx.Pool(
                    processes=num_workers_pool,
                    initializer=_receptor_worker_init,
                    maxtasksperchild=1,
                ) as pool:
                                                                             
                    t = threading.Thread(target=_watchdog, args=(stop_event,))
                    t.daemon = True
                    t.start()
                                                                                                
                    task_args = [(name, cfg, self.moad_dir, lm_embeddings_dir) for name in receptor_names]
                    it = pool.imap_unordered(_build_receptor_from_name, task_args, chunksize=1)
                    with tqdm(
                        total=len(receptor_names),
                        desc=f'building receptors {i}/{len(receptor_names_all)//1000+1}',
                        file=sys.stdout,
                        dynamic_ncols=True,
                        leave=True,
                    ) as pbar2:
                        for status, payload in it:
                            attempted += 1
                            if status == "ok":
                                try:
                                    g = pickle.loads(payload)
                                    receptor_graphs.append(g)
                                    ok_count += 1
                                except Exception:
                                    skipped += 1
                                    skipped_by_reason["parent_unpickle_fail"] = skipped_by_reason.get("parent_unpickle_fail", 0) + 1
                            else:
                                skipped += 1
                                skipped_by_reason[payload] = skipped_by_reason.get(payload, 0) + 1
                            pbar2.update(1)

                                                
                try:
                    if prev_threads is not None:
                        torch.set_num_threads(int(prev_threads))
                    if prev_interop is not None:
                        torch.set_num_interop_threads(int(prev_interop))
                except Exception:
                    pass

                print('Number of receptors: ', len(receptor_graphs))

                with open(batch_path, 'wb') as f:
                    pickle.dump(receptor_graphs, f)

                stats = {
                    'attempted': attempted,
                    'ok': ok_count,
                    'skipped': skipped,
                    'skipped_by_reason': skipped_by_reason,
                    'num_workers': num_workers_pool,
                }
                with open(os.path.join(self.prot_cache_path, f"receptors_batch_{i}_stats.pkl"), 'wb') as sf:
                    pickle.dump(stats, sf)
                                                                     

                del receptor_graphs
                import gc
                gc.collect()
            finally:
                                                        
                try:
                    stop_event.set()
                    if t is not None:
                        t.join(timeout=5)
                except Exception:
                    pass

                print('Attempted:', attempted)
                print('Skipped:', skipped)
                print('Skip reasons:', skipped_by_reason)
        del self._preprocessing_running
        return receptor_names_all

    def check_all_receptors(self):
        complex_names_all = sorted([l for c in self.split_clusters for l in self.cluster_to_ligands[c]])
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        receptor_names_all = [l[:6] for l in complex_names_all]
        receptor_names_all = list(dict.fromkeys(receptor_names_all))
        n_batches = (len(receptor_names_all) + 999) // 1000
        if DEBUG_MAX_RECEPTOR_BATCHES > 0:
            n_batches = min(n_batches, DEBUG_MAX_RECEPTOR_BATCHES)

        for i in range(n_batches):
            if not os.path.exists(os.path.join(self.prot_cache_path, f"receptors_batch_{i}.pkl")):
                return False
        return True

    def collect_receptors(self, receptors_to_keep=None, max_receptor_size=None, remove_promiscuous_targets=None):
        complex_names_all = sorted([l for c in self.split_clusters for l in self.cluster_to_ligands[c]])
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        receptor_names_all = [l[:6] for l in complex_names_all]
        receptor_names_all = sorted(list(dict.fromkeys(receptor_names_all)))

        receptor_graphs_all = []
        total_recovered = 0
        n_keep = len(receptors_to_keep) if receptors_to_keep is not None else "ALL"
        print(f'Loading {len(receptor_names_all)} receptors to keep {n_keep}.')
        for i in range(len(receptor_names_all)//1000+1):
            batch_path = os.path.join(self.prot_cache_path, f"receptors_batch_{i}.pkl")
            print(f'prot path: {batch_path}')
            if not os.path.exists(batch_path):
                print(f'Missing receptor batch {batch_path}, skipping')
                continue
            with open(batch_path, 'rb') as f:
                l = pickle.load(f)
                total_recovered += len(l)

                if receptors_to_keep is not None:
                    l = [t for t in l if t.get("receptor_name", None) in receptors_to_keep]

                                                                      
                # No embedding validation here; accept whatever moad_extract_receptor_structure produced

                receptor_graphs_all.extend(l)

        cur_len = len(receptor_graphs_all)
        print(f"Kept {len(receptor_graphs_all)} receptors out of {len(receptor_names_all)} total and recovered {total_recovered}")

        if max_receptor_size is not None:
            receptor_graphs_all = [rec for rec in receptor_graphs_all if rec["receptor"].pos.shape[0] <= max_receptor_size]
            print(f"Kept {len(receptor_graphs_all)} receptors out of {cur_len} after filtering by size")
            cur_len = len(receptor_graphs_all)

        if remove_promiscuous_targets is not None:
            promiscuous_targets = set()
            for name in complex_names_all:
                l = name.split('_')
                if int(l[3]) > remove_promiscuous_targets:
                    promiscuous_targets.add(name[:6])
            receptor_graphs_all = [rec for rec in receptor_graphs_all if rec.get("receptor_name", None) not in promiscuous_targets]
            print(f"Kept {len(receptor_graphs_all)} receptors out of {cur_len} after removing promiscuous targets")

        self.receptors = {}
        for r in receptor_graphs_all:
            k = r.get("receptor_name", None)
            if k is None:
                continue
            self.receptors[k] = r
        return

                                                                                                    
    def preprocessing_ligands(self):
        print(f'Processing complexes from [{self.split}] and saving it to [{self.lig_cache_path}]')
        self._preprocessing_running = True

        complex_names_all = sorted([l for c in self.split_clusters for l in self.cluster_to_ligands[c]])
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} ligands.')

                                                                                                            
        n_batches = (len(complex_names_all) + 999) // 1000
        list_indices = list(range(n_batches))
        if DEBUG_MAX_LIGAND_BATCHES <= 0:
            random.shuffle(list_indices)
        else:
            list_indices = list_indices[:DEBUG_MAX_LIGAND_BATCHES]
        for i in list_indices:
            batch_lig_path = os.path.join(self.lig_cache_path, f"ligands_batch_{i}.pkl")
            batch_rdkit_path = os.path.join(self.lig_cache_path, f"rdkit_ligands_batch_{i}.pkl")
            if os.path.exists(batch_lig_path) and os.path.exists(batch_rdkit_path):
                continue
            complex_names = complex_names_all[1000*i:1000*(i+1)]
            ligand_graphs, rdkit_ligands = [], []
            attempted = 0
            skipped = 0
            skipped_by_reason = {}

            stop_event = threading.Event()

            def _watchdog_lig(stop_event, ligand_graphs_ref):
                while not stop_event.is_set():
                    tqdm.write(f"[watchdog] lig batch {i}, processed {len(ligand_graphs_ref)}")
                    stop_event.wait(30)

            t = threading.Thread(target=_watchdog_lig, args=(stop_event, ligand_graphs), daemon=True)
            t.start()

            try:
                                                                                           
                with tqdm(
                    total=len(complex_names),
                    desc=f'building ligands {i}/{len(complex_names_all)//1000+1}',
                    file=sys.stdout,
                    dynamic_ncols=True,
                    leave=True,
                ) as pbar:
                    for name in complex_names:
                        attempted += 1
                        if self.split == 'train':
                            lig_path = os.path.join(self.moad_dir, 'pdb_superligand', name + '.pdb')
                        else:
                            lig_path = os.path.join(self.moad_dir, 'pdb_ligand', name + '.pdb')

                        if not os.path.exists(lig_path):
                            skipped += 1
                            skipped_by_reason['missing_file'] = skipped_by_reason.get('missing_file', 0) + 1
                            pbar.update(1)
                            continue

                        raw = {"name": name, "lig_path": lig_path}
                        out = self.build_ligand_graph(raw)
                        del raw
                        if out is None:
                            skipped += 1
                            skipped_by_reason['build_failed'] = skipped_by_reason.get('build_failed', 0) + 1
                            pbar.update(1)
                            continue
                        ligand_graphs.append(out[0])
                        rdkit_ligands.append(out[1])
                        pbar.update(1)

                                                              
                for idx in range(len(ligand_graphs)):
                    try:
                        ligand_graphs[idx] = _scrub_graph_schema(ligand_graphs[idx])
                    except Exception:
                        pass

                with open(batch_lig_path, 'wb') as f:
                    pickle.dump(ligand_graphs, f)
                with open(batch_rdkit_path, 'wb') as f:
                    pickle.dump(rdkit_ligands, f)
                del ligand_graphs, rdkit_ligands
                import gc
                gc.collect()
            finally:
                                                        
                try:
                    stop_event.set()
                    t.join(timeout=5)
                except Exception:
                    pass

                print('Attempted:', attempted)
                print('Skipped:', skipped)
                print('Skip reasons:', skipped_by_reason)
                             
                stats = {
                    'attempted': attempted,
                    'skipped': skipped,
                    'skipped_by_reason': skipped_by_reason,
                }
                with open(os.path.join(self.lig_cache_path, f"ligands_batch_{i}_stats.pkl"), 'wb') as sf:
                    pickle.dump(stats, sf)
        del self._preprocessing_running

        ligand_graphs_all = []
        for i in range(n_batches):
            p = os.path.join(self.lig_cache_path, f"ligands_batch_{i}.pkl")
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                ligand_graphs_all.extend(pickle.load(f))
        with open(os.path.join(self.lig_cache_path, "ligands.pkl"), "wb") as f:
            pickle.dump(ligand_graphs_all, f)

        rdkit_ligands_all = []
        for i in range(n_batches):
            p = os.path.join(self.lig_cache_path, f"rdkit_ligands_batch_{i}.pkl")
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                rdkit_ligands_all.extend(pickle.load(f))
        with open(os.path.join(self.lig_cache_path, "rdkit_ligands.pkl"), "wb") as f:
            pickle.dump(rdkit_ligands_all, f)


    def build_ligand_graph(self, raw):
                                                             
        if mp.current_process().name != "MainProcess":
            raise RuntimeError(
                "Graph construction must run in the main process only."
            )
        name = raw["name"]
        lig = Chem.MolFromPDBFile(raw.get("lig_path"))
        if lig is None:
            return None

        if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
            return None

        try:
            if self.matching:
                smile = Chem.MolToSmiles(lig)
                if '.' in smile:
                    return None

            complex_graph = HeteroData()
            complex_graph['name'] = name

            Chem.SanitizeMol(lig)
            get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                        self.num_conformers, remove_hs=self.remove_hs, tries=self.matching_tries, skip_matching=self.skip_matching)
        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            return None

        if self.split != 'train':
            other_positions = [complex_graph['ligand'].orig_pos]
            nsplit = name.split('_')
            for i in range(100):
                new_file = os.path.join(self.moad_dir, 'pdb_ligand', f'{nsplit[0]}_{nsplit[1]}_{nsplit[2]}_{i}.pdb')
                if os.path.exists(new_file):
                    if i != int(nsplit[3]):
                        lig_alt = Chem.MolFromPDBFile(new_file)
                        lig_alt = RemoveHs(lig_alt, sanitize=True)
                        other_positions.append(lig_alt.GetConformer().GetPositions())
                else:
                    break
            complex_graph['ligand'].orig_pos = np.asarray(other_positions)

        complex_graph = _scrub_graph_schema(complex_graph)
        return complex_graph, lig


def print_statistics(dataset):
    statistics = ([], [], [], [], [], [])
    receptor_sizes = []

    for i in range(len(dataset)):
        complex_graph = dataset[i]
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        receptor_sizes.append(complex_graph['receptor'].pos.shape[0])
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)
        statistics[4].append(int(complex_graph.random_coords) if "random_coords" in complex_graph else -1)
        if "random_coords" in complex_graph and complex_graph.random_coords and "rmsd_matching" in complex_graph:
            statistics[5].append(complex_graph.rmsd_matching)

    if len(statistics[5]) == 0:
        statistics[5].append(-1)
    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching', 'random coordinates', 'random rmsd matching']
    print('Number of complexes: ', len(dataset))
    for i in range(len(name)):
        array = np.asarray(statistics[i])
        if array.size == 0:
            print(f"{name[i]}: mean nan, std nan, max nan (no data)")
        else:
            print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")

    return
