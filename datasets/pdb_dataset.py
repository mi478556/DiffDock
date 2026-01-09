# Significant contribution from Ben Fry

import copy
import os
import os.path
import pickle
import random
import gc
# Used ONLY for vandermers computation. Do NOT use for preprocessing.
from multiprocessing import Pool
from itertools import repeat

import numpy as np
import pandas as pd
# Prevent dataset worker processes from initializing CUDA when using forked multiprocessing.
# This makes dataset workers CUDA-agnostic and avoids hangs caused by CUDA context in child processes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import subgraph
from tqdm import tqdm

import time

from datasets.constants import aa_to_cg_indices, amino_acid_smiles, cg_rdkit_indices
from datasets.parse_chi import aa_long2short, atom_order
from datasets.process_mols import new_extract_receptor_structure, get_lig_graph, generate_conformer
from utils.torsion import get_transformation_mask


def _vandermers_init(coords_map, seq_map):
    """Initializer for worker processes computing vandermers.

    Sets module-global read-only maps so workers can access them without
    relying on forked memory sharing semantics.
    """
    global coords_by_name, seq_by_name
    coords_by_name = coords_map
    seq_by_name = seq_map

def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def compute_num_ca_neighbors(coords, cg_coords, idx, is_valid_bb_node, max_dist=5, buffer_residue_num=7):
    """
    Counts number of residues with heavy atoms within max_dist (Angstroms) of this sidechain that are not
    residues within +/- buffer_residue_num in primary sequence.
    From Ben's code
    Note: Gabriele removed the chain_index
    """

    # Extract coordinates of all residues in the protein.
    bb_coords = coords

    # Compute the indices that we should not consider interactions.
    excluded_neighbors = [idx - x for x in reversed(range(0, buffer_residue_num+1)) if (idx - x) >= 0]
    excluded_neighbors.extend([idx + x for x in range(1, buffer_residue_num+1)])

    # Create indices of an N x M distance matrix where N is num BB nodes and M is num CG nodes.
    e_idx = torch.stack([
        torch.arange(bb_coords.shape[0]).unsqueeze(-1).expand((-1, cg_coords.shape[0])).flatten(),
        torch.arange(cg_coords.shape[0]).unsqueeze(0).expand((bb_coords.shape[0], -1)).flatten()
    ])

    # Expand bb_coords and cg_coords into the same dimensionality.
    bb_coords_exp = bb_coords[e_idx[0]]
    cg_coords_exp = cg_coords[e_idx[1]].unsqueeze(1)

    # Every row is distance of chemical group to each atom in backbone coordinate frame.
    bb_exp_idces, _ = (torch.cdist(bb_coords_exp, cg_coords_exp).squeeze(-1) < max_dist).nonzero(as_tuple=True)
    bb_idces_within_thresh = torch.unique(e_idx[0][bb_exp_idces])

    # Only count residues that are not adjacent or origin in primary sequence and are valid backbone residues (fully resolved coordinate frame).
    bb_idces_within_thresh = bb_idces_within_thresh[~torch.isin(bb_idces_within_thresh, torch.tensor(excluded_neighbors)) & is_valid_bb_node[bb_idces_within_thresh]]

    return len(bb_idces_within_thresh)


def identify_valid_vandermers_raw(args):
    """
    Worker-safe variant that accepts very lightweight arguments:
    (name, max_dist, buffer_residue_num)

    The heavy per-protein data (`coords` and `seq`) are read from module-level
    read-only dictionaries `coords_by_name` and `seq_by_name` which must be
    created before the Pool is spawned. This relies on copy-on-write semantics
    of forked processes to avoid IPC copying.
    Returns (name, numpy_array_of_counts) or None on failure.
    """
    name, max_dist, buffer_residue_num = args

    # Read data from the shared, read-only maps populated by the parent before forking.
    coords_np = coords_by_name.get(name)
    seq = seq_by_name.get(name)
    if coords_np is None or seq is None:
        return None

    coords = torch.from_numpy(coords_np)
    is_valid_bb_node = (coords[:, :4].isnan().sum(dim=(1, 2)) == 0).bool()

    valid_cg_idces = []
    for idx, aa in enumerate(seq):
        if aa not in aa_to_cg_indices:
            valid_cg_idces.append(0)
            continue

        indices = aa_to_cg_indices[aa]
        cg_coordinates = coords[idx][indices]

        # remove chemical group residues that aren't fully resolved.
        if torch.any(cg_coordinates.isnan()).item():
            valid_cg_idces.append(0)
            continue

        nbr_count = compute_num_ca_neighbors(coords, cg_coordinates, idx, is_valid_bb_node,
                                             max_dist=max_dist, buffer_residue_num=buffer_residue_num)
        valid_cg_idces.append(nbr_count)

    return name, np.asarray(valid_cg_idces, dtype=np.int16)


def fast_identify_valid_vandermers(coords, seq, max_dist=5, buffer_residue_num=7):
    offset = 10000 + max_dist

    # Accept either torch.Tensor or np.ndarray
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    else:
        coords = np.asarray(coords)

    R = coords.shape[0]
    coords = coords.reshape(-1, 3)
    pdist_mat = squareform(pdist(coords))
    pdist_mat = pdist_mat.reshape((R, 14, R, 14))
    pdist_mat = np.nan_to_num(pdist_mat, nan=offset)
    pdist_mat = np.min(pdist_mat, axis=(1, 3))

    # compute pairwise distances
    pdist_mat = pdist_mat + np.diag(np.ones(len(seq)) * offset)
    for i in range(1, buffer_residue_num+1):
        pdist_mat += np.diag(np.ones(len(seq)-i) * offset, k=i) + np.diag(np.ones(len(seq)-i) * offset, k=-i)

    # get number of residues that are within max_dist of each other
    nbr_count = np.sum(pdist_mat < max_dist, axis=1)
    return np.asarray(nbr_count, dtype=np.int32)


def identify_valid_vandermers_fast_raw(args):
    """
    Torch-free worker using NumPy + SciPy only.
    Accepts (name, max_dist, buffer_residue_num) and reads coords_by_name
    and seq_by_name populated by the parent before forking.
    FAIL-FAST: any error aborts the entire job.
    """
    name, max_dist, buffer_residue_num = args

    coords = coords_by_name.get(name)
    seq = seq_by_name.get(name)
    if coords is None or seq is None:
        raise RuntimeError(f"Missing coords or seq for {name}")

    counts = fast_identify_valid_vandermers(
        coords,
        seq,
        max_dist=max_dist,
        buffer_residue_num=buffer_residue_num,
    )

    if counts is None:
        raise RuntimeError(f"fast_identify_valid_vandermers returned None for {name}")

    counts = np.asarray(counts, dtype=np.int16)

    if counts.ndim != 1 or counts.shape[0] != len(seq):
        raise RuntimeError(
            f"Invalid vandermers shape for {name}: {counts.shape}, expected ({len(seq)},)"
        )

    return name, counts


# Note: the NumPy-only fast worker was removed to preserve original
# torch-based semantics in workers. Use `identify_valid_vandermers_raw`
# which converts NumPy coords to torch inside the worker and calls
# the original helpers.


def compute_cg_features(aa, aa_smile):
    """
    Given an amino acid and a smiles string returns the stacked tensor of chemical group atom encodings.
    The order of the output tensor rows corresponds to the index the atoms appear in aa_to_cg_indices from constants.
    """

    # Handle any residues that we don't have chemical groups for (ex: GLY if not using bb_cnh and bb_cco)
    aa_short = aa_long2short[aa]
    if aa_short not in aa_to_cg_indices:
        return None

    # Create rdkit molecule from smiles string.
    mol = Chem.MolFromSmiles(aa_smile)

    complex_graph = HeteroData()
    get_lig_graph(mol, complex_graph)

    atoms_to_keep = torch.tensor([i for i, _ in cg_rdkit_indices[aa].items()]).long()
    complex_graph['ligand', 'ligand'].edge_index, complex_graph['ligand', 'ligand'].edge_attr = \
        subgraph(atoms_to_keep, complex_graph['ligand', 'ligand'].edge_index, complex_graph['ligand', 'ligand'].edge_attr, relabel_nodes=True)
    complex_graph['ligand'].x = complex_graph['ligand'].x[atoms_to_keep]

    edge_mask, mask_rotate = get_transformation_mask(complex_graph)
    complex_graph['ligand'].edge_mask = torch.tensor(edge_mask)
    complex_graph['ligand'].mask_rotate = mask_rotate
    return complex_graph


class PDBSidechain(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split='train', limit_complexes=0,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, remove_hs=True, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, sequences_to_embeddings=None,
                 knn_only_graph=True, multiplicity=1, vandermers_max_dist=5, vandermers_buffer_residue_num=7,
                 vandermers_min_contacts=5, remove_second_segment=False, merge_clusters=1, vandermers_extraction=True,
                 add_random_ligand=False):

        super(PDBSidechain, self).__init__(root, transform)
        assert remove_hs == True, "not implemented yet"
        self.root = root
        self.split = split
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.knn_only_graph = knn_only_graph
        self.multiplicity = multiplicity
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.num_workers = num_workers
        self.sequences_to_embeddings = sequences_to_embeddings
        self.remove_second_segment = remove_second_segment
        self.merge_clusters = merge_clusters
        self.vandermers_extraction = vandermers_extraction
        self.add_random_ligand = add_random_ligand
        self.all_atoms = all_atoms
        self.atom_radius = atom_radius
        self.atom_max_neighbors = atom_max_neighbors

        if vandermers_extraction:
            self.cg_node_feature_lookup_dict = {aa_long2short[aa]: compute_cg_features(aa, aa_smile) for aa, aa_smile in
                                           amino_acid_smiles.items()}

        self.cache_path = os.path.join(cache_path, f'PDB3_limit{self.limit_complexes}_INDEX{self.split}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                            + (''if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + ('' if not self.knn_only_graph else '_knnOnly'))
        self.read_split()

        if not self.check_all_proteins():
            os.makedirs(self.cache_path, exist_ok=True)
            self.preprocess()

        self.vandermers_max_dist = vandermers_max_dist
        self.vandermers_buffer_residue_num = vandermers_buffer_residue_num
        self.vandermers_min_contacts = vandermers_min_contacts
        self.collect_proteins()

        filtered_proteins = []
        if vandermers_extraction:
            for complex_graph in tqdm(self.protein_graphs):
                if complex_graph.name in self.vandermers and torch.any(self.vandermers[complex_graph.name] >= self.vandermers_min_contacts):
                    filtered_proteins.append(complex_graph)
            print(f"Computed vandermers and kept {len(filtered_proteins)} proteins out of {len(self.protein_graphs)}")
        else:
            filtered_proteins = self.protein_graphs

        second_filter = []
        for complex_graph in tqdm(filtered_proteins):
            if sequences_to_embeddings is None or complex_graph.orig_seq in sequences_to_embeddings:
                second_filter.append(complex_graph)
        print(f"Checked embeddings available and kept {len(second_filter)} proteins out of {len(filtered_proteins)}")
        
        self.protein_graphs = second_filter

        # filter clusters that have no protein graphs
        self.split_clusters = list(set([g.cluster for g in self.protein_graphs]))
        self.cluster_to_complexes = {c: [] for c in self.split_clusters}
        for p in self.protein_graphs:
            self.cluster_to_complexes[p['cluster']].append(p)
        self.split_clusters = [c for c in self.split_clusters if len(self.cluster_to_complexes[c]) > 0]
        print("Total elements in set", len(self.split_clusters) * self.multiplicity // self.merge_clusters)

        self.name_to_complex = {p.name: p for p in self.protein_graphs}
        self.define_probabilities()

        if self.add_random_ligand:
            # read csv with all smiles
            with open('data/smiles_list.csv', 'r') as f:
                self.smiles_list = f.readlines()
            self.smiles_list = [s.split(',')[0] for s in self.smiles_list]

    def define_probabilities(self):
        if not self.vandermers_extraction:
            return

        if self.vandermers_min_contacts is not None:
            self.probabilities = torch.arange(1000) - self.vandermers_min_contacts + 1
            self.probabilities[:self.vandermers_min_contacts] = 0
        else:
            with open('data/pdbbind_counts.pkl', 'rb') as f:
                pdbbind_counts = pickle.load(f)

            pdb_counts = torch.ones(1000)
            for contacts in self.vandermers.values():
                pdb_counts.index_add_(0, contacts, torch.ones(contacts.shape))
            print(pdbbind_counts[:30])
            print(pdb_counts[:30])

            self.probabilities = pdbbind_counts / pdb_counts
            self.probabilities[:7] = 0

    def len(self):
        return len(self.split_clusters) * self.multiplicity // self.merge_clusters

    def get(self, idx=None, protein=None, smiles=None):
        assert idx is not None or (protein is not None and smiles is not None), "provide idx or protein or smile"

        if protein is None or smiles is None:
            idx = idx % len(self.split_clusters)
            if self.merge_clusters > 1:
                idx = idx * self.merge_clusters
                idx = idx + random.randint(0, self.merge_clusters - 1)
                idx = min(idx, len(self.split_clusters) - 1)
            cluster = self.split_clusters[idx]
            protein_graph = copy.deepcopy(random.choice(self.cluster_to_complexes[cluster]))
        else:
            protein_graph = copy.deepcopy(self.name_to_complex[protein])

        if self.sequences_to_embeddings is not None:
            #print(self.sequences_to_embeddings[protein_graph.orig_seq].shape, len(protein_graph.orig_seq), protein_graph.to_keep.shape)
            if len(protein_graph.orig_seq) != len(self.sequences_to_embeddings[protein_graph.orig_seq]):
                print('problem with ESM embeddings')
                return self.get(random.randint(0, self.len()))

            lm_embeddings = self.sequences_to_embeddings[protein_graph.orig_seq][protein_graph.to_keep]
            protein_graph['receptor'].x = torch.cat([protein_graph['receptor'].x, lm_embeddings], dim=1)

        if self.vandermers_extraction:
            # select sidechain to remove
            vandermers_contacts = self.vandermers[protein_graph.name]
            vandermers_probs = self.probabilities[vandermers_contacts].numpy()

            if not np.any(vandermers_contacts.numpy() >= 10):
                print('no vandarmers >= 10 retrying with new one')
                return self.get(random.randint(0, self.len()))

            sidechain_idx = np.random.choice(np.arange(len(vandermers_probs)), p=vandermers_probs / np.sum(vandermers_probs))

            # remove part of the sequence
            residues_to_keep = np.ones(len(protein_graph.seq), dtype=bool)
            residues_to_keep[max(0, sidechain_idx - self.vandermers_buffer_residue_num):
                             min(sidechain_idx + self.vandermers_buffer_residue_num + 1, len(protein_graph.seq))] = False

            if self.remove_second_segment:
                pos_idx = protein_graph['receptor'].pos[sidechain_idx]
                limit_closeness = 10
                far_enough = torch.sum((protein_graph['receptor'].pos - pos_idx[None, :]) ** 2, dim=-1) > limit_closeness ** 2
                vandermers_probs = vandermers_probs * far_enough.float().numpy()
                vandermers_probs[max(0, sidechain_idx - self.vandermers_buffer_residue_num):
                                 min(sidechain_idx + self.vandermers_buffer_residue_num + 1, len(protein_graph.seq))] = 0
                if np.all(vandermers_probs<=0):
                    print('no second vandermer available retrying with new one')
                    return self.get(random.randint(0, self.len()))
                sc2_idx = np.random.choice(np.arange(len(vandermers_probs)), p=vandermers_probs / np.sum(vandermers_probs))

                residues_to_keep[max(0, sc2_idx - self.vandermers_buffer_residue_num):
                                 min(sc2_idx + self.vandermers_buffer_residue_num + 1, len(protein_graph.seq))] = False

            residues_to_keep = torch.from_numpy(residues_to_keep)
            protein_graph['receptor'].pos = protein_graph['receptor'].pos[residues_to_keep]
            protein_graph['receptor'].x = protein_graph['receptor'].x[residues_to_keep]
            protein_graph['receptor'].side_chain_vecs = protein_graph['receptor'].side_chain_vecs[residues_to_keep]
            protein_graph['receptor', 'rec_contact', 'receptor'].edge_index = \
                subgraph(residues_to_keep, protein_graph['receptor', 'rec_contact', 'receptor'].edge_index, relabel_nodes=True)[0]

            # create the sidechain ligand
            sidechain_aa = protein_graph.seq[sidechain_idx]
            ligand_graph = self.cg_node_feature_lookup_dict[sidechain_aa]
            ligand_graph['ligand'].pos = protein_graph.coords[sidechain_idx][protein_graph.mask[sidechain_idx]]

            for type in ligand_graph.node_types + ligand_graph.edge_types:
                for key, value in ligand_graph[type].items():
                    protein_graph[type][key] = value

            protein_graph['ligand'].orig_pos = protein_graph['ligand'].pos.numpy()
            protein_center = torch.mean(protein_graph['receptor'].pos, dim=0, keepdim=True)
            protein_graph['receptor'].pos = protein_graph['receptor'].pos - protein_center
            protein_graph['ligand'].pos = protein_graph['ligand'].pos - protein_center
            protein_graph.original_center = protein_center
            protein_graph['receptor_name'] = protein_graph.name
        else:
            protein_center = torch.mean(protein_graph['receptor'].pos, dim=0, keepdim=True)
            protein_graph['receptor'].pos = protein_graph['receptor'].pos - protein_center
            protein_graph.original_center = protein_center
            protein_graph['receptor_name'] = protein_graph.name

        if self.add_random_ligand:
            if smiles is not None:
                mol = MolFromSmiles(smiles)
                generate_conformer(mol)
            else:
                success = False
                while not success:
                    smiles = random.choice(self.smiles_list)
                    mol = MolFromSmiles(smiles)
                    success = not generate_conformer(mol)

            lig_graph = HeteroData()
            get_lig_graph(mol, lig_graph)

            edge_mask, mask_rotate = get_transformation_mask(lig_graph)
            lig_graph['ligand'].edge_mask = torch.tensor(edge_mask)
            lig_graph['ligand'].mask_rotate = mask_rotate
            lig_graph['ligand'].smiles = smiles
            lig_graph['ligand'].pos = lig_graph['ligand'].pos - torch.mean(lig_graph['ligand'].pos, dim=0, keepdim=True)

            for type in lig_graph.node_types + lig_graph.edge_types:
                for key, value in lig_graph[type].items():
                    protein_graph[type][key] = value

        for a in ['random_coords', 'coords', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq', 'to_keep', 'chain_ids']:
            if hasattr(protein_graph, a):
                delattr(protein_graph, a)
            if hasattr(protein_graph['receptor'], a):
                delattr(protein_graph['receptor'], a)

        return protein_graph

    def read_split(self):
        # read CSV file
        df = pd.read_csv(self.root + "/list.csv")
        print("Loaded list CSV file")

        # get clusters and filter by split
        if self.split == "train":
            val_clusters = set(read_strings_from_txt(self.root + "/valid_clusters.txt"))
            test_clusters = set(read_strings_from_txt(self.root + "/test_clusters.txt"))
            clusters = df["CLUSTER"].unique()
            clusters = [int(c) for c in clusters if c not in val_clusters and c not in test_clusters]
        elif self.split == "val":
            clusters = [int(s) for s in read_strings_from_txt(self.root + "/valid_clusters.txt")]
        elif self.split == "test":
            clusters = [int(s) for s in read_strings_from_txt(self.root + "/test_clusters.txt")]
        else:
            raise ValueError("Split must be train, val or test")
        print(self.split, "clusters", len(clusters))
        clusters = set(clusters)

        self.chains_in_cluster = []
        complexes_in_cluster = set()
        for chain, cluster in zip(df["CHAINID"], df["CLUSTER"]):
            if cluster not in clusters:
                continue
            # limit to one chain per complex
            if chain[:4] not in complexes_in_cluster:
                self.chains_in_cluster.append((chain, cluster))
                complexes_in_cluster.add(chain[:4])
        print("Filtered chains in cluster", len(self.chains_in_cluster))

        if self.limit_complexes > 0:
            self.chains_in_cluster = self.chains_in_cluster[:self.limit_complexes]

    def check_all_proteins(self):
        for i in range(len(self.chains_in_cluster)//10000+1):
            if not os.path.exists(os.path.join(self.cache_path, f"protein_graphs{i}.pkl")):
                return False
        return True

    def collect_proteins(self):
        self.protein_graphs = []
        self.vandermers = {}
        total_recovered = 0
        print(f'Loading {len(self.chains_in_cluster)} protein graphs.')
        list_indices = list(range(len(self.chains_in_cluster) // 10000 + 1))
        random.shuffle(list_indices)
        for i in tqdm(list_indices, desc="load cached protein_graphs", unit="batch"):
            with open(os.path.join(self.cache_path, f"protein_graphs{i}.pkl"), 'rb') as f:
                l = pickle.load(f)
                total_recovered += len(l)
                self.protein_graphs.extend(l)

            if not self.vandermers_extraction:
                continue

            if os.path.exists(os.path.join(self.cache_path, f'vandermers{i}_{self.vandermers_max_dist}_{self.vandermers_buffer_residue_num}.pkl')):
                with open(os.path.join(self.cache_path, f'vandermers{i}_{self.vandermers_max_dist}_{self.vandermers_buffer_residue_num}.pkl'), 'rb') as f:
                    vandermers = pickle.load(f)
                    self.vandermers.update(vandermers)
                continue

            vandermers = {}
            # Create read-only maps for coords and seqs before forking so children inherit
            # them via copy-on-write (no IPC copying). Do NOT mutate these while the pool is alive.
            global coords_by_name, seq_by_name
            coords_by_name = {g.name: g.coords.numpy() for g in l}
            seq_by_name = {g.name: g.seq for g in l}

            # prepare lightweight argument tuples (name, max_dist, buffer)
            arguments = ((g.name, self.vandermers_max_dist, self.vandermers_buffer_residue_num) for g in l)

            if self.num_workers > 1:
                # use an initializer to populate worker globals reliably across start methods
                with Pool(self.num_workers, initializer=_vandermers_init, initargs=(coords_by_name, seq_by_name)) as p:
                    with tqdm(total=len(l), desc=f'computing vandermers {i}') as pbar:
                        # fail-fast: let worker exceptions propagate; use chunksize=1 to surface failures quickly
                        for name, counts in p.imap_unordered(identify_valid_vandermers_fast_raw, arguments, chunksize=1):
                            vandermers[name] = counts
                            pbar.update()
            else:
                with tqdm(total=len(l), desc=f'computing vandermers {i}') as pbar:
                    for name, counts in map(identify_valid_vandermers_fast_raw, arguments):
                        vandermers[name] = counts
                        pbar.update()

            # Convert numpy arrays to torch tensors after the pool has shut down
            # Fail-fast on any conversion error so issues surface loudly.
            for k, v in list(vandermers.items()):
                vandermers[k] = torch.from_numpy(v)

            t0 = time.time()
            with open(os.path.join(self.cache_path, f'vandermers{i}_{self.vandermers_max_dist}_{self.vandermers_buffer_residue_num}.pkl'), 'wb') as f:
                pickle.dump(vandermers, f)
            print(f"Saved vandermers batch {i}: {len(vandermers)} items -> {os.path.join(self.cache_path, f'vandermers{i}_{self.vandermers_max_dist}_{self.vandermers_buffer_residue_num}.pkl')} in {time.time()-t0:.1f}s")
            self.vandermers.update(vandermers)

        print(f"Kept {len(self.protein_graphs)} proteins out of {len(self.chains_in_cluster)} total")
        return

    def preprocess(self):
        # Preprocess in batches. Load raw files in worker-safe manner, then build graphs
        # in the parent process to avoid CUDA/tensor ownership issues in forked workers.
        list_indices = list(range(len(self.chains_in_cluster) // 10000 + 1))
        random.shuffle(list_indices)

        for i in tqdm(list_indices, desc="preprocess batches", unit="batch"):
            out_path = os.path.join(self.cache_path, f"protein_graphs{i}.pkl")
            if os.path.exists(out_path):
                continue

            chains = self.chains_in_cluster[10000 * i:10000 * (i + 1)]

            print(f"Loading batch {i}, {len(chains)} chains")

            # Load raw records sequentially in parent process (CPU-only)
            raws = self.load_chain_batch(chains, desc=f"load raw batch {i}")

            protein_graphs = []
            for raw in tqdm(raws, desc=f"build graphs batch {i}", unit="prot", leave=False):
                protein_graphs.append(self.build_graph_from_raw(raw))

            # Save with timing to help diagnose slow HDD writes
            t0 = time.time()
            with open(out_path, "wb") as f:
                pickle.dump(protein_graphs, f)
            print(f"Saved batch {i}: {len(protein_graphs)} graphs -> {out_path} in {time.time()-t0:.1f}s")

            # explicit cleanup
            del raws
            del protein_graphs
            gc.collect()

        print("Finished preprocessing and saving protein graphs")

    def load_chain_batch(self, chains, desc=None):
        """Load raw data for a batch of chains.

        Runs in the parent process.
        Returns only plain Python objects and NumPy arrays.
        Must not create torch tensors or HeteroData objects.
        """
        raws = []
        it = tqdm(chains, desc=desc or "loading raw chains", unit="chain", leave=False)
        for chain, cluster in it:
            path = self.root + f"/pdb/{chain[1:3]}/{chain}.pt"
            if not os.path.exists(path):
                # missing file; skip
                continue

            data = torch.load(path, map_location="cpu")

            orig_seq = data.get("seq")
            coords = data.get("xyz")
            mask = data.get("mask")

            # remove residues with NaN backbone coordinates
            to_keep = ~torch.any(torch.isnan(coords[:, :4, 0]), dim=1)
            if not to_keep.any():
                continue

            coords_np = coords[to_keep].numpy().astype("float32")
            mask_np = mask[to_keep].numpy()
            seq = "".join(np.asarray(list(orig_seq))[to_keep.numpy()].tolist())

            raws.append({
                "chain": chain,
                "cluster": cluster,
                "seq": seq,
                "coords": coords_np,
                "mask": mask_np,
                "orig_seq": orig_seq,
                "to_keep": to_keep.numpy(),
            })

        return raws

    def build_graph_from_raw(self, raw):
        """Construct HeteroData graph from a raw record. Runs in parent process only."""
        complex_graph = HeteroData()
        complex_graph["name"] = raw["chain"]

        new_extract_receptor_structure(
            raw["seq"],
            raw["coords"],
            complex_graph=complex_graph,
            neighbor_cutoff=self.receptor_radius,
            max_neighbors=self.c_alpha_max_neighbors,
            knn_only_graph=self.knn_only_graph,
            all_atoms=self.all_atoms,
            atom_cutoff=self.atom_radius,
            atom_max_neighbors=self.atom_max_neighbors,
        )

        complex_graph.seq = raw["seq"]
        complex_graph.coords = torch.from_numpy(raw["coords"]) 
        complex_graph.mask = torch.from_numpy(raw["mask"]).bool()
        complex_graph.cluster = raw["cluster"]
        complex_graph.orig_seq = raw["orig_seq"]
        complex_graph.to_keep = torch.from_numpy(raw["to_keep"]).bool()

        return complex_graph


if __name__ == "__main__":
    dataset = PDBSidechain(root="data/pdb_2021aug02_sample", split="train", multiplicity=1, limit_complexes=150)
    print(len(dataset))
    print(dataset[0])
    for p in dataset:
        print(p)
        pass
