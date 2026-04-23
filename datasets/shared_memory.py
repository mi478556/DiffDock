import torch

from datasets.loader import CombineDatasets
from datasets.moad import MOAD
from datasets.pdb_dataset import PDBSidechain
from datasets.pdbbind import PDBBind


def _share_nested_tensors_(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if torch.is_tensor(obj):
        if obj.device.type == "cpu":
            obj.share_memory_()
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _share_nested_tensors_(v, seen)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _share_nested_tensors_(v, seen)
        return
    if hasattr(obj, "stores"):
        try:
            for store in obj.stores:
                for _, v in store.items():
                    _share_nested_tensors_(v, seen)
        except Exception:
            pass


def share_pdbbind_dataset_(dataset):
    if not isinstance(dataset, PDBBind):
        return dataset
    for graph in dataset.complex_graphs:
        _share_nested_tensors_(graph)
    return dataset


def share_moad_dataset_(dataset):
    if not isinstance(dataset, MOAD):
        return dataset
    ligands = dataset.ligands.values() if isinstance(dataset.ligands, dict) else dataset.ligands
    for graph in ligands:
        _share_nested_tensors_(graph)
    for graph in dataset.receptors.values():
        _share_nested_tensors_(graph)
    return dataset


def share_pdbsidechain_dataset_(dataset):
    if not isinstance(dataset, PDBSidechain):
        return dataset
    if not getattr(dataset, "_is_loaded", False):
        dataset.load()
    if dataset.protein_graphs is not None:
        for graph in dataset.protein_graphs:
            _share_nested_tensors_(graph)
    if dataset.vandermers is not None:
        for contacts in dataset.vandermers.values():
            _share_nested_tensors_(contacts)
    if dataset.sequences_to_embeddings is not None:
        for emb in dataset.sequences_to_embeddings.values():
            _share_nested_tensors_(emb)
    return dataset


def share_dataset_tree_(dataset):
    if isinstance(dataset, CombineDatasets):
        share_dataset_tree_(dataset.dataset1)
        share_dataset_tree_(dataset.dataset2)
        return dataset
    if isinstance(dataset, PDBBind):
        return share_pdbbind_dataset_(dataset)
    if isinstance(dataset, MOAD):
        return share_moad_dataset_(dataset)
    if isinstance(dataset, PDBSidechain):
        return share_pdbsidechain_dataset_(dataset)
    return dataset
