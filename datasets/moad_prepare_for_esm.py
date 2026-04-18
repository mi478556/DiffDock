#!/usr/bin/env python3
"""Prepare MOAD sequences for ESM embedding extraction.

This script scans a MOAD processed folder for protein PDB files, extracts
per-chain amino-acid sequences, and writes a FASTA file suitable for
`esm.scripts.extract`. It preserves original PDB IDs in headers of the
form `<pdbid>_chain_<i>` so the extractor produces per-sequence files with
matching names (e.g. `6upj_chain_1.pt`). Optionally writes a pickle mapping
of id -> sequence as well.

Usage:
  python datasets/moad_prepare_for_esm.py \
    --data_dir data/BindingMOAD_2020_ab_processed_biounit/pdb_protein \
    --out_fasta data/BindingMOAD_2020_ab_processed_biounit/sequences_to_id.fasta \
    --out_pickle data/moad_prepared_for_esm.pkl
"""

import os
import pickle
from argparse import ArgumentParser
from typing import Dict, List

from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm


def find_pdb_ids(data_dir: str) -> List[str]:
    files = os.listdir(data_dir)
    ids = set()
    for fn in files:
        # skip hidden files and macOS resource forks (e.g. .DS_Store, ._filename)
        if fn.startswith('.'):
            continue
        # look for common protein file patterns
        for suffix in ("_protein_chain_removed.pdb", "_protein_processed.pdb", "_protein.pdb"):
            if fn.endswith(suffix):
                # preserve the full prefix (e.g. '6hd6_1' from '6hd6_1_protein.pdb')
                base = fn.rsplit('_protein', 1)[0]
                ids.add(base)
                break
    return sorted(ids)


def choose_pdb_path(data_dir: str, pdb_id: str) -> str:
    # preference order
    candidates = [f"{pdb_id}_protein_chain_removed.pdb",
                  f"{pdb_id}_protein_processed.pdb",
                  f"{pdb_id}_protein.pdb"]
    for c in candidates:
        p = os.path.join(data_dir, c)
        if os.path.exists(p):
            return p
    return ""


def extract_sequences_from_pdb(pdb_path: str) -> List[str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)[0]
    sequences = []
    for chain in structure:
        seq = ''
        for residue in chain:
            if residue.get_resname() == 'HOH':
                continue
            # check backbone atoms present
            atom_names = {atom.get_name() for atom in residue}
            if not ({'CA', 'N', 'C'}.issubset(atom_names)):
                continue
            try:
                # use three_to_one mapping from constants if available; fallback to 'X'
                from datasets.constants import three_to_one
                aa = three_to_one.get(residue.get_resname(), 'X')
            except Exception:
                aa = 'X'
            seq += aa
        if len(seq) > 0:
            sequences.append(seq)
    return sequences


def main():
    p = ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True,
                   help='Directory containing MOAD PDB files (per-complex files)')
    p.add_argument('--out_fasta', type=str, required=True,
                   help='Output FASTA for ESM extractor (headers will be <pdbid>_chain_<i>)')
    p.add_argument('--out_pickle', type=str, default='',
                   help='Optional: write pickle mapping id_chain -> sequence')
    p.add_argument('--skip_missing', action='store_true', default=False,
                   help='If set, do not print warnings for missing PDB files')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_fasta) or '.', exist_ok=True)

    pdb_ids = find_pdb_ids(args.data_dir)
    if not pdb_ids:
        print(f"No pdb files matching expected patterns found in {args.data_dir}")
        return

    records = []
    mapping: Dict[str, str] = {}

    for pdb_id in tqdm(pdb_ids, desc='Scanning MOAD'):
        pdb_path = choose_pdb_path(args.data_dir, pdb_id)
        if not pdb_path:
            if not args.skip_missing:
                print(f"Skipping {pdb_id}: no protein file found")
            continue
        try:
            seqs = extract_sequences_from_pdb(pdb_path)
        except Exception as e:
            print(f"Error parsing {pdb_path}: {e}")
            continue
        for i, seq in enumerate(seqs):
            header = f"{pdb_id}_chain_{i}"
            records.append(SeqRecord(Seq(seq), id=header, description=''))
            mapping[header] = seq

    # write fasta
    SeqIO.write(records, args.out_fasta, 'fasta')
    print(f"Wrote {len(records)} sequences to {args.out_fasta}")

    # optional pickle
    if args.out_pickle:
        with open(args.out_pickle, 'wb') as f:
            pickle.dump(mapping, f)
        print(f"Wrote pickle mapping to {args.out_pickle}")


if __name__ == '__main__':
    main()
