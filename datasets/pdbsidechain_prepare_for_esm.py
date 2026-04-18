#!/usr/bin/env python3
"""Prepare pdbsidechain sequences for ESM extraction and aggregation.

This script scans per-chain .pt files under a pdbsidechain root (default
`data/pdb_2021aug02/pdb/**`) and writes:

- `sequences_to_id.fasta` : plain file with one sequence per line (used by
  `datasets/sidechain_esm_embeddings_to_pt.py` as the sequence -> id map).
- `sidechain_for_esm.fasta` : FASTA suitable for the ESM extractor. Headers
  are numeric indices (`0`, `1`, ...) so the extractor will produce per-id
  files named `0.pt`, `1.pt`, ... which the aggregator can consume directly.
- `useful_sequences.pkl` : optional pickle list of sequences to include

The script is intentionally conservative: it de-duplicates sequences while
keeping a stable ordering.
"""

import os
import glob
import argparse
import pickle
from typing import List

import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm


def collect_sequences(pdb_root: str, limit: int = 0) -> List[str]:
    """Scan all .pt files under `pdb_root/pdb/` and collect unique sequences.

    Returns a list of sequences in stable order.
    """
    pattern = os.path.join(pdb_root, 'pdb', '**', '*.pt')
    files = glob.glob(pattern, recursive=True)
    seen = set()
    sequences = []
    for p in tqdm(files, desc='scanning .pt files'):
        try:
            d = torch.load(p)
        except Exception:
            continue
        seq = d.get('seq')
        if not seq:
            continue
        # normalize sequence to a string if it's stored as list/ndarray/tensor
        if not isinstance(seq, str):
            # recursively extract any string candidates from nested lists/tuples/tensors
            def _extract_strings(x):
                res = []
                if isinstance(x, str):
                    res.append(x)
                elif isinstance(x, (list, tuple)):
                    for e in x:
                        res.extend(_extract_strings(e))
                else:
                    # try to convert numpy/torch arrays to lists
                    try:
                        if hasattr(x, 'tolist'):
                            res.extend(_extract_strings(x.tolist()))
                    except Exception:
                        pass
                return res

            candidates = _extract_strings(seq)
            if candidates:
                # choose the longest candidate (likely the full sequence)
                seq = max(candidates, key=len)
            else:
                # fallback: try to join elements or stringify
                try:
                    if hasattr(seq, 'tolist'):
                        seq_list = seq.tolist()
                    else:
                        seq_list = seq
                    seq = ''.join(map(str, seq_list))
                except Exception:
                    seq = str(seq)

        # final cleanup: remove newlines/spaces and non-letter characters
        if isinstance(seq, str):
            seq = seq.replace('\n', '').replace(' ', '').strip()
            # keep letters and hyphen only
            seq = ''.join([c for c in seq if c.isalpha() or c == '-'])
        if seq not in seen:
            seen.add(seq)
            sequences.append(seq)
            if limit and len(sequences) >= limit:
                break
    return sequences


def write_plain_sequences(sequences: List[str], out_path: str) -> None:
    with open(out_path, 'w') as f:
        for s in sequences:
            f.write(s + '\n')


def write_fasta_numeric_headers(sequences: List[str], out_fasta: str) -> None:
    records = []
    for i, s in enumerate(sequences):
        records.append(SeqRecord(Seq(s), id=str(i), description=''))
    SeqIO.write(records, out_fasta, 'fasta')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb_root', default='data/pdb_2021aug02', help='pdbsidechain root containing `pdb/`')
    p.add_argument('--out_sequences', default='data/pdb_2021aug02/sequences_to_id.fasta',
                   help='Plain sequences file (one sequence per line)')
    p.add_argument('--out_fasta', default='data/pdb_2021aug02/sidechain_for_esm.fasta',
                   help='FASTA for ESM extractor (headers are numeric ids)')
    p.add_argument('--useful_pickle', default='data/pdb_2021aug02/useful_sequences.pkl',
                   help='Pickle list of sequences to include in aggregation')
    p.add_argument('--map_pickle', default='',
                   help='Optional: write pickle mapping id (str) -> sequence')
    p.add_argument('--max_useful', type=int, default=20000, help='Max sequences to include in useful pickle')
    p.add_argument('--limit', type=int, default=0, help='Limit number of unique sequences scanned (0 = no limit)')
    p.add_argument('--shuffle', action='store_true', help='Shuffle sequence list before truncation')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_sequences) or '.', exist_ok=True)

    sequences = collect_sequences(args.pdb_root, limit=args.limit)
    if args.shuffle:
        import random
        random.shuffle(sequences)

    print('Collected', len(sequences), 'unique sequences')

    write_plain_sequences(sequences, args.out_sequences)
    print('Wrote plain sequences to', args.out_sequences)

    write_fasta_numeric_headers(sequences, args.out_fasta)
    print('Wrote ESM FASTA to', args.out_fasta)

    useful = sequences[:args.max_useful]
    with open(args.useful_pickle, 'wb') as f:
        pickle.dump(useful, f)
    print('Wrote useful sequences pickle (count {}) to {}'.format(len(useful), args.useful_pickle))

    if args.map_pickle:
        mapping = {str(i): seq for i, seq in enumerate(sequences)}
        with open(args.map_pickle, 'wb') as f:
            pickle.dump(mapping, f)
        print('Wrote id->sequence mapping to', args.map_pickle)


if __name__ == '__main__':
    main()
