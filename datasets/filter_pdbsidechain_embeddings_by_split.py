#!/usr/bin/env python3
import argparse, sys, pathlib, torch

repo_root = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, repo_root)

from utils.utils import read_strings_from_txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out_emb", required=True)
    ap.add_argument("--out_seq", required=True)
    args = ap.parse_args()

    id_to_emb = torch.load(args.emb, map_location="cpu")
    sequences = read_strings_from_txt(args.seq)

    embedded_ids = sorted(int(k) for k in id_to_emb.keys())

    print(f"Embedding IDs: {len(embedded_ids)}")
    print(f"FASTA lines:   {len(sequences)}")

    # Validate indices
    bad = [i for i in embedded_ids if i < 0 or i >= len(sequences)]
    if bad:
        raise SystemExit(f"Invalid IDs: {bad[:10]}")

    # Build filtered outputs
    out_emb = {}
    out_seqs = []

    # Reindex embeddings so FASTA index i corresponds to embedding key "i".
    out_emb = {}
    out_seqs = []
    for new_i, old_i in enumerate(embedded_ids):
        out_emb[str(new_i)] = id_to_emb[str(old_i)]
        out_seqs.append(sequences[old_i])

    assert len(out_emb) == len(out_seqs)

    torch.save(out_emb, args.out_emb)
    with open(args.out_seq, "w") as f:
        for s in out_seqs:
            f.write(s + "\n")

    print(f"Saved {len(out_emb)} embeddings")
    print(f"Saved {len(out_seqs)} sequences")

    # sanity
    k = next(iter(out_emb))
    print("Sample:", k, out_emb[k].shape)

if __name__ == "__main__":
    main()
