                      

import os
import re
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
import json


_CHAIN_RE = re.compile(r"^(?P<rec>.+?)_chain_(?P<idx>\d+)$")


def _extract_tensor(obj: Any, preferred_layer: Optional[int]) -> torch.Tensor:
    """Strict extraction: accept either a raw tensor or dict with 'representations'.

    Fail fast for any unexpected format.
    """
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict) and "representations" in obj:
        reps = obj["representations"]
        if not isinstance(reps, dict) or len(reps) == 0:
            raise RuntimeError("'representations' present but not a non-empty dict")

        # If user requested a specific layer, require it to exist.
        if preferred_layer is not None:
            if preferred_layer in reps:
                return reps[preferred_layer]
            # try to match string keys like '0'
            if str(preferred_layer) in reps:
                return reps[str(preferred_layer)]
            raise RuntimeError(f"Requested layer {preferred_layer} not found in representations")

        # No preferred layer: pick the representation with the highest integer key
        int_key_map = []
        for k in reps.keys():
            try:
                ik = int(k)
                int_key_map.append((ik, k))
            except Exception:
                continue
        if int_key_map:
            _, best_key = max(int_key_map, key=lambda x: x[0])
            return reps[best_key]

        # If there's exactly one representation, accept it
        if len(reps) == 1:
            return next(iter(reps.values()))

        raise RuntimeError("Unable to select a representation layer from 'representations' dict")

    raise RuntimeError(f"Unexpected embedding format: {type(obj)}")


def _parse_chain_file(stem: str) -> Optional[Tuple[str, int]]:


    m = _CHAIN_RE.match(stem)
    if not m:
        return None
    rec = m.group("rec")
    idx = int(m.group("idx"))
    return rec, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing per-chain .pt files like <receptor>_chain_<idx>.pt",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write per-receptor .pt files like <receptor>.pt",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="If input files contain ESM 'representations', prefer this layer. Default: None (auto).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-receptor outputs.",
    )
    ap.add_argument(
        "--check_shapes",
        action="store_true",
        help="Check that all chain tensors for a receptor share the same last-dim.",
    )
    ap.add_argument(
        "--strict_contiguous",
        action="store_true",
        help="Require chain indices to be contiguous starting at 0. Otherwise, missing indices are allowed.",
    )
    ap.add_argument(
        "--map_location",
        default="cpu",
        help="torch.load map_location (default: cpu).",
    )
    args = ap.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

                    
    files = []
    for fn in os.listdir(in_dir):
        if not fn.endswith(".pt"):
            continue
        stem = os.path.splitext(fn)[0]
        parsed = _parse_chain_file(stem)
        if parsed is None:
            continue
        files.append(fn)

    if not files:
        raise RuntimeError(
            f"No chain files found in {in_dir}. Expected names like <receptor>_chain_<idx>.pt"
        )

                       
    groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for fn in files:
        stem = os.path.splitext(fn)[0]
        rec, idx = _parse_chain_file(stem)                      
        groups[rec].append((idx, fn))

    receptors = sorted(groups.keys())
    skipped_existing = 0
    written = 0
    skipped_bad = 0

    for rec in tqdm(receptors, desc="Writing per-receptor embeddings", total=len(receptors)):
        out_path = os.path.join(out_dir, f"{rec}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            skipped_existing += 1
            continue

        chain_entries = sorted(groups[rec], key=lambda x: x[0])
        chain_indices = [i for i, _ in chain_entries]

        if args.strict_contiguous:
            if chain_indices and (chain_indices[0] != 0 or chain_indices != list(range(chain_indices[-1] + 1))):
                tqdm.write(
                    f"Skipping {rec}: non-contiguous chain indices {chain_indices[:10]}..."
                )
                skipped_bad += 1
                continue

        chain_tensors: List[torch.Tensor] = []
        try:
            for idx, fn in chain_entries:
                p = os.path.join(in_dir, fn)
                obj = torch.load(p, map_location=args.map_location)
                t = _extract_tensor(obj, args.layer)
                # Ensure tensor is detached and on CPU to avoid GPU-saved tensors
                t = t.detach().cpu()
                chain_tensors.append(t)
        except Exception as e:
            tqdm.write(f"Skipping {rec}: failed to load/parse chains ({e})")
            skipped_bad += 1
            continue

        if args.check_shapes:
            try:
                last_dims = [int(t.shape[-1]) for t in chain_tensors]
                if len(set(last_dims)) != 1:
                    tqdm.write(
                        f"Skipping {rec}: inconsistent embedding dims across chains: {last_dims}"
                    )
                    skipped_bad += 1
                    continue
            except Exception as e:
                tqdm.write(f"Skipping {rec}: shape check failed ({e})")
                skipped_bad += 1
                continue

                                                                                    
        try:
            # Atomic write to avoid leaving half-written files on interruption
            tmp_path = out_path + ".tmp"
            torch.save(chain_tensors, tmp_path)
            os.replace(tmp_path, out_path)
            # Also write a lightweight metadata sidecar with chain indices and esm_dim for debugging
            esm_dim = int(chain_tensors[0].shape[-1]) if len(chain_tensors) > 0 else None
            meta = {"chain_indices": chain_indices, "esm_dim": esm_dim}
            meta_tmp = out_path + ".meta.json.tmp"
            with open(meta_tmp, "w") as mf:
                json.dump(meta, mf)
            os.replace(meta_tmp, out_path + ".meta.json")
            written += 1
        except Exception as e:
            tqdm.write(f"Skipping {rec}: failed to write output ({e})")
            skipped_bad += 1
            # Clean up any tmp files if present
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            continue

    print("Done.")
    print(f"Input dir:   {in_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Receptors discovered: {len(receptors)}")
    print(f"Written:     {written}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped bad: {skipped_bad}")
    print()
    print("Example runtime usage in your worker:")
    print("  lm_embeddings = torch.load(f\"{esm_dir}/{receptor_name}.pt\", map_location='cpu')")


if __name__ == "__main__":
    main()
