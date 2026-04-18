import csv
import shutil
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

# Silence RDKit informational/warning messages during bulk validation/processing.
RDLogger.DisableLog('rdApp.*')


@dataclass(frozen=True)
class RawResolution:

    complex_id: str
    raw_dir: str
    protein_src: str
    ligand_src: str
    ligand_kind: str                                
    reason: str                            


def parse_args():
    p = ArgumentParser(description="Process PDBBind for DiffDock using split files as source of truth.")
    
    p.add_argument("--pdbbind_raw_dir", required=True, type=str,
                   help="Root path of raw PDBBind dataset.")
    p.add_argument("--output_dir", required=True, type=str,
                   help="Output processed dataset directory (per-complex folders).")
    p.add_argument("--split_train", required=True, type=str,
                   help="Train split file (one complex ID per line).")
    p.add_argument("--split_val", required=True, type=str,
                   help="Val split file (one complex ID per line).")

                         
    p.add_argument("--split_test", default="", type=str,
                   help="Optional test split file.")

                                 
    p.add_argument("--raw_subset_dirs", nargs="*", default=["refined-set", "general-set", "v2020-other-PL"],
                   help="Candidate subdirectories inside raw root that may contain complex folders.")

                    
    p.add_argument("--splits_out", type=str, default="",
                   help="Directory to write filtered/kept split files. "
                        "Default: sibling 'splits' next to output_dir.")

    p.add_argument("--resolution_report_out", type=str, default="",
                   help="Write a CSV report of ID->raw resolution and skip reasons.")
    p.add_argument("--metadata_out", type=str, default="",
                   help="Write a CSV metadata file with paths to processed assets (outside dataset tree).")

                               
    p.add_argument("--limit_ids", type=int, default=-1,
                   help="If >0, only process first N IDs from the union of splits (after sorting). -1 means no limit.")
    p.add_argument("--resolve_only", action="store_true",
                   help="Only run raw-resolution (locate protein/ligand files) and write resolution report; skip processing.")
    p.add_argument("--validate_only", action="store_true",
                   help="Run resolution + in-memory validation (sanitize checks) without writing processed outputs.")
    p.add_argument("--validation_report_out", type=str, default="",
                   help="Write a CSV validation report when using --validate_only.")

    return p.parse_args()


def read_split_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(s.lower())
    return ids


def validate_split_ids(ids: List[str], label: str) -> None:
    bad = [x for x in ids if len(x) != 4 or (not x.isalnum())]
    if bad:
        print(f"WARNING: {label} has {len(bad)} IDs that are not 4-char alnum PDB codes. Examples: {bad[:10]}")


def write_split_ids(path: str, ids: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for cid in ids:
            f.write(f"{cid}\n")


def iter_candidate_raw_roots(raw_root: Path, subset_dirs: List[str]) -> List[Path]:
    roots: List[Path] = []
                                                                         
    roots.append(raw_root)

    for sd in subset_dirs:
        p = raw_root / sd
        if p.exists() and p.is_dir():
            roots.append(p)

                                                    
    roots = sorted(set(roots), key=lambda x: str(x))
    return roots


def is_pdb_id_dirname(name: str) -> bool:

    if len(name) != 4:
        return False
    return all(c.isalnum() for c in name)


def scan_raw_once(raw_roots: List[Path]) -> Dict[str, List[Path]]:


    idx: Dict[str, List[Path]] = {}
    for root in raw_roots:
        try:
            entries = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name.lower())
        except Exception:
            continue

        for d in entries:
            name = d.name.lower()
            if not is_pdb_id_dirname(name):
                continue
            idx.setdefault(name, []).append(d)

                                 
    for k in idx:
        idx[k] = sorted(idx[k], key=lambda x: str(x))
    return idx


def choose_protein_file(raw_dir: Path, cid: str) -> Optional[Path]:
    candidates = [
        raw_dir / f"{cid}_protein.pdb",
        raw_dir / f"{cid}_protein_processed.pdb",
        raw_dir / f"{cid}_pocket.pdb",
        raw_dir / f"{cid}.pdb",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def choose_ligand_file(raw_dir: Path, cid: str) -> Optional[Tuple[Path, str]]:
    sdf = raw_dir / f"{cid}_ligand.sdf"
    if sdf.exists():
        return sdf, "sdf"

    mol2 = raw_dir / f"{cid}_ligand.mol2"
    if mol2.exists():
        return mol2, "mol2"

    return None


def resolve_id_to_raw_assets(cid: str, raw_dirs: List[Path]) -> RawResolution:
    if not raw_dirs:
        return RawResolution(cid, "", "", "", "", "no_raw_dir")

    any_protein = False
    any_ligand = False
    for rd in raw_dirs:
        protein = choose_protein_file(rd, cid)
        ligand = choose_ligand_file(rd, cid)
        if protein is not None:
            any_protein = True
        if ligand is not None:
            any_ligand = True
        if protein is not None and ligand is not None:
            ligand_path, ligand_kind = ligand
            return RawResolution(cid, str(rd), str(protein), str(ligand_path), ligand_kind, "ok")

                                                                                         
    first_dir = str(raw_dirs[0])

    if not any_protein and not any_ligand:
        return RawResolution(cid, first_dir, "", "", "", "no_protein_or_ligand_candidate")
    if not any_protein:
        return RawResolution(cid, first_dir, "", "", "", "no_protein_candidate")
    if not any_ligand:
        return RawResolution(cid, first_dir, "", "", "", "no_ligand_candidate")
    return RawResolution(cid, first_dir, "", "", "", "unresolved")


def clean_protein(in_pdb: str, out_pdb: str) -> Tuple[bool, str]:
    try:
        shutil.copyfile(in_pdb, out_pdb)
        ok, reason = protein_looks_like_pdb(out_pdb)
        if not ok:
            return False, reason
        return True, "ok"
    except Exception as e:
        return False, f"protein_copy_exception:{type(e).__name__}"


def protein_looks_like_pdb(path: str) -> Tuple[bool, str]:
    try:
        p = Path(path)
        if not p.exists():
            return False, "protein_missing"
        if p.stat().st_size == 0:
            return False, "protein_empty"

        has_atom = False
        with open(p, "r", errors="ignore") as f:
            for _ in range(20000):
                line = f.readline()
                if not line:
                    break
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    has_atom = True
                    break
        if not has_atom:
            return False, "protein_no_atom_records"
        return True, "ok"
    except Exception as e:
        return False, f"protein_check_exception:{type(e).__name__}"


def ligand_sdf_is_sanitizable(path: str) -> Tuple[bool, str]:
    try:
        p = Path(path)
        if not p.exists():
            return False, "ligand_missing"
        if p.stat().st_size == 0:
            return False, "ligand_empty"

        supplier = Chem.SDMolSupplier(str(p), sanitize=True, removeHs=False)
        for i, mol in enumerate(supplier):
            if mol is not None:
                return True, "ok"
            if i >= 4:
                break
        return False, "ligand_not_sanitizable"
    except Exception as e:
        return False, f"ligand_check_exception:{type(e).__name__}"


def process_ligand(lig_src: str,
                   lig_kind: str,
                   out_sdf: str) -> Tuple[bool, str]:
    try:
        if lig_kind == "sdf":
            shutil.copyfile(lig_src, out_sdf)
            return True, "ok"

        if lig_kind == "mol2":
            mol = Chem.MolFromMol2File(lig_src, sanitize=False, removeHs=False)
            if mol is None:
                return False, "mol2_unreadable"
            w = Chem.SDWriter(out_sdf)
            w.write(mol)
            w.close()
            return True, "ok"

        return False, "unsupported_ligand_kind"

    except Exception as e:
        return False, f"ligand_exception:{type(e).__name__}"


def main():
    args = parse_args()

    raw_root = Path(args.pdbbind_raw_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

                                                          
    train_ids = read_split_ids(args.split_train)
    val_ids = read_split_ids(args.split_val)
    test_ids = read_split_ids(args.split_test) if args.split_test else []

    validate_split_ids(train_ids, "train split")
    validate_split_ids(val_ids, "val split")
    if test_ids:
        validate_split_ids(test_ids, "test split")

                         
    S: List[str] = sorted(set(train_ids) | set(val_ids) | set(test_ids))
    if args.limit_ids and args.limit_ids > 0:
        S = S[:args.limit_ids]

    print(f"Target IDs (union of splits): {len(S)}")
    print(f"Train IDs: {len(train_ids)}  Val IDs: {len(val_ids)}  Test IDs: {len(test_ids)}")

                                       
    raw_roots = iter_candidate_raw_roots(raw_root, args.raw_subset_dirs)
    print("Raw roots to scan:")
    for rr in raw_roots:
        print(f"  - {rr}")

    print("Scanning raw dataset (one pass) ...")
    raw_index = scan_raw_once(raw_roots)

                                       
    resolutions: List[RawResolution] = []
    for cid in S:
        res = resolve_id_to_raw_assets(cid, raw_index.get(cid, []))
        resolutions.append(res)

    if args.resolve_only:
        ok = [r for r in resolutions if r.reason == "ok"]
        others = [r for r in resolutions if r.reason != "ok"]
        from collections import Counter
        reasons = Counter([r.reason for r in others])
        print(f"Raw-resolvable IDs (files found by naming rules): {len(ok)} / {len(S)}")
        print("Top non-resolvable reasons:")
        for reason, cnt in reasons.most_common(10):
            print(f"  {reason}: {cnt}")
        if args.resolution_report_out:
            rep_path = Path(args.resolution_report_out).resolve()
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rep_path, "w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["complex_id", "reason", "raw_dir", "protein_src", "ligand_src", "ligand_kind"]
                )
                w.writeheader()
                for r in resolutions:
                    w.writerow({
                        "complex_id": r.complex_id,
                        "reason": r.reason,
                        "raw_dir": r.raw_dir,
                        "protein_src": r.protein_src,
                        "ligand_src": r.ligand_src,
                        "ligand_kind": r.ligand_kind,
                    })
            print(f"Resolution report written: {rep_path}")
        return

    if args.validate_only:
        # Perform in-memory validation for IDs that resolved by naming rules.
        import tempfile
        rows = []
        validated = 0
        for res in tqdm(resolutions, desc="Validating IDs"):
            cid = res.complex_id
            row = {
                "complex_id": cid,
                "resolution_reason": res.reason,
                "raw_dir": res.raw_dir,
                "protein_src": res.protein_src,
                "ligand_src": res.ligand_src,
                "ligand_kind": res.ligand_kind,
                "protein_check": "",
                "protein_reason": "",
                "ligand_check": "",
                "ligand_reason": "",
                "would_write": False,
            }
            if res.reason != "ok":
                rows.append(row)
                continue

            # Protein check (raw file)
            okp, rp = protein_looks_like_pdb(res.protein_src)
            row["protein_check"] = "ok" if okp else "fail"
            row["protein_reason"] = rp

            # Ligand check: for SDF, run sanitizer; for MOL2 try conversion then sanitizer via temp file
            lkind = res.ligand_kind
            if lkind == "sdf":
                okl, rl = ligand_sdf_is_sanitizable(res.ligand_src)
                row["ligand_check"] = "ok" if okl else "fail"
                row["ligand_reason"] = rl
            elif lkind == "mol2":
                # Try sanitize=True conversion first
                mol = Chem.MolFromMol2File(res.ligand_src, sanitize=True, removeHs=False)
                if mol is not None:
                    row["ligand_check"] = "ok"
                    row["ligand_reason"] = "mol2_sanitizable"
                else:
                    mol = Chem.MolFromMol2File(res.ligand_src, sanitize=False, removeHs=False)
                    if mol is None:
                        row["ligand_check"] = "fail"
                        row["ligand_reason"] = "mol2_unreadable"
                    else:
                        # write to temp SDF and test sanitizability
                        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tf:
                            tmp_path = tf.name
                        try:
                            w = Chem.SDWriter(tmp_path)
                            w.write(mol)
                            w.close()
                            okl, rl = ligand_sdf_is_sanitizable(tmp_path)
                            row["ligand_check"] = "ok" if okl else "fail"
                            row["ligand_reason"] = rl
                        finally:
                            try:
                                Path(tmp_path).unlink()
                            except Exception:
                                pass
            else:
                row["ligand_check"] = "fail"
                row["ligand_reason"] = "unsupported_ligand_kind"

            row["would_write"] = (row["protein_check"] == "ok" and row["ligand_check"] == "ok")
            if row["would_write"]:
                validated += 1
            rows.append(row)

        print(f"Validated-writable IDs: {validated} / {len(S)}")
        from collections import Counter
        reasons = Counter()
        for r in rows:
            if r["would_write"]:
                continue
            if r["resolution_reason"] != "ok":
                reasons[r["resolution_reason"]] += 1
            else:
                # validation failures among resolved IDs
                if r["protein_check"] == "fail":
                    reasons[r["protein_reason"]] += 1
                else:
                    reasons[r["ligand_reason"]] += 1

        print("Top validation failures:")
        for reason, cnt in reasons.most_common(10):
            print(f"  {reason}: {cnt}")

        if args.validation_report_out:
            outp = Path(args.validation_report_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", newline="") as f:
                fieldnames = [
                    "complex_id", "resolution_reason", "raw_dir", "protein_src", "ligand_src", "ligand_kind",
                    "protein_check", "protein_reason", "ligand_check", "ligand_reason", "would_write",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"Validation report written: {outp}")
        return

                            
    meta_rows: List[Dict[str, str]] = []

                                                                    
    processed_set: Set[str] = set()

    for i, res in enumerate(tqdm(resolutions, desc="Processing IDs")):
        cid = res.complex_id
        if res.reason != "ok":
            continue

        out_dir = out_root / cid
        out_prot = out_dir / f"{cid}_protein.pdb"
        out_lig = out_dir / f"{cid}_ligand.sdf"

        # Skip only if both outputs exist and pass basic checks
        if out_dir.exists() and out_prot.exists() and out_lig.exists():
            okp, _ = protein_looks_like_pdb(str(out_prot))
            okl, _ = ligand_sdf_is_sanitizable(str(out_lig))
            if okp and okl:
                processed_set.add(cid)
                meta_rows.append({
                    "complex_id": cid,
                    "protein_path": f"{cid}/{cid}_protein.pdb",
                    "ligand_path": f"{cid}/{cid}_ligand.sdf",
                    "raw_dir": res.raw_dir,
                    "protein_src": res.protein_src,
                    "ligand_src": res.ligand_src,
                })
                continue

        # Otherwise rebuild from scratch
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Recompute paths
        out_prot = out_dir / f"{cid}_protein.pdb"
        out_lig = out_dir / f"{cid}_ligand.sdf"

        ok_p, reason_p = clean_protein(res.protein_src, str(out_prot))
        if not ok_p:
            shutil.rmtree(out_dir, ignore_errors=True)
                                                 
            resolutions[i] = RawResolution(
                cid, res.raw_dir, res.protein_src, res.ligand_src, res.ligand_kind, reason_p
            )
            continue

                
        out_lig = out_dir / f"{cid}_ligand.sdf"
        ok_l, reason_l = process_ligand(res.ligand_src, res.ligand_kind, str(out_lig))
        if ok_l:
            ok_read, reason_read = ligand_sdf_is_sanitizable(str(out_lig))
            if not ok_read:
                ok_l = False
                reason_l = reason_read

        if not ok_l:
            shutil.rmtree(out_dir, ignore_errors=True)
            resolutions[i] = RawResolution(
                cid, res.raw_dir, res.protein_src, res.ligand_src, res.ligand_kind, reason_l
            )
            continue

                                                                                             
        processed_set.add(cid)
        meta_rows.append({
            "complex_id": cid,
            "protein_path": f"{cid}/{cid}_protein.pdb",
            "ligand_path": f"{cid}/{cid}_ligand.sdf",
            "raw_dir": res.raw_dir,
            "protein_src": res.protein_src,
            "ligand_src": res.ligand_src,
        })

    print(f"Successfully processed: {len(processed_set)} / {len(S)} target IDs")

                                                                            
    splits_out = Path(args.splits_out).resolve() if args.splits_out else (out_root.parent / "splits")
    splits_out.mkdir(parents=True, exist_ok=True)

    train_kept = [cid.lower() for cid in train_ids if cid.lower() in processed_set]
    val_kept = [cid.lower() for cid in val_ids if cid.lower() in processed_set]
    test_kept = [cid.lower() for cid in test_ids if cid.lower() in processed_set]

                                                                        
    write_split_ids(str(splits_out / "timesplit_no_lig_overlap_train"), train_kept)
    write_split_ids(str(splits_out / "timesplit_no_lig_overlap_val"), val_kept)
    if test_ids:
        write_split_ids(str(splits_out / "timesplit_test"), test_kept)

    write_split_ids(str(splits_out / "timesplit_no_lig_overlap_train.txt"), train_kept)
    write_split_ids(str(splits_out / "timesplit_no_lig_overlap_val.txt"), val_kept)
    if test_ids:
        write_split_ids(str(splits_out / "timesplit_test.txt"), test_kept)

                             
    if args.resolution_report_out:
        rep_path = Path(args.resolution_report_out).resolve()
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["complex_id", "reason", "raw_dir", "protein_src", "ligand_src", "ligand_kind"]
            )
            w.writeheader()
            for r in resolutions:
                w.writerow({
                    "complex_id": r.complex_id,
                    "reason": r.reason,
                    "raw_dir": r.raw_dir,
                    "protein_src": r.protein_src,
                    "ligand_src": r.ligand_src,
                    "ligand_kind": r.ligand_kind,
                })
        print(f"Resolution report written: {rep_path}")

                                           
    if args.metadata_out:
        meta_path = Path(args.metadata_out).resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "complex_id", "protein_path", "ligand_path",
                    "raw_dir", "protein_src", "ligand_src",
                ]
            )
            w.writeheader()
            for row in meta_rows:
                w.writerow(row)
        print(f"Metadata written: {meta_path}")

    print("")
    print("Done.")
    print(f"Processed dataset dir: {out_root}")
    print(f"Filtered splits dir:  {splits_out}")
    print("")
    print("Example training args:")
    print(f"  --data_dir {out_root}")
    print(f"  --split_train {splits_out}/timesplit_no_lig_overlap_train")
    print(f"  --split_val   {splits_out}/timesplit_no_lig_overlap_val")


if __name__ == "__main__":
    main()
