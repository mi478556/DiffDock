#!/usr/bin/env python3
"""
PDBBind Dataset Processing Script for DiffDock

This script processes the raw PDBBind dataset into the format expected by DiffDock's data pipeline.
It creates the correct directory structure, processes protein/ligand files, generates metadata,
creates splits, and prepares ESM2 embeddings.

Usage:
    python process_pdbbind.py --pdbbind_raw_dir /path/to/pdbbind/raw --output_dir data/PDBBind_processed
"""

import os
import sys
import shutil
import csv
import pickle
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Dict, Tuple, Optional
import pandas as pd

# Scientific libraries
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, RemoveHs, AddHs, Descriptors
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch, Selection
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# Import DiffDock constants for amino acid mapping
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets.constants import three_to_one


class NonHetSelect(Select):
    """BioPython selector to remove waters and ligands, keep only protein"""
    def accept_residue(self, residue):
        return residue.id[0] == " "  # Only standard amino acids


class PocketSelect(Select):
    """Select only residues that are part of the binding pocket set."""
    def __init__(self, pocket_residues):
        # pocket_residues is a set of tuples (chain_id, residue.id)
        self.pocket_residues = pocket_residues

    def accept_residue(self, residue):
        try:
            chain_id = residue.get_parent().id
            return (chain_id, residue.id) in self.pocket_residues
        except Exception:
            return False


def parse_args():
    parser = ArgumentParser(description="Process PDBBind dataset for DiffDock training")
    parser.add_argument("--pdbbind_raw_dir", type=str, required=True,
                        help="Path to raw PDBBind directory (with v2020-other-PL, refined-set, etc.)")
    parser.add_argument("--output_dir", type=str, default="data/PDBBind_processed",
                        help="Output directory for processed data")
    parser.add_argument("--subset", type=str, default="refined", choices=["core", "refined", "general"],
                        help="PDBBind subset to process")
    parser.add_argument("--max_protein_size", type=int, default=2000,
                        help="Maximum number of residues in protein")
    parser.add_argument("--max_ligand_size", type=int, default=79, 
                        help="Maximum number of heavy atoms in ligand")
    parser.add_argument("--min_ligand_size", type=int, default=5,
                        help="Minimum number of heavy atoms in ligand")
    parser.add_argument("--limit_complexes", type=int, default=None,
                        help="Limit number of complexes for testing")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes")
    parser.add_argument("--generate_splits", action="store_true", default=True,
                        help="Generate train/val/test splits")
    parser.add_argument("--splits_out", type=str, default="",
                        help="Optional path to write split files. If empty, splits are written beside the output_dir (sibling 'splits' folder) to avoid placing them inside the dataset output directory.")
    parser.add_argument("--prepare_esm", action="store_true", default=False,
                        help="(disabled by default) Prepare sequences for ESM2 embedding generation. Disabled to avoid overlap with datasets/esm_embedding_preparation.py")
    parser.add_argument("--metadata_out", type=str, default="",
                        help="Optional path to write metadata CSV. If empty, metadata will not be written to the output directory to avoid overlap with other tooling.")
    parser.add_argument("--fragment_policy", type=str, default="largest",
                        choices=["largest", "all", "fail"],
                        help="How to handle multi-fragment ligands in mol2 files: 'largest' selects the largest fragment (default), 'all' keeps the original molecule, 'fail' rejects multi-fragment ligands.")
    parser.add_argument("--min_protein_residues", type=int, default=0,
                        help="Minimum number of standard residues required after protein cleaning. 0 disables this check.")
    parser.add_argument("--generate_pockets", action="store_true", default=False,
                        help="Generate pocket.pdb files for each complex (default: off)")
    parser.add_argument("--pocket_cutoff", type=float, default=6.0,
                        help="Distance cutoff (Å) to define pocket residues around ligand atoms")
    return parser.parse_args()


def get_complex_ids_from_index(pdbbind_raw_dir: str, subset: str) -> List[str]:
    """Extract complex IDs from PDBBind index file"""
    # Find index files - they can be in different locations
    possible_index_dirs = [
        os.path.join(pdbbind_raw_dir, "refined-set", "index"),  # v2020 structure
        os.path.join(pdbbind_raw_dir, "index"),  # Alternative structure
        pdbbind_raw_dir  # Index files in root
    ]
    
    if subset == "core":
        index_filename = "INDEX_core_set.2020"
    elif subset == "refined":
        index_filename = "INDEX_refined_set.2020"
    else:  # general
        index_filename = "INDEX_general_PL.2020"
    
    index_file = None
    for index_dir in possible_index_dirs:
        potential_path = os.path.join(index_dir, index_filename)
        if os.path.exists(potential_path):
            index_file = potential_path
            break
    
    if index_file is None:
        raise FileNotFoundError(f"Index file {index_filename} not found in any of: {possible_index_dirs}")
    
    print(f"Using index file: {index_file}")
    
    complex_ids = []
    id_to_year = {}  # Also extract years for splitting
    
    with open(index_file, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            # Format: PDB_code  resolution  release_year  binding_data  reference  ligand_name
            parts = line.strip().split()
            if len(parts) >= 3:
                pdb_id = parts[0].lower()
                try:
                    year = int(parts[2])
                    complex_ids.append(pdb_id)
                    id_to_year[pdb_id] = year
                except (ValueError, IndexError):
                    # Skip lines with invalid year
                    continue
    
    print(f"Found {len(complex_ids)} complexes in {subset} set")
    return complex_ids, id_to_year


def clean_protein(protein_pdb_path: str, output_path: str, max_residues: int = 2000, min_residues: int = 0) -> bool:
    """Clean protein structure: remove waters/ligands, keep only standard amino acids"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", protein_pdb_path)

        # ----- ALTLOC handling: for atoms with the same name in a residue,
        # keep the atom with highest occupancy or prefer altloc 'A' -----
        for model in structure:
            for chain in model:
                for residue in list(chain):
                    # build mapping atom_name -> [atoms]
                    atom_groups = {}
                    for atom in list(residue):
                        name = atom.get_name()
                        atom_groups.setdefault(name, []).append(atom)

                    # for each group, keep best atom and remove others
                    for name, atoms in atom_groups.items():
                        if len(atoms) <= 1:
                            continue
                        # prefer altloc 'A' if present, otherwise highest occupancy
                        best = None
                        best_occ = -1.0
                        for a in atoms:
                            try:
                                alt = a.get_altloc()
                            except Exception:
                                alt = ''
                            try:
                                occ = a.get_occupancy()
                            except Exception:
                                occ = None
                            if occ is None:
                                occ = 0.0
                            if alt == 'A':
                                best = a
                                break
                            if occ > best_occ:
                                best = a
                                best_occ = occ

                        for a in atoms:
                            if a is not best:
                                try:
                                    residue.detach_child(a.get_id())
                                except Exception:
                                    # best-effort cleanup; ignore failures
                                    pass

        # ----- Backbone completeness: remove residues that lack CA, N or C -----
        for model in structure:
            for chain in model:
                for residue in list(chain):
                    atom_names = {atom.get_name() for atom in residue}
                    if not ({'CA', 'N', 'C'}.issubset(atom_names)):
                        # remove residue (it may be incomplete/altloc-only)
                        try:
                            chain.detach_child(residue.id)
                        except Exception:
                            pass

        # Count residues after cleaning
        residue_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == " ":  # Standard amino acid
                        residue_count += 1

        if residue_count == 0:
            print(f"Protein has no standard residues after cleaning: {protein_pdb_path}")
            return False
        if min_residues and residue_count < min_residues:
            print(f"Protein too short after cleaning: {residue_count} < min_residues ({min_residues})")
            return False
        if residue_count > max_residues:
            print(f"Protein too large: {residue_count} > {max_residues} residues")
            return False

        # Save cleaned protein
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path, NonHetSelect())
        return True
        
    except Exception as e:
        print(f"Error cleaning protein {protein_pdb_path}: {e}")
        return False


def process_ligand(ligand_mol2_path: str, output_sdf_path: str, 
                  min_atoms: int = 5, max_atoms: int = 100, fragment_policy: str = "largest") -> bool:
    """Process ligand: convert mol2 to sdf, validate size"""
    try:
        # Read ligand from mol2 file
        mol = Chem.MolFromMol2File(ligand_mol2_path, sanitize=True, removeHs=False)
        if mol is None:
            print(f"Failed to read ligand: {ligand_mol2_path}")
            return False
        # Fragment/connectivity handling according to policy
        try:
            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                if fragment_policy == 'largest':
                    sizes = [f.GetNumAtoms() for f in frags]
                    largest_idx = int(max(range(len(frags)), key=lambda i: sizes[i]))
                    print(f"Warning: ligand file {ligand_mol2_path} contains {len(frags)} fragments; selecting largest fragment with {sizes[largest_idx]} atoms")
                    mol = frags[largest_idx]
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        pass
                elif fragment_policy == 'fail':
                    print(f"Rejecting {ligand_mol2_path}: contains multiple fragments and fragment_policy='fail'")
                    return False
                elif fragment_policy == 'all':
                    # keep original mol (no change)
                    pass
        except Exception:
            # if RDKit fragment handling fails for any reason, continue and let later checks catch it
            pass
        
        # Check size constraints
        heavy_atoms = mol.GetNumHeavyAtoms()
        total_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        if heavy_atoms < min_atoms or heavy_atoms > max_atoms:
            print(f"Ligand size out of range: {heavy_atoms} atoms (range: {min_atoms}-{max_atoms})")
            return False
            
        # Filter out overly complex ligands that cause DiffDock conformer matching to hang
        # Multiple criteria to catch different types of complexity that cause genetic algorithm issues
        
        # Basic size limits
        if total_atoms > 80 or num_bonds > 100:
            print(f"Ligand too complex for DiffDock processing: {total_atoms} atoms, {num_bonds} bonds")
            return False
            
        # Advanced complexity filters for multiprocessing genetic algorithm stability
        from rdkit.Chem import Descriptors

        # Rotatable bonds - allow more flexible ligands; increased limit to 19
        # DiffDock handles flexibility, but extremely flexible ligands can still be problematic.
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        if rotatable_bonds > 19:
            print(f"Ligand too flexible: {rotatable_bonds} rotatable bonds (limit 19)")
            return False
        
        # Add hydrogens and generate conformer if needed
        mol = AddHs(mol)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
        
        # Write to SDF
        writer = Chem.SDWriter(output_sdf_path)
        writer.write(mol)
        writer.close()
        return True
        
    except Exception as e:
        print(f"Error processing ligand {ligand_mol2_path}: {e}")
        return False


def extract_protein_sequence(protein_pdb_path: str) -> List[str]:
    """Extract protein sequences from PDB file"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', protein_pdb_path)
        
        sequences = []
        for chain in structure[0]:  # First model only
            seq = ''
            for residue in chain:
                if residue.id[0] == " ":  # Standard amino acid
                    try:
                        seq += three_to_one[residue.get_resname()]
                    except KeyError:
                        seq += 'X'  # Unknown amino acid
            if len(seq) > 0:
                sequences.append(seq)
        
        return sequences
    except Exception as e:
        print(f"Error extracting sequence from {protein_pdb_path}: {e}")
        return []


def create_time_splits(processed_complexes: List[str], id_to_year: Dict[str, int], 
                      output_dir: str, splits_root: Optional[str] = None) -> None:
    """Create time-based splits based on PDB release year.

    Note: This function expects a list of successfully processed complex IDs
    (e.g. `processed_complexes`) rather than the raw index list. Passing only
    processed complexes prevents split files from containing IDs that were
    skipped during processing and aren't present on disk.
    """

    # Filter to processed complexes and sort by year
    valid_complexes = [(cid, id_to_year.get(cid, 2020)) for cid in processed_complexes if cid in id_to_year]
    valid_complexes.sort(key=lambda x: x[1])  # Sort by year
    
    # Time-based splits: train (before 2015), val (2015-2017), test (2018+)
    train_ids = [cid for cid, year in valid_complexes if year < 2015]
    val_ids = [cid for cid, year in valid_complexes if 2015 <= year < 2018]
    test_ids = [cid for cid, year in valid_complexes if year >= 2018]
    
    print(f"Time splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    # Write split files
    # If caller provided an explicit splits_root, use it. Otherwise write splits
    # to a sibling `splits/` folder next to the `output_dir` to avoid polluting
    # the processed dataset directory (some downstream tooling expects the
    # dataset dir to contain only per-complex folders).
    if splits_root:
        splits_dir = splits_root
    else:
        splits_dir = os.path.join(os.path.dirname(output_dir), "splits")
    os.makedirs(splits_dir, exist_ok=True)
    # Also write files with names that the training code historically expects
    # (e.g. timesplit_no_lig_overlap_train) so users can point --split_train to
    # the default path without renaming files.
    alt_names = {
        'timesplit_train': 'timesplit_no_lig_overlap_train',
        'timesplit_val': 'timesplit_no_lig_overlap_val',
        'timesplit_test': 'timesplit_test'
    }

    for split_name, ids in [("timesplit_train", train_ids), 
                           ("timesplit_val", val_ids), 
                           ("timesplit_test", test_ids)]:
        # Primary file with a clear .txt extension
        txt_path = os.path.join(splits_dir, f"{split_name}.txt")
        with open(txt_path, 'w') as f:
            f.write('\n'.join(ids))

        # Also write alternate filename expected by training defaults (no extension)
        alt_name = alt_names.get(split_name, split_name)
        alt_path = os.path.join(splits_dir, alt_name)
        # write the same contents to the alternate path so training can use the default names
        with open(alt_path, 'w') as f:
            f.write('\n'.join(ids))


def prepare_esm_sequences(processed_complexes: List[str], output_dir: str) -> None:
    """Prepare protein sequences for ESM2 embedding generation"""
    print("Preparing sequences for ESM2 embeddings...")
    
    records = []
    
    for complex_id in tqdm(processed_complexes, desc="Extracting sequences"):
        protein_path = os.path.join(output_dir, complex_id, f"{complex_id}_protein.pdb")
        if not os.path.exists(protein_path):
            continue
            
        sequences = extract_protein_sequence(protein_path)
        for i, seq in enumerate(sequences):
            if len(seq) > 0:
                record = SeqRecord(Seq(seq), f'{complex_id}_chain_{i}')
                record.description = ''
                records.append(record)
    
    # Write FASTA file
    fasta_path = os.path.join(output_dir, "prepared_for_esm.fasta")
    SeqIO.write(records, fasta_path, "fasta")
    print(f"Wrote {len(records)} sequences to {fasta_path}")
    
    # Note: we intentionally do not create an instruction shell script here.
    # The repository includes separate tooling for generating ESM embeddings.
    # If you need a helper script, generate it externally or re-enable this
    # behavior manually. This keeps the output directory free of helper files.


def extract_binding_pocket(protein_path: str, ligand_path: str, output_path: str, cutoff: float = 6.0) -> bool:
    """Extract residues within `cutoff` Å of any ligand atom and write a pocket PDB.

    Returns True on success, False on error (non-fatal for pipeline).
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('prot', protein_path)

        # get all atoms from protein structure
        atoms = Selection.unfold_entities(structure, 'A')  # list of Atom objects

        # load ligand (RDKit) and get atom coordinates
        ligand = Chem.MolFromMolFile(ligand_path, removeHs=False)
        if ligand is None:
            print(f"Warning: failed to read ligand for pocket extraction: {ligand_path}")
            return False
        conf = ligand.GetConformer()
        if conf is None or conf.GetNumAtoms() == 0:
            print(f"Warning: ligand has no conformer for pocket extraction: {ligand_path}")
            return False

        ligand_coords = conf.GetPositions()

        # neighbor search and collect residues
        ns = NeighborSearch(atoms)
        pocket_residues = set()
        for coord in ligand_coords:
            close_atoms = ns.search(coord, cutoff)
            for atom in close_atoms:
                residue = atom.get_parent()
                chain_id = residue.get_parent().id
                pocket_residues.add((chain_id, residue.id))

        if len(pocket_residues) == 0:
            # nothing found; still write an empty file or skip
            print(f"No pocket residues found within {cutoff}Å for {protein_path} / {ligand_path}")
            return False

        # write pocket PDB containing only selected residues
        io = PDBIO()
        io.set_structure(structure)
        selector = PocketSelect(pocket_residues)
        io.save(output_path, selector)
        return True

    except Exception as e:
        print(f"Error extracting pocket for {protein_path}: {e}")
        return False


def main():
    args = parse_args()
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get complex IDs from PDBBind index
    complex_ids, id_to_year = get_complex_ids_from_index(args.pdbbind_raw_dir, args.subset)
    
    if args.limit_complexes:
        complex_ids = complex_ids[:args.limit_complexes]
        print(f"Limited to {len(complex_ids)} complexes for testing")
    
    # Determine PDBBind data directory - handle different structures
    possible_data_dirs = [
        os.path.join(args.pdbbind_raw_dir, "refined-set"),  # v2020 structure
        os.path.join(args.pdbbind_raw_dir, f"v2020-other-PL"),  # Alternative
        args.pdbbind_raw_dir  # Data files in root
    ]
    
    pdbbind_data_dir = None
    for data_dir in possible_data_dirs:
        if os.path.exists(data_dir):
            # Check if it contains complex directories
            # compare lowercase names to be robust to uppercase/lowercase directory names
            wanted_ids = {cid.lower() for cid in complex_ids[:5]}
            sample_complexes = [d for d in os.listdir(data_dir) if d.lower() in wanted_ids]
            if sample_complexes:
                pdbbind_data_dir = data_dir
                break
    
    if pdbbind_data_dir is None:
        raise FileNotFoundError(f"PDBBind data directory not found. Checked: {possible_data_dirs}")

    print(f"Using PDBBind data directory: {pdbbind_data_dir}")
    
    # Convert to Path object for consistent handling
    pdbbind_data_dir = Path(pdbbind_data_dir)

    # Process complexes
    processed_complexes = []
    metadata_rows = []

    print(f"Processing {len(complex_ids)} complexes...")
    for complex_id in tqdm(complex_ids, desc="Processing complexes"):
        
        complex_dir = pdbbind_data_dir / complex_id
        if not complex_dir.exists():
            print(f"Complex directory not found: {complex_dir}")
            continue
        
        # Find protein and ligand files
        protein_pdb = complex_dir / f"{complex_id}_protein.pdb"
        ligand_mol2 = complex_dir / f"{complex_id}_ligand.mol2"
        
        if not protein_pdb.exists() or not ligand_mol2.exists():
            print(f"Missing files for {complex_id}")
            continue
        
        # Create output directory for this complex (DiffDock expects flat structure: complex_id/)
        output_complex_dir = output_dir / complex_id
        output_complex_dir.mkdir(exist_ok=True)

        # Process protein (DiffDock expects: complex_id_protein.pdb)
        output_protein = output_complex_dir / f"{complex_id}_protein.pdb"
        protein_ok = clean_protein(str(protein_pdb), str(output_protein), args.max_protein_size, args.min_protein_residues)
        if not protein_ok:
            # Remove any partial output to avoid downstream scripts seeing this complex
            print(f"Protein processing failed for {complex_id}; removing partial output directory")
            shutil.rmtree(output_complex_dir, ignore_errors=True)
            continue

        # Process ligand (DiffDock expects: complex_id_ligand.sdf)
        output_ligand = output_complex_dir / f"{complex_id}_ligand.sdf"
        ligand_ok = process_ligand(str(ligand_mol2), str(output_ligand), 
                                   args.min_ligand_size, args.max_ligand_size, args.fragment_policy)
        if not ligand_ok:
            print(f"Ligand processing failed for {complex_id}; removing partial output directory")
            shutil.rmtree(output_complex_dir, ignore_errors=True)
            continue

        # Preserve original MOL2 in output directory for compatibility with other tools
        try:
            shutil.copy(str(ligand_mol2), str(output_complex_dir / f"{complex_id}_ligand.mol2"))
        except Exception as e:
            print(f"Warning: failed to copy original mol2 for {complex_id}: {e}")

        # Optionally generate pocket.pdb (residues within cutoff Å of ligand atoms)
        if args.generate_pockets:
            pocket_path = output_complex_dir / f"{complex_id}_pocket.pdb"
            try:
                ok = extract_binding_pocket(str(output_protein), str(output_ligand), str(pocket_path), cutoff=args.pocket_cutoff)
                if not ok:
                    # non-fatal; just warn
                    print(f"Warning: pocket extraction didn't produce a pocket for {complex_id}")
            except Exception as e:
                print(f"Warning: pocket extraction failed for {complex_id}: {e}")

        # Add to successful processing list
        processed_complexes.append(complex_id)
        metadata_rows.append({
            'complex_id': complex_id,
            'protein_path': f"{complex_id}/{complex_id}_protein.pdb",
            'ligand_path': f"{complex_id}/{complex_id}_ligand.sdf"
        })
    
    print(f"Successfully processed {len(processed_complexes)} complexes")
    
    # Write metadata if requested explicitly via --metadata_out (keeps output_dir clean)
    if args.metadata_out:
        metadata_path = Path(args.metadata_out)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['complex_id', 'protein_path', 'ligand_path'])
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"Metadata written to {metadata_path}")
        metadata_msg = str(metadata_path)
    else:
        print("Skipping writing metadata.csv to output directory (no --metadata_out provided)")
        metadata_msg = "not written (use --metadata_out to save elsewhere)"
    
    # Generate splits
    if args.generate_splits:
        # Determine splits output path: prefer explicit --splits_out, otherwise
        # write to a sibling `splits/` directory beside the dataset output_dir.
        if args.splits_out:
            splits_out_path = args.splits_out
        else:
            splits_out_path = str(output_dir.parent / "splits")
        create_time_splits(processed_complexes, id_to_year, str(output_dir), splits_root=splits_out_path)
    
    # Prepare ESM sequences
    if args.prepare_esm:
        # This action is intentionally disabled by default to avoid overlap
        # with `datasets/esm_embedding_preparation.py`. If you really want the
        # FASTA generated here, set --prepare_esm and then move the file or
        # incorporate it into your workflow manually. We do not create the
        # prepared_for_esm.fasta automatically to keep directories clean.
        print("Note: --prepare_esm was set but FASTA generation is intentionally disabled in this script to avoid overlap with datasets/esm_embedding_preparation.py")
    
    print(f"\n✅ PDBBind processing complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Processed complexes: {len(processed_complexes)}")
    print(f"📝 Metadata: {metadata_msg}")
    print("\n🔬 Next steps:")
    print("1. Run the ESM embedding generation script if needed")
    print("2. Test with DiffDock:")
    # Inform user where splits were written
    if args.generate_splits:
        splits_display = args.splits_out if args.splits_out else str(output_dir.parent / "splits")
    else:
        splits_display = "(no splits generated)"
    print(f"   python train.py --dataset pdbbind --data_dir {output_dir} --split_train {splits_display}/timesplit_train.txt")


if __name__ == "__main__":
    main()