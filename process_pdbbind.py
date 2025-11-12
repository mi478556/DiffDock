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
from Bio.PDB import PDBParser, PDBIO, Select
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
    parser.add_argument("--prepare_esm", action="store_true", default=True,
                        help="Prepare sequences for ESM2 embedding generation")
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


def clean_protein(protein_pdb_path: str, output_path: str, max_residues: int = 2000) -> bool:
    """Clean protein structure: remove waters/ligands, keep only standard amino acids"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", protein_pdb_path)
        
        # Count residues first
        residue_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == " ":  # Standard amino acid
                        residue_count += 1
        
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
                  min_atoms: int = 5, max_atoms: int = 100) -> bool:
    """Process ligand: convert mol2 to sdf, validate size"""
    try:
        # Read ligand from mol2 file
        mol = Chem.MolFromMol2File(ligand_mol2_path, sanitize=True, removeHs=False)
        if mol is None:
            print(f"Failed to read ligand: {ligand_mol2_path}")
            return False
        
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
        
        # Molecular complexity score (BertzCT) - high values indicate complex topology
        complexity_score = Descriptors.BertzCT(mol)
        if complexity_score > 2500:  # 1yet has ~2686, causing issues
            print(f"Ligand too topologically complex: BertzCT={complexity_score:.0f}")
            return False
            
        # Rotatable bonds - high flexibility can cause conformer generation issues
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        if rotatable_bonds > 12:  # 1yet has 12, at the limit
            print(f"Ligand too flexible: {rotatable_bonds} rotatable bonds")
            return False
            
        # Ring count - complex ring systems can cause genetic algorithm issues
        num_rings = mol.GetRingInfo().NumRings()
        if num_rings > 4:
            print(f"Ligand has too many rings: {num_rings} rings")
            return False
            
        # Molecular weight threshold
        mol_weight = Descriptors.MolWt(mol)
        if mol_weight > 550:  # 1yet has ~560
            print(f"Ligand too heavy: MW={mol_weight:.1f}")
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


def create_time_splits(complex_ids: List[str], id_to_year: Dict[str, int], 
                      output_dir: str) -> None:
    """Create time-based splits based on PDB release year"""
    
    # Filter to processed complexes and sort by year
    valid_complexes = [(cid, id_to_year.get(cid, 2020)) for cid in complex_ids if cid in id_to_year]
    valid_complexes.sort(key=lambda x: x[1])  # Sort by year
    
    # Time-based splits: train (before 2015), val (2015-2017), test (2018+)
    train_ids = [cid for cid, year in valid_complexes if year < 2015]
    val_ids = [cid for cid, year in valid_complexes if 2015 <= year < 2018]
    test_ids = [cid for cid, year in valid_complexes if year >= 2018]
    
    print(f"Time splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    # Write split files
    splits_dir = os.path.join(output_dir, "splits")
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
        protein_path = os.path.join(output_dir, complex_id, f"{complex_id}_protein_processed.pdb")
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
    
    # Create instruction file for ESM embedding generation
    instruction_path = os.path.join(output_dir, "generate_esm_embeddings.sh")
    with open(instruction_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Generate ESM2 embeddings for processed PDBBind dataset\n")
        f.write("# Install ESM first: pip install fair-esm\n\n")
        f.write(f"python -c \"\n")
        f.write("import torch\n")
        f.write("import esm\n")
        f.write("from Bio import SeqIO\n")
        f.write("import os\n\n")
        f.write("# Load ESM2 model\n")
        f.write("model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()\n")
        f.write("batch_converter = alphabet.get_batch_converter()\n")
        f.write("model.eval()\n\n")
        f.write(f"# Load sequences\n")
        f.write(f"sequences = []\n")
        f.write(f"for record in SeqIO.parse('{fasta_path}', 'fasta'):\n")
        f.write(f"    sequences.append((record.id, str(record.seq)))\n\n")
        f.write("# Generate embeddings in batches\n")
        f.write("embeddings = {}\n")
        f.write("batch_size = 16\n")
        f.write("for i in range(0, len(sequences), batch_size):\n")
        f.write("    batch = sequences[i:i+batch_size]\n")
        f.write("    _, _, batch_tokens = batch_converter(batch)\n")
        f.write("    with torch.no_grad():\n")
        f.write("        results = model(batch_tokens, repr_layers=[33], return_contacts=False)\n")
        f.write("    for j, (seq_id, _) in enumerate(batch):\n")
        f.write("        embeddings[seq_id] = results['representations'][33][j, 1:-1]  # Remove start/end tokens\n")
        f.write("    print(f'Processed {min(i+batch_size, len(sequences))}/{len(sequences)} sequences')\n\n")
        f.write(f"# Save embeddings\n")
        f.write(f"torch.save(embeddings, '{os.path.join(output_dir, 'esm2_embeddings.pt')}')\n")
        f.write(f"print('Embeddings saved to esm2_embeddings.pt')\n")
        f.write("\"\n")
    
    os.chmod(instruction_path, 0o755)
    print(f"ESM embedding generation script saved to {instruction_path}")


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
            sample_complexes = [d for d in os.listdir(data_dir) if d in complex_ids[:5]]
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
        
        # Process protein (DiffDock expects: complex_id_protein_processed.pdb)
        output_protein = output_complex_dir / f"{complex_id}_protein_processed.pdb"
        if not clean_protein(str(protein_pdb), str(output_protein), args.max_protein_size):
            continue
        
        # Process ligand (DiffDock expects: complex_id_ligand.sdf)
        output_ligand = output_complex_dir / f"{complex_id}_ligand.sdf"
        if not process_ligand(str(ligand_mol2), str(output_ligand), 
                             args.min_ligand_size, args.max_ligand_size):
            continue
        
        # Add to successful processing list
        processed_complexes.append(complex_id)
        metadata_rows.append({
            'complex_id': complex_id,
            'protein_path': f"{complex_id}/{complex_id}_protein_processed.pdb",
            'ligand_path': f"{complex_id}/{complex_id}_ligand.sdf"
        })
    
    print(f"Successfully processed {len(processed_complexes)} complexes")
    
    # Write metadata
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['complex_id', 'protein_path', 'ligand_path'])
        writer.writeheader()
        writer.writerows(metadata_rows)
    
    print(f"Metadata written to {metadata_path}")
    
    # Generate splits
    if args.generate_splits:
        create_time_splits(processed_complexes, id_to_year, str(output_dir))
    
    # Prepare ESM sequences
    if args.prepare_esm:
        prepare_esm_sequences(processed_complexes, str(output_dir))
    
    print(f"\n✅ PDBBind processing complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Processed complexes: {len(processed_complexes)}")
    print(f"📝 Metadata: {metadata_path}")
    print("\n🔬 Next steps:")
    print("1. Run the ESM embedding generation script if needed")
    print("2. Test with DiffDock:")
    print(f"   python train.py --dataset pdbbind --data_dir {output_dir} --split_train {output_dir}/splits/timesplit_train.txt")


if __name__ == "__main__":
    main()