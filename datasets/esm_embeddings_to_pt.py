import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--esm_embeddings_path', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new', help='')
parser.add_argument('--output_path', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.pt', help='')
args = parser.parse_args()

dict = {}
# Prefer a deterministic order and show a tqdm progress bar with a description
files = sorted([f for f in os.listdir(args.esm_embeddings_path) if os.path.isfile(os.path.join(args.esm_embeddings_path, f))])
for filename in tqdm(files, desc='Converting embeddings', total=len(files)):
    try:
        key = os.path.splitext(filename)[0]
        data = torch.load(os.path.join(args.esm_embeddings_path, filename))
        # keep only the desired layer representation
        dict[key] = data['representations'][33]
    except Exception as e:
        # skip problematic files but show a short message on error
        tqdm.write(f"Skipping {filename}: {e}")
torch.save(dict, args.output_path)