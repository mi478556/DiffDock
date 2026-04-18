
import os
import pickle
from argparse import ArgumentParser

import torch
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--esm_embeddings_path', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new', help='')
parser.add_argument('--output_path', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.pt', help='')
parser.add_argument('--repr_layer', type=int, default=33,
                    help='Which ESM representation layer to extract (default 33)')
parser.add_argument('--sequences_fasta', type=str, default='data/pdb_2021aug02/sequences_to_id.fasta', help='FASTA mapping file for numeric ids')
parser.add_argument('--useful_sequences', type=str, default='data/pdb_2021aug02/useful_sequences.pkl', help='Pickle with sequences list')
args = parser.parse_args()

dic = {}

# load sequences FASTA mapping (headers and sequences)
headers = []
sequences = []
with open(args.sequences_fasta) as fh:
    cur_header = None
    cur_seq = []
    for line in fh:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if cur_header is not None:
                headers.append(cur_header)
                sequences.append(''.join(cur_seq))
            cur_header = line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_header is not None:
        headers.append(cur_header)
        sequences.append(''.join(cur_seq))

# read sequences list
with open(args.useful_sequences, 'rb') as f:
    useful_sequences = pickle.load(f)

available = set([os.path.splitext(filename)[0] for filename in os.listdir(args.esm_embeddings_path)])

for filename in tqdm(sorted(os.listdir(args.esm_embeddings_path)), desc='Aggregating'):
    base = os.path.splitext(filename)[0]
    try:
        data = torch.load(os.path.join(args.esm_embeddings_path, filename), map_location='cpu')
        reps = data.get('representations', {})
        if not reps:
            raise KeyError('representations missing')
        layer = args.repr_layer if args.repr_layer in reps else next(iter(reps))
        rep = reps[layer]

        # resolve sequence key: numeric ids -> sequences list, header-like keys -> headers mapping
        seq_key = None
        if base.isdigit():
            idx = int(base)
            if 0 <= idx < len(sequences):
                seq_key = sequences[idx]
        elif base in headers:
            seq_key = sequences[headers.index(base)]
        else:
            # if extractor used header-like names, use that header as the key
            seq_key = base

        dic[seq_key] = rep
    except Exception as e:
        tqdm.write(f"Skipping {filename}: {e}")

torch.save(dic, args.output_path)