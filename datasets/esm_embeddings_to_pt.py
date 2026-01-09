import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--esm_embeddings_path', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new', help='')
parser.add_argument('--output_path', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.pt', help='')
parser.add_argument('--layer', type=int, default=33,
                    help='Which representation layer to extract from each .pt file (default: 33). If the layer is not present, the script will fall back to the first available layer.')
parser.add_argument('--sequences_fasta', type=str, default='', help='Optional FASTA file used to map numeric ids to sequences')
args = parser.parse_args()

# load sequences list if provided (used when extractor produced numeric filenames '0.pt','1.pt',...)
sequences = None
headers = None
if args.sequences_fasta:
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

out = {}
files = sorted([f for f in os.listdir(args.esm_embeddings_path) if os.path.isfile(os.path.join(args.esm_embeddings_path, f))])
for filename in tqdm(files, desc='Converting embeddings', total=len(files)):
    try:
        key = os.path.splitext(filename)[0]
        # load on CPU to avoid GPU/serialization issues
        data = torch.load(os.path.join(args.esm_embeddings_path, filename), map_location='cpu')
        reps = data.get('representations', {})
        if not reps:
            raise KeyError('representations missing')
        # prefer user-requested layer, otherwise fall back to the first available layer
        if args.layer in reps:
            chosen_layer = args.layer
        else:
            chosen_layer = next(iter(reps))
            tqdm.write(f"Warning: requested layer {args.layer} not found in {filename}; using layer {chosen_layer} instead")
        rep = reps[chosen_layer]

        # determine canonical sequence key:
        seq_key = None
        if key.isdigit() and sequences is not None:
            idx = int(key)
            if 0 <= idx < len(sequences):
                seq_key = sequences[idx]
        # if header-style key and sequences_fasta provided, map header -> sequence
        if seq_key is None and headers is not None and key in headers:
            seq_key = sequences[headers.index(key)]
        # fallback: use the filename key itself (header-like)
        if seq_key is None:
            seq_key = key

        out[seq_key] = rep
    except Exception as e:
        tqdm.write(f"Skipping {filename}: {e}")

torch.save(out, args.output_path)