import esm
import torch
import numpy as np
import math
import os
import pandas as pd


def next_batch(X1, X2, batch_size):
    """Return data for next batch"""
    tot = len(X1)
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx]
        batch_x2 = X2[start_idx: end_idx]
        yield (batch_x1, batch_x2, (i + 1))


def protein_representation_gen(input_file, output_file):
    sequences = []
    number_ids = []
    o_df = pd.read_csv(input_file)
    o_df = o_df.to_dict(orient='records')
    for line in o_df:
        number_id = line['number_id']
        sequence = line['sequence']

        number_ids.append(number_id)
        sequences.append(sequence[:1022])

    # Load ESM-1 model
    protein_encoder, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    protein_encoder.eval()  # disables dropout for deterministic results

    sequence_representations = []

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        for batch_x1, batch_x2, batch_No in next_batch(number_ids, sequences, 5):
            # Prepare data
            protein_data = list(zip(batch_x1, batch_x2))
            batch_labels, batch_strs, batch_tokens = batch_converter(protein_data)
            results = protein_encoder(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, (_, seq) in enumerate(protein_data):
                sequence_representations.append(token_representations[i, 1: len(seq) + 1].mean(0))

    np.save(output_file, sequence_representations)
    print('success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate protein sequence representations.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output .npy file.')
    args = parser.parse_args()

    protein_representation_gen(args.input_file, args.output_file)
