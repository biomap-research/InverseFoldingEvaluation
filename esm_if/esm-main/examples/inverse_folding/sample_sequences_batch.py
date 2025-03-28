# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Sample sequences based on a given structure (multinomial sampling, no beam search).
#
# usage: sample_sequences.py [-h] [--chain CHAIN] [--temperature TEMPERATURE]
# [--outpath OUTPATH] [--num-samples NUM_SAMPLES] pdbfile

import argparse
import numpy as np
from pathlib import Path
import torch

import esm
import esm.inverse_folding
import pandas as pd
from tqdm import tqdm
import pandas as pd
import os


def sample_seq_singlechain(model, alphabet, args):
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    coords, native_seq = esm.inverse_folding.util.load_coords(args.pdbfile, args.chain)
    print('Native sequence loaded from structure file:')
    print(native_seq)

    print(f'Saving sampled sequences to {args.outpath}.')

    Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outpath, 'w') as f:
        for i in range(args.num_samples):
            print(f'\nSampling.. ({i+1} of {args.num_samples})')
            sampled_seq = model.sample(coords, temperature=args.temperature, device=torch.device('cuda'))
            print('Sampled sequence:')
            print(sampled_seq)
            f.write(f'>sampled_seq_{i+1}\n')
            f.write(sampled_seq + '\n')

            recovery = np.mean([(a==b) for a, b in zip(native_seq, sampled_seq)])
            print('Sequence recovery:', recovery)


def sample_seq_multichain_batch(model, alphabet, args):
        if torch.cuda.is_available() and not args.nogpu:
                model = model.cuda()
                print("Transferred model to GPU")
        
        
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        print(f"working dir is {args.outpath}")

        df_input = pd.read_csv(args.input_csv)
        print(f"working on csv file: {args.input_csv}")
        pdb_dir = Path(args.pdb_dir)

        for _, row in tqdm(df_input.iterrows()):
                pdb_name = row['pdb_name']
                print(f"working on {pdb_name}")
                pdb_path = pdb_dir / f"{pdb_name}.pdb"
                vh_chain = row['vh_chain']

                try:
                        structure = esm.inverse_folding.util.load_structure(pdb_path)
                        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
                        target_chain_id = vh_chain
                        # native_seq = native_seqs[target_chain_id]


                        structure = esm.inverse_folding.util.load_structure(pdb_path)
                        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
                        target_chain_id = vh_chain
                        native_seq = native_seqs[target_chain_id]
                        print(f'Native sequence loaded from structure file: {pdb_path}')
                        # print(native_seq)
                        # print('\n')

                        

                        result_dir = Path(args.outpath)/Path(f"{pdb_name}.fasta")
                        print(f'Saving sampled sequences to: {result_dir}')

                        with open(result_dir, 'w') as f:
                                for i in range(args.num_samples):
                                        # print(f'\nSampling.. ({i+1} of {args.num_samples})')
                                        sampled_seq = esm.inverse_folding.multichain_util.sample_sequence_in_complex(
                                                model, coords, target_chain_id, temperature=args.temperature)
                                        # print('Sampled sequence:')
                                        # print(sampled_seq)
                                        f.write(f'>sampled_seq_{i+1}\n')
                                        f.write(sampled_seq + '\n')

                                        recovery = np.mean([(a==b) for a, b in zip(native_seq, sampled_seq)])
                                # print('Sequence recovery:', recovery)

                except Exception as e:
                        print(f"failed with {pdb_name} - {e}")

def main():
    parser = argparse.ArgumentParser(
            description='Sample sequences based on a given structure.'
    )
    parser.add_argument(
            '--input_csv', type=str,
            help='csv path, must contains "pdb_name"',
    )
    parser.add_argument(
            '--pdb_dir', type=str,
            help='dir where the input pdb files saved, file name must same with pdb_name in csv_path',
    )
    parser.add_argument(
            '--temperature', type=float,
            help='temperature for sampling, higher for more diversity',
            default=1.,
    )
    parser.add_argument(
            '--outpath', type=str,
            help='output filepath for saving sampled sequences',
            default='output/sampled_seqs.fasta',
    )
    parser.add_argument(
            '--num-samples', type=int,
            help='number of sequences to sample',
            default=1,
    )
    parser.set_defaults(multichain_backbone=False)
    parser.add_argument(
            '--multichain-backbone', action='store_true',
            help='use the backbones of all chains in the input for conditioning'
    )
    parser.add_argument(
            '--singlechain-backbone', dest='multichain_backbone',
            action='store_false',
            help='use the backbone of only target chain in the input for conditioning'
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
  
    args = parser.parse_args()

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    if args.multichain_backbone:
        sample_seq_multichain_batch(model, alphabet, args)
    else:
        sample_seq_singlechain(model, alphabet, args)


if __name__ == '__main__':
    main()
