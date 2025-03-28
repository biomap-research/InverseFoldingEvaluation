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
import logging

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
                logging.info("Transferred model to GPU")
        
        
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"working dir is {args.work_dir}")
        results_dir = Path(args.work_dir)/Path('results')
        logging.info(f"working dir is {results_dir}")

        df_input = pd.read_csv(args.input_csv)
        logging.info(f"working on csv file: {args.input_csv}")

        pdb_dir = Path(args.pdb_dir)
        for _, row in tqdm(df_input.iterrows()):
                pdb_name = row['pdb_name']
                logging.info(f"working on {pdb_name}")
                pdb_path = pdb_dir / f"{pdb_name}.pdb"
                vh_chain = row['vh_chain']
                vl_chain = row['vl_chain']

                try:
                        structure = esm.inverse_folding.util.load_structure(pdb_path)
                        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

                        logging.info(f'Native sequence loaded from structure file: {pdb_path}')


                        fasta_path = Path(results_dir)/Path(f"{pdb_name}.fasta")
                        logging.info(f'Saving sampled sequences to: {fasta_path}')

                        with open(fasta_path, 'w') as f:
                                for i in range(args.num_samples):
                                        logging.info(f'\nSampling.. ({i+1} of {args.num_samples})')
                                        sampled_seq_vh = esm.inverse_folding.multichain_util.sample_sequence_in_complex(
                                                model, coords, vh_chain, temperature=args.temperature)
                                        
                                        sampled_seq_vl = esm.inverse_folding.multichain_util.sample_sequence_in_complex(
                                                model, coords, vl_chain, temperature=args.temperature)
            
                                        recovery_vh = np.mean([(a==b) for a, b in zip(native_seqs[vh_chain], sampled_seq_vh)])
                                        recovery_vl = np.mean([(a==b) for a, b in zip(native_seqs[vl_chain], sampled_seq_vl)])
                                        f.write(f'>sampled_seq_{i+1} - {recovery_vh} - {recovery_vl} \n')
                                        f.write(f"{sampled_seq_vh}/{sampled_seq_vl}\n")


                except Exception as e:
                        logging.error(f"failed with {pdb_name} - {e}")

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
            '--work_dir', type=str,
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

    log_file = Path(args.work_dir)/Path("esm_if.log")
    logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    if args.multichain_backbone:
        sample_seq_multichain_batch(model, alphabet, args)
    else:
        sample_seq_singlechain(model, alphabet, args)


if __name__ == '__main__':
    main()
