# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Scores sequences based on a given structure.
#
# usage:
# score_log_likelihoods.py [-h] [--outpath OUTPATH] [--chain CHAIN] pdbfile seqfile

import argparse
from biotite.sequence.io.fasta import FastaFile, get_sequences
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os

import esm
import esm.inverse_folding


def score_singlechain_backbone(model, alphabet, args):
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    coords, native_seq = esm.inverse_folding.util.load_coords(args.pdbfile, args.chain)
    print('Native sequence loaded from structure file:')
    print(native_seq)
    print('\n')

    ll, _ = esm.inverse_folding.util.score_sequence(
            model, alphabet, coords, native_seq) 
    print('Native sequence')
    print(f'Log likelihood: {ll:.2f}')
    print(f'Perplexity: {np.exp(-ll):.2f}')

    print('\nScoring variant sequences from sequence file..\n')
    infile = FastaFile()
    infile.read(args.seqfile)
    seqs = get_sequences(infile)
    Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outpath, 'w') as fout:
        fout.write('seqid,log_likelihood\n')
        for header, seq in tqdm(seqs.items()):
            ll, _ = esm.inverse_folding.util.score_sequence(
                    model, alphabet, coords, str(seq))
            fout.write(header + ',' + str(ll) + '\n')
    print(f'Results saved to {args.outpath}') 


def score_multichain_backbone(model, alphabet, args):
        if torch.cuda.is_available() and not args.nogpu:
                model = model.cuda()
                print("Transferred model to GPU")

        df_input = pd.read_csv(args.input_csv)

        for _, row in tqdm(df_input.iterrows()):
                pdb_name = row['pdb_name']
                print(f"working on {pdb_name}")
                pdb_path = row['pdb_path']
                fasta_path = row['fasta_path']
                chain = row['vh_chain']
                structure = esm.inverse_folding.util.load_structure(pdb_path)
                coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
                target_chain_id = chain
                native_seq = native_seqs[target_chain_id]
                # print('Native sequence loaded from structure file:')
                # print(native_seq)
                # print('\n')

                ll_native, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                        model, alphabet, coords, target_chain_id, native_seq) 
                # print('Native sequence')
                # print(f'Log likelihood: {ll:.2f}')
                # print(f'Perplexity: {np.exp(-ll):.2f}')

                # print('\nScoring variant sequences from sequence file..\n')
                infile = FastaFile()
                infile.read(fasta_path)
                seqs = get_sequences(infile)
                
                # Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
                task_output_csv = os.path.join(args.outpath, f"{pdb_name}.csv")

                with open(task_output_csv, 'w') as fout:
                        fout.write('seqid,log_likelihood,wt_ll\n')
                        for header, seq in tqdm(seqs.items()):
                                ll, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                                        model, alphabet, coords, target_chain_id, str(seq))
                                fout.write(header + ',' + str(ll) + ',' + str(ll_native) + '\n')
                print(f'Results saved to {task_output_csv}') 


def main():
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--input_csv', type=str,
            help='input csv for the batch of designs',
    )
#     parser.add_argument(
#             'seqfile', type=str,
#             help='input filepath for variant sequences in a .fasta file',
#     )
    parser.add_argument(
            '--outpath', type=str,
            help='output filepath for scores of variant sequences',
            default='output/sequence_scores.csv',
    )
#     parser.add_argument(
#             '--chain', type=str,
#             help='chain id for the chain of interest', default='A',
#     )
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
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    if args.multichain_backbone:
        score_multichain_backbone(model, alphabet, args)
    else:
        score_singlechain_backbone(model, alphabet, args)



if __name__ == '__main__':
    main()
