import io
import os
import warnings
from typing import Dict, Any
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio import BiopythonWarning
import pandas as pd
import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, Superimposer, Structure, PDBIO
from typing import Dict, Any
from abnumber import Chain as ABNUM_Chain

import seaborn as sns
import matplotlib.pyplot as plt

from abnumber import Chain, exceptions
from biopandas.pdb import PandasPdb
import sys
import pandas as pd
import json
from Bio.PDB import PDBParser, Superimposer, Structure
from biopandas.pdb import PandasPdb
import shutil
import subprocess as sp
from datetime import datetime, timezone
from pathlib import Path

import abnumber
from collections import Counter


from scripts.utils import get_cdr_residue_idx_list, calculate_seq_identity, fasta2seq, calculate_seq_similarity_blosum62, parse_blosum62, get_hydropathy
BLOSUM62_MATRIX = parse_blosum62('../data/resources/BLOSUM62.txt')



restype_1to3 = {
    "R": "ARG",
    "K": "LYS",
    "H": "HIS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "G": "GLY",
    "P": "PRO",
    "A": "ALA",
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP",
}
aa_list = restype_1to3.keys()



class Inverse_Folding_Design:
    def __init__(self, args, method = 'esm-if', antibody_type = 'vhh'):
        self.pdb_name = args['pdb_name']
        # self.pdb_path = args['pdb_path']
        self.design_path = args['design_path']
        self.antibody_type = antibody_type

        design_results_basename = os.path.basename(self.design_path).split('.')[0]
        if method == 'antifold':
            design_results_basename = design_results_basename.rsplit('_', 1)[0]

        if not self.pdb_name == design_results_basename:
            raise ValueError(f"pdb_name {self.pdb_name} does not match design name {design_results_basename}")

        # this part fix the bug of incompatible seq from df vs se from pdb file
        # we only observe incompatible problem in some vh in fab

        _vh_seq = args['vh_seq']

        if self.antibody_type == 'fab': # fab design
            self.wt_seq = ({
                'H': _vh_seq,
                'L': args['vl_seq']
            })
        else: # vhh design
            self.wt_seq = ({
                'H': args['vh_seq'],
                'L': '-'
            })

        self.seqs = fasta2seq(self.design_path)

        if method in ['antifold', 'mpnn', 'abmpnn']: # the 1st seq of antifold and mpnn output is the wild type sequence, also included abmpnn
            frist_id = list(self.seqs)[0]
            del self.seqs[frist_id] # remove the 1st wt sequence
        
        if self.antibody_type == 'fab':
            self.seqs = {k: {'H': v.split('/')[0], 'L': v.split('/')[1]} for k, v in self.seqs.items()}
        elif self.antibody_type == 'vhh':
            try: 
                self.seqs = {k: {'H': v.split('/')[0], 'L': v.split('/')[1]} for k, v in self.seqs.items()}
            except:
                self.seqs = {k: {'H': v, 'L': ''} for k, v in self.seqs.items()}

        if self.antibody_type == 'fab':
            self._cdr_list = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']
        elif self.antibody_type == 'vhh':
            self._cdr_list = ['H_CDR1', 'H_CDR2', 'H_CDR3']

        self.cdr_range, self.cdr_pos, self.cdr_seq = {}, {}, {}
        for cdr in self._cdr_list:
            self.cdr_range[cdr] = range(int(args[cdr].split('-')[0]), int(args[cdr].split('-')[1]))
            self.cdr_pos[cdr] = list(self.cdr_range[cdr])
            self.cdr_pos[cdr] = [self.cdr_pos[cdr][0] - 1] + self.cdr_pos[cdr] # this is to include the -1 position in the list, beacause the cdr_seq number from zero ABNUM_Chain

            if "H" in cdr:
                self.cdr_seq[cdr] = ''.join([self.wt_seq['H'][i] for i in self.cdr_pos[cdr]])
            elif "L" in cdr:
                self.cdr_seq[cdr] = ''.join([self.wt_seq['L'][i] for i in self.cdr_pos[cdr]])
            else:
                raise ValueError(f'unrecognized cdr type {cdr}')

        self.cdr_concat = ''.join([self.cdr_seq[cdr] for cdr in self._cdr_list])
        # self.cdr_concat_H = ''.join([self.cdr_seq[cdr] for cdr in self._cdr_list[:3]])
        self.df_all_design = self._get_df_all_design()
        self.df_all_design['wt_H_length'] = len(self.wt_seq['H'])
        self.df_all_design['wt_L_length'] = len(self.wt_seq['L'])


    def _get_df_all_design(self):
        
        # generate df from self.seqs, which is a dict
        df = pd.DataFrame.from_dict(self.seqs, orient='index').reset_index()

        df.columns = ['design_id', 'design_seq_H', 'design_seq_L']
        df['pdb_name'] = self.pdb_name

        # collect cdrs from designed seqs
        for cdr in self._cdr_list:
            if "H" in cdr:
                df[f"seq-{cdr}"] = df['design_seq_H'].apply(lambda s: ''.join([s[i] for i in self.cdr_pos[cdr]]))
            elif "L" in cdr:
                df[f"seq-{cdr}"] = df['design_seq_L'].apply(lambda s: ''.join([s[i] for i in self.cdr_pos[cdr]]))
            else:
                raise ValueError(f'unrecognized cdr type {cdr}')
        
        # concat all cdrs to get seq_cdr_concat
        df['seq_cdr_concat'] = df.apply(lambda row: ''.join([row[f'seq-{cdr}'] for cdr in self._cdr_list]), axis=1)

        # calculate cdr identity and cdr_concat identity
        for cdr in self._cdr_list:
            df[f"identity-{cdr}"] = df[f"seq-{cdr}"].apply(lambda s: calculate_seq_identity(s, self.cdr_seq[cdr]))

        df[f"identity-cdr_concat"] = df["seq_cdr_concat"].apply(lambda s: calculate_seq_identity(s, self.cdr_concat))
        
        # calculate cdr similarity and cdr_concat similarity
        for cdr in self._cdr_list:
            _wt_blosum62 = calculate_seq_similarity_blosum62(self.cdr_seq[cdr], self.cdr_seq[cdr], BLOSUM62_MATRIX)
            df[f"similarity_B62-{cdr}"] = df[f"seq-{cdr}"].apply(lambda s: calculate_seq_similarity_blosum62(s, self.cdr_seq[cdr], BLOSUM62_MATRIX))
            df[f"similarity_B62-{cdr}"] = df[f"similarity_B62-{cdr}"] / _wt_blosum62

        _wt_blosum62 = calculate_seq_similarity_blosum62(self.cdr_concat, self.cdr_concat, BLOSUM62_MATRIX)
        df["similarity_B62-cdr_concat"] = df["seq_cdr_concat"].apply(lambda s: calculate_seq_similarity_blosum62(s, self.cdr_concat, BLOSUM62_MATRIX))
        df["similarity_B62-cdr_concat"] = df["similarity_B62-cdr_concat"] / _wt_blosum62

        # calculate cdr hydropathy
        for cdr in self._cdr_list:
            df[f"hydropathy-{cdr}"] = df[f"seq-{cdr}"].apply(lambda s: get_hydropathy(s).mean())
            df[f"hydropathy_wt-{cdr}"] = get_hydropathy(self.cdr_seq[cdr]).mean()
            

        return df

    def get_df_pos(self):
        dfs = []
        for cdr in self._cdr_list:
            for pos in self.cdr_pos[cdr]:
                c = Counter(dict(
                    zip(restype_1to3.keys(), [0]*len(restype_1to3))
                ))
                if "H" in cdr:
                    c.update([s['H'][pos-1] for s in self.seqs.values()])
                elif "L" in cdr:
                    c.update([s['L'][pos-1] for s in self.seqs.values()])
                else:
                    raise ValueError(f'unrecognized cdr type {cdr}')

                df = pd.DataFrame.from_dict(c, orient='index').transpose()
                df.insert(0, 'pdb_name', self.pdb_name)
                df.insert(1, 'cdr', cdr)
                df.insert(2, 'pos', pos)

                chain = cdr.split('_')[0]
                df.insert(3, 'wt_aa', self.wt_seq[chain][pos-1])

                dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)



class Batch_Designs:
    def __init__(self, df, results_dir, method, antibody_type):
        self.results_dir = results_dir
        self.batch = os.path.basename(results_dir)
        self.method = method

        self.antibody_type = antibody_type

        
        files = os.listdir(self.results_dir)
        if method in ['mpnn', 'abmpnn']: # included both mpnn and abmpnn
            file_dirs = [Path(self.results_dir)/Path(f)/Path('seqs')/Path(f"{f}.fa") for f in files]
            pdb_names = files
        else:
            files = [f for f in files if f.endswith('.fasta')] # remove files in case some .csv in the results_dir
            file_dirs = [os.path.join(self.results_dir, f) for f in files]
            pdb_names = [f.split('.')[0] for f in files]

        if method == 'antifold': # the output of antifold have additional "_HC" after the pdb name
            pdb_names = [f.rsplit('_', 1)[0] for f in pdb_names]

        df_results = pd.DataFrame({
            'pdb_name': pdb_names,
            'design_path': file_dirs
        })

        self.df_info = pd.merge(df, df_results, on='pdb_name', how='inner')
        print(f"df_info of shape {self.df_info.shape}")

    def analyze_results(self):
        if_designs = []
        results_dfs = []
        pos_dfs = []
        for _, row in self.df_info.iterrows():

            args = row.to_dict()
            if_design = Inverse_Folding_Design(args, method=self.method, antibody_type=self.antibody_type)
            pos_dfs.append(if_design.get_df_pos())

            results_dfs.append(if_design.df_all_design)

            


        self.df_raw = pd.concat(results_dfs, ignore_index=True)
        print(f"collected df_raw with shape: {self.df_raw.shape}")
        self.df_pos = pd.concat(pos_dfs, ignore_index=True)
        print(f"collected df_pos with shape: {self.df_pos.shape}")

        cols = ['identity-cdr_concat', 'similarity_B62-cdr_concat',
            'identity-H_CDR1', 'identity-H_CDR2', 'identity-H_CDR3',
            'similarity_B62-H_CDR1', 'similarity_B62-H_CDR2','similarity_B62-H_CDR3',
            'hydropathy-H_CDR1','hydropathy-H_CDR2','hydropathy-H_CDR3',
            'hydropathy_wt-H_CDR1','hydropathy_wt-H_CDR2','hydropathy_wt-H_CDR3']

        if self.antibody_type == 'fab': # additional cols if antibody is fab
            cols += ['identity-L_CDR1', 'identity-L_CDR2', 'identity-L_CDR3',
                'similarity_B62-L_CDR1', 'similarity_B62-L_CDR2','similarity_B62-L_CDR3',
                'hydropathy-L_CDR1','hydropathy-L_CDR2','hydropathy-L_CDR3',
                'hydropathy_wt-L_CDR1','hydropathy_wt-L_CDR2','hydropathy_wt-L_CDR3']

        col_metric = dict(zip(cols, ['mean']*len(cols)))
        self.df_pdb = self.df_raw.groupby('pdb_name').agg(col_metric)
        print(f"collected df_pdb with shape: {self.df_pdb.shape}")

        # calculate df_pos_gp
        df_pos_group = self.df_pos.groupby('wt_aa').agg('sum').reset_index()
        if 'X' in df_pos_group.columns:
            df_pos_group = df_pos_group.drop(columns=['X'])
        if '-' in df_pos_group.columns:
            df_pos_group = df_pos_group.drop(columns=['-'])
        df_pos_group['wt_aa'] = pd.Categorical(df_pos_group['wt_aa'], categories=list(restype_1to3.keys()), ordered=True)
        df_pos_group = df_pos_group.sort_values('wt_aa')
        df_pos_group.index = df_pos_group['wt_aa']
        df_pos_group = df_pos_group.iloc[:, 4:]
        self.df_pos_gp = df_pos_group.apply(lambda x: x / x.sum(), axis=1)

        

    def plot_confusion_heat_map(self):
        plt.figure(figsize=(15, 12))
        sns.heatmap(self.df_pos_gp, annot=True, fmt=".3f", linewidth=.5, cmap="Blues") # cmap="BuPu" cmap="coolwarm"
        plt.show()
    

def get_cdr(row, cdr='H_CDR3'):
    vh_seq = row['vh_seq']
    start, end = row[cdr].split('-')
    cdr = vh_seq[int(start)-1: int(end)]
    return cdr
    
class Dataset:
    def __init__(self, df, df_info = None, antibody_type = 'vhh'):
        self.antibody_type = antibody_type
        if df_info is not None:
            self.df_info = df_info
        self.df_raw = df
        self.df_raw['year'] = [int(d.split('-')[0]) for d in self.df_raw['date']]
        self.df_raw['H_CDR3_seq'] = self.df_raw.apply(lambda x: get_cdr(x, cdr='H_CDR3'), axis=1)


        # self.df_2023 = self._filter_df()
        

    def _filter_df(self):
        """ this is the first version of filter, the filtered df are used for inverse folding design """

        df_2023 = self.df_raw[self.df_raw['year'] >= 2023]
        df_before_2023 = self.df_raw[self.df_raw['year'] < 2023]
        print(f"all pdb after 2023: {len(df_2023)}")
        print(f"df_bafore: {len(df_before_2023)}")

        # removing antibody with CDR3 seq in dataset before 2023
        df_2023 = df_2023[~df_2023['H_CDR3_seq'].isin(df_before_2023['H_CDR3_seq'])]
        print(f"after removing antibody with identical cdr2 before 2023: {len(df_2023)}")

        df_2023 = df_2023.drop_duplicates(subset='H_CDR3_seq')
        print(f"after dropping duplicate cdr3 seq within df_2023: {len(df_2023)}")
        
        df_2023 = df_2023[df_2023['antigen_chains'] == 'A'] # removing complex with multiple antigen chains
        print(f"after removing complex with multiple antigen chains: {len(df_2023)}")

        return df_2023

    def filter_high_quality_pdb(self):
        """
        This filter is used to filter pdb for the follwoing criteria 
        1. PDB with low resolution and PDB method NMR
        2. PDB with broken chain
        
        """
        if self.df_info is None:
            raise Exception('need to define self.df_info before this function')
        print(f"------------------filtering high quality pdb for {self.antibody_type} ------------------")
        high_quality_pdb = self._filter_high_quality_pdb(resolution_cutoff = 3.5)
        print(f"original df length: {len(self.df_2023)}")
        df_hq = self.df_2023[self.df_2023['entry'].isin(high_quality_pdb)].copy()
        print(f"df length after removing low resolution pdb: {len(df_hq)}")

        df_hq['chain_break_num'] = df_hq.apply(lambda row: check_pdb_for_chain_breaks(row['pdb_path']), axis=1)
        df_hq = df_hq[df_hq['chain_break_num'] == 0]
        print(f"df length after removing broken chain: {len(df_hq)}")

        print(f"use self.df_hq to check the df after filtering")
        self.df_hq = df_hq

    def filter_unwanted_pdb(self, filtered_pdbs_dict):
        """
        This filter is used to filter pdb for the follwoing criteria 
        1. PDB with low resolution and PDB method NMR
        2. PDB with broken chain
        
        """
        if self.df_info is None:
            raise Exception('need to define self.df_info before this function')
        print(f"------------------filtering high quality pdb for {self.antibody_type} ------------------")

        df_hq = self.df_hq
        print(f"input df shape: {len(df_hq)}")
        for k in filtered_pdbs_dict:
            if k in ['covid_set']:
                df_hq = df_hq[~df_hq['pdb'].isin(filtered_pdbs_dict[k])]
            else:
                df_hq = df_hq[~df_hq['pdb_name'].isin(filtered_pdbs_dict[k])]
            print(f"filter with {k}: {len(df_hq)}")

        print(f"use self.df_filter to check the df after filtering")
        self.df_filter = df_hq


    def _filter_high_quality_pdb(self, resolution_cutoff = 3.5):

        # removing pdb without resolution value, these pdb are usually those analyzed by NMR
        df = self.df_info[self.df_info['resolution'] != 'NOT']
        df.loc[:, 'resolution'] = [_.split(',')[0] for _ in df['resolution']] # some pdb have resolution value such as "3.0, 3.0", remove duplicated value
        df.loc[:, 'resolution'] = df['resolution'].astype(float)

        df_good_quality = df[df['resolution'] < resolution_cutoff]
        return set(df_good_quality['pdb'].to_list())




class Exps:
    def __init__(self, df_vhh, df_fab, exp_info_json):
        self.df_vhh = df_vhh
        self.df_fab = df_fab

        self.exp_info = self._get_default_info(exp_info_json)
        print("Got the following exp_info")
        for k in self.exp_info:
            print(f"design id: {k}")
            print(self.exp_info[k])

            

    def _get_default_info(self, json_file):
        with open(json_file, 'r') as json_file:
            return json.load(json_file)

    def collect_results(self):

        self.results = {}
        for k in self.exp_info.keys():
            method = self.exp_info[k]['method']
            results_dir = self.exp_info[k]['results_dir']
            antibody_type = self.exp_info[k]['antibody_type']

            if antibody_type == 'vhh':
                df = self.df_vhh 
            elif antibody_type == 'fab':
                df = self.df_fab 
            else:
                raise ValueError(f"unrecognized antibody_type: {antibody_type}")
            
            self.results[k] = Batch_Designs(df, results_dir, method=method, antibody_type = antibody_type)
            print(f"collecting results for {k} - {self.exp_info[k]}")
            self.results[k].analyze_results()
    


def calculate_distance(atom1, atom2):
    """calculate distance between two atoms"""
    diff_vector = atom1.coord - atom2.coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

DISTANCE_THRESHOLD = 2.0  
def check_pdb_for_chain_breaks(pdb_file, distance_threshold=DISTANCE_THRESHOLD):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    chain_breaks = []
    
    for model in structure:
        for chain in model:
            if chain.id =='H' or chain.id=='L':
                prev_c_atom = None
                
                for residue in chain:
                    # check for atoms in main chain
                    if "N" in residue and "CA" in residue and "C" in residue:
                        n_atom = residue["N"]
                        ca_atom = residue["CA"]
                        c_atom = residue["C"]
                        
                        if prev_c_atom is not None:
                            distance = calculate_distance(prev_c_atom, n_atom)
                            
                            if distance > distance_threshold:
                                chain_breaks.append((chain.id, prev_residue_id, residue.id[1], distance))
                        
                        prev_c_atom = c_atom
                        prev_residue_id = residue.id[1]

    return len(chain_breaks)