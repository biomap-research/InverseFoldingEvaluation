import os
from pathlib import Path

from Bio import SeqIO
from biopandas.pdb import PandasPdb
import abnumber
from typing import List, Dict
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser
from Bio.Data import SCOPData
from Bio.PDB import Model, Chain, Residue, Selection, Structure
import  scipy.spatial.distance as distance

import shutil
import gzip
import numpy as np

pwd = os.path.dirname(__file__)

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


def pdb2seq(pdb_path: Path) -> dict[str, str]:
    return (
        PandasPdb()
        .read_pdb(str(pdb_path))
        .amino3to1()
        .groupby("chain_id")
        .agg({"residue_name": sum})
        .to_dict()["residue_name"]
    )

def calculate_seq_identity(seq1, seq2):

    if len(seq1) != len(seq2):
        raise ValueError("两个序列的长度必须相同")

    matches = sum(a == b for a, b in zip(seq1, seq2))

    identity = (matches / len(seq1))

    return identity

def parse_blosum62(filename):

    matrix = {}
    with open(filename, 'r') as file:
        amino_acids = file.readline().split()
        for line in file:
            values = line.split()
            row_acid = values[0]
            for col_acid, score in zip(amino_acids, values[1:]):
                matrix[(row_acid, col_acid)] = int(score)
    return matrix

blosum62_matrix = parse_blosum62('%s/../data/resources/BLOSUM62.txt' % pwd)

def calculate_seq_similarity_blosum62(seq1, seq2, blosum62_matrix):
    if len(seq1) != len(seq2):
        raise ValueError("seqs must have same length")

    score = 0

    for a, b in zip(seq1, seq2):
        score += blosum62_matrix.get((a, b), 0)

    return score

def fasta2seq(fasta_path: Path) -> dict[str, str]:
    res: dict[str, str] = {}
    for idx, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        res[f"{idx}-{record.id}"] = str(record.seq)

    return res
    
def get_cdr_residue_idx_list(seq: str | abnumber.Chain, offset=1, cdr=3):
    if isinstance(seq, str):
        chain = abnumber.Chain(seq, scheme="imgt")
    elif isinstance(seq, abnumber.Chain):
        chain = seq
    else:
        raise NotImplementedError()
    cdr_ranges = []
    if cdr == "all":
        for i in range(3):
            cdr_ranges.append(
                get_cdr_residue_idx_list(seq, offset, cdr=i+1)
            )
    else:
        cdr = eval(f"chain.cdr{cdr}_seq")
        start = seq.index(cdr)
        for idx in range(start, start + len(cdr)):
            cdr_ranges.append(idx + offset)
    return cdr_ranges


def number_antibody(
    seq: str,
    scheme: str,
    assign_germline: bool = True,
    **kwargs: dict[str, str | list[str] | None],
) -> dict[str, int | str | None]:
    """Run numbering on the given antibody Fv sequence.

    Args:
        seq: The Fv sequence.
        scheme: One of ``imgt``, ``chothia``, ``kabat``, or ``aho``.
        assign_germline: Include the V and J gene germlines in the returned dict.
        kwargs: Passed to [``abnumber.Chain``](https://abnumber.readthedocs.io/en/latest/#chain).

    Returns:
        A dictionary containing:
            1) the starting position of the Fv region in ``seq``,
            2) sequences of FR1-4 and CDR1-3,
            3) optionally germlines of V and J genes, and
            4) error messages during numbering.
    """
    # TODO: deal with abnumber's bug of slicing a few leading/trailing residues
    try:
        from abnumber import Chain

        chain = Chain(seq, scheme=scheme, assign_germline=assign_germline, **kwargs)
        fv_start_pos = seq.index(chain.seq)  # offset of numbered sequence
        res: dict[str, str | int | None] = {
            "fv_offset": fv_start_pos,
            "FR1": chain.fr1_seq,
            "CDR1": chain.cdr1_seq,
            "FR2": chain.fr2_seq,
            "CDR2": chain.cdr2_seq,
            "FR3": chain.fr3_seq,
            "CDR3": chain.cdr3_seq,
            "FR4": chain.fr4_seq,
        }
        if assign_germline:
            res = res | {"V_gene": chain.v_gene, "J_gene": chain.j_gene}
        res["numbering_error_msg"] = None

    except Exception as e:
        res_keys = ["fv_offset", "FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
        if assign_germline:
            res_keys.extend(["V_gene", "J_gene"])
        res = {k: None for k in res_keys}

        if isinstance(e.args[0], bytes):
            error_msg = e.args[0].decode("utf-8")
        else:
            error_msg = str(e)

        res["numbering_error_msg"] = error_msg

    return res


def get_sequence_by_biopython(pdb_path) -> Dict:
    """Return pdb sequences by dict."""
    parser = PDBParser(QUIET=True)

    if pdb_path.endswith('.gz'):
        with gzip.open(pdb_path, 'rt') as f:
            structure = parser.get_structure('PDB_structure', f)

    elif pdb_path.endswith('.pdb'):
        structure = parser.get_structure(None, pdb_path)
    else:
        raise TypeError("do not support this type")


    model = structure[0]
    seq_dict = {}
    for chain in model:
        seq = "".join(
            [
                SCOPData.protein_letters_3to1.get(residue.resname, "X")
                for residue in Selection.unfold_entities(chain, "R")
            ]
        )
        seq_dict[chain.id] = seq
    return seq_dict

def find_mutations(seq1, seq2):

    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length.")
    
    mutations = []
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:

            mutation = f"{seq1[i]}{i+1}{seq2[i]}"
            mutations.append(mutation)
    
    return tuple(mutations)


def make_mutations(seq, mutations):
    mut_seq = [ char for char in seq ]
    for mutation in mutations:
        if mutation == '':
            pass
        else:
            wt, pos, mt = mutation[0], int(mutation[1:-1]) - 1, mutation[-1]
            assert(seq[pos] == wt)
            mut_seq[pos] = mt
    mut_seq = ''.join(mut_seq).replace('-', '')
    return mut_seq

hydropathy_dict = {
    "C": 2.5,  "D": -3.5, "S": -0.8, "Q": -3.5, "K": -3.9,
    "I": 4.5,  "P": -1.6, "T": -0.7, "F": 2.8,  "N": -3.5,
    "G": -0.4, "H": -3.2, "L": 3.8,  "R": -4.5, "W": -0.9,
    "A": 1.8,  "V": 4.2,  "E": -3.5, "Y": -1.3, "M": 1.9,
    "X": 0,    "-": 0
}




def get_hydropathy(aa_seq):
    return np.array([hydropathy_dict[a] for a in aa_seq])


