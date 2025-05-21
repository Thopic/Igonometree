import polars as pl
import tempfile
import random
import numpy as np
import os
import subprocess
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import ete4
import json
import re
from tqdm import tqdm
import shutil


def extract_trees(df):
    """
    Returns a dictionary mapping each group_id to its corresponding ete4 tree.

    Assumes input `df` is the output of `infer_trees`, containing at least `group_id` and `tree` columns.
    Each tree includes an artificial root node added to satisfy Newick format constraints.
    This function removes that node and returns the actual subtree.

    Returns:
        dict: {group_id: ete4.Tree} with one child stripped from the root.
    """
    ddf = df[['group_id', 'tree']].unique()
    dct =  dict(zip(ddf['group_id'], ddf['tree']))

    # check that the structure is right
    for k in dct:
        assert len(ete4.Tree(dct[k]).children) == 1, f"One of the tree is malformed \n{dct[k]}"
    
    return {k: ete4.Tree(dct[k]).children[0] for k in dct}


def set_nucleotide(nuc):
    """
    Maps a nucleotide or IUPAC ambiguity code to its corresponding nucleotide set.
    Returns a set of possible nucleotides.
    """
    iupac = {
        'A': {'A'},
        'C': {'C'},
        'G': {'G'},
        'T': {'T'},
        'R': {'A', 'G'},
        'Y': {'C', 'T'},
        'S': {'G', 'C'},
        'W': {'A', 'T'},
        'K': {'G', 'T'},
        'M': {'A', 'C'},
        'B': {'C', 'G', 'T'},
        'D': {'A', 'G', 'T'},
        'H': {'A', 'C', 'T'},
        'V': {'A', 'C', 'G'},
        'N': {'A', 'C', 'G', 'T'},
        '-': {'-'},  # gap
        '.': {'.'},
    }
    return iupac[nuc.upper()]


def get_ambiguous_nucleotide(nuc_set):
    """
    Given a set of nucleotides, returns the corresponding IUPAC ambiguity code.
    Defaults to 'N' if combination is not recognized.
    """

    if len(nuc_set) == 1:
        return list(nuc_set)[0]
    
    nuc_tuple = tuple(sorted(nuc_set))
    code_map = {
        ('A',): 'A',
        ('C',): 'C',
        ('G',): 'G',
        ('T',): 'T',
        ('-',): '-',
        ('.',): '.',
        ('A', 'G'): 'R',
        ('C', 'T'): 'Y',
        ('C', 'G'): 'S',
        ('A', 'T'): 'W',
        ('G', 'T'): 'K',
        ('A', 'C'): 'M',
        ('C', 'G', 'T'): 'B',
        ('A', 'G', 'T'): 'D',
        ('A', 'C', 'T'): 'H',
        ('A', 'C', 'G'): 'V',
        ('A', 'C', 'G', 'T'): 'N',
    }
    return code_map.get(nuc_tuple, 'N')



def hamming(s1, s2):
    """
    Computes Hamming distance between two strings of equal length.
    """
    assert len(s1) == len(s2), "Can't compute the hamming distance for sequences of different length"
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def mutation_string(ref: str, alt: str) -> str:
    """
    ref, alt: aligned sequences of equal length (with '-' for gaps)
    Returns comma‑separated mutations of form (MIXCR style):
      S[from]pos[to]  (substitution)
      D[from]pos      (deletion)
      Ipos[to]        (insertion)
    Positions are zero‑based, for insertions the pos is where the inserted
    nucleotide sits in the final (inserted) sequence. 
    """
    assert len(ref) == len(alt), "Aligned sequences must be same length"

    muts = []
    for i, (r, a) in enumerate(zip(ref, alt)):
        assert r in {'A', 'T', 'G', 'C', 'N', 'R','Y','S','W','K','M','B','D','H','V', '-'}, f"Invalid character {r} in the sequence"
        assert a in {'A', 'T', 'G', 'C', 'N', 'R','Y','S','W','K','M','B','D','H','V', '-'}, f"Invalid character {a} in the sequence"
        if r == a:
            continue
        if r != "-" and a != "-":
            muts.append(f"S{r}{i}{a}")
        elif r != "-" and a == "-":
            muts.append(f"D{r}{i}")
        elif r == '-' and a != '-':
            muts.append(f"I{i}{a}")
    return ",".join(muts)

def count_mutations(x) -> int:
    """
    Counts the number of mutations in a MIXCR-style mutation string.
    """
    if len(x) == 0:
        return 0
    t = x.split(',')
    return len(t)



def read_fasta(fn):
    """
    Reads a FASTA file and returns a DataFrame with 'name' and 'sequence' columns.

    Parameters:
        fn (str): Path to FASTA file.

    Returns:
        pl.DataFrame
    """
    names, seqs = [], []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                names.append(line[1:])
                seqs.append("")
            else:
                seqs[-1] += line
    return pl.DataFrame({"name": names, "sequence": seqs}).with_columns(sequence = pl.col('sequence').str.to_uppercase())


def get_fasta(names, seqs):
    """
    Converts lists of names and sequences into FASTA format string.

    Parameters:
        names (list[str]): Sequence identifiers.
        seqs (list[str]): Corresponding sequences.

    Returns:
        str: FASTA-formatted string.
    """
    if len(names) != len(seqs):
        raise ValueError("names and seqs must have the same length")

    return "\n".join(f">{n}\n{s.upper()}" for n, s in zip(names, seqs)) + "\n"


def parse_jplace_tree(newick_str):
    """
    Parses a jplace-style Newick string, replacing edge number annotations
    for compatibility with ETE4.
    """
    newick_str = re.sub(r"\{(\d+)\}", r"[&&NHX:edge_number=\1]", newick_str)
    t = ete4.Tree(newick_str)
    return t
    
