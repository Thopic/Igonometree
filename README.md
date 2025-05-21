# Igonometree

Python package for tree inference from AIRR-seq data.

## Overview

Antibody sequences are relatively short, highly similar, and can sometimes form extremely large clonal families. Standard likelihood-based phylogenetic methods are ill-suited for large families (slow + overfit). Igonometree uses a pragmatic approach:

* Subsample representative sequences from each clonal lineage.
* Infer a tree on this subset.
* Map the remaining sequences onto the tree.

This allows scaling to large datasets while maintaining reasonable accuracy.

## What it does

* Builds trees for each clonal lineage. If the lineage has fewer than `N` sequences (default: 100), all are used.
* Infers internal ancestral sequences based on parsimony + germline.
* For each input sequence, determines:
  * Its placement on the tree.
  * Nearest ancestral node.
  * Mutations.

The input is a DataFrame in AIRR format with the following required columns:

* `sequence_id`: unique identifier for each sequence
* `sequence_alignment`: the sequence (not necessarily aligned)
* `germline_alignment`: the corresponding germline sequence
* `group_id`: clonal lineage identifier

Note:

* Both `sequence_alignment` and `germline_alignment` must be present but do not need to be pre-aligned — alignment is recomputed internally.
* You can restrict these two columns to a specific sequence region (e.g., FR1 to FR3), no need to use the full sequence. 

## Install

If you're lucky, clone the folder, cd into it and:
```
pip install -e .
```

If you're not lucky, you may also need to install the following tools and place them in the `tools/` directory (replacing the existing ones):

* **raxml-ng** (v1.2.2):
  Install from [github.com/amkozlov/raxml-ng](https://github.com/amkozlov/raxml-ng)
  → Place the binary as: `tools/raxml-ng`

* **epa-ng** (v0.3.8):
  Install from [github.com/pierrebarbera/epa-ng](https://github.com/pierrebarbera/epa-ng)
  → Place the binary as: `tools/epa-ng`

* **mafft** (v7.526):
  Download from [mafft.cbrc.jp](https://mafft.cbrc.jp/alignment/software/)
  → Required files:

  * `tools/mafft.bat`
  * `tools/mafftdir/` (including `bin/mafft` and `libexec/`)


## Usage

```py
## Test with real sequences
import polars as pl
from igonometree import infer_trees, extract_trees


df = pl.read_csv('example_airr.csv')
df = df.rename({'cdr1fwr3_sequence_alignment': 'sequence_alignment', 
                'cdr1fwr3_germline_alignment': 'germline_alignment', 
                'clonal_family_hilary': 'group_id'})

# n_subsample is the size of the "core" tree, but all the sequences are analyzed
df = infer_trees(df, n_subsample=50)
trees = extract_trees(df)

# to show the tree, ete4 uses pyQt6 by default, which should be installed (pip install PyQt6)
key = df['group_id'][0]
trees[key].show()
```

## Technical details

* **Alignment:** Uses `mafft` for fast multiple sequence alignment.
* **Subsampling:** For large lineages, computes a distance matrix and clusters sequences with AgglomerativeClustering. `Ns` sequences are selected from clusters.
* **Tree inference and placement:** Uses `raxml-ng` and `epa-ng`. Return trees that are usually non-binary (collapsed) (which is very frequent for antibodies).
* **Ancestral reconstruction:** For ASR (and just for ASR), we use parsimony rather than likelihood. Likelihood-based ASR methods don't really exist for multifurcating trees, and the existence of the germline make parsimony based methods fairly good.
* **Germline required.** Needed for rooting and ASR.

