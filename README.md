# Igonometree

Tree inference from AIRR-seq data.

## Overview

Antibody sequences are relatively short, highly similar, and exist in large clonal families. Standard likelihood-based phylogenetic methods are ill-suited for this setting (slow +overfit). Igonometree uses a pragmatic approach:

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

```{py}
## Test with real sequences
import polars as pl
from igonometree import infer_trees, extract_trees


df = pl.read_csv('example_airr.csv')
df = df.rename({'cdr1fwr3_sequence_alignment': 'sequence_alignment', 
                'cdr1fwr3_germline_alignment': 'germline_alignment', 
                'clonal_family_hilary': 'group_id'})

df = infer_trees(df, n_subsample=50)
trees = extract_trees(df)

# to show the tree, ete4 uses pyQt6 by default, which should be installed (pip install PyQt6)
key = df['group_id'][0]
trees[key].show()
```

## Technical details

* **Alignment:** Uses `mafft` for fast multiple sequence alignment.
* **Subsampling:** For large lineages, computes a distance matrix and clusters sequences. `Ns` sequences are selected from clusters.
* **Ancestral reconstruction:** Based on parsimony and the known germline sequence. Likelihood-based ASR is complicated on multifurcating trees.
* **Tree inference and placement:** Uses `raxml-ng` and `epa-ng`. 
* **Tree type:** Non-binary (collapsed) trees are used.
* **Germline required.** Needed for rooting and ASR.

## Caveats

* The method prioritizes scalability and robustness over exactness.
* Works well for qualitative structure and lineage dynamics, but don’t expect perfect branch support.
