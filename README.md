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
  * Novel mutations.


## Usage



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
