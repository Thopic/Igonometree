import uuid
import tempfile
import random
import os
import polars as pl
from importlib.resources import files
import warnings
import subprocess
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json
from tqdm import tqdm
import shutil
from igonometree.utils import *

# location of the tools
tools_location = files("igonometree").joinpath("..", "tools")
epang = os.path.join(tools_location, "epa-ng")
raxml= os.path.join(tools_location, "raxml-ng")
mafft = os.path.join(tools_location, "mafft.bat")
gappa = os.path.join(tools_location, "gappa")


def infer_trees(df, n_subsample=100, isotype_order=False, no_leaves = False, nb_threads=1,
                seed=42, scratch_folder='./', log_file=None,
                keep_tmp_files=False):
    """
    Batch-infers phylogenetic trees for each group in a DataFrame.

    For each unique clone_id in `df`:
      1. Calls infer_tree(...) on its subset.
      2. Tags resulting tree data with clone_id.
      3. Concatenates all per‑group results into one DataFrame.
      4. Optionally writes per‑group logs to `log_file`.

    Parameters:
        df (pl.DataFrame): Must contain a ‘clone_id’ column and the columns
        required by infer_tree.
        n_subsample (int): Number of sequences to sample per group for tree
        inference.
        isotype_order (bool): If true, try to keep the right isotype order [unimplemented]
        no_leaves (bool): If true, return the final tree without the leaves
        nb_threads (int): Threads for the tree‑building step.
        seed (int): Random seed for reproducibility.
        scratch_folder (str): Directory for temporary files.
        log_file (str or None): If set, write the log output here.
        keep_tmp_files (bool): If true, keep the temporary folder with all the
        intermediate data

    Returns:
        pl.DataFrame: Combined tree annotations for all groups, with a
        ‘clone_id’ column.
    """

    try:
        df = pl.from_pandas(df)
    except:
        pass

    # cast the clone_id column to str, just in case.
    df = df.with_columns(pl.col('clone_id').cast(str))

    
    all_tree_data = None
    for clone_id, data in tqdm(df.group_by('clone_id'), total=df['clone_id'].n_unique()):
        if len(data) < 5:
            warnings.warn("Clonal lineages with less than 5 sequences are skipped & removed.", stacklevel=2)
            continue


        
        log, _, _, tree_data = try_infer_tree(data, n_subsample=n_subsample,
                                           isotype_order=isotype_order,
                                              no_leaves = no_leaves,
                                           nb_threads=nb_threads,
                                           seed=seed,
                                           scratch_folder=scratch_folder,
                                           keep_tmp_files=keep_tmp_files)
        
        


            
        tree_data = tree_data.with_columns(
            clone_id=pl.lit(clone_id[0]))

        all_tree_data = (tree_data if all_tree_data is None
                         else pl.concat([all_tree_data, tree_data],
                                        how='diagonal'))
        if log_file is not None:
            with open(log_file, 'a') as fw:
                fw.write(clone_id[0] + "\n" + log + "\n" + "#"*20)
    return all_tree_data



def try_infer_tree(df, n_subsample=100, no_leaves = False, isotype_order=False,
               nb_threads=1, seed=42, scratch_folder='./',
               keep_tmp_files=False):
    """ Try twice to infer the tree, just in case """

    log = ""
    for attempt in range(2):
        try:
            return infer_tree(df, n_subsample=n_subsample,
                              isotype_order=isotype_order,
                              no_leaves = no_leaves,
                              nb_threads=nb_threads,
                              seed=seed,
                              scratch_folder=scratch_folder,
                              keep_tmp_files=keep_tmp_files)
        
        except Exception as e:
            log += f"\n FAILURE OF INFERENCE, attempt #{attempt}, exception {e} \n"

    # in case it didn't work, return the og dataframe
    return log, None, None, df


def infer_tree(df, n_subsample=100, isotype_order=False, no_leaves=False, 
               nb_threads=1, seed=42, scratch_folder='./',
               keep_tmp_files=False):
    """
    Infers a phylogenetic tree from a single group of aligned sequences.

    Steps:
      1. Runs MAFFT alignment on all sequences + germline.
      2. Subsamples `n_subsample` sequences.
      3. Builds tree with RAxML and infer the ancestral sequences
      using parsimony.
      4. Replaces original sequences onto the tree.
      5. Returns raw logs, Newick string, and annotated DataFrame.

    Parameters:
        df (pl.DataFrame): Must include `sequence_alignment`
        and `germline_alignment`, plus `sequence_id` (unique)
        and `clone_id` (only one)
        n_subsample (int): Number of sequences to sample for tree inference.
        isotype_order (bool): If true, try to keep the right isotype order
        no_leaves (bool): If true the trees returned won't have leaves
        nb_threads (int): Threads for RAxML; should not exceed CPU cores.
        seed (int or None): Random seed for reproducibility.
        scratch_folder (str): Directory for temporary working files.
        keep_tmp_files (bool): If true, keep the temporary folder with all the
        intermediate data

    Returns:
        tuple:
            full_output (str): Combined stdout/stderr from alignment,
            tree build, and placement.
            newick_str (str): Newick representation of the final tree.
            df (pl.DataFrame): Input `df` enriched with placement and
            mutation annotations.
    """

    full_output = ""

    # seed random
    if seed is not None:
        random.seed(seed)

    # all the alignment are lower letters only
    df = df.with_columns(
        sequence_alignment=pl.col('sequence_alignment').str.to_uppercase(),
        germline_alignment=pl.col('germline_alignment').str.to_uppercase())

    # check that all the sequence_id are distincts and only 1 clone_id
    assert df['sequence_id'].n_unique() == len(df), "Some sequence_id are the same"
    assert df['clone_id'].n_unique() == 1
    
    # Use a temporary directory for intermediate files
    with tempfile.TemporaryDirectory(dir=scratch_folder, delete=(not keep_tmp_files)) as out_directory:

        # Step 1: Multiple sequence alignment
        output = align(df, out_directory, nb_threads=nb_threads)
        full_output += "ALIGN ------ \n" + output.stdout.decode() + output.stderr.decode()
        if output.returncode != 0:
            raise RuntimeError(f"Alignment failed:\n{full_output}")

        # Step 2: Subsample sequences for tree inference (clustering on the distance matrix)
        sample_representative_sequences(out_directory, n_subsample)
        
        # Step 3: Infer tree using the aligned + sampled sequences
        output, newick_str = infer_subsampled_tree(directory=out_directory,
                                                   df=df,
                                                   isotype_order=isotype_order,
                                                   nb_threads=nb_threads,
                                                   seed=seed)
        full_output += output
        

        # Step 4: Locate all the sequences on the tree and annotate them
        # with their inferred properties
        output, df = locate_sequences(df, out_directory)
        full_output += "PLACEMENT --- \n" + output.stdout.decode() + output.stderr.decode()

        if output.returncode != 0:
            raise RuntimeError(f"Placement algorithm failed:\n{full_output}")

        # Step 5: remove the leaves (germline excepted) if needed
        if no_leaves:
            tree = ete4.Tree(newick_str)
            all_leaves = [n.name for n in tree if n.name != 'germline']
            for n in tree:
                if n.name in all_leaves:
                    n.detach()
            newick_str = tree.write(props=['name', 'sequence'])
            # replace by the new tree
            df = df.with_columns(pl.lit(newick_str).alias('tree'))

        complete_tree = None
        # Step 6, probably not necessary: Create a tree with all the sequences
        # output, complete_tree = add_sequences_to_trees(out_directory)
        # full_output += "ADD TO TREES --- \n" + output.stdout.decode() + output.stderr.decode()
        # if output.returncode != 0:
        #     raise RuntimeError(f"Tree update algorithm failed:\n{full_output}")
        # df = df.with_columns(complete_tree = pl.lit(complete_tree))
        

        return full_output, newick_str, complete_tree, df

    raise RuntimeError("Temporary directory creation failed")


def collapse_placement(jplace_file, collapsed_tree_file):
    """
    Maps placement data from a jplace file onto a simplified (collapsed) tree using edge numbers,
    assigns each placement to the nearest internal node in the collapsed tree (via MRCA),
    and aggregates placement likelihoods per internal node.

    Parameters:
        jplace_file (str): Path to the .jplace file with sequence placements.
        collapsed_tree_file (str): Path to the Newick tree file used to group placements.

    Returns:
        pl.DataFrame: One row per query sequence, including:
            - best matching ancestor_id (collapsed tree node),
            - summed placement likelihoods,
            - average placement distances,
            - sequence ID.
    """
        
    with open(jplace_file) as f:
        jplace_data = json.load(f)

    tree_jplace = parse_jplace_tree(jplace_data['tree'])
    tree_collapse = ete4.Tree(collapsed_tree_file)

    for node in tree_collapse:
        if node.is_leaf:
            continue
        node.add_prop("edge_numbers", [])
    
    # identify ancestor of a given edge in the collapsed tree
    mapping_edges = {}
    for node in tree_jplace.traverse():
        if "edge_number" not in node.props:
            continue
        if node.is_leaf:
            node_collapse = list(tree_collapse.search_leaves_by_name(node.name))[0]
            mapping_edges[int(node.props["edge_number"])] = node_collapse.up.name
        else:
            # if not a leave, then it has at least 2 descendant leaves
            # both trees have the same label names 
            leaves_list = [list(tree_collapse.search_leaves_by_name(l.name))[0] for l in node]
            common_ancestor = tree_collapse.common_ancestor(leaves_list)
            mapping_edges[int(node.props["edge_number"])] = common_ancestor.up.name
        
    placements = jplace_data['placements']

    # group the placement data
    all_placements = None
    for dct in placements:
        name = dct['n'][0]
        # don't add the germline sequence
        if name == 'germline':
            continue

        df_placement = pl.DataFrame(dct['p'],
                                     schema=['edge_number', 'likelihood',
                                             'likelihood_weight_ratio',
                                             'distal_length', 'pendant_length'],
                                    orient="row"
                                     )\
                         .with_columns(
                             ancestor_id =
                             pl.col('edge_number').replace_strict(mapping_edges)
                         )

        df_placement = df_placement.group_by('ancestor_id').agg(
            pl.col('likelihood'),
            pl.col('likelihood_weight_ratio').sum(),
            pl.col('distal_length').mean(),
            pl.col('pendant_length').mean()
        ).with_columns(pl.col('likelihood').map_elements(
            lambda g: np.logaddexp.reduce(list(g)),return_dtype=float)
                      ).sort(by='likelihood_weight_ratio', descending=True
               ).with_columns(sequence_id = pl.lit(name)).head(1)

        all_placements = df_placement if all_placements is None else pl.concat([all_placements, df_placement]) 
        
    return all_placements
    

def locate_sequences(df, out_directory):
    """
    Performs phylogenetic placement of query sequences onto a reference tree using EPA-NG,
    annotates the placements, and computes mutations.

    This function executes the following steps:
    1. Runs EPA-NG to place query sequences onto the reference tree.
    2. Parses and collapses placement results to associate each query with the collapsed tree
    3. Computes mutations of each query sequence
    
    Parameters:
        df (pl.DataFrame): DataFrame containing at least 'sequence_id' and 'sequence_alignment' columns.
        out_directory (str): Path to the directory where intermediate and output files will be stored.

    Returns:
        tuple:
            - subprocess.CompletedProcess: Result of the EPA-NG execution.
            - pl.DataFrame: DataFrame with the following columns
    Raises:
        RuntimeError: If EPA-NG execution fails or if required files are missing.
    """

    
    full_alignment_file = os.path.join(out_directory, "clonal_alignment.fa")
    reduced_alignment_file = os.path.join(out_directory, "clonal_alignment_reduced.fa")

    # here we need to use "bestTree" rather than "bestTreeCollapsed" sadly
    reduced_tree_file = os.path.join(out_directory, "tree.raxml.bestTree")
    model_file = os.path.join(out_directory, "tree.raxml.bestModel")
    command = (f"{epang} --redo --ref-msa {reduced_alignment_file}"
           f" --tree {reduced_tree_file}"
           f" --query {full_alignment_file}  --outdir {out_directory}"
           f" --model {model_file}")

    result = subprocess.run(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Alignment failed:\n{result.stdout.decode()}\n{result.stderr.decode()}")

    # load and correct the placement file
    jplace_file = os.path.join(out_directory, "epa_result.jplace")
    collapsed_tree_file = os.path.join(out_directory, "tree.raxml.bestTreeCollapsed.asr")
    df_placement = collapse_placement(jplace_file, collapsed_tree_file)
    tree_collapse = ete4.Tree(collapsed_tree_file)
    newick_str = tree_collapse.write(props=['name', 'sequence', 'edge_number'])
    asr_sequences = {n.name: n.props['sequence'] for n in tree_collapse.traverse() if n.name is not None}

    # merge with the alignment info 
    df_alignment = read_fasta(full_alignment_file)
    df_alignment = df_alignment.rename({'name': 'sequence_id',
                                        'sequence': 'sequence_alignment_tree'})
    df = df.join(df_alignment, on='sequence_id')
    df = df.join(df_placement, on='sequence_id')
    kept_names = read_fasta(reduced_alignment_file)['name'].to_list()
    df = df.with_columns(in_tree = pl.col('sequence_id').is_in(kept_names))

    # identify the mrca
    mrca_sequence = tree_collapse.common_ancestor(
        [l for l in tree_collapse if l.name != 'germline']).props['sequence']
    
    germline_sequence = df_alignment.filter(
        pl.col('sequence_id') == 'germline')['sequence_alignment_tree'][0]

    # identify the mutations
    df = df.with_columns(
        germline_alignment_tree = pl.lit(germline_sequence),
        mrca_alignment_tree = pl.lit(mrca_sequence),
        parent_sequence = pl.col('ancestor_id').replace_strict(asr_sequences)
    ).with_columns(
        pl.struct(['sequence_alignment_tree', 'parent_sequence']).map_elements(
                      lambda r:
            mutation_string(r['parent_sequence'],
                            r['sequence_alignment_tree']
                            )
        , return_dtype=str).alias('mutations_from_nearest'),
        pl.struct(['sequence_alignment_tree', 'germline_alignment_tree']).map_elements(
                      lambda r:
            mutation_string(r['germline_alignment_tree'],
                            r['sequence_alignment_tree']
                            )
       , return_dtype=str ).alias('mutations_from_germline'),
        pl.struct(['sequence_alignment_tree', 'mrca_alignment_tree']).map_elements(        
                      lambda r:
            mutation_string(r['mrca_alignment_tree'],
                            r['sequence_alignment_tree']
                            )
        , return_dtype=str).alias('mutations_from_mrca')
    ).with_columns(
        pl.col('mutations_from_mrca').map_elements(lambda x:
                                                   count_mutations(x),
                                                   return_dtype=int
                                                   ).alias('nb_mutations_from_mrca'),
        pl.col('mutations_from_nearest').map_elements(lambda x:
                                                      count_mutations(x),
                                                      return_dtype=int
                                                      ).alias('nb_mutations_from_nearest'),
        pl.col('mutations_from_germline').map_elements(lambda x:
                                                       count_mutations(x),
                                                       return_dtype=int
                                                       ).alias('nb_mutations_from_germline'),
        # and add the tree
        pl.lit(newick_str).alias('tree')
    )

    return result, df
    

    
def align(df, out_directory, nb_threads=1):
    """
    Aligns sequences with MAFFT, including germline sequence.

    Parameters:
    - df (pl.DataFrame): Must contain 'sequence_id', 'sequence_alignment', and 'germline_alignment'.
    - out_directory (str): Where to write output files.
    - nb_threads (int): Number of threads in use

    Returns:
    - subprocess.CompletedProcess: Result of the MAFFT run.
    """
    indexes = df['sequence_id'].to_list()
    sequences = df['sequence_alignment'].to_list()
    result = None
    
    indexes.append("germline")
    sequences.append(df['germline_alignment'][0])
    sequence_path = os.path.join(out_directory, "original_sequences.fa")
    with open(sequence_path, 'w') as fw:
        fw.write(get_fasta(indexes, sequences))

    output_path = os.path.join(out_directory, "clonal_alignment.fa")

    
    parameters = ""
    # recommended methods by mafft for less than 200 sequences
    if len(df) < 200:
        parameters = "--nuc --globalpair --maxiterate 1000 --ep 0.248 --op 20"
    # very large alignment
    elif len(df) > 10000:
        parameters = "--nuc --retree 1 --maxiterate 0 --nofft --parttree --ep 0.123 --op 10"
    else:
        parameters = "--nuc --retree 2 --maxiterate 0 --ep 0.248 --op 20 "
    
    
    command = f"{mafft}  --thread {nb_threads} {parameters} {sequence_path} > {output_path}"
    result = subprocess.run(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)


    return result


def sample_representative_sequences(out_directory, n_subsample):
    """
    Reads aligned sequences from clonal_alignment.msa.
    If number of sequences > n_subsample:
    - Computes full pairwise Hamming distances.
    - Clusters sequences into n_subsample groups using agglomerative clustering.
    - Picks one representative per cluster (first sequence found).
    Else:
    - Uses all sequences as representatives.
    Writes selected sequences to clonal_alignment.fa.
    """
    df = read_fasta(os.path.join(out_directory, "clonal_alignment.fa"))
    germline = df.filter(pl.col('name') == 'germline')['sequence'][0]
    df = df.filter(pl.col('name') != 'germline')

    sequences = df["sequence"].to_list()
    names = df["name"].to_list()
    
    if len(df) > n_subsample:
        num_sequences = len(sequences)
        pairwise_distances = np.zeros((num_sequences, num_sequences))
        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                distance = hamming(sequences[i], sequences[j])
                pairwise_distances[i, j] = distance
                pairwise_distances[j, i] = distance

        # Clustering
        clustering = AgglomerativeClustering(n_clusters=n_subsample - 1,
                                             metric='precomputed',
                                             linkage='average')
        labels = clustering.fit_predict(pairwise_distances)
        idxs = [np.where(labels == l)[0][0] for l in np.unique(labels)]
        
    else:
        idxs = range(len(df))
            
    output_path = os.path.join(out_directory, "clonal_alignment_reduced.fa")
    with open(os.path.join(out_directory, "clonal_alignment_reduced.fa"), "w") as fw:
        fw.write(get_fasta(['germline'] + [names[i] for i in idxs],
                           [germline] + [sequences[i] for i in idxs]))
      

    
    
def infer_subsampled_tree(directory, df,
                          isotype_order=False,
                          nb_threads=1,
                          nb_start_trees=20,
                          seed=42):
    """
    Infer a phylogenetic tree from a subsampled multiple sequence alignment.

    This function runs `raxml-ng` to construct a tree from sequences stored in a
    clonal alignment file (`clonal_alignment_reduced.fa`) within the given directory.
    It performs tree inference using a GTR+G model, with both parsimony and random
    starting trees, and uses a specified outgroup ("germline").

    After tree inference, the tree is loaded, collapsed if necessary, and internal nodes
    are annotated with ancestral sequences inferred via maximum parsimony.

    Parameters
    ----------
    directory : str
        Path to the working directory containing input/output files.
    df : polars.DataFrame
        Not used in this function but expected for interface consistency.
    isotype_order : bool, optional
        Unused; placeholder for potential future logic on isotype ordering.
    nb_threads : int, default=1
        Number of threads to use for `raxml-ng`.
    nb_start_trees : int, default=20
        Number of parsimony and random starting trees to use.
    seed : int, default=42
        Random seed for reproducibility in tree inference.

    Returns
    -------
    full_output : str
        The full stdout and stderr output of the `raxml-ng` run.
    newick_str : str
        The resulting ASR-annotated tree in Newick format.

    Raises
    ------
    RuntimeError
        If `raxml-ng` fails to produce a tree.
    """
        
    # run raxml
    msa_file = os.path.join(directory, "clonal_alignment_reduced.fa")
    cmd = (f"{raxml} --seed {seed} --threads auto{nb_threads}"
           f" --force perf_threads" # shouldn't be necessary but I had issue with this in setup where it should not fail
           f" --model 'GTR+G'"
           f" --log DEBUG"
           f" --outgroup germline"
           f" --tree pars{{{nb_start_trees}}},rand{{{nb_start_trees}}}" 
           f" --msa {msa_file} --prefix {directory}/tree")

    
    output = subprocess.run(cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            shell=True)

    if isotype_order:
        pass
        #pick_trees(directory, df)

    collapsed_tree_path = os.path.join(directory, 'tree.raxml.bestTreeCollapsed')
    bipartite_tree_path = os.path.join(directory, 'tree.raxml.bestTree')
    # if bestTreeCollapsed is not created (when it's not needed), we still create a copy
    if not os.path.exists(collapsed_tree_path):
        shutil.copy(bipartite_tree_path, collapsed_tree_path)
    
    full_output = ("TREE RECONSTRUCTION ------ \n"
                    + output.stdout.decode() + output.stderr.decode())
    if output.returncode != 0:
        raise RuntimeError(f"Tree reconstruction failed:\n{full_output}")


    tree = load_tree_with_sequences(collapsed_tree_path,
                                    os.path.join(directory, 'clonal_alignment_reduced.fa')
                                    )
    tree = asr_parsimony(tree)

    newick_str = tree.write(props=['name', 'sequence'])
    with open(os.path.join(directory, "tree.raxml.bestTreeCollapsed.asr"), "w") as fw:
        fw.write(newick_str)
        
    
    return full_output, newick_str


def add_sequences_to_trees(out_directory):
    """ Add sequences to an already created tree """
    command = f"{gappa} examine graft --name-prefix EPA_added_ --fully-resolve --jplace-path {out_directory}/epa_result.jplace --out-dir {out_directory};" 
    result = subprocess.run(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)

    output = f"{result.stdout.decode()}\n{result.stderr.decode()}"
    if result.returncode != 0:
        raise RuntimeError(f"Graft failed:\n{output}")

    with open(f'{out_directory}/epa_result.newick') as f:
        tree = f.readline()
    return output, tree


def load_tree_with_sequences(tree_file, align_file):
    t = ete4.Tree(tree_file)
    df = read_fasta(align_file)
    mapping = dict(df[['name', 'sequence']].iter_rows())
    outgroup = None
    for leaf in t:
        if leaf.name == 'germline':
            t.set_outgroup(leaf)
        leaf.add_props(sequence=mapping[leaf.name])

    # add a tag for the node name, so we're sure it's unique
    uid = uuid.uuid4().hex

    for ii, node in enumerate(t.traverse()):
        if node.name is None:
            node.name = f'{uid}_Node{ii}'

    return t


def asr_parsimony(tree, seed=None):
    L = None
    for leaf in tree:
        L = len(leaf.props['sequence'])
        break
    
    if L is None:
        return tree

    # leaves to root
    germline_seq_estimate = None
    for node in tree.traverse(strategy="postorder"):
        # sequence_estimate contains all the possible character at a position [{A}, {T, C}, {C}, ...]
        if node.is_leaf:
            node.add_props(sequence_estimate=
                           [set_nucleotide(c) for c in node.props['sequence']])
            if node.name == 'germline':
                germline_seq_estimate = node.props['sequence_estimate']
        else:
            # here we just use the characters in the children (ie, not the root)
            sequence_estimate = [set.union(*[child.props['sequence_estimate'][ii] for child in node.children]) for ii in range(L)]
            node.add_props(sequence_estimate = sequence_estimate)

    # root to leaves
    for node in tree.traverse(strategy='preorder'):
        # for the root, we juste use the germline
        if node.up is None:
            node.add_prop('sequence_estimate', germline_seq_estimate)
            continue
        sequence_estimate = []
        for s_node, s_parent in zip(node.props['sequence_estimate'],
                                    node.up.props['sequence_estimate']):
            # no ambiguity: we choose the child
            if len(s_node) == 1:
                sequence_estimate += [s_node]
            # no intersection between parent and child, we choose the parent
            elif len(s_node & s_parent) == 0: 
                sequence_estimate += [s_parent]
            # ambiguous -- or trivial -- intersection between parent and child, we keep the ambiguity 
            else:
                sequence_estimate += [s_node & s_parent]
        node.add_prop('sequence_estimate', sequence_estimate)
                
    # now go from the sequence estimate to the actual sequence
    for node in tree.traverse():
        node.add_props(sequence = "".join([get_ambiguous_nucleotide(cs) for cs in node.props['sequence_estimate']]))
        node.del_prop('sequence_estimate')

    # before saving add a top root to avoid the issue with the disapparition of the root feature
    new_tree = ete4.Tree()
    new_tree.name = "root"
    tree.dist = 0
    new_tree.add_child(tree)
    
    return new_tree
