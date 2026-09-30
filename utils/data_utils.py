import os
import re
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from inmoose.pycombat import pycombat_seq

from utils.plot_utils import plot_pca


def load_timepoint_dir(timepoint_dir: str, cleaned_only=False, original=True):
    replicates = {}
    c = 0
    for dir in os.listdir(timepoint_dir):
        dir_path = os.path.join(timepoint_dir, dir)
        if not os.path.isdir(dir_path):
            continue
        c += 1
        timepoint_files = {}
        for file in os.listdir(dir_path):
            if file.startswith("."):
                continue
            if cleaned_only and "cleaned" not in file:
                continue
            if original and not file.endswith("rsem.counts.txt"):
                continue
            if "mock" in file:
                timepoint_files["0"] = os.path.join(dir_path, file)
                continue
            tp = file.split("dpi")[0][-1]
            timepoint_files[tp] = os.path.join(dir_path, file)
        replicates[c] = timepoint_files
    print(f"Found {len(replicates.keys())} replicates across {len(replicates[1].keys())} timepoints.")
    return replicates


def process_expression(count_data: str, protein_map: str, virus_map: str, column="TPM", threshold=1):
    mapping = pd.read_csv(protein_map, sep="\t")
    g2n_map = mapping.set_index("Gene stable ID")["NCBI gene (formerly Entrezgene) ID"].to_dict()
    g2n_map = {k: int(v) for k, v in g2n_map.items() if not pd.isna(v)}

    # viral genes
    virus_mapping = pd.read_csv(virus_map, sep="\t", header=None)
    v2n_map = virus_mapping.set_index(0)[1].to_dict()
    v2n_map = {k: int(v) for k, v in v2n_map.items() if not pd.isna(v)}

    # combine
    g2n_map.update(v2n_map)
    mappable_gene_ids = set(g2n_map.keys())

    sample = pd.read_csv(count_data, sep="\t")
    sample.set_index("gene_id", inplace=True)

    reachable_nodes = {}
    for idx, row in sample.iterrows():
        if idx not in mappable_gene_ids:
            continue
        if row[column] > threshold:
            reachable_nodes[idx] = row[column]
        else:
            reachable_nodes[idx] = 0

    reachable_nodes_ncbi = {g2n_map[x]: reachable_nodes[x] for x in reachable_nodes}
    return reachable_nodes_ncbi


def batch_correction(
    replicates: dict,
    protein_map,
    virus_map,
    max_timepoints: int,
    save=False,
    replicates_to_keep="all",
    return_counts=False,
):
    # lets skip batch correction while not changing the function name, what could go wrong?
    # dict {1: {}, 2: {}, 3: {}}
    if os.path.exists("data/batch_corrected_counts.csv"):
        print("Loading existing batch corrected data")
        corrected = pd.read_csv("data/batch_corrected_counts.csv")
        corrected.set_index("Unnamed: 0", inplace=True)
    else:
        corrected, _, _ = load_timepoint_data(replicates, protein_map, virus_map, max_timepoints)
        # corrected = apply_pycombat(replicates, protein_map, virus_map, max_timepoints)

        if save:
            corrected.to_csv("data/batch_corrected_counts.csv")

    gene_length_map = process_expression(
        replicates[1]["0"], protein_map, virus_map, column="effective_length", threshold=-1
    )
    # if gene length is 0, set to na
    gene_length_map = {k: pd.NA if v == 0 else v for k, v in gene_length_map.items()}
    tpm_corrected = calculate_tpm(corrected, gene_length_map)
    # tpm_corrected = calculate_alr(corrected)

    if replicates_to_keep != "all":
        remaining_cols = [col for col in tpm_corrected.columns if "r_1" in col]
        tpm_corrected = tpm_corrected[remaining_cols]
        print(tpm_corrected.columns)

    final = format_expression(tpm_corrected)
    count_format = format_expression(corrected)

    if return_counts:
        return count_format, corrected
    return final, tpm_corrected


def filter_by_expr(counts, cpm_cutoff=1, sample_cutoff=3):
    lib_size = counts.sum(axis=0)
    cpm = counts.div(lib_size, axis=1) * 1e6
    keep = (cpm > cpm_cutoff).sum(axis=1) >= sample_cutoff
    counts_filtered = counts.loc[keep]
    return counts_filtered


def load_timepoint_data(replicates: dict, protein_map: str, virus_map: str, max_timepoints: int):
    collected = {}
    replicate = []
    tp_cov = []
    for key, paths in replicates.items():
        for i in range(max_timepoints):  # seven time points
            if f"{i}" in paths:
                # expected_count because pycombat_seq uses count data
                data = process_expression(paths[f"{i}"], protein_map, virus_map, "expected_count", -1)
                collected[f"r_{key}_tp_{i}"] = data
                replicate.append(f"batch {key}")
                tp_cov.append(f"tp {i}")

    return pd.DataFrame(collected), replicate, tp_cov


def apply_pycombat(df, replicate, covariates):
    return pycombat_seq(df, replicate, covariates)


def calculate_tpm(batch_corrected, gene_lengths):
    lengths = pd.Series(gene_lengths).reindex(batch_corrected.index)
    missing = lengths.isna().sum()
    if missing > 0:
        print(f"Warning: {missing} genes have no length entry and will be dropped.")
        batch_corrected = batch_corrected.loc[lengths.notna()]
        lengths = lengths.dropna()

    # Step 1: RPK — reads per kilobase
    rpk = batch_corrected.divide(lengths / 1e3, axis=0)

    # Step 1a: Replace Na values with zero (when using effective gene length we may end up dividing by 0)
    rpk = rpk.fillna(0)

    # Step 2: TPM — normalise each sample to sum to 1e6
    tpm = rpk.divide(rpk.sum(axis=0) / 1e6, axis=1)
    return tpm


def calculate_alr(batch_corrected, ref_gene_id=7756):
    # get indices of rows where std == 0 or mean < 1
    mask = (batch_corrected.std(axis=1) == 0) | (batch_corrected.mean(axis=1) < 1)
    df_adj = batch_corrected + 1

    ref_values = df_adj.loc[ref_gene_id]

    # Compute ALR per sample column
    alr_df = np.log(df_adj.div(ref_values, axis=1))

    alr_df.loc[mask, :] = 0

    return alr_df


def format_expression(df: pd.DataFrame):
    result = defaultdict(dict)

    # Get unique timepoints from column names
    timepoints = set()
    for col in df.columns:
        match = re.search(r"tp_(\d+)", col)
        if match:
            timepoints.add(match.group(1))

    for tp in timepoints:
        # Get all columns for this timepoint, sorted by replicate number
        tp_cols = sorted(
            [
                col for col in df.columns
                if (m := re.search(r"tp_(\d+)", col)) and m.group(1) == tp
            ],
            key=lambda x: int(re.search(r"r_(\d+)", x).group(1)),
        )


        for idx in df.index:
            result[f"tp{tp}"][idx] = df.loc[idx, tp_cols].tolist()

    return dict(result)


def format_tpm_counts(file_path, protein_map, virus_map, replicates="all"):
    df = pd.read_csv(file_path)

    # human genes
    mapping = pd.read_csv(protein_map, sep="\t")
    g2n_map = mapping.set_index("Gene stable ID")["NCBI gene (formerly Entrezgene) ID"].to_dict()
    g2n_map = {k: int(v) for k, v in g2n_map.items() if not pd.isna(v)}

    # viral genes
    virus_mapping = pd.read_csv(virus_map, sep="\t", header=None)
    v2n_map = virus_mapping.set_index(0)[1].to_dict()
    v2n_map = {k: int(v) for k, v in v2n_map.items() if not pd.isna(v)}

    # combine
    g2n_map.update(v2n_map)

    df = df.set_index(df.columns[0])

    # use g2n_map to filter df and then convert index
    df = df[df.index.isin(g2n_map.keys())]
    df.index = df.index.map(g2n_map)
    df = df[~df.index.duplicated(keep="first")]

    if replicates != "all":
        remaining_cols = [col for col in df.columns if "r_1" in col]
        df = df[remaining_cols]

    return format_expression(df), df


def batch_correct_tp(df, timepoints):
    df = pycombat_seq(df, timepoints)
    return df


def write_graph(graph: nx.Graph, file: str, nodes=False):
    if nodes:
        edge_attributes = list(next(iter(graph.edges(data=True)))[2].keys())
        with open(f"{file}.edges.tsv", "w") as f:
            f.write("src\tdest\t" + "\t".join(edge_attributes) + "\n")
            for u, v, d in graph.edges(data=True):
                if u == v:  # exclude connections to itself
                    continue
                if "weight" in d:
                    f.write(f"{u}\t{v}\t{d['weight'] * 100}\t{d.get('real_weight', 'none')}\n")
                else:
                    f.write(f"{u}\t{v}\n")

        one_node = next(iter(graph.nodes(data=True)))
        node_attributes = list(one_node[1].keys())
        if "timepoints" in node_attributes:
            node_attributes.remove("timepoints")
        if "time" in node_attributes:
            node_attributes.remove("time")

        with open(f"{file}.nodes.tsv", "w") as f:
            f.write("node_id\t" + "\t".join(node_attributes) + "\n")
            for node_id, attribs in graph.nodes(data=True):
                row = [str(node_id)] + [str(attribs.get(attr, "None")) for attr in node_attributes]
                f.write("\t".join(row) + "\n")

    else:
        with open(file, "w") as f:
            for u, v in graph.edges():
                f.write(f"{u}\t{v}\n")
    print(f"Wrote graph to {file}!")


def get_timepoint_means(df: pd.DataFrame, virus_ids: list, cutoff=1):
    df = df.loc[~df.index.isin(virus_ids)]
    df = df[df.mean(axis=1) > cutoff]
    means = df.mean(axis=0).to_list()
    return means


def write_sif(G, filepath, gene_map: str, interaction_label="interacts_with"):
    mapping = (
        pd.read_csv(gene_map, sep="\t").set_index("NCBI gene (formerly Entrezgene) ID")["Gene stable ID"].to_dict()
    )
    with open(filepath, "w") as f:
        f.write(f"ID_interactor_a\t{interaction_label}\tID_interactor_b\n")
        for u, v, data in G.edges(data=True):
            interaction = data.get("interaction", interaction_label)
            try:
                u_name = mapping[u]
                v_name = mapping[v]
            except KeyError:
                continue
            f.write(f"{u_name}\t{interaction}\t{v_name}\n")

        # Handle isolated nodes (no edges) - SIF allows a line with just the node
        for n in nx.isolates(G):
            try:
                n_name = mapping[n]
            except KeyError:
                continue
            f.write(f"{n_name}\n")


def write_active_nodes(G, filepath, gene_map):
    mapping = (
        pd.read_csv(gene_map, sep="\t").set_index("NCBI gene (formerly Entrezgene) ID")["Gene stable ID"].to_dict()
    )
    with open(filepath, "w") as f:
        for node in G.nodes():
            try:
                f.write(f"{mapping[node]}\n")
            except KeyError:
                continue
