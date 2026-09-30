import math
import os
import random
from copy import deepcopy

import networkx as nx
import pandas as pd
from dotenv import load_dotenv

from gene_regulatory.grn_graph import (
    adjusted_pvals,
    build_graph,
    build_random_graph,
    filter_alert_nodes,
    find_alert_nodes,
    get_reachable_graph,
)
from utils.data_utils import get_timepoint_means, write_active_nodes, write_sif
from utils.graph_utils import (
    calculate_edge_metric,
    constant_nodes,
    cut_edges_below_cutoff,
)
from utils.plot_utils import plot_hist


def override_virus_scores(graph, virus_nodes):
    # override virus scores to one since measurement is unreliable or something
    for node in virus_nodes:
        for succ in graph.successors(node):
            graph[node][succ]["real_weight"] = 1
    return graph


def get_gene_sets(file: str):
    pathways = pd.read_csv(file)
    gene_sets = pathways.groupby("1")["0"].apply(set).to_dict()  # for each pathway, have a set of genes its involved in
    pathway_sets = (
        pathways.groupby("0")["1"].apply(set).to_dict()
    )  # for each gene, have a set of pathways its involved in
    pathway_names = pathways.groupby("1")["3"].first().to_dict()

    pathway_sets = {k: v for k, v in pathway_sets.items()}
    return gene_sets, pathway_sets, pathway_names


def match_genes(graph: nx.Graph, pathway_sets: dict):
    """
    Return for each pathway the genes that could be found in the supplied graph
    """
    gene_set_hit = {}
    for node in graph.nodes():
        involved_pathways = pathway_sets.get(str(node), set())
        if len(involved_pathways) == 0:
            continue
        for pw in involved_pathways:
            if pw not in gene_set_hit:
                gene_set_hit[pw] = []
            gene_set_hit[pw].append(node)
    return dict(sorted(gene_set_hit.items(), key=lambda x: len(x[1])))


def pathway_fills(gene_set_hits: dict, gene_sets: dict):
    """
    Return for each pathway the percentage of genes that were found in the graph
    """
    pathway_match = {}
    for pathway in gene_set_hits:
        found_genes = gene_set_hits[pathway]
        all_genes_in_pw = gene_sets[pathway]
        pathway_match[pathway] = len(found_genes) / len(all_genes_in_pw)
    return dict(sorted(pathway_match.items(), key=lambda x: x[1], reverse=True))


def pathway_leads(gene_set_hits: dict, gene_sets: dict, min_size=15):
    """
    Filter pathways for some minimum size and return those that exceed it
    """
    pathway_lead = {}
    for pathway in gene_set_hits:
        found_genes = gene_set_hits[pathway]
        if len(gene_sets[pathway]) < min_size:  # skip small pathways
            continue
        pathway_lead[pathway] = found_genes
    return pathway_lead


def format_hits(pathway_lead, name_map):
    formatted_hits = []
    for pw, genes in pathway_lead.items():
        name_format = f"{name_map[pw]} {pw}"
        gene_format = ";".join([str(x) for x in genes])
        formatted_hits.append({"pathway_ids": pw, "Term": name_format, "Lead_genes": gene_format})

    return pd.DataFrame(formatted_hits)


def remove_necessary(graph: nx.Graph, hits, gene_sets, virus_nodes):
    neighbours = {x for vn in virus_nodes for x in graph.neighbors(vn)}
    neighbours.update(set(virus_nodes))

    adj_hits = {}
    adj_gene_set = {}
    removed = 0

    for k, v in hits.items():
        corrected_leads = set(v) - neighbours
        remaining_genes = {int(x) for x in gene_sets[k] if x.isdigit()} - neighbours

        if len(remaining_genes) > 0:
            adj_hits[k] = corrected_leads
            adj_gene_set[k] = remaining_genes
        else:
            removed += 1
    print(f"Removed {removed} pathways due to 0 length.")

    return adj_hits, adj_gene_set


def cut_random_graph(random_graph: nx.DiGraph, real_graph: nx.DiGraph, match_edges=True):
    real_graph_size = real_graph.number_of_edges() if match_edges else real_graph.number_of_nodes()

    r_edge_scores = {}
    for u, v, d in random_graph.edges(data=True):
        if u == v:
            continue
        r_edge_scores[(u, v)] = abs(d["real_weight"])

    sorted_r_edge_scores = dict(sorted(r_edge_scores.items(), key=lambda x: x[1], reverse=True))
    shuffle_r_edge_scores = list(sorted_r_edge_scores.keys())
    random.shuffle(shuffle_r_edge_scores)

    included_edges = []
    unique_nodes = set()
    for edge in shuffle_r_edge_scores:
        if match_edges and len(included_edges) < real_graph_size:
            included_edges.append(edge)
        elif not match_edges and len(unique_nodes) < real_graph_size:
            unique_nodes.add(edge[0])
            unique_nodes.add(edge[1])
            included_edges.append(edge)
        else:
            break

    r_graph = random_graph.edge_subgraph(included_edges).copy()
    return r_graph


def print_sars(hits, gene_set):
    print("| Pathway ID | Score | Pathway Name | Pathway # genes |")
    print("| ---------- | ----- | ------------ | --------------- |")
    for k, v in hits.items():
        if "SARS" not in name_map[k]:
            continue
        score = len(v) / len(gene_set[k])

        print(f"| {k} | {score:.2f} | {name_map[k]} | {len(gene_set[k])} |")


def gene_sets_to_maximum(graph: nx.Graph, gene_sets: dict):
    available_nodes = {str(x) for x in graph.nodes()}
    max_gene_sets = {}
    for g_set in gene_sets:
        full_gene_set = set(gene_sets[g_set])
        filtered_gene_set = full_gene_set & available_nodes
        max_gene_sets[g_set] = filtered_gene_set

    return max_gene_sets


def write_pathways(real_pw, random_pw, name_map, filename):
    with open(filename, "w") as f:
        f.write("pathway_id\treal_pct\trandom_pct\tname\n")
        f.writelines(f"{key}\t{real_pw[key]}\t{random_pw[key]}\t{name_map[key]}\n" for key in real_pw)


if __name__ == "__main__":
    full_graph, data_matrix = build_graph(no_batch_corr=(os.getenv("TAXID") == "10298"), return_counts=False)
    filter_graph = constant_nodes(full_graph, exclusion_score=0.0, logp1=False, mean_expression_threshold=1)
    virus_nodes = [n for n, d in filter_graph.nodes(data=True) if d["is_virus"]]
    reachable_graph = get_reachable_graph(filter_graph.copy(), virus_nodes)
    timepoint_means = get_timepoint_means(data_matrix, virus_nodes)

    scored_graph = calculate_edge_metric(deepcopy(reachable_graph), metric="rho_prop", tp_means=timepoint_means)
    scored_graph = override_virus_scores(scored_graph, virus_nodes)

    random_graphs = []
    for i in range(5):
        random_g = build_random_graph(scored_graph, randomize_edges=False, shuffle_nodes=False, shuffle_order=True)
        scored_random_g = calculate_edge_metric(random_g.copy(), metric="rho_prop", tp_means=timepoint_means)
        random_graphs.append(scored_random_g)

    cutoff = adjusted_pvals(scored_graph, random_graphs)
    scored_graph = cut_edges_below_cutoff(scored_graph, cutoff)
    print(scored_graph)
    reachable_scored_graph = get_reachable_graph(scored_graph, virus_nodes)
    print(reachable_scored_graph)

    ### ACTUAL STUFF HAPPENING
    gene_sets, pathway_sets, name_map = get_gene_sets(os.getenv("PATHWAY_FILE"))
    gene_sets = gene_sets_to_maximum(scored_graph, gene_sets)

    # true graph
    matched_genes = match_genes(scored_graph, pathway_sets)
    leads = pathway_leads(matched_genes, gene_sets, min_size=10)
    adj_leads, adj_gene_sets = remove_necessary(scored_graph, leads, gene_sets, virus_nodes)
    # adj_leads, adj_gene_sets = leads, gene_sets
    pathway_finds = pathway_fills(adj_leads, adj_gene_sets)

    # repeat for random graph
    r_graph = cut_random_graph(scored_random_g, reachable_scored_graph, match_edges=False)
    print("Random graph:", r_graph)
    r_matched_genes = match_genes(r_graph, pathway_sets)
    r_leads = pathway_leads(r_matched_genes, gene_sets, min_size=10)
    r_adj_leads, r_adj_gene_sets = remove_necessary(scored_random_g, r_leads, gene_sets, virus_nodes)
    r_pathway_finds = pathway_fills(r_adj_leads, r_adj_gene_sets)

    write_pathways(pathway_finds, r_pathway_finds, name_map, "data/pathways_all")

    plot_hist(
        list(pathway_finds.values()),
        list(r_pathway_finds.values()),
        discrete=False,
        labels=("Real", "Random"),
        title="Matched gene % in pathway excluding trivial genes (min size = 10)",
    )
