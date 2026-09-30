import math
import os
import pickle
import sys
from copy import deepcopy

import networkx as nx
import numpy as np
import tqdm
from dotenv import load_dotenv
from zarr import full

from gene_regulatory.grn_graph import (
    adjusted_pvals,
    build_graph,
    build_random_graph,
    filter_alert_nodes,
    find_alert_nodes,
    get_reachable_graph,
    virus_path,
)
from utils.data_utils import get_timepoint_means, write_graph
from utils.graph_utils import (
    calculate_edge_metric,
    constant_nodes,
    cut_edges_below_cutoff,
)
from utils.plot_utils import plot_edge_weights, plot_hist, plot_virus_score_distributions

load_dotenv(override=True)


def paths_of_length_to_target(G, target, k, max_paths=1000):
    Grev = G.reverse(copy=False)  # cheap view, no copy of data
    results = []

    if k == 0:
        k = 2

    def dfs(node, path):
        if len(results) >= max_paths:
            return
        if len(path) == k:
            results.append(list(reversed(path)))
            return
        preds = sorted(
            Grev.successors(node),
            key=lambda pred: Grev[node][pred].get("weight", 0),
        )
        for pred in preds:  # predecessors in the original graph
            if len(results) >= max_paths:
                return
            if pred not in path:  # keep paths simple (no repeated nodes)
                path.append(pred)
                dfs(pred, path)
                path.pop()

    dfs(target, [target])
    return results

def pathways_to_target(G, target, k, allowed_nodes: set):
    Grev = G.reverse(copy=False)  # cheap view, no copy of data
    results = []

    if k == 0:
        k = 2

    def dfs(node, path):
        if len(path) == k:
            results.append(list(reversed(path)))
            return
        preds = sorted(
            Grev.successors(node),
            key=lambda pred: Grev[node][pred].get("weight", 0),
        )
        for pred in preds:  # predecessors in the original graph
            if pred not in path and pred in allowed_nodes:  # keep paths simple (no repeated nodes)
                path.append(pred)
                dfs(pred, path)
                path.pop()

    dfs(target, [target])
    return results

def compare_paths(virus_graph, human_graph, alert_node, virus_nodes):
    best_score = 0
    path_len = 2  # in case virus has nothing the smallest path will be 2
    best_path = None
    path_lens = []
    for node in virus_nodes:
        try:
            virus_path = nx.shortest_path(virus_graph, source=node, target=alert_node)
        except nx.NetworkXNoPath:
            continue
        score = 0
        path_lens.append(len(virus_path))
        for i in range(1, len(virus_path) - 1):
            score += abs(virus_graph[virus_path[i]][virus_path[i + 1]]["real_weight"])
        if score > best_score:
            best_score = score
            path_len = len(virus_path)
            best_path = virus_path

    best_human_score = 0
    best_human_path = None
    human_scores = []
    for path in paths_of_length_to_target(
        human_graph, alert_node, path_len - 1
    ):  # -1 because we don't count first virus step
        score = 0
        for i in range(len(path) - 1):
            score += abs(human_graph[path[i]][path[i + 1]]["real_weight"])
        if score > best_human_score:
            best_human_score = score
            best_human_path = path
        human_scores.append(score / (path_len - 1))

    best_score = best_score / (max(path_len - 2, 1))
    best_human_score = best_human_score / (path_len - 1)

    if best_score == 0 == best_human_score:
        return {
            "best": "exclude",
            "best_v_score": best_score,
            "best_h_score": best_human_score,
            "path_lengths": path_lens,
            "best_path": None,
            "bg_path_scores": human_scores,
        }

    return {
        "best": "virus" if best_score > best_human_score else "human",
        "best_v_score": best_score,
        "best_h_score": best_human_score,
        "path_lengths": path_lens,
        "best_path": best_path if best_score > best_human_score else best_human_path,
        "bg_path_scores": human_scores,
    }


def compare_paths_all(virus_graph, human_graph, alert_node, virus_nodes, cutoff=5):
    path_lens = []
    v_paths = []
    v_path_scores = []
    for node in virus_nodes:
        try:
            # virus_path = nx.shortest_path(virus_graph, source=node, target=alert_node)
            virus_paths = nx.all_simple_paths(virus_graph, source=node, target=alert_node, cutoff=cutoff)
        except nx.NetworkXNoPath:
            continue
        for vp in virus_paths:
            score = 0
            path_lens.append(len(vp))
            for i in range(1, len(vp) - 1):
                score += abs(virus_graph[vp[i]][vp[i + 1]]["real_weight"])
            v_paths.append(vp)
            v_path_scores.append(
                score / (max(len(vp) - 2, 1))
            )  # no reasonable score between virus and gene; one less edge than nodes

    path_len = max(path_lens) if path_lens else 2  # in case virus has nothing the smallest path will be 2
    h_paths = []
    human_scores = []
    for path in paths_of_length_to_target(
        human_graph, alert_node, path_len - 1, max_paths=(len(v_paths) + 1)
    ):  # -1 because we don't count first virus step
        score = 0
        for i in range(len(path) - 1):
            score += abs(human_graph[path[i]][path[i + 1]]["real_weight"])
        if score > 0:
            h_paths.append(path)
            human_scores.append(score / (path_len - 1))

    if len(human_scores) == 0 and not any(v_path_scores):
        return {
            "exclude": True,
            "virus_paths": v_paths,
            "virus_scores": v_path_scores,
            "path_lengths": path_lens,
            "human_paths": h_paths,
            "human_scores": human_scores,
        }

    return {
        "exclude": False,
        "virus_paths": v_paths,
        "virus_scores": v_path_scores,
        "path_lengths": path_lens,
        "human_paths": h_paths,
        "human_scores": human_scores,
    }


def write_node_scores(graph, virus_nodes):
    with open("data/covid_edge_scores_raw_counts.tsv", "w") as f:
        f.write("vir_src\thum_dest\tedge_weight\n")
        for vn in virus_nodes:
            succ = graph.successors(vn)
            f.writelines(f"{vn}\t{s}\t{graph[vn][s]['real_weight']}\n" for s in succ)


def override_virus_scores(graph, virus_nodes):
    # override virus scores to one since measurement is unreliable or something
    for node in virus_nodes:
        for succ in graph.successors(node):
            graph[node][succ]["weight"] = 1
            graph[node][succ]["real_weight"] = 1
    return graph


def isolate_path(graph: nx.DiGraph, path: list) -> nx.DiGraph:
    preds = [p for node in path for p in graph.predecessors(node)]
    rel_nodes = set(preds) | set(path)
    sub_g: nx.DiGraph = nx.subgraph(graph, rel_nodes).copy()
    sub_g.remove_edges_from(nx.selfloop_edges(sub_g))  # remove self loops
    sub_g.remove_edges_from(
        [(path[-1], v) for v in sub_g.successors(path[-1])]
    )  # remove outgoing edges of the final node
    sub_g.remove_edges_from(
        [(u, v) for u, v in sub_g.edges() if v not in path]
    )  # remove other outgoing edges from path
    return sub_g


def up_or_down(x: list, y: list):
    slope, _ = np.polyfit(x, y, 1)
    return slope


def setup_graph(random_repeat=5, plus1=False, plot_weights=False, save_graph=None):
    full_graph, data_matrix = build_graph(return_counts=False)
    filter_graph = constant_nodes(full_graph, exclusion_score=0.0, logp1=False, mean_expression_threshold=1)
    virus_nodes = [n for n, d in filter_graph.nodes(data=True) if d["is_virus"]]
    reachable_graph = get_reachable_graph(filter_graph.copy(), virus_nodes)
    timepoint_means = get_timepoint_means(data_matrix, virus_nodes)

    scored_graph = calculate_edge_metric(
        deepcopy(reachable_graph), metric="rho_prop", tp_means=timepoint_means, desc="viral", plus1=plus1
    )
    scored_graph = override_virus_scores(scored_graph, virus_nodes)

    # get graph with no viral elements
    ho_graph = nx.subgraph(filter_graph, set(filter_graph.nodes()) - set(virus_nodes))
    ho_scored_graph = calculate_edge_metric(
        ho_graph.copy(), metric="rho_prop", tp_means=timepoint_means, desc="human only", plus1=plus1
    )

    random_graphs = []
    for i in range(random_repeat):
        random_g = build_random_graph(scored_graph, randomize_edges=False, shuffle_nodes=True, shuffle_order=False)
        scored_random_g = calculate_edge_metric(
            random_g.copy(), metric="rho_prop", tp_means=timepoint_means, desc=f"random {i}", plus1=plus1,
        )
        random_graphs.append(scored_random_g)

    cutoff = adjusted_pvals(scored_graph, random_graphs)

    if plot_weights:
        plot_edge_weights(scored_graph, scored_random_g, cutoff)

    if save_graph:
        pickle.dump(scored_graph.copy(), open(f"data/{save_graph}_{cutoff}.pkl", "wb"))

    scored_graph = cut_edges_below_cutoff(scored_graph, cutoff)

    if sum([scored_graph.degree(vn) for vn in virus_nodes]) == 0:
        print("Virus nodes are not connected to the host graph. Please check")
        sys.exit()

    return filter_graph, scored_graph, ho_scored_graph, virus_nodes


def score_paths(filter_graph, scored_graph, ho_scored_graph, virus_nodes, plot=False, leaf_only=True):
    # get alert nodes
    scored_nodes = find_alert_nodes(filter_graph, leaf_only=leaf_only, fold_change=False)
    alert_nodes, _, _ = filter_alert_nodes(scored_nodes)
    in_graph_alert = [n for n in alert_nodes if n in scored_graph and not scored_graph.nodes[n]["is_virus"]]

    better_explained = {"human": 0, "virus": 0, "exclude": 0}
    path_lens = []
    bg_path_scores = []
    v_scores = []
    nodes = []
    all_alert_node_info = []
    for node in in_graph_alert:
        out = compare_paths(scored_graph, ho_scored_graph, node, virus_nodes)
        path_lens.extend(out["path_lengths"])
        if out["best"] != "exclude":
            print(
                f"Better: {out['best']}, Score virus {out['best_v_score']:.2f}, Score human {out['best_h_score']:.2f}"
                f": {' -> '.join([str(x) for x in out['best_path']])}"
            )
            bg_path_scores.append(out["bg_path_scores"])
            v_scores.append(out["best_v_score"])
            nodes.append(scored_graph.nodes[node]["name"])
            iso_graph_base = scored_graph if out["best"] == "virus" else ho_scored_graph
            alert_node_g = isolate_path(iso_graph_base, out["best_path"])
            all_alert_node_info.append(
                {"node_name": scored_graph.nodes[node]["name"], "alert_graph": alert_node_g, **out}
            )
        better_explained[out["best"]] += 1

    if plot:
        plot_virus_score_distributions(bg_path_scores, v_scores, nodes)
        plot_hist(path_lens, title="Path lengths to alert nodes")
    print(better_explained)
    return all_alert_node_info


def score_paths_all(
    filter_graph: nx.DiGraph,
    scored_graph: nx.DiGraph,
    ho_scored_graph: nx.DiGraph,
    virus_nodes: list,
    leaf_only=True,
    paths_only=False,
    max_path_len=5,
):
    scored_nodes = find_alert_nodes(filter_graph, leaf_only=leaf_only, fold_change=False)
    alert_nodes, _, _ = filter_alert_nodes(scored_nodes)
    in_graph_alert = [n for n in alert_nodes if n in scored_graph and not scored_graph.nodes[n]["is_virus"]]

    all_alert_node_info = []
    for node in tqdm.tqdm(in_graph_alert, desc="Scoring paths to alert"):
        out = compare_paths_all(scored_graph, ho_scored_graph, node, virus_nodes, cutoff=max_path_len)
        if out["exclude"]:
            continue

        partial_h_graphs = []
        h_nodes = set()
        for path in out["human_paths"]:
            partial_h_graphs.append(isolate_path(ho_scored_graph, path))
            h_nodes.update(path)

        partial_v_graphs = []
        v_nodes = set()
        for path in out["virus_paths"]:
            partial_v_graphs.append(isolate_path(scored_graph, path))
            v_nodes.update(path)

        h_alert_node_g = nx.compose_all(partial_h_graphs) if partial_h_graphs else None
        v_alert_node_g = nx.compose_all(partial_v_graphs) if partial_v_graphs else None

        if paths_only:
            h_alert_node_g = nx.subgraph(h_alert_node_g, h_nodes) if h_alert_node_g is not None else None
            v_alert_node_g = nx.subgraph(v_alert_node_g, v_nodes) if v_alert_node_g is not None else None

        all_alert_node_info.append(
            {
                "node_name": scored_graph.nodes[node]["name"],
                "node_id": node,
                "h_alert_graph": h_alert_node_g,
                "v_alert_graph": v_alert_node_g,
                **out,
            }
        )

    return all_alert_node_info


def possible_pathways(graph: nx.DiGraph, pathway_list: dict, alert_nodes: list, fixed=False):
    possible_pathways = {}
    for pathway, genes in tqdm.tqdm(pathway_list.items()):
        allowed_nodes = {int(gene) for gene in genes if gene.isdigit()}
        for alert_node in alert_nodes:
            if str(alert_node) not in genes:
                continue
            all_paths = pathways_to_target(graph, alert_node, 4, allowed_nodes=allowed_nodes)
            if pathway not in possible_pathways:
                possible_pathways[pathway] = 0
            possible_pathways[pathway] += len(all_paths)
            if fixed:
                continue
            all_paths_3 = pathways_to_target(graph, alert_node, 3, allowed_nodes=allowed_nodes)
            possible_pathways[pathway] += len(all_paths_3)

    possible_pathways = {k:v for k, v in possible_pathways.items() if v > 0}
    return possible_pathways

if __name__ == "__main__":
    filter_graph, scored_graph, ho_scored_graph, virus_nodes = setup_graph(save_graph="covid_big_shuf_scored")
    all_info = score_paths_all(
        filter_graph, scored_graph, ho_scored_graph, virus_nodes, paths_only=True, max_path_len=4
    )