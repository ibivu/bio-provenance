import math
import os
import pickle
from copy import deepcopy

import networkx as nx
import numpy as np
import statsmodels.api as sm
import tqdm
from dotenv import load_dotenv
from kneed import KneeLocator

from utils.data_utils import (
    batch_correction,
    format_tpm_counts,
    get_timepoint_means,
    load_timepoint_dir,
    write_graph,
)
from utils.graph_utils import (
    add_temporal_information,
    adjusted_pvals,
    build_random_graph,
    calculate_edge_metric,
    constant_nodes,
    score_graph,
    uniform_attributes,
    virus_host_network,
)
from utils.plot_utils import (
    plot_alert_nodes,
    plot_edge_distribution,
    plot_edge_weights,
    plot_hist,
    plot_path,
    plot_timepoints,
    plot_timepoints_line,
)

load_dotenv(override=True)


def grn_network(path):
    nodes_path = path + ".nodes.tsv"
    edges_path = path + ".edges.tsv"

    graph = nx.DiGraph()
    columns = []
    is_first = True
    with open(nodes_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            if is_first:
                columns.extend(line)
                is_first = False
                continue
            # graph.add_node(int(line[0]), **{columns[1]: line[1], columns[2]: line[2], columns[3]: line[3]})
            graph.add_node(int(line[0]), name=line[1], is_virus=False, kind=line[3])

    with open(edges_path, "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            if len(line) > 2:
                graph.add_edge(int(line[0]), int(line[1]), weight=float(line[2]), real_weight=float(line[3]))
            else:
                graph.add_edge(int(line[0]), int(line[1]))
    return graph


def combine_graphs(vhi_graph: nx.Graph, grn_graph: nx.DiGraph, taxid: int):
    vh_graph = nx.compose(vhi_graph, grn_graph)

    if taxid is not None:
        irrelevant_viral_node = {
            n
            for n in vh_graph.nodes()
            if vh_graph.nodes[n]["is_virus"] and vh_graph.nodes[n].get("taxid", taxid) != taxid
        }

        vh_graph.remove_nodes_from(irrelevant_viral_node)
    return vh_graph


def build_graph(replicates="all", return_counts=False):
    # prepare graphs
    vh_graph = virus_host_network(
        os.getenv("VIRUS_NODE_PATH"), os.getenv("VIRUS_EDGE_PATH"), virus_map=os.getenv("VIRUS_MAP"), directed=True
    )
    grn_graph = grn_network(path=os.getenv("GRN_PATH"))
    c_graph = combine_graphs(vh_graph, grn_graph, int(os.getenv("TAXID")))

    if not os.path.isdir(os.getenv("EXPRESSION_DATA")): # if its a singular count file
        bc_timepoints, tpm_matrix = format_tpm_counts(
            os.getenv("EXPRESSION_DATA"), os.getenv("PROTEIN_MAP"), os.getenv("VIRUS_MAP"), replicates=replicates
        )
    else:
        timepoints = load_timepoint_dir(os.getenv("EXPRESSION_DATA"), cleaned_only=True)

        # NO BATCH CORRECTION HAPPENING ANYMORE, JUST TOO LAZY TO CHANGE FUNCTION NAME
        bc_timepoints, tpm_matrix = batch_correction(
            timepoints,
            os.getenv("PROTEIN_MAP"),
            os.getenv("VIRUS_MAP"),
            max_timepoints=7,
            save=True,
            replicates_to_keep=replicates,
            return_counts=return_counts,
        )

    exp_tp = {x.replace("tp", ""): {} for x in bc_timepoints} # get the number after tp
    for point in bc_timepoints:
        # bc_timepoints = {tp1: {geneid1: [1, 2, 3], geneid2: [1,2,3] ...}, tp2: {...}}
        for gene in bc_timepoints[point]:
            exp_tp[point.replace("tp", "")][gene] = bc_timepoints[point][gene]

    full_graph = add_temporal_information(c_graph, exp_tp)

    return full_graph, tpm_matrix


def find_best_path(graph, alert, taxid):
    possible_dest = [n for n, d in graph.nodes(data=True) if d.get("taxid", 0) == int(taxid)]

    paths = []
    for dest in possible_dest:
        irrelevant_nodes = {x for x in possible_dest if x != dest}
        relevant_nodes = {x for x in graph.nodes() if x not in irrelevant_nodes}
        reachable_graph = graph.subgraph(relevant_nodes).copy()

        try:
            path = nx.dijkstra_path(reachable_graph, source=dest, target=alert)
        except nx.NetworkXNoPath:
            continue

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            graph[u][v]["color"] = "blue"

        weights = [reachable_graph[path[i]][path[i + 1]].get("real_weight", 1) for i in range(len(path) - 1)]

        degrees = [reachable_graph.degree(node) for node in path]

        result = {"path": path, "weights": weights, "degrees": degrees}
        paths.append(result)

    return paths


def calculate_best_path(paths, selected_node=None, least_corr_score=False):
    best_path = None
    best_score = 0
    for path in paths:
        path_len = len(path["path"])
        path_weight = abs(sum(path["weights"]))
        if least_corr_score:
            path_score = min(path_weight)
        else:
            path_score = path_weight / path_len
        if selected_node is not None and selected_node in path["path"]:
            best_path = path
            best_score = path_score
            break
        if best_score < path_score:
            best_path = path
            best_score = path_score
    return best_path, best_score


def find_alert_nodes(graph: nx.DiGraph, leaf_only=False, fold_change=False):
    potentials = {}
    for node, data in graph.nodes(data=True):
        if data["is_virus"]:
            continue
        if leaf_only and len(list(graph.successors(node))) > 1:
            continue
        tp = data["timepoints"]
        rep_info = data["time"]
        tp_avg = []
        rep_values = []
        prev_i = -1
        for idx, i in enumerate(rep_info):
            if i == prev_i:
                rep_values.append(tp[idx])
            else:
                if len(rep_values) > 0:
                    tp_avg.append(np.mean(rep_values))  # TODO: FIX: I dont think this is what you want to do
                rep_values = [tp[idx]]
                prev_i = i
        tp_avg.append(np.mean(rep_values))

        # get largest diff (abs(end-start))
        if fold_change:
            tp_avg = [math.log2(x + 1 / (tp_avg[0] + 1)) for x in tp_avg[1:]]
            if max(tp_avg) > abs(min(tp_avg)):
                diff = max(tp_avg)
            else:
                diff = min(tp_avg)
        else:
            diff = abs(max(tp_avg) - min(tp_avg))
        potentials[node] = diff
    return dict(sorted(potentials.items(), key=lambda x: x[1], reverse=True))


def select_alert_node(alert_nodes, random=True):
    if random:
        shuffle = list(alert_nodes.keys())
        np.random.shuffle(shuffle)
        return shuffle[0]
    return next(iter(alert_nodes.keys()))


def virus_path(graph, timepoints=False, plot=False, write=True):
    paths = find_best_path(scored_graph, starter_node, os.getenv("TAXID"))

    best_path, best_score = calculate_best_path(paths)
    print(f"Best path has score {best_score}")
    if best_score == 0:
        return
    for i in range(len(best_path["path"]) - 1):
        print(
            f"{best_path['path'][i]} ({best_path['degrees'][i]}) - ({best_path['weights'][i]:.2f}) -"
            f"> {best_path['path'][i + 1]} ({best_path['degrees'][i + 1]})"
        )

    if plot:
        plot_path(graph, best_path)

    for i in range(len(best_path["path"])):
        if graph.nodes[best_path["path"][i]]["is_virus"]:
            continue

        print(best_path["path"][i], end=": ")
        print(
            " ".join([f"{x:.2f}" for i, x in enumerate(graph.nodes[best_path["path"][i]]["timepoints"]) if i % 2 == 0])
        )

    subgraph_nodes = set()
    for node in best_path["path"]:
        subgraph_nodes.add(node)
        if scored_graph.nodes[node].get("is_virus"):
            continue
        else:
            neighbors = scored_graph.neighbors(node)
            subgraph_nodes.update(set(neighbors))

    best_path_graph = nx.subgraph(scored_graph, subgraph_nodes)
    for u, v in best_path_graph.edges():
        if u == v:
            continue
        if best_path_graph[u][v].get("weight", 0) < 0.5:
            best_path_graph[u][v]["weight"] = 0.001
    if write:
        write_graph(best_path_graph, "graphs/grn_best_short", nodes=True)


def avg_paths_to_sinks(G, starts):
    sinks = {n for n in G.nodes if G.out_degree(n) == 0}
    counts = []
    for s in starts:
        for sink in sinks:
            num_paths = len([x for x in nx.all_simple_paths(G, s, sink)])
            if num_paths > 0:
                counts.append(num_paths)
    plot_hist(counts, title="Number of paths from virus to a sink node")
    return counts


def get_reachable_graph(graph: nx.DiGraph, virus_nodes):
    reachable = set()
    for v in virus_nodes:
        reachable.add(v)
        reachable.update(nx.descendants(graph, v))

    sub_g = nx.subgraph(graph, reachable)
    return sub_g


def virus_central_attack(graph: nx.DiGraph, virus_nodes):
    affected = {}
    for node in virus_nodes:
        succ = graph.successors(node)
        for s in succ:
            if s not in affected:
                affected[s] = 0
            affected[s] += 1

    potential_central = {n: c_affected for n, c_affected in affected.items() if c_affected > 1}
    return dict(sorted(potential_central.items(), key=lambda x: x[1], reverse=True))


def filter_alert_nodes(alert_nodes):
    # find good knees
    alert_node_val = list(alert_nodes.values())
    nodes_x = np.arange(len(alert_node_val))
    kl_low = KneeLocator(nodes_x, alert_node_val, curve="convex", direction="increasing")
    kl_high = KneeLocator(nodes_x, alert_node_val, curve="convex", direction="decreasing")

    important_alert = {}
    excluded = 0
    for k, v in alert_nodes.items():
        if v > kl_high.knee_y or v < kl_low.knee_y:
            important_alert[k] = v
        else:
            excluded += 1
    return important_alert, kl_high, kl_low


def explained_behavior_node(graph: nx.DiGraph, node: int):
    pred = graph.predecessors(node)

    node_vals = graph.nodes[node]["timepoints"]
    predictors = []
    for n in pred:
        if graph.nodes[n]["is_virus"]:
            continue
        predictors.append(graph.nodes[n]["timepoints"])

    predictors = np.array(predictors).T
    if len(predictors) == 0:
        return None, False

    biggest_r2 = 0
    if predictors.shape[1] > 1:
        for i in range(predictors.shape[1]):
            predictor = predictors[:, i]
            X = sm.add_constant(predictor)
            model = sm.OLS(node_vals, X).fit()
            biggest_r2 = max(biggest_r2, model.rsquared)

    X = sm.add_constant(predictors)
    model = sm.OLS(node_vals, X).fit()
    # model.summary()
    biggest_r2 = max(biggest_r2, model.rsquared_adj)
    single = biggest_r2 != model.rsquared_adj
    return biggest_r2, single


def downstream_activity(graph: nx.DiGraph, alert_nodes: list, scored_nodes: dict):
    downstream_activity = []
    downstream_nodes = set()
    for alert in alert_nodes:
        if not graph.successors(alert):
            continue
        succ = graph.successors(alert)
        for s in succ:
            if not graph.has_edge(alert, s):  # only direct successors
                continue
            downstream_nodes.add(s)
            score = scored_nodes[s]
            downstream_activity.append(score)

    all_activity = []
    for node in graph.nodes():
        if graph.nodes[node]["is_virus"]:
            continue
        if node not in alert_nodes and node not in downstream_nodes:
            all_activity.append(scored_nodes[node])

    plot_hist(
        downstream_activity,
        all_activity,
        title="Downstream Activity Score (Alert nodes v non)",
        discrete=False,
        labels=("alert", "non"),
    )


def score_node_activity(filter_graph, scored_graph, alert_nodes, scored_nodes):
    # check how many of those nodes are in the scored_graph
    in_scored_g = 0
    for n in alert_nodes:
        if n in scored_graph:
            scored_graph.nodes[n]["is_alert"] = "Yes"
            in_scored_g += 1
    for n, d in scored_graph.nodes(data=True):
        if not d.get("is_alert", False):
            d["is_alert"] = "No"
    print(f"{in_scored_g} out of {len(scored_graph)} nodes in reachable graph are alert")

    for n in scored_graph.nodes():
        if not scored_graph.nodes[n].get("kind", False):
            scored_graph.nodes[n]["kind"] = "Virus" if scored_graph.nodes[n]["is_virus"] else "Unknown"
        if n in scored_nodes:
            scored_graph.nodes[n]["activity_score"] = scored_nodes[n]
        else:
            scored_graph.nodes[n]["activity_score"] = 0
    return filter_graph, scored_graph


def compare_scores(graph1: nx.DiGraph, graph2):
    diffs = []
    for u, v, d in graph1.edges(data=True):
        if graph1.nodes[u]["is_virus"] or graph1.nodes[v]["is_virus"]:
            continue
        pred = graph1.predecessors(v)  # if u is the only predecessor of v, the correlation will remain and diff is 0
        if len(list(pred)) == 1:  # must be u
            continue
        score = d.get("real_weight")
        score2 = graph2[u][v].get("real_weight")
        score_diff = abs(score - score2)
        if score_diff > 0.7:
            print(f"Edge: {u} -> {v} spcorr: {score2:.2f}  pcorr: {score:.2f}")
        diffs.append(score_diff)
    return diffs


def alert_paths(graph, alert_nodes, virus_nodes):
    path_lengths = []
    for node in tqdm.tqdm(alert_nodes):
        if graph.nodes[node]["is_virus"]:
            continue
        for viral_node in virus_nodes:
            if nx.has_path(graph, source=viral_node, target=node):
                all_paths = list(nx.all_simple_paths(graph, source=viral_node, target=node, cutoff=4))
                path_lengths.extend([len(x) for x in all_paths])
    return path_lengths


def non_viral_path(full_graph, viral_nodes, alert_node, path_len=0):
    nv_graph = nx.subgraph(full_graph, [n for n in full_graph.nodes if n not in viral_nodes])

    possible_sources = nx.single_source_shortest_path_length(nv_graph.reverse(), alert_node, cutoff=path_len)
    for src in possible_sources:
        all_paths = list(nx.all_simple_paths(nv_graph, target=alert_node, source=""))


def build_non_viral_graph(filter_graph, scored_graph, timepoint_means=None):
    inclusion_nodes = set(filter_graph.nodes()) - set(scored_graph.nodes())
    non_viral_graph = nx.subgraph(filter_graph, inclusion_nodes)
    scored_nv_graph = calculate_edge_metric(non_viral_graph.copy(), metric="rho_prop", tp_means=timepoint_means)
    return scored_nv_graph


if __name__ == "__main__":
    full_graph, data_matrix = build_graph(return_counts=True)
    filter_graph = constant_nodes(full_graph, exclusion_score=0.0, logp1=False, mean_expression_threshold=3)
    virus_nodes = [n for n, d in filter_graph.nodes(data=True) if d["is_virus"]]
    reachable_graph = get_reachable_graph(filter_graph.copy(), virus_nodes)
    timepoint_means = get_timepoint_means(data_matrix, virus_nodes)
    scored_graph = calculate_edge_metric(deepcopy(reachable_graph), metric="rho_prop", tp_means=timepoint_means)
    # pickle.dump(scored_graph, open("data/scored_grn_graph.pkl", "wb"))

    # plot_timepoints(scored_graph, exclude_direct=False, relative=True)
    # plot_hist(diffs, title="Abs difference pcorr vs spcorr", discrete=False)

    random_graphs = []
    for i in range(5):
        random_g = build_random_graph(scored_graph, randomize_edges=False, shuffle_nodes=False, shuffle_order=True)
        scored_random_g = calculate_edge_metric(random_g.copy(), metric="rho_prop", tp_means=timepoint_means, desc="random")
        random_graphs.append(scored_random_g)

    # random_graphs = [build_non_viral_graph(filter_graph, scored_graph, timepoint_means)]

    cutoff = adjusted_pvals(scored_graph, random_graphs)

    # write_graph(scored_graph, "graphs/rho_prop_scored_grn", nodes=True)

    # finding alert nodes
    # avg_paths_to_sinks(scored_graph, virus_nodes)
    scored_nodes = find_alert_nodes(filter_graph, leaf_only=False, fold_change=False)
    alert_nodes, k1, k2 = filter_alert_nodes(scored_nodes)
    in_graph_alert = [n for n in alert_nodes if n in scored_graph and not scored_graph.nodes[n]["is_virus"]]
    # plot_alert_nodes(scored_nodes, knees=[k1.knee, k2.knee])
    filter_graph, scored_graph = score_node_activity(filter_graph, scored_graph, alert_nodes, scored_nodes)

    # path lengths to alert nodes
    # path_lengths = alert_paths(scored_graph, in_graph_alert, virus_nodes)
    # plot_hist(path_lengths, title='Path lengths to alert nodes')

    # downstream_activity(scored_graph, virus_nodes, scored_nodes)

    starter_node = -1
    while starter_node not in scored_graph:
        starter_node = select_alert_node(alert_nodes, random=True)

    # plot_hist(explanations, explanations_random, title="Explanation from pred vs null", discrete=False, labels=("Real", "Random"))

    # top_100 = [k for i, (k, v) in enumerate(alert_nodes.items()) if i < 100]
    # in_final = [n for n in top_100 if n in scored_graph]
    # plot_timepoints(nx.subgraph(filter_graph, top_100), in_final, exclude_direct=False, relative=True)

    central = virus_central_attack(scored_graph, virus_nodes)
    for i, (k, v) in enumerate(central.items()):
        print(f"{k}: {v}")
        if i >= 5:
            break

    virus_distances = {}
    for v in virus_nodes:
        try:
            virus_distances[v] = nx.shortest_path_length(scored_graph, source=v, target=starter_node)
        except nx.NetworkXNoPath:
            # print(f"No path found for {v}")
            continue
    if len(virus_distances.keys()) > 0:
        print(f"Distance from starter node to virus node: {max(virus_distances.values())} steps")
        virus_path(scored_graph, plot=False, write=False)

    plot_edge_weights(scored_graph, random_graphs[0], cutoff, fig_name="grn_random")
    # plot_edge_distribution(scored_graph)
