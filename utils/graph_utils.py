import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pingouin as pg
import tqdm
from natsort import natsorted


def virus_host_network(node_path: str, edges_path: str, sep=",", virus_map=None, directed=False):
    nodes = pd.read_csv(node_path, sep=sep)
    edges = pd.read_csv(edges_path, sep=sep)
    if virus_map:
        v_map = pd.read_csv(virus_map, sep="\t", header=None).set_index(1)[0].to_dict()

    # drop rows where entrez is duplicated
    nodes = nodes.drop_duplicates(subset="entrez")
    nodes = nodes.set_index("entrez").to_dict("index")

    removed_nodes = set()
    if directed:
        # edges in the original file are already sorted so we get this for free
        graph = nx.DiGraph(zip(edges["entrez_A"].values, edges["entrez_B"].values))
    else:
        graph = nx.Graph(zip(edges["entrez_A"].values, edges["entrez_B"].values))

    for node in list(graph.nodes):
        if node not in nodes:  # if entrez is missing
            removed_nodes.add(node)
            graph.remove_node(node)
            continue
        name = f"{v_map.get(node, 'You Made A Mistake')} (v)" if virus_map else nodes[node]["name"]
        # we have uniprot id, name, and taxonomy id
        graph.nodes[node]["uniprot_id"] = nodes[node]["id"]
        graph.nodes[node]["name"] = name
        graph.nodes[node]["taxid"] = nodes[node]["taxonomy"]
        graph.nodes[node]["is_virus"] = nodes[node]["name"] != "human"

    if len(removed_nodes) > 0:
        print(f"Removed {len(removed_nodes)} nodes: {','.join([str(x) for x in removed_nodes][:5])}")
    return graph


def constant_nodes(graph: nx.Graph, exclusion_score=2.0, logp1=False, mean_expression_threshold=1):
    # do z score normalisation
    z_scores_per_sample = []
    boring_nodes = set()
    zero_exp = 0
    virus_zero_exp = 0
    for node, data in graph.nodes(data=True):
        if data["is_virus"]:
            continue

        expr = data["timepoints"]
        mean_expr = np.mean(expr)
        std_expr = np.std(expr)
        if std_expr == 0 or abs(mean_expr) < mean_expression_threshold:
            boring_nodes.add(node)
            zero_exp += 1
            if data["is_virus"]:
                virus_zero_exp += 1
            continue
        z_scores = [(e - mean_expr) / std_expr for e in expr]
        z_scores_per_sample.extend(z_scores)
        if all(-exclusion_score < score < exclusion_score for score in z_scores):
            boring_nodes.add(node)
        if logp1:
            graph.nodes[node]["timepoints"] = np.log1p(graph.nodes[node]["timepoints"])

    print(f"Removed {len(boring_nodes)}/{len(list(graph.nodes()))} nodes")
    print(f"{zero_exp} nodes without exp, {virus_zero_exp} are virus")
    graph.remove_nodes_from(boring_nodes)
    return graph


def compare_nodes(node_a: tuple, node_b: tuple, metric="correlation"):
    values_a = node_a["timepoints"]
    values_b = node_b["timepoints"]
    if metric == "correlation":
        return np.corrcoef(values_a, values_b)[0, 1]


def score_graph(graph):
    for u, v in graph.edges():
        if graph.nodes[u].get("is_virus", False) or graph.nodes[v].get("is_virus", False):
            graph[u][v]["weight"] = 0  # to virus is always zero
            graph[u][v]["real_weight"] = 1
            continue
        correlation = compare_nodes(graph.nodes[u], graph.nodes[v])
        graph[u][v]["weight"] = -np.log(abs(correlation))
        graph[u][v]["real_weight"] = correlation

    return graph


def protein_protein_network(ppi_path: str):
    with open(ppi_path, "rb") as f:
        ppi_graph = pickle.load(f)

    for node in ppi_graph.nodes:
        ppi_graph.nodes[node]["is_virus"] = False  # its a human ppi network
    return ppi_graph


def add_temporal_information(graph: nx.Graph, timepoint_expr: dict, attr_name=None):
    # all_non_viral_nodes = [n for n, d in graph.nodes(data=True) if not d.get("is_virus")]
    if attr_name is None:
        attr_name = "timepoints"
    timepoints = natsorted(timepoint_expr.keys())
    for node in list(graph.nodes()):
        if node not in timepoint_expr[timepoints[0]]:
            if not graph.nodes[node]["is_virus"]:
                graph.remove_node(node)
            continue
        node_values = []
        time = []
        for timepoint in timepoints:
            node_values.extend(timepoint_expr[timepoint][node])
            time.extend([int(timepoint)] * len(timepoint_expr[timepoint][node]))

        assert len(node_values) == len(time), (
            f"Node {node} has too few values. \
        Expected {len(timepoints)} got {len(node_values)}"
        )

        graph.nodes[node][attr_name] = node_values
        graph.nodes[node]["time"] = time

    return graph


def filter_by_species(graph: nx.Graph, taxid=None):
    if not taxid:
        return graph

    taxon_nodes = {n for n, d in graph.nodes(data=True) if d.get("taxid") == taxid}
    taxon_neighbours = {n for n in graph.nodes if any(m in taxon_nodes for m in graph.neighbors(n))}
    relevant_nodes = taxon_neighbours | taxon_nodes
    return graph.subgraph(relevant_nodes).copy()


def filter_by_expression(graph: nx.Graph, expression_data=None):
    if not expression_data:
        return graph

    # go through nodes, if they are virus skip, if not check if theyre in the expression data
    # if they are, assign the corresponding tpm. if it's zero remove edges from node (but do not remove!)
    all_non_viral_nodes = [n for n, d in graph.nodes(data=True) if not d.get("is_virus")]
    for node in all_non_viral_nodes:
        if node not in expression_data:
            graph.remove_node(node)
            continue
        graph.nodes[node]["expression"] = expression_data[node]

        if graph.nodes[node]["expression"] == 0:
            graph.remove_edges_from([e for e in graph.edges(node)])

    return graph


def filter_graph(graph: nx.Graph, timepoint: int | None = None, taxid=None, expression_data=None):
    f_graph = graph.copy()

    f_graph = filter_by_species(f_graph, taxid)

    f_graph = filter_by_expression(f_graph, expression_data)

    # rename graph ids to the respective time point
    if timepoint is not None:
        # rename graph ids to the
        tp_nodes = {node: f"{node}.{timepoint}" for node in f_graph.nodes}
        f_graph = nx.relabel_nodes(f_graph, mapping=tp_nodes, copy=True)

    return f_graph


def randomize_graph(graph: nx.Graph, tries=None):
    graph_randomized = graph.copy()
    n_swaps = len(graph.edges())
    return nx.double_edge_swap(graph_randomized, nswap=n_swaps, max_tries=n_swaps * 50 if tries is None else tries)


def partial_correlation_per_node(graph: nx.DiGraph, node: int, semi=False):
    neighbours = list(graph.predecessors(node))
    data = {}
    partial_corrs = {}
    if len(neighbours) == 0:
        return {}
    data[node] = graph.nodes[node]["timepoints"]
    for n in neighbours:
        # if graph.nodes[node].get("is_virus", True) or graph.nodes[n].get("is_virus", True):
        #   partial_corrs[n] = 1.0
        #    continue

        # get node data
        data[n] = graph.nodes[n]["timepoints"]

    df = pd.DataFrame(data)
    cols = list(df.columns)
    other_cols = [c for c in cols if c != node]

    for col_name in other_cols:
        covars = [c for c in other_cols if c != col_name]
        if covars and semi:
            corr = pg.partial_corr(data=df, x=node, y=col_name, x_covar=covars)
        elif covars:
            corr = pg.partial_corr(data=df, x=node, y=col_name, covar=covars)
        else:
            corr = pg.partial_corr(data=df, x=node, y=col_name)
        partial_corrs[col_name] = corr.loc["pearson", "r"]

    return partial_corrs


def rho_prop(a1: np.array, a2: np.array):
    var_a1_minus_a2 = np.var(a1 - a2)
    var_a1 = np.var(a1)
    var_a2 = np.var(a2)
    rho = 1 - ((var_a1_minus_a2) / (var_a1 + var_a2))
    return rho


def rho_per_node(graph: nx.DiGraph, node, means: list, plus1=False):
    neighbours = list(graph.predecessors(node))
    data = {}
    vlrs = {}
    if len(neighbours) == 0:
        return {}
    data[node] = graph.nodes[node]["timepoints"]
    for n in neighbours:
        if graph.nodes[n]["is_virus"]:
            continue
        data[n] = graph.nodes[n]["timepoints"]

    df = pd.DataFrame(data)
    cols = list(df.columns)
    other_cols = [c for c in cols if c != node]

    df = df.div(means, axis=0)
    df = np.log2(df + 0.1 if plus1 else df) # adding +1 here to avoid a bunch of 0 vals

    node_vals = np.array(df[node].values)
    for col in other_cols:
        vlr_score = rho_prop(node_vals, np.array(df[col].values))
        vlrs[col] = vlr_score
    return vlrs


def calculate_edge_metric(graph: nx.Graph, metric="partial", **kwargs):
    node_list = list(graph.nodes)
    desc_str = "all edges in" if not kwargs.get("desc", None) else kwargs.get("desc")
    nan_vals = 0
    for u in tqdm.tqdm(node_list, total=len(node_list), unit="nodes", desc=f"Scoring {desc_str} graph"):
        if metric == "partial":
            corrs = partial_correlation_per_node(graph, u, semi=kwargs.get("semi", False))
        elif metric == "rho_prop":
            corrs = rho_per_node(graph, u, means=kwargs.get("tp_means"), plus1=kwargs.get("plus1"))
        else:
            raise RuntimeError("No calculation selected")

        for v, c in corrs.items():
            # we're looking at predecessors so its v -> u not u -> v
            if graph[v][u].get("weight", None) is not None:  # skip if there is already a weight for this edge
                continue

            if np.isnan(c):
                c = 0 + 1e-6
                nan_vals += 1

            assert c is not None, f"Edge {v} -> {u} has score of None."
            graph[v][u]["weight"] = -np.log(abs(c))
            graph[v][u]["real_weight"] = c

    if nan_vals:
        print(f"Warning: {nan_vals} NaN values created. Replacing with 0+ε weight for safety")

    return graph


def empirical_pval(r, null_dist, alternative="two-sided"):
    n_null = len(null_dist)
    if alternative == "two-sided":
        count = np.sum(null_dist >= np.abs(r))
    elif alternative == "greater":
        count = np.sum(null_dist >= r)
    elif alternative == "less":
        count = np.sum(null_dist <= r)
    # +1 / +1 correction avoids p=0, standard for permutation tests
    return (count + 1) / (n_null + 1)


def adjusted_pvals(
    graph,
    graph_bs,
    weight_col="real_weight",
):
    edge_weights_real = []
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        if graph.nodes[u]["is_virus"] or graph.nodes[v]["is_virus"]:
            continue
        if data.get(weight_col) is None:
            raise ValueError(f"Found no real_weight attribute for edge: {u} -> {v}")
        edge_weights_real.append(data.get(weight_col))

    edge_weights_random = []
    c = 0
    for graph_b in graph_bs:
        for u, v, data in graph_b.edges(data=True):
            if u == v:
                continue
            if graph_b.nodes[u]["is_virus"] or graph_b.nodes[v]["is_virus"]:
                continue
            if data.get(weight_col) is None:
                raise ValueError(f"Found no real_weight attribute for edge: {u} -> {v}")
            if np.isnan(data[weight_col]):
                c+=1
                continue
            edge_weights_random.append(data[weight_col])

    if c > 0:
        print(f"Warning: skipped {c}/{len(edge_weights_random)} edges due to NaN values.")
    abs_edge_weight_random = np.abs(edge_weights_random)
    assert not any(np.isnan(edge_weights_random)), "Found NaN values in edge weights"
    pvals = np.array([empirical_pval(r, abs_edge_weight_random) for r in edge_weights_real])
    reject, _ = pg.multicomp(pvals, method="fdr_bh")

    # find p-value of the most extreme case that still got rejected
    if reject.any(): 
        p_cutoff = pvals[reject].max()  # largest raw p-value among rejected cases
    else:
        p_cutoff = 0.0  # nothing survived

    # translate that p-value back into a correlation cutoff using the null distribution
    # for two-sided test:
    r_cutoff = np.percentile(np.abs(edge_weights_random), 100 * (1 - p_cutoff))
    print(f"Cases with |ρ| > {r_cutoff:.3f} survive BH correction")

    return r_cutoff


def uniform_attributes(graph: nx.Graph):
    possible_attributes = {key for _, attr in graph.nodes(data=True) for key in attr}

    for attr in possible_attributes:
        for node in graph.nodes():
            if graph.nodes[node].get(attr) is not None:
                continue
            if attr == "timepoints" and graph.nodes[node]["is_virus"]:
                continue
            if attr == "kind" and graph.nodes[node]["is_virus"]:
                graph.nodes[node][attr] = "Virus"
                continue
            graph.nodes[node][attr] = "none"
    return graph


def build_random_graph(existing_graph: nx.DiGraph, randomize_edges=True, shuffle_nodes=False, shuffle_order=False):
    ex_graph = existing_graph.copy()
    r_graph = nx.DiGraph()
    r_graph.add_nodes_from(ex_graph.nodes(data=True))
    all_nodes = list(ex_graph.nodes())
    num_edges = list(ex_graph.edges())

    if randomize_edges:
        for i in range(len(num_edges)):
            node_1 = np.random.choice(all_nodes)
            node_2 = np.random.choice(all_nodes)
            while ex_graph.nodes[node_2]["is_virus"]:  # make sure virus nodes stay without predecessors
                node_2 = np.random.choice(all_nodes)
            r_graph.add_edge(node_1, node_2)
    if shuffle_nodes:
        r_graph.add_edges_from(ex_graph.edges())
        for n in all_nodes:
            tp_data = r_graph.nodes[n].get("timepoints")
            if tp_data is None:
                continue
            tp_data = tp_data.copy()
            np.random.shuffle(tp_data)
            r_graph.nodes[n]["timepoints"] = tp_data

    if shuffle_order:
        r_graph.add_edges_from(ex_graph.edges())
        for n in all_nodes:
            tp_data = r_graph.nodes[n].get("timepoints")
            if tp_data is None:
                continue
            timepoints = r_graph.nodes[n]["time"]
            unique_cats = np.unique(timepoints)
            np.random.shuffle(unique_cats)

            # Build index order by concatenating indices for each category in shuffled order
            order = np.concatenate([np.where(timepoints == cat)[0] for cat in unique_cats])

            r_graph.nodes[n]["time"] = np.array(timepoints)[order]
            r_graph.nodes[n]["timepoints"] = np.array(tp_data)[order]
    return r_graph


def cut_edges_below_cutoff(graph: nx.Graph, cutoff=0):
    graph = graph.copy()
    edges_to_remove = []
    for u, v in graph.edges():
        if u == v:
            continue
        weight = graph[u][v]["real_weight"]
        if abs(weight) < cutoff:
            edges_to_remove.append((u, v))

    print(f"Removing {len(edges_to_remove):,} edges out of {len(graph.edges()):,} total")
    graph.remove_edges_from(edges_to_remove)
    return graph
