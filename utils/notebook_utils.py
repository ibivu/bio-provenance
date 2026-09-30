import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# PATH SETUP STUFF


def get_node_by_name(graph, name):
    matches = [n for n, d in graph.nodes(data=True) if d.get("name") == name]
    if len(matches) == 0:
        raise KeyError(f"Name {name} not found")
    if len(matches) > 1:
        print("Warning: more than one match for name, picking first one.")
    return matches[0]


def nodes_per_path(node_info, scored_graph, paths="virus"):
    total_alert_paths = 0
    passed_genes = {}
    score_per_gene = {}
    for i, path in enumerate(node_info[f"{paths}_paths"]):
        total_alert_paths += 1
        for node in path:
            name = scored_graph.nodes[node]["name"]
            if name not in passed_genes:
                passed_genes[name] = 0
                score_per_gene[name] = []
            passed_genes[name] += 1
            score_per_gene[name].append(node_info[f"{paths}_scores"][i])
    return total_alert_paths, passed_genes, score_per_gene


def alert_genes_to_df(alert_genes, alert_gene_scores, alert_paths):
    rows = []
    for i, (n, p) in enumerate(alert_genes.items()):
        rows.append(
            {
                "Gene": n,
                "n_paths": p,
                "pct_included": (p / alert_paths) * 100,
                "avg_score": np.mean(alert_gene_scores[n]),
            }
        )
    return pd.DataFrame(rows, columns=["Gene", "n_paths", "pct_included", "avg_score"])


# SANKEY STUFF


def sankey_convert(node_info, scored_graph, paths="virus", top_n=100):
    node_pairs = {}
    top_n_indices = sorted(range(len(node_info[f"{paths}_scores"])), 
                           key=lambda i: node_info[f"{paths}_scores"][i], reverse=True)[:top_n]
    top_n_indices = set(top_n_indices)
    for i, path in enumerate(node_info[f"{paths}_paths"]):
        if i not in top_n_indices:
            continue
        for j in range(len(path) - 1):
            node1 = path[j]
            node2 = path[j + 1]
            name1 = scored_graph.nodes[node1]["name"]
            name2 = scored_graph.nodes[node2]["name"]
            if (name1, name2) not in node_pairs:
                node_pairs[(name1, name2)] = 0
            node_pairs[(name1, name2)] += 1

    # for pair, freq in node_pairs.items():
    #    print(f"{pair[0]} [{freq}] {pair[1]}")
    return node_pairs


def hex_to_rgba(hex_color, alpha=0.5):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_sankey(node_pairs, all_info, selected_alert):
    # Build a unique list of node labels
    labels = sorted({n for pair in node_pairs for n in pair})
    label_to_idx = {label: i for i, label in enumerate(labels)}

    sources = [label_to_idx[pair[0]] for pair in node_pairs]
    targets = [label_to_idx[pair[1]] for pair in node_pairs]
    values = list(node_pairs.values())

    palette = px.colors.qualitative.Plotly
    node_colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
    link_colors = [hex_to_rgba(node_colors[pair[0]], alpha=0.4) for pair in node_pairs]  # color by source

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 5,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                    "label": labels,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": link_colors,
                },
                arrangement="freeform",
            )
        ]
    )

    fig.update_layout(
        title_text=f"Viral Paths Flowchart to {all_info[selected_alert]['node_name']}", font_size=12, height=500
    )
    fig.show()


# ALL PATH STUFF


def random_paths(graph: nx.DiGraph, tries=1000, add_pseudo=False):
    all_nodes = list(graph.nodes())
    node_appearance = {n: [] for n in all_nodes}

    succ_tries = 0
    while succ_tries < tries:
        start = np.random.choice(all_nodes)
        end = np.random.choice(all_nodes)
        if start == end:
            continue
        try:
            path = nx.shortest_path(graph, start, end)
        except nx.NetworkXNoPath:
            continue

        succ_tries += 1
        for n in all_nodes:
            if n in path:
                node_appearance[n].append(1)
            else:
                node_appearance[n].append(0)
    if add_pseudo:
        for n in all_nodes:
            node_appearance[n].extend([0, 1])  # add a pseudocount such that we can have p = k+1 / N + 2
    return node_appearance


def z_score(obs, mean, std):
    if obs is None:
        return 0
    return (obs - mean) / std


def merge_sum(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value
        elif isinstance(value, list):
            target[key] = target.get(key, [])
            target[key].extend(value)
    return target


# PATHWAY MATCHING


def is_sublist(a, b):
    """True if list a appears as a contiguous run inside list b."""
    n, m = len(a), len(b)
    if n > m:
        return False
    if n == 0:
        return True
    for i in range(m - n + 1):
        if b[i : i + n] == a:
            return True
    return False


def match_genes_to_pathways(path_set: set|tuple, gene_set: set, all_but=0):
    if all_but == 0:
        return all(str(p) in gene_set for p in path_set)
    if len(path_set) - all_but < 2:
        return False
    but = 0
    for p in path_set:
        if str(p) not in gene_set:
            but += 1
        if but > all_but:
            return False
    return True


def find_pathways(
    selected_alert, all_info, gene_sets, pathway_names, source="virus", print_out=False, longest_path=False, all_but=0
):
    if longest_path:
        path_list = [path[1:] if source == "virus" else path for path in all_info[selected_alert][f"{source}_paths"]]
        path_list = sorted(path_list, key=len, reverse=True)
        kept = []
        for lst in path_list:
            if not any(is_sublist(lst, bigger) for bigger in kept):
                kept.append(lst)

        unique_paths = {frozenset(path) for path in kept}
    else:
        unique_paths = {
            tuple(path[1:] if source == "virus" else path) for path in all_info[selected_alert][f"{source}_paths"]
        }

    v_path_sets = {}
    nt_paths = 0  # non-trivial paths
    hit = 0
    paths = {}
    for i, path in enumerate(unique_paths):
        path_set = path
        if len(path_set) <= 2:  # we don't care about trivial cases where virus just connects to alert
            continue
        nt_paths += 1
        hit_stop = False
        for pw in gene_sets:
            if match_genes_to_pathways(path_set, gene_sets[pw], all_but):
                if not hit_stop:
                    hit += 1
                    hit_stop = True
                if i not in v_path_sets:
                    v_path_sets[i] = []
                v_path_sets[i].append(pw)
                if pw not in paths:
                    paths[pw] = []
                paths[pw].append(path)

    if print_out:
        for i, v in v_path_sets.items():
            print(i, [pathway_names[x] for x in v], v)

        if len(v_path_sets) > 0:
            print(f"{len(v_path_sets) / nt_paths * 100:.2f}% of paths found in a pathway")
    return v_path_sets, nt_paths, hit, paths


def plot_path_weights(G1, G2, path, labels=("Small", "Large"), pathway="Pathway",
                       weight_attr="real_weight", ax=None, band_value=None):

    edges = list(zip(path[:-1], path[1:]))
    if not edges:
        raise ValueError("Path must contain at least one edge.")
 
    node_labels = [G1.nodes[n]["name"] for n in path]
    x = list(range(len(path)))  # one x position per node
 
    w1 = [G1.edges[u, v][weight_attr] for u, v in edges]
    w2 = [G2.edges[u, v][weight_attr] for u, v in edges]
 
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, len(path) * 1.5), 5))
 
    cmap = plt.get_cmap("tab10")
    color1, color2 = cmap(0), cmap(1)

    # Shaded background band spanning the full x-axis, bounded by +/- band_value
    if band_value is not None:
        if isinstance(band_value, tuple):
            ax.axhspan(-band_value[0], band_value[0], xmin=0, xmax=1,
                   color="lightblue", alpha=0.15, zorder=0, label="Not significant (small)")
            ax.axhspan(-band_value[1], band_value[1], xmin=0, xmax=1,
                   color="lightsalmon", alpha=0.15, zorder=0, label="Not significant (large)")
        else:
            ax.axhspan(-band_value, band_value, xmin=0, xmax=1,
                   color="grey", alpha=0.15, zorder=0, label="Not significant")
 
    # Each edge is its own line segment between two consecutive node x-positions,
    # at height = that edge's weight. Segments for the same graph share a color/label.
    for i, (u, v) in enumerate(edges):
        xi = [x[i], x[i + 1]]
        ax.plot(xi, [w1[i], w1[i]], marker="o", color=color1, linewidth=2, markersize=7,
                label=labels[0] if i == 0 else None, zorder=3)
        ax.plot(xi, [w2[i], w2[i]], marker="o", color=color2, linewidth=2, markersize=7,
                label=labels[1] if i == 0 else None, zorder=3)

        if w1[i] > w2[i]:
            y_p1 = 10 
            y_p2 = -10
        else:
            y_p1 = -10
            y_p2 = 10
 
        mid = (x[i] + x[i + 1]) / 2
        ax.annotate(f"{w1[i]:.2f}", (mid, w1[i]),
                    textcoords="offset points", xytext=(0, y_p1), ha="center", va="center", color=color1)
        ax.annotate(f"{w2[i]:.2f}", (mid, w2[i]),
                    textcoords="offset points", xytext=(0, y_p2), ha="center", va="center", color=color2)

 
    ax.set_xticks(x)
    ax.set_xticklabels(node_labels)
    ax.set_xlim(-0.5, len(path) - 0.5)
    ax.set_ylim(0, 1) if all(x > 0 for x in w1) and all(x > 0 for x in w2) else ax.set_ylim(-1, 1)
    ax.set_xlabel("Edge")
    ax.set_ylabel("Edge weight")
    ax.set_title(f"Edge weights along path: {pathway}")
    ax.legend(title="Dataset", loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
 
    return ax

def order_path_by_distance_from_end(nodes, graph, end):
    node_set = set(nodes)

    def backtrack(current, ordered):
        if len(ordered) == len(nodes):
            return ordered

        candidates = [
            p for p in graph.predecessors(current)
            if p in node_set and p not in ordered
        ]

        for candidate in candidates:
            result = backtrack(candidate, ordered + [candidate])
            if result is not None:
                return result

        # no candidate worked from here — signal caller to backtrack
        return None

    result = backtrack(end, [end])

    if result is None:
        raise ValueError(
            f"Could not find a valid ordering of {nodes} ending at '{end}'"
        )

    return list(reversed(result))


def get_path_score(all_info, path):
    alert_gene_id = path[-1]

    for ag in all_info:
        if ag["node_id"] != alert_gene_id:
            continue
        all_paths = [path[1:] for path in ag["virus_paths"]]
        idx = all_paths.index(list(path))
        score = ag["virus_scores"][idx]
        return score


def get_all_scores(all_info, src="virus"):
    scores = []
    for ag in all_info:
        scores.extend(ag[f"{src}_scores"])
    return scores


def find_fdr_cutoff(null_scores, target_scores, fdr=0.05,
                    scale=None):

    null_scores = np.asarray(null_scores, dtype=float)
    target_scores = np.asarray(target_scores, dtype=float)
 
    if len(null_scores) == 0 or len(target_scores) == 0:
        raise ValueError("null_scores and target_scores must be non-empty")
 
    if scale is None:
        scale = len(target_scores) / len(null_scores)
 
    n = null_scores
    t = target_scores
 
    # Candidate thresholds: every observed target score, most stringent first.
    thresholds = np.sort(np.unique(t))[::-1]
 
    null_sorted = np.sort(n)
    target_sorted = np.sort(t)
 
    # "At least as good as threshold" counts via searchsorted.
    n_null_ge = len(null_sorted) - np.searchsorted(null_sorted, thresholds, side='left')
    n_target_ge = len(target_sorted) - np.searchsorted(target_sorted, thresholds, side='left')
 
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_fdr = np.where(n_target_ge > 0, scale * n_null_ge / n_target_ge, 0.0)
 
    # Monotonicity correction (step-up procedure): true FDR cannot decrease
    # as the threshold is relaxed, so we walk from most stringent to least
    # stringent (the order `thresholds` is already in) and take a running
    # maximum. This smooths out noisy dips in the raw empirical estimate.
    monotone_fdr = np.maximum.accumulate(raw_fdr)
 
    passing = np.where(monotone_fdr <= fdr)[0]
    if len(passing) == 0:
        cutoff = None
    else:
        idx = passing.max()  # most permissive threshold that still passes
        cutoff = thresholds[idx]
 
    info = {
        'thresholds': thresholds,
        'fdr': monotone_fdr,
        'n_target': n_target_ge,
        'n_null': n_null_ge,
    }
    return cutoff, info
