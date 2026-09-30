from cProfile import label

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_edge_weights(graph, graph_b=None, cutoff=0.0, weight_col="real_weight", fig_name="edge_weight_random_dist"):
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        edge_weights.append(data.get(weight_col))

    tpm_counts = []
    for _, data in graph.nodes(data=True):
        if "timepoints" not in data:
            continue
        tpm_counts.extend(data["timepoints"])

    if graph_b:
        edge_weights_b = []
        for u, v, data in graph_b.edges(data=True):
            edge_weights_b.append(data.get("real_weight"))
        plot_two(
            edge_weights_b,
            edge_weights,
            tpm_counts,
            cutoff,
            ("Random graph", "Real graph"),
            "Comparison null model edge weights",
        )
        return

    plt.hist(edge_weights, bins=20)
    plt.title("Distribution of edge weights")
    plt.ylabel("Count")
    plt.xlabel("Partial correlation")
    if fig_name is not None:
        plt.savefig(f"plots/{fig_name}.png")
    plt.show()


def plot_two(arr1, arr2, arr3, cutoff, labels, title, kde=True, stat="density", figsize=(10, 5)):

    arr1 = np.asarray(arr1, dtype=float)
    arr2 = np.asarray(arr2, dtype=float)
    arr3 = np.asarray(arr3, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    palette = sns.color_palette("Set2")

    # ── Left panel: overlapping histograms + KDE ──────────────────────────
    for arr, label, color in zip([arr1, arr2], labels, palette):
        sns.histplot(
            arr,
            ax=axes[0],
            label=label,
            color=color,
            kde=kde,
            alpha=0.45,
            edgecolor="white",
            linewidth=0.5,
        )

    axes[0].axvline(cutoff, color="red", linestyle="--", label="cut off after BH correction")
    axes[0].axvline(-cutoff, color="red", linestyle="--")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Correlation")
    axes[0].set_ylabel(stat.capitalize())
    axes[0].legend()
    sns.despine(ax=axes[0])

    # ── Right panel: Distribution of counts ─────────────
    sns.histplot(arr3, ax=axes[1], color=palette[0], fill=True, alpha=0.35, log_scale=True)

    axes[1].set_title("TPM distribution")
    axes[1].set_xlabel("TPM")
    axes[1].set_ylabel("Count")
    sns.despine(ax=axes[1])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/null_graph_corr_comparison.png")
    plt.show()


def plot_edge_distribution(graph: nx.Graph):
    edge_degrees = []
    node_range = [i for i in range(len(graph))]
    for n in graph.nodes():
        edge_degrees.append(graph.degree(n))

    edge_degrees = sorted(edge_degrees, reverse=True)

    palette = sns.color_palette("Set2")
    sns.lineplot(x=node_range, y=edge_degrees, color=palette[0])
    plt.title("Edge Degree Distribution")
    plt.xlabel("Nodes")
    plt.ylabel("Degree")
    plt.show()


def plot_hist(counts: list, counts2: list | None = None, title="Histogram", discrete=True, labels=("c1", "c2")):
    if counts2 is None:
        sns.histplot(counts, discrete=discrete)
        plt.title(title)
        plt.xlabel("Categories")
        plt.ylabel("Counts")
        plt.show()
        return

    _fig, ax = plt.subplots(figsize=(6, 6))

    sns.histplot(counts, label=labels[0], color="steelblue", alpha=0.5, ax=ax)
    sns.histplot(counts2, label=labels[1], color="coral", alpha=0.5, ax=ax)

    ax.legend()
    plt.title(title)
    # plt.yscale("log")
    plt.tight_layout()
    plt.show()


# plot expression over time for each of the timepoints as a visual indicator of what genes are worth being an alert
def plot_timepoints(graph: nx.DiGraph, is_in_final=None, exclude_direct=True, relative=False):
    nodes_data = []
    y_labels = []
    x_labels = []
    for node, data in graph.nodes(data=True):
        if data["is_virus"]:
            continue
        direct = False
        for p in graph.predecessors(node):
            if graph.nodes[p]["is_virus"]:
                direct = bool(exclude_direct)
                break
        if direct:
            continue
        nodes_data.append(data["timepoints"])
        if is_in_final and node in is_in_final:
            node = f"{node} !"
        y_labels.append(node)
        x_labels = data["time"]

    timepoint_data = np.array(nodes_data)
    sort_idx = np.argsort(y_labels)

    y_labels = np.array(y_labels)[sort_idx]
    timepoint_data = timepoint_data[sort_idx, :]

    if relative:
        # set all values relative to 0 # TODO: MAKE THIS AGNOSTIC OF RELPLICATES
        timepoint_data[:, ::2] /= timepoint_data[:, [0]]
        timepoint_data[:, 1::2] /= timepoint_data[:, [1]]

    n_rows = timepoint_data.shape[0]
    n_cols = timepoint_data.shape[1]

    # plot_timepoints_line(timepoint_data, y_labels, x_labels)

    min_box_height = 0.2  # inches per row, adjust to taste
    min_box_width = 0.5  # inches per column, optional

    fig_height = max(6, n_rows * min_box_height)
    fig_width = max(8, n_cols * min_box_width)

    _fig, _ax = plt.subplots(figsize=(fig_width, fig_height))

    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap = sns.color_palette("viridis", as_cmap=True)

    sns.heatmap(timepoint_data, cmap=cmap, yticklabels=y_labels, xticklabels=x_labels, annot=False)
    plt.savefig("plots/heatmap.png")


def plot_timepoints_line(timepoints, xlabel, ylabels):
    arr = np.array(timepoints)  # shape (n_sequences, seq_len)
    median = np.median(arr, axis=0)
    q1, q3 = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
    iqr = q3 - q1

    THRESHOLD = 1.5  # ← tune this (lower = more sensitive)
    is_outlier = np.any((arr < q1 - THRESHOLD * iqr) | (arr > q3 + THRESHOLD * iqr), axis=1)

    # ── long-form dataframe ───────────────────────────────────────────────────
    df = (
        pd.DataFrame(arr)
        .assign(
            seq_id=range(len(timepoints)),
            is_outlier=is_outlier.astype(str),  # seaborn hue needs a string/category
        )
        .melt(id_vars=["seq_id", "is_outlier"], var_name="step", value_name="value")
    )

    # ── plot ──────────────────────────────────────────────────────────────────
    _fig, ax = plt.subplots(figsize=(10, 5))

    PAL = {"False": "#c8c8c8", "True": "#e05c3a"}

    for label, grp in df.groupby("is_outlier"):
        is_out = label == "True"
        for sid, sub in grp.groupby("seq_id"):
            ax.plot(
                sub["step"],
                sub["value"],
                color=PAL[label],
                alpha=0.85 if is_out else 0.25,
                lw=1.8 if is_out else 0.8,
                zorder=3 if is_out else 1,
            )

    # median ± IQR band
    steps = np.arange(arr.shape[1])
    ax.fill_between(steps, q1 - THRESHOLD * iqr, q3 + THRESHOLD * iqr, alpha=0.08, color="#888", label="IQR band")
    ax.plot(steps, median, color="#555", lw=2, ls="--", label="median")

    # legend proxy
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#e05c3a", lw=2, label=f"outlier  (n={is_outlier.sum()})"),
        Line2D([0], [0], color="#c8c8c8", lw=1, label=f"normal   (n={(~is_outlier).sum()})"),
        Line2D([0], [0], color="#555", lw=2, ls="--", label="median"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    ax.set(xlabel="step", ylabel="value", title="Sequence outlier plot")
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/outlier_sequences.png", dpi=150)
    plt.show()


def plot_path(graph: nx.Graph, path: dict):
    node_labels = []
    timepoints = []
    replicates = []
    for i in range(len(path["path"])):
        if graph.nodes[path["path"][i]]["is_virus"]:
            continue
        node_labels.append(path["path"][i])

        timepoints.append(graph.nodes[path["path"][i]]["timepoints"])

        curr = -1
        for time in graph.nodes[path["path"][i]]["time"]:
            pass

    timepoints = np.array(timepoints)

    x = np.arange(timepoints.shape[1])
    for row, label in zip(timepoints, node_labels):
        sns.lineplot(x=x, y=row, label=label)

    plt.legend()
    plt.show()


def plot_alert_nodes(alert_nodes, knees=None):
    scores = list(alert_nodes.values())
    scores = sorted(scores, reverse=True)

    sns.lineplot(scores)
    plt.axhline(0, color="red", linestyle="--")
    if knees:
        for knee in knees:
            plt.axvline(knee, color="blue", linestyle="--")
    plt.title("Log2Fold changes in all nodes")
    plt.ylabel("Score")
    plt.xlabel("Gene")
    plt.show()


def plot_pca(df, batches, covariates, n_components=2, scale=True, figsize=(8, 6), point_size=80, alpha=0.85, ax=None):

    df = df.T

    # use only numeric columns for PCA input
    X = df.select_dtypes(include=[np.number]).copy()

    if scale:
        X = StandardScaler().fit_transform(X)
    else:
        X = X.values

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    pca_df = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"], index=df.index)
    pca_df["batch"] = np.asarray(batches)
    pca_df["covariate"] = np.asarray(covariates)

    var_exp = pca.explained_variance_ratio_ * 100

    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="batch",
        style="covariate",
        s=point_size,
        alpha=alpha,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
    ax.set_title("PCA")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_virus_score_distributions(background_dist_list, virus_scores, names):
    num_plots = len(background_dist_list)

    if num_plots != len(virus_scores) or num_plots != len(names):
        raise ValueError("The length of background_dist_list, virus_scores, and names must be the same.")

    # Calculate the number of rows and columns for subplots
    cols = min(num_plots, 3)  # Maximum of 3 columns for better visualization
    rows = (num_plots + cols - 1) // cols  # Ensure all plots are included

    _fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))

    if num_plots == 1:
        ax = axes
    else:
        axes: list = axes.flatten()

    for i in range(num_plots):
        sns.histplot(background_dist_list[i], kde=True, ax=axes[i])
        axes[i].axvline(x=virus_scores[i], color="red", linestyle="--", label="Virus score")
        axes[i].set_title(f"Alert: {names[i]}; human path scores")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("plots/paths_to_alerts.png")
    plt.show()

def plot_path_score_distributions(scores_virus, scores_human, cutoff=None, labels=("Virus scores", "Human scores"), bins=30):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(scores_virus, kde=True, stat="density", bins=bins, label=labels[0], alpha=0.5, ax=ax)
    sns.histplot(scores_human, kde=True, stat="density", bins=bins, label=labels[1], alpha=0.5, ax=ax)
    if cutoff is not None:
        plt.axvline(cutoff, color="red", linestyle="--", label="5% FDR")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Path Scores Distributions")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_box(ev ,nev):
    df = pd.DataFrame({
        'score': list(ev) + list(nev),
        'group': ['evidence'] * len(ev) + ['no evidence'] * len(nev)
    })

    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x='group', y='score', hue='group', palette='Set2', legend=False)
    sns.stripplot(data=df, x='group', y='score', color='black', alpha=0.4, jitter=True, size=3)

    plt.title('Comparison of ev vs no ev scores')
    plt.xlabel('')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()