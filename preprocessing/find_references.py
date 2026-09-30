import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biothings_client import get_client

mg = get_client("gene")


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
        reachable_nodes[idx] = row[column]

    reachable_nodes_ncbi = {g2n_map[x]: reachable_nodes[x] for x in reachable_nodes}
    return reachable_nodes_ncbi


def calculate_tpm(batch_corrected, gene_lengths):
    lengths = pd.Series(gene_lengths).reindex(batch_corrected.index)
    missing = lengths.isna().sum()
    if missing > 0:
        print(f"Warning: {missing} genes have no length entry and will be dropped.")
        batch_corrected = batch_corrected.loc[lengths.notna()]
        lengths = lengths.dropna()

    # Step 1: RPK — reads per kilobase
    rpk = batch_corrected.divide(lengths / 1e3, axis=0)

    # Step 2: TPM — normalise each sample to sum to 1e6
    tpm = rpk.divide(rpk.sum(axis=0) / 1e6, axis=1)
    return tpm


def find_housekeeping(df):
    # filter out values that are zero everywhere
    df = df[df.sum(axis=1) > 0]

    # low expr filter
    min_cpm = 5
    expressed = df[df.mean(axis=1) > df.mean(axis=1).median()]
    expressed = expressed.astype(float)
    log_expr = np.log2(expressed + 1)

    print(df.mean(axis=1).median())

    log_sd = log_expr.std(axis=1)

    log_mean = np.log2(expressed.mean(axis=1))
    log_sd = log_expr.std(axis=1)

    highest_diff = log_expr.max(axis=1) - log_expr.mean(axis=1)

    expressed = expressed[highest_diff < 2]
    log_sd = log_sd[highest_diff < 2]
    highest_diff = highest_diff[highest_diff < 2]

    stability = pd.DataFrame(
        {"mean_tpm": expressed.mean(axis=1), "log2_sd": log_sd, "highest_diff": highest_diff}
    ).sort_values("log2_sd")

    high_expr = stability[stability["mean_tpm"] > 5]
    final_candidates = high_expr.sort_values("log2_sd").head(50)

    if len(high_expr) == 0:
        raise ValueError("No genes")

    gene_ids = list(final_candidates.index)
    gene_names = mg.getgenes(gene_ids, fields="symbol")
    name_map = {}

    for output in gene_names:
        name_map[output["_id"]] = output["symbol"]

    rev_name_map = {v: k for k, v in name_map.items()}

    final_candidates["name"] = [name_map[str(gene_id)] for gene_id in final_candidates.index]
    mt_genes = [not g.startswith("MT-") for g in final_candidates["name"]]
    final_candidates = final_candidates[mt_genes]
    print(final_candidates)
    return rev_name_map


def compute_pairwise_variation(expr_df, ranked_genes, max_n=None):
    if max_n is None:
        max_n = len(ranked_genes) - 1

    v_values = []

    # NF_n = geometric mean of top n genes' expression, per sample
    def geometric_mean(sub_df):
        # sub_df: genes x samples
        log_vals = np.log2(sub_df)
        return 2 ** log_vals.mean(axis=0)  # geometric mean across genes, per sample

    for n in range(2, max_n + 1):
        genes_n = ranked_genes[:n]
        genes_n1 = ranked_genes[: n + 1]

        NFn = geometric_mean(expr_df.loc[genes_n])
        NFn1 = geometric_mean(expr_df.loc[genes_n1])

        # V_n/n+1 = standard deviation of log2(NFn / NFn+1) across samples
        log_ratio = np.log2(NFn / NFn1)
        V = log_ratio.std(ddof=1)

        v_values.append({"comparison": f"V{n}/{n + 1}", "n": n, "V_value": V})

    return pd.DataFrame(v_values)


def check_variation(rev_name_map, ranked_genes):
    ranked_genes = ["EEF1A1", "B2M", "SCGB1A1", "BPIFB1", "SLPI", "SAT1", "HSP90AA1"]

    ranked_genes = [int(rev_name_map[x]) for x in ranked_genes]

    print(ranked_genes)

    v_df = compute_pairwise_variation(df, ranked_genes, max_n=6)
    print(v_df)

    print(ranked_genes[:4])

    plt.plot(v_df["comparison"], v_df["V_value"], marker="o")
    plt.axhline(0.15, color="red", linestyle="--", label="V = 0.15 threshold")
    plt.xlabel("Pairwise comparison")
    plt.ylabel("V value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def check_mean_expr(df, housekeeping=None):
    if housekeeping is not None:
        in_df = [x for x in housekeeping if x in df.index]
    else:
        in_df = df.index
    rel_df = df.loc[in_df, :]
    for col in rel_df.columns:
        column = rel_df[col]
        # drop all 0 entries
        column = column[column != 0]
        print(col, column.mean())
    # mean_exprs = rel_df.mean(axis=0)
    # print(mean_exprs)


def read_virus_pct(df, viral_genes):
    for column in df.columns:
        total_reads = df[column].sum()
        viral_reads = df.loc[viral_genes, column].sum()
        viral_pct = viral_reads / total_reads * 100
        print(f"{column} {total_reads:.0f}\t{viral_pct:.2f}%")


def prefilter_genes(expr: pd.DataFrame, top_n: int = 500, log_transform: bool = True, pseudocount: float = 1.0) -> list:
    """
    Rank genes by their own individual CV across samples (low CV = stable
    expression on its own) and return the top_n most stable genes.
    Use this to shrink the search space before running select_stable_genes
    on a large input matrix.
    """
    data = np.log2(expr + pseudocount) if log_transform else expr.copy()
    means = data.mean(axis=1)
    stds = data.std(axis=1, ddof=1)
    cv = (stds / means.abs()).replace([np.inf, -np.inf], np.nan)
    ranked = cv.sort_values()  # ascending: most stable first
    return list(ranked.head(top_n).index)


def select_stable_genes(
    expr: pd.DataFrame,
    candidate_genes: list | None = None,
    min_genes: int = 5,
    target_cv: float | None = None,
    log_transform: bool = True,
    pseudocount: float = 1.0,
    sample_groups: pd.Series = None,
    verbose: bool = True,
):
    """
    Greedily remove genes to minimize the CV of the subset's per-sample mean.

    Parameters
    ----------
    expr : DataFrame, genes x samples, TPM values.
    candidate_genes : list, optional. Restrict the search to these genes
        (e.g. output of prefilter_genes). Defaults to all genes in expr.
    min_genes : int. Stop once the subset reaches this size (lower bound).
    target_cv : float, optional. If set, stop as soon as the objective CV
        drops at or below this value (may stop before min_genes is hit).
    log_transform : bool. Log2(TPM + pseudocount) before computing stats.
    sample_groups : pandas Series, optional. Index = sample names (must
        match expr.columns), values = group label (e.g. "core",
        "critical"). When provided, the optimization target is the
        WORST-CASE (max) CV across groups rather than the pooled CV over
        all samples. This prevents a small, biologically distinct minority
        of samples from being effectively ignored just because it barely
        moves the pooled average. Without this, plain pooled CV is
        dominated by whichever group has more samples, and a gene set can
        look "stable" purely because it's stable in the majority group
        while being unstable in a smaller group you may care about most.
    verbose : bool. Print progress.

    Returns
    -------
    selected_genes : list of gene names in the final stable subset.
    history : DataFrame tracking n_genes, the objective CV, per-group CVs
        (log and linear scale), and which gene was removed at each step,
        so you can pick a different size/CV tradeoff after the fact and
        see the stability of each group individually.
    """
    data = expr.copy()
    if log_transform:
        data = np.log2(data + pseudocount)

    current = list(candidate_genes) if candidate_genes is not None else list(data.index)

    if sample_groups is not None:
        sample_groups = sample_groups.reindex(expr.columns)
        if sample_groups.isna().any():
            missing = list(sample_groups[sample_groups.isna()].index)
            raise ValueError(f"sample_groups is missing labels for samples: {missing}")
        group_labels = list(pd.unique(sample_groups))
    else:
        group_labels = None

    def group_cvs(genes, df):
        """Per-sample mean of `genes`, then CV within each group.
        Returns dict {group_label: cv}. If no groups given, one entry
        'all' covering every sample."""
        m = df.loc[genes].mean(axis=0)
        if group_labels is None:
            return {"all": float(m.std(ddof=1) / abs(m.mean()))}
        out = {}
        for g in group_labels:
            sub = m[sample_groups[sample_groups == g].index]
            out[g] = float(sub.std(ddof=1) / abs(sub.mean()))
        return out

    def objective(genes, df):
        gcvs = group_cvs(genes, df)
        return max(gcvs.values()), gcvs  # worst-case group is what we optimize

    def flatten(prefix, gcvs):
        return {f"{prefix}_{k}": v for k, v in gcvs.items()}

    cv, gcvs = objective(current, data)
    _, lin_gcvs = objective(current, expr)
    history = [
        {
            "n_genes": len(current),
            "cv": cv,
            "removed_gene": None,
            **flatten("cv", gcvs),
            **flatten("linear_cv", lin_gcvs),
        }
    ]

    if verbose:
        detail = ", ".join(f"{k}={v:.1%}" for k, v in lin_gcvs.items())
        print(f"Start: {len(current)} genes, worst-case log-CV={cv:.4f} | linear CV by group: {detail}")

    while len(current) > min_genes:
        if target_cv is not None and cv <= target_cv:
            break

        best_gene, best_cv = None, np.inf
        for g in current:
            trial_cv, _ = objective([x for x in current if x != g], data)
            if trial_cv < best_cv:
                best_cv, best_gene = trial_cv, g

        # If removing the worst offender no longer helps, stop early
        # (only relevant if target_cv is None; otherwise we keep going
        # toward min_genes regardless).
        if target_cv is None and best_cv >= cv and len(current) <= min_genes + 1:
            break

        current.remove(best_gene)
        cv, gcvs = objective(current, data)
        _, lin_gcvs = objective(current, expr)
        history.append(
            {
                "n_genes": len(current),
                "cv": cv,
                "removed_gene": best_gene,
                **flatten("cv", gcvs),
                **flatten("linear_cv", lin_gcvs),
            }
        )

        if verbose:
            detail = ", ".join(f"{k}={v:.1%}" for k, v in lin_gcvs.items())
            print(
                f"Removed {best_gene} -> {len(current)} genes, worst-case log-CV={cv:.4f} | linear CV by group: {detail}"
            )

    return current, pd.DataFrame(history)


def filter_low_tpm(df):
    exclude = df.mean(axis=1) < 1
    df = df.loc[~exclude]
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/batch_corrected_counts.csv", index_col=0)
    gene_lengths = "data/sars_cov_2_time_counts/replicate1/GSM7050960_11-mock_GT21-29957_CCGCGGTT-CTAGCGCT_S393_L002.cleaned.tpm.tsv"
    prot_map = "data/mart_export.txt"
    vir_map = "data/sars_cov_2_gene_map.tsv"
    gene_lengths = process_expression(gene_lengths, prot_map, vir_map, "effective_length")
    gene_length_map = {k: pd.NA if v == 0 else v for k, v in gene_lengths.items()}
    viral_genes = pd.read_csv(vir_map, sep="\t", header=None)
    viral_genes = viral_genes[1].to_list()

    viral_genes = [x for x in viral_genes if x in df.index]

    # read_virus_pct(df, viral_genes)

    housekeeping = [
        100033819,
        10010,
        100507436,
        10060,
        10077,
        10084,
        101180976,
        10146,
        10155,
        10206,
        1021,
        10269,
        103,
        10346,
        10379,
        104,
        10410,
        10417,
        10475,
        10521,
        10533,
        10542,
        10561,
        10577,
        10581,
        10621,
        10622,
        10623,
        10625,
        107181291,
        1072,
        10758,
        10771,
        10859,
        10868,
        10875,
        10955,
        10964,
        11035,
        11040,
        11043,
        11074,
        11100,
        11128,
        11213,
        112703,
        11274,
        11277,
        1129,
        1130,
        11331,
        114548,
        1147,
        115004,
        115330,
        115362,
        1174,
        117854,
        1191,
        1386,
        140564,
        1432,
        146167,
        146857,
        148022,
        149233,
        151636,
        1525,
        163702,
        164668,
        1653,
        1654,
        1660,
        1665,
        1667,
        1668,
        1687,
        170506,
        171389,
        171568,
        1755,
        1937,
        197259,
        1984,
        200315,
        200316,
        201292,
        201294,
        201626,
        2023,
        2074,
        2131,
        2150,
        220001,
        2268,
        22861,
        22863,
        22900,
        22948,
        22954,
        22978,
        23076,
        23087,
        2332,
        23478,
        23515,
        23586,
        23607,
        23765,
        24138,
        246778,
        2475,
        2537,
        255520,
        256957,
        256987,
        25939,
        25942,
        26065,
        26146,
        26175,
        2625,
        2633,
        2634,
        2635,
        27297,
        27350,
        2736,
        282616,
        282617,
        282618,
        29072,
        29078,
        29108,
        29110,
        29115,
        2962,
        29800,
        29883,
        29915,
        30009,
        3091,
        3280,
        3315,
        3320,
        3373,
        338,
        338321,
        338376,
        340061,
        3428,
        3429,
        3433,
        3434,
        3437,
        3439,
        3440,
        3441,
        3442,
        3443,
        3444,
        3445,
        3446,
        3448,
        3449,
        3451,
        3452,
        3454,
        3455,
        3456,
        345611,
        3458,
        3459,
        3460,
        3467,
        3551,
        3553,
        3569,
        3588,
        3592,
        3593,
        3594,
        3600,
        3609,
        3627,
        3659,
        3660,
        3661,
        3663,
        3665,
        3669,
        3687,
        3694,
        3696,
        375,
        3764,
        377630,
        388419,
        388646,
        389856,
        3921,
        3925,
        3934,
        3964,
        3965,
        406919,
        406933,
        406938,
        406991,
        407017,
        407024,
        407028,
        407031,
        4088,
        4153,
        4277,
        4283,
        4321,
        4343,
        439996,
        440275,
        4486,
        4599,
        4600,
        4615,
        4686,
        4690,
        4790,
        4828,
        4829,
        4938,
        4939,
        4940,
        4953,
        4986,
        50489,
        50506,
        5094,
        50943,
        51191,
        51251,
        51257,
        51284,
        51297,
        51311,
        51428,
        51561,
        51652,
        51728,
        5179,
        5245,
        5359,
        5366,
        5371,
        53840,
        53905,
        54106,
        54476,
        5450,
        54512,
        5452,
        54541,
        54606,
        5465,
        54726,
        54765,
        54920,
        5495,
        550,
        55054,
        55210,
        55284,
        55324,
        55337,
        55421,
        55509,
        5551,
        55601,
        55703,
        55718,
        55763,
        5600,
        5610,
        5611,
        566,
        56829,
        5683,
        56832,
        56915,
        56927,
        57169,
        57337,
        57379,
        57506,
        57663,
        57678,
        57705,
        57724,
        5788,
        581,
        58487,
        58515,
        59,
        59067,
        596,
        5970,
        598,
        5987,
        6009,
        602,
        6035,
        6036,
        6039,
        6041,
        60489,
        60509,
        6210,
        6351,
        6352,
        6355,
        6356,
        6363,
        6367,
        6375,
        6387,
        64081,
        64108,
        64135,
        64170,
        64332,
        64343,
        64771,
        6502,
        653082,
        6609,
        661,
        664,
        665,
        6675,
        6693,
        6737,
        6772,
        6773,
        684,
        6895,
        7040,
        708,
        7097,
        7098,
        7124,
        7128,
        7178,
        7187,
        7189,
        7268,
        728358,
        7292,
        730249,
        7334,
        768212,
        7706,
        7726,
        7728,
        7852,
        7874,
        79102,
        79132,
        79591,
        79594,
        79798,
        79844,
        80149,
        80212,
        80342,
        8061,
        8091,
        81030,
        811,
        81559,
        81570,
        81576,
        81603,
        81622,
        8165,
        81786,
        81844,
        834,
        83666,
        8372,
        83737,
        83862,
        8399,
        84101,
        84166,
        84282,
        8449,
        8467,
        84851,
        84871,
        84932,
        84993,
        8517,
        8519,
        85363,
        85364,
        8547,
        8575,
        8638,
        8673,
        8678,
        8692,
        8725,
        8729,
        8772,
        8780,
        8815,
        89870,
        9020,
        90865,
        90933,
        9111,
        91351,
        9140,
        91445,
        91543,
        91607,
        9188,
        9201,
        9213,
        9217,
        9447,
        9474,
        951,
        958,
        9582,
        9636,
        9641,
        9755,
        9821,
        9868,
    ]

    tpm_df = calculate_tpm(df, gene_length_map)
    tpm_df = tpm_df.astype(float)

    tpm_df = filter_low_tpm(tpm_df)

    not_tp_6 = [x for x in tpm_df.columns if "tp_6" not in x]
    sample_importance = {x: "critical" if "tp_6" in x else "core" for x in tpm_df.columns}
    groups = pd.Series(sample_importance)

    candidates = prefilter_genes(tpm_df)

    selected = set(tpm_df.index) - set(viral_genes)
    selected = list(selected)

    # selected, history = select_stable_genes(
    #     tpm_df,
    #     candidate_genes=candidates,   # or omit to search all genes
    #     min_genes=20,                 # stop once you're down to 20 genes
    #     # target_cv=0.03,         # alternatively stop once CV <= 0.02
    #     sample_groups=groups
    # )
    linear_mean = tpm_df.loc[selected].mean(axis=0)  # raw TPM, no log transform
    linear_cv = linear_mean.std(ddof=1) / linear_mean.mean()

    raw_mean = df.loc[selected].mean(axis=0)
    raw_cv = raw_mean.std(ddof=1) / raw_mean.mean()
    print(f"TPM CV: {linear_cv:.3f}, raw CV: {raw_cv:.3f}")

    check_mean_expr(tpm_df, selected)
