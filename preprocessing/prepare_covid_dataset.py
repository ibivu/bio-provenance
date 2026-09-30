import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from natsort import natsorted


def filter_samples(participants=None, time=0, symptom=None, samples_per_timepoint=None):
    df = pd.read_csv("/Users/elias/Downloads/GSE198449_samples.csv")
    df = df[df["enrollment batch"] == "Later Surveillance"]
    df = df[df["PCR test for SARS-Cov-2"] == "Detected"]
    if participants:
        df = df[df["participant ID"].isin(participants)]
    if time:
        df = df[df["biocollection ID"].str.split("T").str[1].astype(int) < time]
        seen_pat_ids = df[df["biocollection ID"].str.split("T").str[1].astype(int) > 0]["participant ID"].values
        seen_pat_ids = dict(Counter(seen_pat_ids))
        seen_pat_ids = dict(sorted(seen_pat_ids.items(), key=lambda x: x[1], reverse=True))
        remove_pat_ids = {k for i, k in enumerate(seen_pat_ids) if i > 4}
        print(len(remove_pat_ids), len(seen_pat_ids))
        tp = df["biocollection ID"].str.split("T").str[1].astype(int)
        mask = df["participant ID"].isin(remove_pat_ids) & (tp == 0)
        df = df[~mask]
    if symptom:
        df = df[~df["symptom"].isna()]
        df = df[df["symptom"].str.split(";").apply(lambda x: symptom in x)]

    b_ids = list(df["biocollection ID"].values)
    pat_ids = list(df["participant ID"].values)

    with open("/Users/elias/Downloads/GSE198449_featureCounts.txt", "r") as f:
        header = f.readline().strip().split("\t")

    header = set(header)
    header_short = {s[:-3] for s in header}
    plates = {x.split("_")[-1] for x in header}
    print(f"{len(header) - 1} samples, {len(plates)} plates")

    b_ids = [b_id for b_id in b_ids if b_id in header_short]

    if samples_per_timepoint:
        df['tp'] = df["biocollection ID"].str.split("T").str[1].astype(int)
        df = df[df["tp"] < 36]
        df = df[df["biocollection ID"].isin(b_ids)]
        df = (
            df.groupby('tp')
            .filter(lambda g: len(g) >= 4)
            .groupby('tp')
            .apply(lambda g: g.sample(n=4, random_state=0))
            .reset_index(drop=True)
        )
        b_ids = list(df["biocollection ID"].values)
        pat_ids = list(df["participant ID"].values)

    print(f"{len(b_ids)} samples to process")

    timepoints = {}
    for bid in b_ids:
        tp = bid.split("-T")[1]
        tp = int(tp)
        if tp not in timepoints:
            timepoints[tp] = 0
        timepoints[tp] += 1

    sorted_tp = dict(sorted(timepoints.items(), key=lambda item: item[1], reverse=True))
    return header, b_ids, sorted_tp, pat_ids

def plot_samples(sorted_tp):
    ax = sns.barplot(x=list(sorted_tp.keys()), y=list(sorted_tp.values()))
    ax.bar_label(ax.containers[0])
    plt.xlabel("Timepoint")
    plt.ylabel("Count samples")
    plt.title("Samples per timepoint")
    plt.show()

def get_ensembl_gene_lengths(gff3_path):
    ensembl_pattern = re.compile(r'\b(ENS[A-Z]*G\d+)(?:\.\d+)?\b')
    # ensembl_pattern = re.compile(r'\b(ENS[A-Z]*G\d+.\d+)\b')
    gene_lengths = {}
    with open(gff3_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue

            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9:
                continue

            feature_type = fields[2]
            if feature_type != 'gene':
                continue

            start, end = fields[3], fields[4]
            attributes = fields[8]

            # Try to find an Ensembl-style gene ID in the attributes column
            match = ensembl_pattern.search(attributes)
            if not match:
                continue

            ensembl_id = match.group(1)

            try:
                length = int(end) - int(start) + 1
            except ValueError:
                continue

            gene_lengths[ensembl_id] = length
    return gene_lengths

def calculate_tpm(matrix, gene_lengths):
    matrix = matrix[matrix.index.isin(gene_lengths)]

    lengths = pd.Series({g: gene_lengths[g] for g in matrix.index}, index=matrix.index)
    lengths_kb = lengths / 1000.0

    # Ensure numeric counts
    counts = matrix.apply(pd.to_numeric, errors='coerce')

    # Step 1: reads per kilobase (RPK)
    rpk = counts.div(lengths_kb, axis=0)

    # Step 2: scale each sample so RPKs sum to 1 million
    scaling_factors = rpk.sum(axis=0) / 1e6
    tpm = rpk.div(scaling_factors, axis=1)

    return tpm

# subset sample file
def subset_sample_file(b_ids, header, filename="covid_set2.csv"):
    relevant_samples = {"Geneid": "gene_id_0"}
    b_ids = set(b_ids)
    timepoints = set()
    pat_map = {}
    c = 0
    for sample in header:
        if sample[:-3] not in b_ids:
            continue
        sample_split = sample.split("_")[1].split("-")
        sample_tp = int(sample_split[1][1:])
        timepoints.add(sample_tp)
        sample_patient = sample_split[0]
        if sample_patient not in pat_map:
            c+=1
            pat_map[sample_patient] = c
        ordered_col_name = f"r_{pat_map[sample_patient]}_tp_{sample_tp}"
        if ordered_col_name in set(relevant_samples.values()):
            continue

        relevant_samples[sample] = ordered_col_name

    assert len(set(relevant_samples.values())) == len(relevant_samples.values()), "Sample mapping should be unique!"

    col_order = []
    for tp in timepoints:
        tp_samples = [x for x in relevant_samples.values() if int(x.split("_")[-1]) == tp]
        sorted_tp = natsorted(tp_samples)
        col_order.extend(sorted_tp)

    print(f"Selected {len(relevant_samples) - 1} samples")

    gene_lengths = get_ensembl_gene_lengths("/Users/elias/Downloads/gencode.v34.annotation.gff3")

    full_df =  pd.read_csv('/Users/elias/Downloads/GSE198449_featureCounts.txt', usecols=relevant_samples, sep="\t")
    full_df = full_df.rename(columns=relevant_samples)
    full_df = full_df[col_order]
    full_df.set_index("gene_id_0", inplace=True)
    full_df.index = full_df.index.str.split(".").str[0]

    overlap = {x for x in full_df.index if x in gene_lengths}
    print(f"Can match {len(overlap)} of {len(full_df)} ensembl ids to gff3 file")

    full_df = calculate_tpm(full_df, gene_lengths)

    full_df.to_csv(f'data/sars_cov_2_big_time_counts/{filename}', header=True, index=True)

def sort_patients(pat_ids, b_ids):
    patient_count = {}
    for patient in pat_ids:
        if patient not in patient_count:
            patient_count[patient] = 0
        patient_count[patient] += 1

    s_pats = dict(sorted(patient_count.items(), key=lambda x: x[1], reverse=True))
    sum_samples = 0
    selected = []
    for i, (k, v) in enumerate(s_pats.items()):
        if i > 20:
            break
        print(k, v)
        selected.append(k)
        sum_samples += v
    print(f"Selected {len(selected)}/{len(patient_count.keys())} patients, at {sum_samples} individual points")
    return set(selected)

if __name__ == "__main__":
    header, b_ids, sorted_tps, patients = filter_samples(samples_per_timepoint=True)

    #good_patients = sort_patients(patients, b_ids)
    #header, b_ids, sorted_tps, patients = filter_samples(good_patients)
    plot_samples(sorted_tps)
    subset_sample_file(b_ids, header, filename="covid_set_4_per_tp.csv")