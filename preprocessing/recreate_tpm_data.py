import os

import numpy as np
import pandas as pd


def calculate_tpm(df, gene_col="gene_id", length_col="length", count_col="expected_count"):
    df = df.copy()

    # Step 1: reads per kilobase
    rpk = df[count_col] / (df[length_col] / 1000)

    # Step 2: scaling factor (sum of RPK, per million)
    scaling_factor = rpk.sum() / 1e6

    # Step 3: TPM
    df["TPM"] = rpk / scaling_factor

    return df


def calculate_alr(df, ref_gene_id, gene_col="gene_id", value_col="TPM"):
    df = df.copy()

    ref_match = df.loc[df[gene_col] == ref_gene_id, value_col]
    if ref_match.empty:
        raise ValueError(f"Reference gene '{ref_gene_id}' not found in '{gene_col}' column.")
    ref_value = ref_match.iloc[0]

    # add pseudo count
    df[value_col] += 1

    # Compute ALR
    df["ALR"] = np.log(df[value_col] / ref_value)

    return df


to_process = []
for path, dir, files in os.walk("data/sars_cov_2_time_counts"):
    for file in files:
        if "combined" not in file:
            continue
        to_process.append(os.path.join(path, file))


for file in to_process:
    df = pd.read_csv(file, sep="\t")
    df = calculate_tpm(df)
    df = calculate_alr(df, "ENSG00000010244", value_col="expected_count")  # 7756 = ENSG00000010244
    df = df[[x for x in df.columns if x != "FPKM"]]
    raw_name = file.split(".")[0]
    df.to_csv(f"{raw_name}.cleaned.tpm.tsv", sep="\t")
