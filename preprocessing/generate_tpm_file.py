import os

import pandas as pd
from dotenv import load_dotenv
from natsort import natsorted

from utils.data_utils import calculate_tpm, load_timepoint_dir

load_dotenv(override=True)

timepoints = load_timepoint_dir(os.getenv("EXPRESSION_DATA"), cleaned_only=False, original=True)

tp_map = {}
tps = set()
for replicate in timepoints:
    for tp in timepoints[replicate]:
        df = pd.read_csv(timepoints[replicate][tp], sep="\t")
        tp_map[f"r_{replicate}_tp_{tp}"] = df.set_index("gene_id")["TPM"].to_dict()
        tps.add(tp)

df = pd.DataFrame(tp_map)

columns = df.columns
col_order = []
tps = sorted(tps)
for tp in tps:
    tp_samples = [x for x in columns if int(x.split("_")[-1]) == int(tp)]
    sorted_tp = natsorted(tp_samples)
    col_order.extend(sorted_tp)

assert len(columns) == len(col_order), f"Missed something: {len(columns)} vs {len(col_order)}"

df = df[col_order]

print(df.sum())
df.to_csv("data/sars_cov_2_time_counts/covid_set.csv", index=True)