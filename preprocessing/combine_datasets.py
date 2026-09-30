import os

import pandas as pd


def get_files(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".gz") or file.startswith("."):
            continue

        file_path = os.path.join(folder, file)
        timepoint = file.split("_")[1].split(".")[0][1:]
        replicate = 2 if "_2" in file else 1

        files.append({"path": file_path, "timepoint": int(timepoint), "replicate": replicate})

    files = sorted(files, key=lambda f: (f["timepoint"], f["replicate"]))
    return files


def combine_files(files):
    data = {}
    index = []
    for file in files:
        col_name = f"r_{file['replicate']}_tp_{file['timepoint']}"
        single = pd.read_csv(file["path"], sep="\t", header=None)
        if len(index) == 0:
            index = list(single[0].values)
        else:
            assert index == list(single[0].values)
        data[col_name] = list(single[1].values)

    df = pd.DataFrame(data, index=index)
    return df


def to_tpm(df):
    # convert to TPM column wise
    return df.div(df.sum(axis=0), axis=1) * 1e6


if __name__ == "__main__":
    files = get_files("/Users/elias/Downloads/GSE59717_RAW")
    df = combine_files(files)
    df = to_tpm(df)
    df.to_csv("../data/hsv1_s17_counts/hsv1_s17_counts.csv")
