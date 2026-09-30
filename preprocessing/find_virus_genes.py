import json

import requests


def for_hsv():
    virus_quant = []
    with open("/Users/elias/code/provenance/data/hsv1_s17_counts/hsv1_s17_counts.csv", "r") as f:
        for line in f:
            if not line.startswith("ENSG"):
                virus_quant.append(line.strip().split(",")[0])

    gene_ids = ",".join(virus_quant[1:])

    print(gene_ids)

    LINK = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=protein&db=gene&retmode=json&id="

    res = requests.get(LINK + gene_ids)
    data = json.loads(res.text)

    orig_ids = []
    ncbi_ids = []
    for item in data["linksets"]:
        orig_ids = item["ids"]
        ncbi_ids = item["linksetdbs"][0]["links"]

    with open("/Users/elias/code/provenance/data/hsv_gene_map.tsv", "w") as f:
        f.writelines(f"GI:{k}\t{v}\n" for k, v in zip(orig_ids, ncbi_ids))


def for_covid():

    with open(
        "/Users/elias/Downloads/GSM7050949_9-mock_GT21-25881_CCGCGGTT-CTAGCGCT_S34_L004.rsem.counts.txt", "r"
    ) as f:
        for line in f:
            if not line.startswith("ENSG"):
                print(line)


if __name__ == "__main__":
    for_covid()
