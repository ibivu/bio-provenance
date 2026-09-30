import networkx as nx
import pandas as pd
import tqdm

from utils.data_utils import write_graph

regulatory_network_file_path = "data/human_core_Regulator_Gene.txt"
regulatory_tf_file_path = "data/human_core_Regulator_TF.txt"
measured_genes_path = (
    "data/sars_cov_2_time_counts/replicate1/GSM7050960_11-mock_GT21-29957_CCGCGGTT-CTAGCGCT_S393_L002.rsem.counts.txt"
)
gene_map = "data/mart_export.txt"

measured_genes = pd.read_csv(measured_genes_path, sep="\t")
reg_network = pd.read_csv(regulatory_network_file_path, sep="\t", header=None)
reg_tf = pd.read_csv(regulatory_tf_file_path, sep="\t", header=None)
gene_map = pd.read_csv(gene_map, sep="\t")
g2n_map = gene_map.set_index("Gene stable ID")["NCBI gene (formerly Entrezgene) ID"].to_dict()

unique_genes = set(measured_genes["gene_id"].values)
unique_genes_ncbi = {g2n_map[x] for x in unique_genes if g2n_map.get(x, None) is not None}

reg_network_genes = set(reg_network[1].values).union(set(reg_network[3].values))
reg_network_genes_names = {int(x[5:]) for x in reg_network_genes if x.startswith("NCBI:") and x[5:].isdigit()}

reg_tf_genes = set(reg_tf[1].values).union(set(reg_tf[3].values))
reg_tf_genes_names = {int(x[5:]) for x in reg_network_genes if x.startswith("NCBI:") and x[5:].isdigit()}

# combine both sources
reg_network_genes_names = reg_network_genes_names.union(reg_tf_genes_names)
reg_network = pd.concat([reg_network, reg_tf], ignore_index=True)

grn = nx.DiGraph()

for idx, row in tqdm.tqdm(reg_network.iterrows(), total=len(reg_network)):
    u, v = row[1], row[3]

    if not u.startswith("NCBI:") or not u[5:].isdigit():
        continue
    if not v.startswith("NCBI:") or not v[5:].isdigit():
        continue

    u, v = int(u[5:]), int(v[5:])

    if u not in unique_genes_ncbi or v not in unique_genes_ncbi:
        continue

    if not grn.has_node(u):
        grn.add_node(u, name=row[0], is_virus=False, kind=row[4])
    if not grn.has_node(v):
        grn.add_node(v, name=row[2], is_virus=False, kind=row[5])
    grn.add_edge(u, v)

if __name__ == "__main__":
    write_graph(grn, "data/grn_tf_ncbi", True)
