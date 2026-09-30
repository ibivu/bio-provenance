from os import write

import networkx as nx
from utils.data_utils import write_graph

tf_tf = []
skipped = 0
total = 0
with open("data/reg_network_tf_tf_with_evidence.txt") as f:
    header = f.readline().strip().split("\t")
    for line in f:
        line_split = line.strip().split("\t")
        total += 1
        if line_split[-1] == "N/A":
            skipped += 1
            continue
        a, b = line_split[1].split(":")[1], line_split[3].split(":")[1]
        tf_tf.append((a, b))
print(f"{skipped} skipped out of {total} total ({skipped/total:.2f}) TF/TF")

tf_gene = []
skipped, total = 0, 0
with open("data/reg_network_tf_gene_with_evidence.txt") as f:
    _ = f.readline()
    for line in f:
        line_split = line.strip().split("\t")
        total += 1
        if line_split[-1] == "N/A":
            skipped += 1
            continue
        a, b = line_split[1].split(":")[1], line_split[3].split(":")[1]
        tf_gene.append((a, b))
print(f"{skipped} skipped out of {total} total ({skipped/total:.2f}) TF/Gene")


total = set(tf_gene + tf_tf)
print(f"Left: {len(total)}")

def grn_network(path):
    nodes_path = path + ".nodes.tsv"
    edges_path = path + ".edges.tsv"

    graph = nx.DiGraph()
    columns = []
    is_first = True
    with open(nodes_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            if is_first:
                columns.extend(line)
                is_first = False
                continue
            # graph.add_node(int(line[0]), **{columns[1]: line[1], columns[2]: line[2], columns[3]: line[3]})
            graph.add_node(int(line[0]), name=line[1], is_virus=False, kind=line[3])

    with open(edges_path, "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            if len(line) > 2:
                graph.add_edge(int(line[0]), int(line[1]), weight=float(line[2]), real_weight=float(line[3]))
            else:
                graph.add_edge(int(line[0]), int(line[1]))
    return graph

g = grn_network("data/grn_tf_ncbi")

reliable_edges = []
for u, v in g.edges():
    if (f"{u}", f"{v}") in total:
        reliable_edges.append((u, v))

print(len(reliable_edges))
g_sub = g.edge_subgraph(reliable_edges).copy()
print(g_sub)

# write_graph(g_sub, "data/grn_tf_ncbi_evidence", nodes=True)