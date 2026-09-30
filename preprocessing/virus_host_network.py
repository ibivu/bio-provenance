import csv

import networkx as nx
import pandas as pd


def split_interactions(df, edges_file="edges.csv", nodes_file="nodes.csv", filter=None):
    edges = []
    nodes = {}  # keyed by protein ID to deduplicate

    for row in df.to_dict(orient="records"):
        # Collect edge
        if filter:
            if row["taxonomy_A"] == filter:
                edges.append((row["entrez_A"], row["entrez_B"]))
        else:
            edges.append((row["A"], row["B"]))

        # Collect node info for A and B (deduplicated by ID)
        if row["A"] not in nodes:
            nodes[row["A"]] = {
                "id": row["A"],
                "taxonomy": row["taxonomy_A"],
                "entrez": int(row["entrez_A"]),
                "name": row["name_A"],
            }
        if row["B"] not in nodes:
            nodes[row["B"]] = {
                "id": row["B"],
                "taxonomy": row["taxonomy_B"],
                "entrez": int(row["entrez_B"]),
                "name": row["name_B"],
            }

    # Write edges file
    with open(edges_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["A", "B"])
        writer.writerows(edges)

    # Write nodes file
    with open(nodes_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "taxonomy", "entrez", "name"])
        writer.writeheader()
        writer.writerows(nodes.values())

    print(f"Done. Written {len(edges)} edges to '{edges_file}' and {len(nodes)} unique nodes to '{nodes_file}'.")


if __name__ == "__main__":
    interactions = pd.read_csv("../data/interactions_annotated.biogrid.csv")
    # drop rows where there are nans
    interactions = interactions.dropna()

    filter_virus = "2697049"

    split_interactions(
        interactions, f"../data/virus_edges_{filter_virus}.csv", "../data/virus_nodes.biogrid.csv", filter_virus
    )
