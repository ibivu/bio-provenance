# one source is intact, the other biogrid
import pandas as pd


def human_nodes(df1, df2):
    # check number of overlapping nodes where taxid is 9606
    human_df1 = df1[df1["taxonomy"] == 9606]
    human_df2 = df2[df2["taxonomy"] == 9606]

    df1_entrez = set(human_df1["entrez"].values)
    df2_entrez = set(human_df2["entrez"].values)

    print(f"Number of human nodes in df1: {len(df1_entrez)}; in df2: {len(df2_entrez)}")
    print(f"Overlapping nodes: {len(df1_entrez.intersection(df2_entrez))}")


def non_human_nodes(df1, df2):
    nh_df1 = df1[df1["taxonomy"] != 9606]
    nh_df2 = df2[df2["taxonomy"] != 9606]

    unique_ia = set(nh_df1["taxonomy"].values)
    unique_bg = set(nh_df2["taxonomy"].values)

    print(f"Number of viruses in intact: {len(unique_ia)}; biogrid: {len(unique_bg)}")


def interaction_counts(df1, df2):
    print(f"Total interactions in intact: {len(df1)}; biogrid {len(df2)}")

    # group by taxonomy and count interactions
    # A,B,taxonomy_A,taxonomy_B,entrez_A,entrez_B,name_A,name_B
    counts = pd.concat([df1["taxonomy_A"], df1["taxonomy_B"]]).value_counts()
    counts = counts[counts >= 5]

    print("IntAct Interactions per virus (>5 only)")
    for taxonomy_id, count in counts.items():
        print(f"{taxonomy_id}: {count}")

    counts = pd.concat([df2["taxonomy_A"], df2["taxonomy_B"]]).value_counts()
    counts = counts[counts >= 0]

    print("Biogrid Interactions per virus (>5 only)")
    for taxonomy_id, count in counts.items():
        print(f"{taxonomy_id}: {count}")


if __name__ == "__main__":
    df_ia = pd.read_csv("../data/virus_nodes.csv")
    df_bg = pd.read_csv("../data/virus_nodes.biogrid.csv")

    edges_ia = pd.read_csv("../data/interactions_annotated.csv")
    edges_bg = pd.read_csv("../data/interactions_annotated.biogrid.csv")

    human_nodes(df_ia, df_bg)
    non_human_nodes(df_ia, df_bg)

    interaction_counts(edges_ia, edges_bg)
