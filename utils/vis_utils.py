import networkx as nx


def change_graph_ids(G: nx.Graph):
    for n, d in G.nodes(data=True):
        if d["is_virus"] and d["name"].endswith("SARS-CoV-2"):
            d["name"] = f"{d['name']}:{str(n)[-3:]}"
    mapping = nx.get_node_attributes(G, "name")
    G = nx.relabel_nodes(G, mapping)
    return G


def colour_path(G: nx.Graph, path: list, alert=None):
    for node, data in G.nodes(data=True):
        if data["is_virus"]:
            data["color"] = "lightgreen"
        elif node not in path:
            data["color"] = "lightgray"
        elif alert is None and node == path[-1] or alert and node == alert:
            data["color"] = "orange"
        else:
            data["color"] = "lightblue"

    for u, v, d in G.edges(data=True):
        if G.nodes[u]["is_virus"] and v in path:
            d["color"] = "green"
        elif u in path and v in path:
            d["color"] = "blue"
        else:
            d["color"] = "gray"
