import json
import sys

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def show_graph(json_file=""):
    data = {}
    with open(json_file, "r") as f:
        data = json.load(f)

    graph = nx.DiGraph()
    for node in data["nodes"]:
        graph.add_node(node["name"])

    for edge in data["edges"]:
        graph.add_edge(edge["from"], edge["to"])

    #pos = nx.spring_layout(graph)
    pos = graphviz_layout(graph, prog="dot")
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    plt.title("online recommend DAG")
    plt.show()
    plt.savefig


if __name__ == "__main__":
    json_file = sys.argv[1]
    show_graph(json_file)
