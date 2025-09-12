import pickle
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np

def convert_to_pyg(graph_path: str):
    # Load NetworkX graph
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Map node ids to integers
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    nx.set_node_attributes(G, node_mapping, "node_id")

    # Extract edges
    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_attr.append([data.get("emails", 0), data.get("sensitive_count", 0)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Placeholder node features (use more later in Step 3)
    x = torch.ones((len(G.nodes()), 1), dtype=torch.float)

    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print("PyG graph summary:")
    print(pyg_data)

    # ✅ Save PyG data object
    torch.save(pyg_data, "backend/app/data/pyg_graph.pt")
    print("✅ Saved PyG graph as backend/app/data/pyg_graph.pt")

    return pyg_data


if __name__ == "__main__":
    pyg_graph = convert_to_pyg("backend/app/data/user_graph.pkl")
