# backend/app/services/convert_to_pyg.py
import os
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import pickle

def convert_to_pyg(graph_pkl_path, csv_path, save_path="backend/app/data/pyg_graph_basic.pt"):
    # Load NetworkX graph (from Step 1)
    with open(graph_pkl_path, "rb") as f:
        G = pickle.load(f)

    # Load dataset
    df = pd.read_csv(csv_path)

    # 🔹 Select only numeric columns safely
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # 🔹 Force conversion of suspicious "numeric-looking" strings
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate numeric features per node (mean)
    node_features = df.groupby("from_user")[numeric_cols].mean().fillna(0)

    # Mapping: node -> index
    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    # Edge list
    edge_index = []
    for u, v in G.edges():
        if pd.notna(u) and pd.notna(v) and u in node_mapping and v in node_mapping:
            edge_index.append([node_mapping[u], node_mapping[v]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Node features → tensor
    features = torch.tensor(node_features.values, dtype=torch.float)

    # Build PyG graph
    data = Data(x=features, edge_index=edge_index)

    # Save it
    torch.save(data, save_path)

    print(f"✅ PyG graph saved to {save_path}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Feature dim: {data.num_node_features}")
    return data


if __name__ == "__main__":
    pyg_graph = convert_to_pyg(
        graph_pkl_path="backend/app/data/user_graph.pkl",
        csv_path="backend/app/data/Emails.csv",
        save_path="backend/app/data/pyg_graph_basic.pt"
    )
