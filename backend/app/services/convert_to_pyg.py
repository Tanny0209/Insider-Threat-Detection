# backend/app/services/convert_to_pyg_basic.py
import argparse
import pickle
import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx


def enhance_basic(graph_pkl, csv_path, out_pyg="backend/app/data/pyg_graph_basic.pt", out_map="backend/app/data/node_mapping_basic.pkl"):
    print("Loading NetworkX graph:", graph_pkl)
    with open(graph_pkl, "rb") as f:
        G = pickle.load(f)

    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    # Choose numeric columns: prefer known/named numeric list, but also include numeric dtypes
    expected_numeric = [
        "emails_sent_day", "avg_daily_emails", "attachment_count", "email_size",
        "subject_length", "body_length", "num_recipients", "contains_sensitive_keywords",
        "has_attachment", "suspicious_attachment", "is_attachment_only", "large_email",
        "new_contact_flag"
    ]
    numeric_from_df = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in expected_numeric if c in df.columns] + [c for c in numeric_from_df if c not in expected_numeric]

    # Force numeric conversion for those cols
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregate per from_user (mean)
    node_features = df.groupby("from_user")[numeric_cols].mean().fillna(0)

    # Create node mapping and ensure order is stable
    node_list = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(node_list)}

    # Build features matrix aligned to node_list
    X = np.zeros((len(node_list), len(numeric_cols)), dtype=np.float32)
    for i, node in enumerate(node_list):
        if node in node_features.index:
            X[i, :] = node_features.loc[node].values
        else:
            # zeros (already)
            pass

    data = from_networkx(G)  # create PyG Data from the NetworkX graph
    data.x = torch.tensor(X, dtype=torch.float32)

    os.makedirs(os.path.dirname(out_pyg), exist_ok=True)
    torch.save(data, out_pyg, _use_new_zipfile_serialization=True)
    with open(out_map, "wb") as f:
        pickle.dump(node_mapping, f)

    print(f"✅ Saved PyG basic graph: {out_pyg}")
    print(f" - Nodes: {data.num_nodes}, Features: {data.num_node_features}")
    print(f"✅ Saved node mapping: {out_map}")

    return data, node_mapping


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph_pkl", default="backend/app/data/user_graph.pkl")
    p.add_argument("--csv", default="backend/app/data/Emails.csv")
    p.add_argument("--out_pyg", default="backend/app/data/pyg_graph_basic.pt")
    p.add_argument("--out_map", default="backend/app/data/node_mapping_basic.pkl")
    args = p.parse_args()

    enhance_basic(args.graph_pkl, args.csv, args.out_pyg, args.out_map)
