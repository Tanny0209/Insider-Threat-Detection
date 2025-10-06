# backend/app/services/build_user_graph.py
import argparse
import pickle
import pandas as pd
import networkx as nx
import os
import torch
from torch_geometric.utils import from_networkx


def extract_domain(email):
    try:
        return email.split("@")[-1].lower().strip()
    except Exception:
        return "unknown"


def build_graph(csv_path: str, out_graph_pkl: str, out_pyg_pt: str, out_node_list: str):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    if "from_user" not in df.columns or "to_users" not in df.columns:
        raise ValueError("CSV must contain 'from_user' and 'to_users' columns.")

    # Normalize
    df["from_user"] = df["from_user"].astype(str).str.lower().str.strip()
    df["to_users"] = df["to_users"].fillna("").astype(str)

    G = nx.DiGraph()
    dropped_senders = 0
    dropped_recipients = 0

    for _, row in df.iterrows():
        sender = row["from_user"]
        if pd.isna(sender) or sender == "" or str(sender).lower() == "nan":
            dropped_senders += 1
            continue

        recipients = [r.strip().lower() for r in str(row["to_users"]).split(";") if r and str(r).strip().lower() != "nan"]

        if len(recipients) == 0:
            dropped_recipients += 1
            continue

        for r in recipients:
            # add node attrs optionally
            if not G.has_node(sender):
                G.add_node(sender, domain=extract_domain(sender))
            if not G.has_node(r):
                G.add_node(r, domain=extract_domain(r))

            if G.has_edge(sender, r):
                # increment existing
                G[sender][r]["emails"] = G[sender][r].get("emails", 0) + 1
                if row.get("contains_sensitive_keywords", False):
                    G[sender][r]["sensitive_count"] = G[sender][r].get("sensitive_count", 0) + 1
            else:
                G.add_edge(sender, r,
                           emails=1,
                           sensitive_count=1 if row.get("contains_sensitive_keywords", False) else 0)

    print("Graph summary:")
    print(f" Nodes: {G.number_of_nodes()}")
    print(f" Edges: {G.number_of_edges()}")
    print(f" Dropped invalid senders: {dropped_senders}")
    print(f" Dropped invalid recipients: {dropped_recipients}")

    # Save NetworkX graph .pkl
    os.makedirs(os.path.dirname(out_graph_pkl), exist_ok=True)
    with open(out_graph_pkl, "wb") as f:
        pickle.dump(G, f)
    print(f"✅ Saved NetworkX graph: {out_graph_pkl}")

    # Also convert to PyG Data object and save (keeps parity)
    data = from_networkx(G)
    torch.save(data, out_pyg_pt, _use_new_zipfile_serialization=True)
    print(f"✅ Saved PyG graph: {out_pyg_pt}")

    # Save node list ordering (useful for alignment later)
    node_list = list(G.nodes())
    with open(out_node_list, "wb") as f:
        pickle.dump(node_list, f)
    print(f"✅ Saved node list: {out_node_list}")

    return G, data, node_list


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="backend/app/data/Emails.csv")
    p.add_argument("--out_graph_pkl", default="backend/app/data/user_graph.pkl")
    p.add_argument("--out_pyg_pt", default="backend/app/data/pyg_graph.pt")
    p.add_argument("--out_node_list", default="backend/app/data/node_list.pkl")
    args = p.parse_args()

    build_graph(args.csv, args.out_graph_pkl, args.out_pyg_pt, args.out_node_list)
