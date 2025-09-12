import pandas as pd
import networkx as nx
import pickle


def build_graph(csv_path: str):
    # Load dataset
    df = pd.read_csv(csv_path, low_memory=False)

    # Init directed graph
    G = nx.DiGraph()

    dropped_senders = 0
    dropped_recipients = 0

    for _, row in df.iterrows():
        sender = row['from_user']

        # Skip invalid senders
        if pd.isna(sender) or not str(sender).strip() or str(sender).lower() == "nan":
            dropped_senders += 1
            continue

        recipients = str(row['to_users']).split(';') if pd.notna(row['to_users']) else []

        for r in recipients:
            r = str(r).strip()
            if not r or r.lower() == "nan":
                dropped_recipients += 1
                continue

            if G.has_edge(sender, r):
                G[sender][r]['emails'] += 1
                if row['contains_sensitive_keywords']:
                    G[sender][r]['sensitive_count'] += 1
            else:
                G.add_edge(
                    sender,
                    r,
                    emails=1,
                    sensitive_count=1 if row['contains_sensitive_keywords'] else 0
                )

    print("Graph summary:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Dropped invalid senders: {dropped_senders}")
    print(f"Dropped invalid recipients: {dropped_recipients}")

    # Print a few edges with attributes
    for u, v, d in list(G.edges(data=True))[:5]:
        print(f"{u} -> {v} {d}")

    # ✅ Save graph to pickle file
    with open("backend/app/data/user_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    print("✅ Saved graph as backend/app/data/user_graph.pkl")

    return G


if __name__ == "__main__":
    graph = build_graph("backend/app/data/Emails.csv")
