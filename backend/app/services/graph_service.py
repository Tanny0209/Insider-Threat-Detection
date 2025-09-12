import pandas as pd
import networkx as nx
import pickle

def build_graph(csv_path: str):
    # Load dataset
    df = pd.read_csv(csv_path, low_memory=False)

    # Init directed graph
    G = nx.DiGraph()

    for _, row in df.iterrows():
        sender = row['from_user']
        recipients = str(row['to_users']).split(';') if pd.notna(row['to_users']) else []

        for r in recipients:
            if not r.strip():
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

    # Print a few edges with attributes
    for u, v, d in list(G.edges(data=True))[:5]:
        print(f"{u} -> {v} {d}")

    # ✅ Save graph to pickle file
    with open("backend/app/data/user_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    print("✅ Saved graph as backend/app/data/user_graph.pkl")

    return G


if __name__ == "__main__":
    graph = build_graph("backend/app/data/Final_Emails.csv")
