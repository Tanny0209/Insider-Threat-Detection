import os
import json
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

# -----------------------
# Helpers / defaults
# -----------------------
DEFAULT_GRAPH_DIR = "backend/app/data/graph_outputs"
DEFAULT_FULL_DF = "backend/app/data/Final_Emails.csv"
DEFAULT_OUT_DIR = DEFAULT_GRAPH_DIR
ANOMALIES_CSV = "anomalies.csv"
ANOMALIES_JSON = "anomalies.json"

# For numeric stable scaling
_EPS = 1e-9


# -----------------------
# I/O functions
# -----------------------
def load_nodes_edges(graph_dir):
    nodes_path = os.path.join(graph_dir, "nodes.csv")
    edges_path = os.path.join(graph_dir, "edges.csv")

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing nodes or edges in {graph_dir}. Expected nodes.csv and edges.csv")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    return nodes_df, edges_df


def build_nx_graph(nodes_df, edges_df, directed=True):
    """
    Build a directed graph. Handles both integer node_ids and string-based IDs (emails).
    """
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes
    for _, r in nodes_df.iterrows():
        nid = r.get("node_id", r.get("id", None))
        if pd.isna(nid):
            continue
        # Keep string if it's email, else cast to int safely
        try:
            nid = int(nid)
        except Exception:
            nid = str(nid).lower()

        attrs = r.to_dict()
        G.add_node(nid, **attrs)

    # Add edges
    for _, r in edges_df.iterrows():
        s = r.get("source")
        t = r.get("target")
        if pd.isna(s) or pd.isna(t):
            continue
        try:
            s = int(s)
        except Exception:
            s = str(s).lower()
        try:
            t = int(t)
        except Exception:
            t = str(t).lower()

        attrs = r.to_dict()
        weight = float(r.get("count", 1))
        attrs["weight"] = weight
        G.add_edge(s, t, **attrs)

    return G


# -----------------------
# Graph metrics
# -----------------------
def compute_basic_metrics(G, top_n=20):
    in_deg = dict(G.in_degree()) if G.is_directed() else dict(G.degree())
    out_deg = dict(G.out_degree()) if G.is_directed() else dict(G.degree())
    deg = dict(G.degree())

    top_degree = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_n]

    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    return {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "degree": deg,
        "top_degree": top_degree,
        "pagerank": pr
    }


def compute_approx_betweenness(G, k=None, seed=42):
    try:
        bc = nx.betweenness_centrality(G, k=k, seed=seed, weight="weight")
    except Exception as e:
        print("Betweenness computation failed:", e)
        bc = {n: 0.0 for n in G.nodes()}
    return bc


# -----------------------
# Temporal activity anomalies
# -----------------------
def compute_daily_activity_spikes(full_df_path, z_thresh=3.0):
    if not os.path.exists(full_df_path):
        print(f"Full dataset not found at {full_df_path}. Skipping activity spikes.")
        return pd.DataFrame(columns=["from_user","last_count","mean_count","std_count","z_score","spike"])

    df = pd.read_csv(full_df_path, usecols=["from_user","date"], low_memory=False)

    # 🔥 Ensure datetime parsing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where date could not be parsed
    df = df.dropna(subset=["date"])

    df["date_only"] = df["date"].dt.date

    daily = df.groupby(["from_user","date_only"]).size().reset_index(name="count")
    if daily.empty:
        return pd.DataFrame(columns=["from_user","last_count","mean_count","std_count","z_score","spike"])

    last_day = daily["date_only"].max()
    hist = daily[daily["date_only"] < last_day]
    if hist.empty:
        return pd.DataFrame(columns=["from_user","last_count","mean_count","std_count","z_score","spike"])

    stats = hist.groupby("from_user")["count"].agg(["mean","std"]).reset_index().rename(columns={"mean":"mean_count","std":"std_count"})
    last = daily[daily["date_only"]==last_day].rename(columns={"count":"last_count"})[["from_user","last_count"]]
    merged = last.merge(stats, how="left", on="from_user").fillna(0)
    merged["z_score"] = (merged["last_count"] - merged["mean_count"]) / (merged["std_count"] + _EPS)
    merged["spike"] = ((merged["z_score"] >= z_thresh) | (merged["last_count"] >= merged["mean_count"] * 5))
    return merged


# -----------------------
# Risk scoring
# -----------------------
def normalize_series(s):
    arr = np.array(list(s.values())) if isinstance(s, dict) else np.array(s)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn < _EPS:
        return {k: 0.0 for k in (s.keys() if isinstance(s, dict) else range(len(arr)))}
    norm = (arr - mn) / (mx - mn)
    if isinstance(s, dict):
        return {k:v for k,v in zip(s.keys(), norm)}
    else:
        return norm


def build_risk_scores(nodes_df, degree_dict, pagerank_dict, betweenness_dict, external_ratio_col="external_sent_ratio", activity_spikes_df=None):
    deg_norm = normalize_series(degree_dict)
    pr_norm = normalize_series(pagerank_dict)
    bc_norm = normalize_series(betweenness_dict)

    rows = []
    for _, r in nodes_df.iterrows():
        user = r.get("user") or r.get("email") or r.get("from_user") or r.get("node_id")
        if pd.isna(user):
            continue
        node_id = r.get("node_id", user)
        try:
            node_id = int(node_id)
        except Exception:
            node_id = str(node_id).lower()

        deg = deg_norm.get(node_id, 0)
        pr = pr_norm.get(node_id, 0)
        bc = bc_norm.get(node_id, 0)
        ext_ratio = float(r.get(external_ratio_col, 0)) if external_ratio_col in r else 0.0

        spike_flag = False
        last_count = 0
        if activity_spikes_df is not None:
            row_s = activity_spikes_df[activity_spikes_df["from_user"]==user]
            if not row_s.empty:
                spike_flag = bool(row_s.iloc[0]["spike"])
                last_count = int(row_s.iloc[0]["last_count"])

        w_activity = 0.4
        w_external = 0.25
        w_pagerank = 0.15
        w_newcontact = 0.2
        new_contact_val = 1.0 if spike_flag else 0.0

        risk_score = (
            w_activity * (1.0 if spike_flag else 0.0)
            + w_external * ext_ratio
            + w_pagerank * pr
            + w_newcontact * new_contact_val
        )
        risk_score = float(min(max(risk_score, 0.0), 1.0))

        rows.append({
            "user": user,
            "node_id": node_id,
            "deg": deg,
            "pagerank": pr,
            "betweenness": bc,
            "external_ratio": ext_ratio,
            "activity_spike": spike_flag,
            "last_count": last_count,
            "risk_score": risk_score
        })

    risk_df = pd.DataFrame(rows)
    return risk_df.sort_values("risk_score", ascending=False)


# -----------------------
# Main orchestration
# -----------------------
def main(args):
    graph_dir = args.graph_dir
    out_dir = args.out_dir or graph_dir
    full_df = args.full_df

    print("📂 Loading nodes/edges from:", graph_dir)
    nodes_df, edges_df = load_nodes_edges(graph_dir)

    print("🔨 Building networkx graph...")
    G = build_nx_graph(nodes_df, edges_df)

    print("📊 Computing basic metrics...")
    metrics = compute_basic_metrics(G, top_n=args.top_n)

    print(f"⏳ Computing betweenness (k={args.betweenness_k})...")
    bc = compute_approx_betweenness(G, k=args.betweenness_k)

    print("📈 Checking activity spikes...")
    activity_spikes = compute_daily_activity_spikes(full_df, z_thresh=args.z_thresh)

    print("⚖️ Building risk scores...")
    risk_df = build_risk_scores(nodes_df, metrics["degree"], metrics["pagerank"], bc, activity_spikes_df=activity_spikes)

    anomalies = risk_df.head(args.top_n_output)
    anomalies_path = os.path.join(out_dir, ANOMALIES_CSV)
    anomalies_json = os.path.join(out_dir, ANOMALIES_JSON)
    anomalies.to_csv(anomalies_path, index=False)
    anomalies.to_json(anomalies_json, orient="records", indent=2)
    print(f"✅ Saved top {args.top_n_output} anomalies → {anomalies_path}, {anomalies_json}")

    print("🔍 Top flagged users:")
    print(anomalies[["user","node_id","risk_score","external_ratio","activity_spike"]].head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph EDA + Anomaly Detector")
    parser.add_argument("--graph-dir", type=str, default=DEFAULT_GRAPH_DIR, help="Directory with nodes.csv and edges.csv")
    parser.add_argument("--full-df", type=str, default=DEFAULT_FULL_DF, help="Full email CSV (Final_Emails.csv)")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Where to save anomalies")
    parser.add_argument("--top-n", type=int, default=20, help="Top N for degree/pagerank listing")
    parser.add_argument("--top-n-output", type=int, default=100, help="How many top anomalies to save")
    parser.add_argument("--betweenness-k", type=int, default=200, help="k for betweenness; None=exact")
    parser.add_argument("--z-thresh", type=float, default=3.0, help="z-threshold for activity spikes")
    args = parser.parse_args()
    main(args)
