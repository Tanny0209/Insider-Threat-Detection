# backend/app/services/convert_to_pyg_full.py
import argparse
import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx


def ensure_cols(df, cols_defaults):
    for c, default in cols_defaults.items():
        if c not in df.columns:
            df[c] = default
    return df


def extract_domain(email):
    try:
        return email.split('@')[-1].lower().strip()
    except Exception:
        return "unknown"


def hour_bin_label(h):
    try:
        h = int(h)
    except Exception:
        return "unknown"
    if 0 <= h < 6:
        return "night"
    if 6 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "afternoon"
    if 18 <= h < 24:
        return "evening"
    return "unknown"


def safe_reindex_to_series(obj, cols):
    """
    Return a pd.Series indexed by cols (fill missing with 0).
    obj may be a Series or a DataFrame row (DataFrame.loc[...] returning Series or DataFrame).
    """
    if isinstance(obj, pd.Series):
        return obj.reindex(cols, fill_value=0)
    # if it's a DataFrame row (single-row DF), pick the first row and reindex
    try:
        return obj.iloc[0].reindex(cols, fill_value=0)
    except Exception:
        # fallback: zeros
        return pd.Series([0.0] * len(cols), index=cols)


def build_node_numeric(df):
    numeric_cols = [
        "emails_sent_day", "avg_daily_emails", "attachment_count", "email_size",
        "subject_length", "body_length", "num_recipients", "contains_sensitive_keywords",
        "has_attachment", "suspicious_attachment", "is_attachment_only", "large_email",
        "new_contact_flag"
    ]
    # force numeric where present
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.groupby("from_user")[ [c for c in numeric_cols if c in df.columns] ].mean().fillna(0)


def build_node_dists(df):
    if "day_of_week" in df.columns:
        dow = df.groupby(["from_user", "day_of_week"]).size().unstack(fill_value=0)
        dow = dow.div(dow.sum(axis=1).replace(0, 1), axis=0)
    else:
        dow = pd.DataFrame()

    if "hour" in df.columns:
        df2 = df.copy()
        df2["hour_bin"] = df2["hour"].apply(hour_bin_label)
        hour_bin = df2.groupby(["from_user", "hour_bin"]).size().unstack(fill_value=0)
        hour_bin = hour_bin.div(hour_bin.sum(axis=1).replace(0, 1), axis=0)
    else:
        hour_bin = pd.DataFrame()

    return dow, hour_bin


def build_top_domains(df, top_k):
    if "from_domain" in df.columns:
        df["from_domain_extracted"] = df["from_domain"].fillna(df["from_user"].apply(extract_domain)).astype(str).str.lower()
    else:
        df["from_domain_extracted"] = df["from_user"].apply(extract_domain)
    top_domains = df["from_domain_extracted"].value_counts().nlargest(top_k).index.tolist()
    return top_domains, df


def build_top_attachment_types(df, top_k):
    if "attachment_types" not in df.columns:
        return []
    df["attachment_types_clean"] = df["attachment_types"].fillna("").astype(str).str.lower()
    exploded = df.assign(att_list=df["attachment_types_clean"].str.split(";")).explode("att_list")
    exploded["att_list"] = exploded["att_list"].astype(str).str.strip()
    top_att = exploded[exploded["att_list"] != ""]["att_list"].value_counts().nlargest(top_k).index.tolist()
    return top_att


def build_edge_aggregates(df_exploded):
    numeric_cols = ["attachment_count", "subject_length", "body_length", "email_size", "num_recipients"]
    for c in numeric_cols:
        if c in df_exploded.columns:
            df_exploded[c] = pd.to_numeric(df_exploded[c], errors="coerce")

    group = df_exploded.groupby(["from_user", "to_user"])
    edge_basic = group.agg(
        emails=("email_id", "count"),
        sensitive_count=("contains_sensitive_keywords", "sum"),
        attachment_count=("attachment_count", "mean"),
        subject_length=("subject_length", "mean"),
        body_length=("body_length", "mean"),
        is_after_hours=("is_after_hours", "mean"),
        is_attachment_only=("is_attachment_only", "mean"),
        suspicious_attachment=("suspicious_attachment", "mean"),
        is_weekend=("is_weekend", "mean")
    ).fillna(0)
    return edge_basic


def build_edge_dists(df_exploded):
    if "day_of_week" in df_exploded.columns:
        edge_dow = df_exploded.groupby(["from_user", "to_user", "day_of_week"]).size().unstack(fill_value=0)
        edge_dow = edge_dow.div(edge_dow.sum(axis=1).replace(0, 1), axis=0)
    else:
        edge_dow = pd.DataFrame()

    if "hour" in df_exploded.columns:
        df2 = df_exploded.copy()
        df2["hour_bin"] = df2["hour"].apply(hour_bin_label)
        edge_hour = df2.groupby(["from_user", "to_user", "hour_bin"]).size().unstack(fill_value=0)
        edge_hour = edge_hour.div(edge_hour.sum(axis=1).replace(0, 1), axis=0)
    else:
        edge_hour = pd.DataFrame()

    return edge_dow, edge_hour


def build_edge_attachment_features(df_exploded, top_att_types):
    df_exploded = df_exploded.copy()
    df_exploded["attachment_types"] = df_exploded.get("attachment_types", "").fillna("").astype(str).str.lower()
    att_df = df_exploded.assign(att_list=df_exploded["attachment_types"].str.split(";")).explode("att_list")
    att_df["att_list"] = att_df["att_list"].astype(str).str.strip()
    att_df = att_df[att_df["att_list"].isin(top_att_types)]
    if att_df.empty:
        return pd.DataFrame([], columns=top_att_types)
    edge_att_counts = att_df.groupby(["from_user", "to_user", "att_list"]).size().unstack(fill_value=0)
    edge_att_counts = edge_att_counts.reindex(columns=top_att_types, fill_value=0)
    edge_att_counts = edge_att_counts.div(edge_att_counts.sum(axis=1).replace(0, 1), axis=0)
    return edge_att_counts


def main(args):
    graph_path = args.graph  # expects user_graph.pkl (NetworkX)
    csv_path = args.csv
    top_k_domains = args.top_k_domains
    top_k_att = args.top_k_att
    out_pyg = args.out_pyg

    os.makedirs(os.path.dirname(out_pyg), exist_ok=True)
    out_map_dir = os.path.dirname(out_pyg)

    print("Loading NetworkX graph:", graph_path)
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    defaults = {
        "emails_sent_day": 0, "avg_daily_emails": 0, "new_contact_flag": 0,
        "is_external": 0, "subject_length": 0, "body_length": 0,
        "attachment_count": 0, "large_email": 0, "contains_sensitive_keywords": 0,
        "has_attachment": 0, "is_after_hours": 0, "is_attachment_only": 0,
        "suspicious_attachment": 0, "hour": 0, "day_of_week": 0,
        "is_weekend": 0, "attachment_types": ""
    }
    df = ensure_cols(df, defaults)

    df["from_user"] = df["from_user"].astype(str).str.lower().str.strip()
    df["to_users"] = df["to_users"].fillna("").astype(str)
    if "from_domain" not in df.columns:
        df["from_domain"] = df["from_user"].apply(extract_domain)
    else:
        df["from_domain"] = df["from_domain"].fillna(df["from_user"].apply(extract_domain)).astype(str).str.lower()

    # Node-level
    print("Building node numeric aggregates...")
    node_numeric = build_node_numeric(df)

    print("Building node day-of-week and hour-bin distributions...")
    node_dow, node_hourbin = build_node_dists(df)

    print(f"Selecting top {top_k_domains} domains...")
    top_domains, df = build_top_domains(df, top_k=top_k_domains)

    print("Building node domain one-hot...")
    node_domain_series = df.groupby("from_user")["from_domain_extracted"].first().fillna("unknown")
    node_domain_series = node_domain_series.apply(lambda x: x if x in top_domains else "other")
    domain_dummies = pd.get_dummies(node_domain_series).reindex(columns=top_domains + ["other"], fill_value=0)

    # Edge-level
    print("Exploding recipients...")
    df_exploded = df.assign(to_user=df["to_users"].str.split(";")).explode("to_user")
    df_exploded["to_user"] = df_exploded["to_user"].astype(str).str.lower().str.strip()
    df_exploded = df_exploded[df_exploded["to_user"] != ""]

    print("Building edge numeric aggregates...")
    edge_basic = build_edge_aggregates(df_exploded)

    print("Building edge day-of-week and hour-bin distributions...")
    edge_dow, edge_hour = build_edge_dists(df_exploded)

    print(f"Selecting top {top_k_att} attachment types...")
    top_att = build_top_attachment_types(df_exploded, top_k=top_k_att)
    edge_att_feats = build_edge_attachment_features(df_exploded, top_att) if top_att else pd.DataFrame([], columns=[])

    # Node mapping
    print("Creating node mapping from NetworkX graph...")
    node_list = list(G.nodes())
    node_mapping = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)
    print("Num nodes in graph:", num_nodes)

    numeric_cols = node_numeric.columns.tolist()
    dow_cols = node_dow.columns.tolist() if not node_dow.empty else []
    hourbin_cols = node_hourbin.columns.tolist() if not node_hourbin.empty else []
    domain_cols = domain_dummies.columns.tolist()

    node_feature_dim = len(numeric_cols) + len(dow_cols) + len(hourbin_cols) + len(domain_cols)
    print("Node feature dimension will be:", node_feature_dim)

    node_feats = np.zeros((num_nodes, node_feature_dim), dtype=np.float32)

    for user, idx in node_mapping.items():
        # numeric block
        if user in node_numeric.index:
            node_feats[idx, :len(numeric_cols)] = node_numeric.loc[user].values
        offset = len(numeric_cols)

        # dow
        if len(dow_cols) and user in node_dow.index:
            vals = safe_reindex_to_series(node_dow.loc[user], dow_cols)
            node_feats[idx, offset: offset + len(dow_cols)] = vals.values
        offset += len(dow_cols)

        # hourbin
        if len(hourbin_cols) and user in node_hourbin.index:
            vals = safe_reindex_to_series(node_hourbin.loc[user], hourbin_cols)
            node_feats[idx, offset: offset + len(hourbin_cols)] = vals.values
        offset += len(hourbin_cols)

        # domain
        if user in domain_dummies.index:
            vals = safe_reindex_to_series(domain_dummies.loc[user], domain_cols)
            node_feats[idx, offset: offset + len(domain_cols)] = vals.values

    # Build edges and attributes
    edge_rows = []
    edge_attrs = []
    for (u, v), row in edge_basic.iterrows():
        if u not in node_mapping or v not in node_mapping:
            continue
        uid, vid = node_mapping[u], node_mapping[v]

        basic_vals = [
            row.get("emails", 0), row.get("sensitive_count", 0), row.get("attachment_count", 0),
            row.get("subject_length", 0), row.get("body_length", 0),
            row.get("is_after_hours", 0), row.get("is_attachment_only", 0),
            row.get("suspicious_attachment", 0), row.get("is_weekend", 0)
        ]

        if (u, v) in edge_dow.index:
            dow_vals = safe_reindex_to_series(edge_dow.loc[(u, v)], dow_cols).values.tolist()
        else:
            dow_vals = [0.0] * len(dow_cols)

        if (u, v) in edge_hour.index:
            hour_vals = safe_reindex_to_series(edge_hour.loc[(u, v)], hourbin_cols).values.tolist()
        else:
            hour_vals = [0.0] * len(hourbin_cols)

        if (len(top_att) > 0) and ((u, v) in edge_att_feats.index):
            att_vals = safe_reindex_to_series(edge_att_feats.loc[(u, v)], top_att).values.tolist()
        else:
            att_vals = [0.0] * len(top_att)

        attr = np.array(basic_vals + dow_vals + hour_vals + att_vals, dtype=np.float32)
        edge_rows.append([uid, vid])
        edge_attrs.append(attr)

    if len(edge_rows) == 0:
        raise RuntimeError("No edges produced for PyG conversion. Check data/graph.")

    edge_index = torch.tensor(edge_rows, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.vstack(edge_attrs), dtype=torch.float32)
    x = torch.tensor(node_feats, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print("PyG full graph built:")
    print(f" - Nodes: {data.num_nodes}, Node feature dim: {data.num_node_features}")
    print(f" - Edges: {data.num_edges}, Edge feature dim: {data.num_edge_features}")

    # Save
    torch.save(data, out_pyg, _use_new_zipfile_serialization=True)
    with open(os.path.join(out_map_dir, "node_mapping.pkl"), "wb") as f:
        pickle.dump(node_mapping, f)
    with open(os.path.join(out_map_dir, "top_domains.pkl"), "wb") as f:
        pickle.dump(top_domains, f)
    with open(os.path.join(out_map_dir, "top_attachment_types.pkl"), "wb") as f:
        pickle.dump(top_att, f)

    print("Saved:")
    print(" - PyG graph:", out_pyg)
    print(" - node_mapping.pkl")
    print(" - top_domains.pkl")
    print(" - top_attachment_types.pkl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="backend/app/data/user_graph.pkl")
    p.add_argument("--csv", default="backend/app/data/Emails.csv")
    p.add_argument("--top_k_domains", type=int, default=50)
    p.add_argument("--top_k_att", type=int, default=10)
    p.add_argument("--out_pyg", default="backend/app/data/pyg_graph_full.pt")
    args = p.parse_args()
    main(args)
