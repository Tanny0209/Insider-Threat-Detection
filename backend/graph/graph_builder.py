# backend/graph/graph_builder.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
import re
import os
import networkx as nx
from tqdm import tqdm


# --- CONFIG ---
ORG_DOMAIN = "enron.com"
PROCESSED_CSV = "datasets/processed_emails.csv"
USER_META_CSV = "datasets/user_metadata.csv"  # optional: columns => email, department, role
OUTPUT_PT = "backend/graph/graph_data.pt"      # updated path

# regexes
url_regex = re.compile(r"http[s]?://\S+")
attach_ext_regex = re.compile(r"\b\w+\.(pdf|docx?|xls[xm]?|pptx?|zip|csv|txt)\b", re.IGNORECASE)
sensitive_keywords = {"confidential","password","ssn","salary","privileged","secret",
                      "proprietary","invoice","credentials","download","urgent","transfer","wire"}

# --- UTILITIES ---
def safe_split_emails(field):
    if pd.isna(field) or str(field).strip()=="":
        return []
    parts = [p.strip().lower() for p in str(field).split(",") if p.strip()]
    return parts

def heuristic_nlp_score(subject, body):
    text = (str(subject or "") + " " + str(body or "")).lower()
    score = 0.05
    for kw in sensitive_keywords:
        if kw in text:
            score += 0.25
    if attach_ext_regex.search(text):
        score += 0.15
    if url_regex.search(text):
        score += 0.1
    if len(text.split()) > 200:
        score += 0.05
    return float(min(1.0, round(score,3)))

# --- MAIN FUNCTION ---
def build_graph(processed_file=PROCESSED_CSV, user_meta_file=USER_META_CSV, output_file=OUTPUT_PT, approx_bc=True, bc_samples=1000):
    print("🔹 Loading processed CSV...")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"{processed_file} not found. Run parser first.")

    dtype_map = {
        "from": str, "to": str, "cc": str, "bcc": str, "subject": str,
        "body": str, "hour_of_day": "float32", "day_of_week": str, "nlp_score": "float32"
    }
    df = pd.read_csv(processed_file, dtype=dtype_map, low_memory=False)

    # Optional user metadata
    user_meta = {}
    if os.path.exists(user_meta_file):
        um = pd.read_csv(user_meta_file, dtype=str, low_memory=False)
        for _, r in um.iterrows():
            email = str(r.get("email","")).strip().lower()
            user_meta[email] = {
                "department": str(r.get("department","")) or "",
                "role": str(r.get("role","")) or ""
            }

    # --- MAP EMAIL TO NODE ID ---
    email_to_id = {}
    id_to_email = []
    def get_node_id(email):
        email = str(email).strip().lower()
        if email == "" or email == "nan":
            return None
        if email not in email_to_id:
            email_to_id[email] = len(id_to_email)
            id_to_email.append(email)
        return email_to_id[email]

    # --- ACCUMULATORS ---
    edge_counter = defaultdict(lambda: {"count":0, "hours":[], "has_link":0, "has_attach":0, "nlp_score_sum":0.0})
    node_stats = defaultdict(lambda: {
        "sent":0, "recv":0, "ext_contacts":0, "hours":[], "nlp_scores":[], "body_lengths":0,
        "unique_recipients": set(), "weekend_count":0
    })

    # --- PROCESS EMAILS ---
    print("🔹 Processing emails and building edge/node stats...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sender = str(row.get("from","")).strip().lower()
        if sender in ("", "nan"):
            continue

        all_recipients = safe_split_emails(row.get("to","")) + \
                         safe_split_emails(row.get("cc","")) + \
                         safe_split_emails(row.get("bcc",""))
        if len(all_recipients) == 0:
            continue

        nlp_score = None
        if "nlp_score" in df.columns and pd.notna(row.get("nlp_score")):
            try: nlp_score = float(row.get("nlp_score",0.0))
            except: nlp_score = None
        if nlp_score is None:
            nlp_score = heuristic_nlp_score(row.get("subject",""), row.get("body",""))

        text_combined = str(row.get("subject","")) + " " + str(row.get("body",""))
        has_attach = 1 if attach_ext_regex.search(text_combined) else 0
        has_link = 1 if url_regex.search(text_combined) else 0

        hour = None
        if pd.notna(row.get("hour_of_day")):
            try: hour = int(row.get("hour_of_day"))
            except: pass

        is_weekend = False
        dow = row.get("day_of_week")
        if pd.notna(dow):
            try:
                if isinstance(dow,str):
                    is_weekend = dow.strip().lower() in ("sat","saturday","sun","sunday")
                else:
                    is_weekend = int(dow)>=5
            except: pass

        # Update sender node
        s_id = get_node_id(sender)
        node_stats[s_id]["sent"] += 1
        node_stats[s_id]["nlp_scores"].append(nlp_score)
        node_stats[s_id]["body_lengths"] += len(str(row.get("body","")))
        if hour is not None: node_stats[s_id]["hours"].append(hour)
        if is_weekend: node_stats[s_id]["weekend_count"] += 1

        # Update edges
        for r in all_recipients:
            r_id = get_node_id(r)
            if r_id is None: continue
            edge_key = (s_id,r_id)
            edge_counter[edge_key]["count"] +=1
            if hour is not None: edge_counter[edge_key]["hours"].append(hour)
            edge_counter[edge_key]["has_link"] = max(edge_counter[edge_key]["has_link"], has_link)
            edge_counter[edge_key]["has_attach"] = max(edge_counter[edge_key]["has_attach"], has_attach)
            edge_counter[edge_key]["nlp_score_sum"] += nlp_score
            node_stats[r_id]["recv"] += 1
            node_stats[r_id]["body_lengths"] += len(str(row.get("body","")))
            node_stats[s_id]["unique_recipients"].add(r)
            try:
                s_dom = sender.split("@")[-1] if "@" in sender else ""
                r_dom = r.split("@")[-1] if "@" in r else ""
                if s_dom != r_dom:
                    node_stats[s_id]["ext_contacts"] += 1
            except: pass

    # --- BUILD NODE FEATURES ---
    print("🔹 Building node feature matrix...")
    num_nodes = len(id_to_email)
    x = np.zeros((num_nodes,11), dtype=np.float32)
    for nid in range(num_nodes):
        stats = node_stats[nid]
        sent = stats.get("sent",0)
        recv = stats.get("recv",0)
        ext = stats.get("ext_contacts",0)
        hours = stats.get("hours",[])
        avg_hour = float(np.mean(hours)) if hours else 0.0
        std_hour = float(np.std(hours)) if hours else 0.0
        avg_nlp = float(np.mean(stats.get("nlp_scores",[]))) if stats.get("nlp_scores") else 0.0
        avg_body = float(stats.get("body_lengths",0)/max(1,sent+recv))
        unique_contacts = len(stats.get("unique_recipients", set()))
        external_ratio = float(ext/sent) if sent>0 else 0.0
        after_hours = sum(1 for h in hours if h<9 or h>=18)/len(hours) if hours else 0.0
        weekend_ratio = float(stats.get("weekend_count",0)/max(1,sent))
        burstiness = std_hour
        x[nid,:] = np.array([sent, recv, ext, avg_hour, avg_nlp, avg_body,
                             unique_contacts, external_ratio, after_hours,
                             weekend_ratio, burstiness], dtype=np.float32)

    # --- BUILD EDGES ---
    print("🔹 Building edge index and edge attributes...")
    if len(edge_counter)==0:
        raise ValueError("No edges found. Check CSV recipients")

    edge_list, edge_attr_list = [], []
    max_edge_count = max(vals["count"] for vals in edge_counter.values())
    for (s_id,r_id), vals in edge_counter.items():
        edge_list.append([s_id,r_id])
        avg_hour = float(np.mean(vals["hours"])) if vals["hours"] else -1.0
        count = float(vals["count"])
        norm_count = count/max_edge_count if max_edge_count>0 else 0.0
        avg_nlp = float(vals["nlp_score_sum"]/max(1,vals["count"]))
        edge_attr_list.append([count, norm_count, avg_hour, float(vals["has_link"]),
                               float(vals["has_attach"]), avg_nlp])

    edge_index = torch.tensor(edge_list,dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list,dtype=torch.float32)
    x_tensor = torch.tensor(x,dtype=torch.float32)

    # --- GRAPH STRUCTURAL FEATURES ---
    print("🔹 Computing graph structural features...")
    G_und = nx.Graph()
    G_und.add_nodes_from(range(num_nodes))
    G_und.add_edges_from([(u,v) for (u,v) in edge_list])

    out_deg = np.zeros((num_nodes,), dtype=np.float32)
    in_deg = np.zeros((num_nodes,), dtype=np.float32)
    for (u,v), vals in edge_counter.items():
        out_deg[u] += vals["count"]
        in_deg[v] += vals["count"]

    clustering_arr = np.array([v for k,v in nx.clustering(G_und).items()], dtype=np.float32)

    print("🔹 Computing betweenness centrality...")
    if approx_bc:
        # Approximate using k random samples
        bc_dict = nx.betweenness_centrality(G_und, k=min(bc_samples, num_nodes), normalized=True, seed=42)
    else:
        bc_dict = nx.betweenness_centrality(G_und, normalized=True)
    bc_arr = np.array([bc_dict.get(n,0.0) for n in range(num_nodes)], dtype=np.float32)

    struct_feats = np.stack([out_deg, in_deg, clustering_arr, bc_arr], axis=1)
    x_tensor = torch.cat([x_tensor, torch.tensor(struct_feats,dtype=torch.float32)], dim=1)

    # --- APPEND USER META ---
    if os.path.exists(user_meta_file):
        dept_vocab, role_vocab = {}, {}
        dept_col = np.full((num_nodes,), -1, dtype=np.int32)
        role_col = np.full((num_nodes,), -1, dtype=np.int32)
        for nid, email in enumerate(id_to_email):
            meta = user_meta.get(email,{})
            d, r = meta.get("department","").strip(), meta.get("role","").strip()
            if d:
                if d not in dept_vocab: dept_vocab[d] = len(dept_vocab)
                dept_col[nid] = dept_vocab[d]
            if r:
                if r not in role_vocab: role_vocab[r] = len(role_vocab)
                role_col[nid] = role_vocab[r]
        dept_feat = ((dept_col.astype(np.float32)+1)/(max(1, dept_col.max())+1)).reshape(-1,1)
        role_feat = ((role_col.astype(np.float32)+1)/(max(1, role_col.max())+1)).reshape(-1,1)
        x_tensor = torch.cat([x_tensor, torch.tensor(dept_feat,dtype=torch.float32),
                              torch.tensor(role_feat,dtype=torch.float32)], dim=1)

    # --- SAVE GRAPH DATA ---
    print(f"🔹 Saving graph to {output_file}...")
    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
    data.email_to_id = email_to_id
    data.id_to_email = id_to_email
    data.node_stats = node_stats
    data.user_meta = user_meta

    torch.save(data, output_file)
    print(f"✅ Graph saved! Nodes: {len(id_to_email)}, Edges: {edge_index.size(1)}, Node-features: {data.x.shape[1]}, Edge-features: {data.edge_attr.shape[1]}")
    return data

# --- ENTRY POINT ---
if __name__ == "__main__":
    build_graph()
