# backend/app/services/graph_builder.py
import os, ast
import pandas as pd
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

INPUT_CSV = "backend/app/data/Final_Emails.csv"
OUT_DIR = "backend/app/data/graph_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

def parse_list_field(cell):
    if pd.isna(cell) or cell == "":
        return []
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return [x.lower() for x in ast.literal_eval(s)]
        except Exception:
            pass
    # fallback split
    parts = [p.strip().lower() for p in s.replace(";",",").split(",") if p.strip()]
    return parts

def build_graph():
    print("Loading CSV (this may take a while)...")
    df = pd.read_csv(INPUT_CSV, usecols=['email_id','date','from_user','to_users','cc_users','bcc_users','subject_length','body_length','is_external','dataset_source'], parse_dates=['date'], low_memory=False)
    print("Rows:", len(df))

    G = nx.DiGraph()
    edge_aggregates = defaultdict(lambda: {"count":0, "total_subject_len":0, "total_body_len":0, "external_count":0})
    node_stats = defaultdict(lambda: {"sent":0, "received":0, "total_subject_len_sent":0, "total_body_len_sent":0, "external_sent":0})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph"):
        sender = str(row['from_user']).lower()
        to_list = parse_list_field(row.get('to_users', ''))
        cc_list = parse_list_field(row.get('cc_users', ''))
        bcc_list = parse_list_field(row.get('bcc_users', ''))
        recipients = to_list + cc_list + bcc_list
        subj_len = int(row.get('subject_length') or 0)
        body_len = int(row.get('body_length') or 0)
        is_external = bool(row.get('is_external') == True or str(row.get('is_external')).lower() == 'true')

        # update node stats for sender
        node_stats[sender]['sent'] += 1
        node_stats[sender]['total_subject_len_sent'] += subj_len
        node_stats[sender]['total_body_len_sent'] += body_len
        node_stats[sender]['external_sent'] += int(is_external)

        for r in recipients:
            if not r:
                continue
            r = r.lower()
            edge_key = (sender, r)
            agg = edge_aggregates[edge_key]
            agg['count'] += 1
            agg['total_subject_len'] += subj_len
            agg['total_body_len'] += body_len
            agg['external_count'] += int(is_external)
            node_stats[r]['received'] += 1

    # create graph and write edges
    nodes = list(node_stats.keys())
    node_to_id = {n:i for i,n in enumerate(nodes)}

    edges_rows = []
    for (s,r), agg in edge_aggregates.items():
        edges_rows.append({
            "source": s,
            "target": r,
            "source_id": node_to_id[s],
            "target_id": node_to_id[r],
            "count": agg['count'],
            "avg_subject_len": agg['total_subject_len'] / agg['count'] if agg['count'] else 0,
            "avg_body_len": agg['total_body_len'] / agg['count'] if agg['count'] else 0,
            "external_count": agg['external_count']
        })

    nodes_rows = []
    for n, stats in node_stats.items():
        sent = stats['sent']
        nodes_rows.append({
            "user": n,
            "node_id": node_to_id[n],
            "sent": sent,
            "received": stats['received'],
            "avg_subject_len_sent": stats['total_subject_len_sent'] / sent if sent else 0,
            "avg_body_len_sent": stats['total_body_len_sent'] / sent if sent else 0,
            "external_sent_ratio": stats['external_sent'] / sent if sent else 0
        })

    edges_df = pd.DataFrame(edges_rows)
    nodes_df = pd.DataFrame(nodes_rows)
    mapping_df = pd.DataFrame(list(node_to_id.items()), columns=['user','node_id'])

    edges_df.to_csv(os.path.join(OUT_DIR, "edges.csv"), index=False)
    nodes_df.to_csv(os.path.join(OUT_DIR, "nodes.csv"), index=False)
    mapping_df.to_csv(os.path.join(OUT_DIR, "user_node_map.csv"), index=False)

    print("Graph built. Nodes:", len(nodes_df), "Edges:", len(edges_df))
    print("Saved to", OUT_DIR)

if __name__ == "__main__":
    build_graph()
