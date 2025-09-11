import os
import json
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

# Import functions from your graph_eda.py
from backend.app.services.graph_eda import (
    load_nodes_edges,
    build_nx_graph,
    compute_basic_metrics,
    compute_approx_betweenness,
    compute_daily_activity_spikes,
    build_risk_scores,
    DEFAULT_GRAPH_DIR,
    DEFAULT_FULL_DF,
    ANOMALIES_CSV,
    ANOMALIES_JSON
)

# -------------------------
# FastAPI app init
# -------------------------
app = FastAPI(
    title="Internal Eye – Graph Risk API",
    description="REST API to serve graph-based anomaly detection results",
    version="1.0.0",
)

# Cache results to avoid recomputing every request
risk_cache = None
nodes_cache = None
edges_cache = None

# -------------------------
# Utility function
# -------------------------
def compute_and_cache_results(
    graph_dir: str = DEFAULT_GRAPH_DIR,
    full_df: str = DEFAULT_FULL_DF,
    top_n_output: int = 100,
    betweenness_k: int = 200,
    z_thresh: float = 3.0
):
    global risk_cache, nodes_cache, edges_cache

    nodes_df, edges_df = load_nodes_edges(graph_dir)
    G = build_nx_graph(nodes_df, edges_df)

    metrics = compute_basic_metrics(G, top_n=20)
    bc = compute_approx_betweenness(G, k=betweenness_k)
    activity_spikes = compute_daily_activity_spikes(full_df, z_thresh=z_thresh)

    risk_df = build_risk_scores(
        nodes_df,
        metrics["degree"],
        metrics["pagerank"],
        bc,
        activity_spikes_df=activity_spikes
    )

    # Save anomalies
    anomalies = risk_df.head(top_n_output)
    anomalies.to_csv(os.path.join(graph_dir, ANOMALIES_CSV), index=False)
    anomalies.to_json(os.path.join(graph_dir, ANOMALIES_JSON), orient="records", indent=2)

    risk_cache = risk_df
    nodes_cache = nodes_df
    edges_cache = edges_df
    return risk_df


# -------------------------
# API Endpoints
# -------------------------

@app.get("/health")
def health_check():
    """Simple health check."""
    return {"status": "ok", "service": "Graph Risk API"}

@app.get("/graph/anomalies")
def get_anomalies(limit: int = Query(50, description="Number of anomalies to return")):
    """Return top anomalies (users with highest risk score)."""
    global risk_cache
    if risk_cache is None:
        risk_cache = compute_and_cache_results()
    df = risk_cache.head(limit)
    return JSONResponse(content=json.loads(df.to_json(orient="records")))

@app.get("/graph/user/{email}")
def get_user_risk(email: str):
    """Return risk score and metrics for a specific user/email."""
    global risk_cache
    if risk_cache is None:
        risk_cache = compute_and_cache_results()
    row = risk_cache[risk_cache["user"].str.lower() == email.lower()]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"User {email} not found")
    return JSONResponse(content=json.loads(row.to_json(orient="records"))[0])

@app.post("/graph/recompute")
def recompute_graph():
    """Force recomputation of graph metrics and anomalies."""
    risk_df = compute_and_cache_results()
    return {"status": "recomputed", "total_users": len(risk_df)}


