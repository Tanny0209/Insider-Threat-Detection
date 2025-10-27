import torch
from torch_geometric.data import Data
import os

DATA_PATH = "backend/app/data/graph_data.pt"
OUT_PATH = "backend/app/data/graph_data_normalized.pt"

print("[fix] Loading existing graph...")
data = torch.load(DATA_PATH, weights_only=False)

# --- Normalize node features ---
if hasattr(data, 'x'):
    x = data.x
    print(f"[fix] Normalizing node features: mean={x.mean():.4f}, std={x.std():.4f}")
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    data.x = x
    print("[fix] Node features normalized ✅")

# --- Normalize edge attributes (optional but helpful) ---
if hasattr(data, 'edge_attr') and data.edge_attr is not None:
    edge_attr = data.edge_attr
    print(f"[fix] Normalizing edge attributes: mean={edge_attr.mean():.4f}, std={edge_attr.std():.4f}")
    edge_attr = (edge_attr - edge_attr.mean(dim=0)) / (edge_attr.std(dim=0) + 1e-6)
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=-1.0)
    data.edge_attr = edge_attr
    print("[fix] Edge attributes normalized ✅")

# --- Save the fixed graph ---
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
torch.save(data, OUT_PATH)
print(f"[fix] Saved normalized graph to {OUT_PATH}")
