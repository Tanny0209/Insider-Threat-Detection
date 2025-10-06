# backend/app/services/smart_negative_sampling.py

import torch
from torch import nn, optim
from torch_geometric.nn import GAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from backend.app.services.gnn_model import GNNEncoder
import os

# -----------------------------
# Custom MLP Decoder
# -----------------------------
class MLPDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, edge_index):
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        return self.fc(h).squeeze(-1)

# -----------------------------
# Training step
# -----------------------------
def train_step(model, optimizer, data, device, neg_ratio=2.0, clip_value=1.0):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x.to(device), data.edge_index.to(device))

    pos_edge_index = data.edge_label_index.to(device).long()
    pos_edge_label = data.edge_label.to(device).float()

    pos_out = model.decoder(z, pos_edge_index).view(-1)
    pos_loss = nn.functional.binary_cross_entropy_with_logits(pos_out, pos_edge_label)

    num_neg = int(pos_edge_index.size(1) * neg_ratio)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg,
        method="sparse"
    ).to(device)
    neg_out = model.decoder(z, neg_edge_index).view(-1)
    neg_loss = nn.functional.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

    loss = pos_loss + neg_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
    return loss.item()

# -----------------------------
# Compute anomaly scores
# -----------------------------
@torch.no_grad()
def compute_anomaly_scores(model, data, device):
    model.eval()
    z = model.encode(data.x.to(device), data.edge_index.to(device))
    edge_index = data.edge_index.to(device)
    scores = model.decoder(z, edge_index).view(-1).sigmoid()
    return scores, edge_index

# -----------------------------
# Get top suspicious edges
# -----------------------------
def get_top_suspicious_edges(scores, edge_index, top_k=20):
    top_indices = torch.topk(scores, k=top_k).indices
    top_edges = edge_index[:, top_indices]
    top_scores = scores[top_indices]
    return top_edges, top_scores

# -----------------------------
# Main
# -----------------------------
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRAPH_PATH = "backend/app/data/graph_data.pt"
    MODEL_PATH = "backend/app/data/gnn_trained_model.pt"
    OUTPUT_PATH = "backend/app/data/anomalies_scores.pt"

    print(f"Using device: {DEVICE}")

    # Load graph
    data = torch.load(GRAPH_PATH, map_location=DEVICE, weights_only=False)
    print(f"✅ Graph loaded! Nodes: {data.num_nodes}, Features: {data.num_node_features}, Edges: {data.num_edges}")

    # Split edges for negative sampling
    split = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = split(data)
    train_data, val_data, test_data = train_data.to(DEVICE), val_data.to(DEVICE), test_data.to(DEVICE)

    # Build model
    encoder = GNNEncoder(
        in_channels=data.num_node_features,
        hidden_channels=128,
        out_channels=64,
        dropout=0.2,
        num_layers=2
    )
    model = GAE(encoder, decoder=MLPDecoder(64)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Model loaded!")

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # Train step (optional, one epoch enough)
    print("\n🚀 Starting smart negative sampling training...")
    train_step(model, optimizer, train_data, DEVICE, neg_ratio=2.0)

    # Compute anomaly scores
    scores, edge_index = compute_anomaly_scores(model, data, DEVICE)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(scores, OUTPUT_PATH)
    print(f"💾 Anomaly scores saved to {OUTPUT_PATH}")

    # Show top suspicious edges
    top_edges, top_scores = get_top_suspicious_edges(scores, edge_index, top_k=20)
    print("\n🔥 Top suspicious edges:")
    for i in range(top_edges.size(1)):
        print(f"Edge: {top_edges[0,i].item()} -> {top_edges[1,i].item()}, Score: {top_scores[i]:.4f}")

if __name__ == "__main__":
    main()
