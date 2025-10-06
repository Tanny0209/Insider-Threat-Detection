# backend/app/services/train_gnn.py

import argparse
import torch
from torch import optim, nn
from torch_geometric.nn import GAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from backend.app.services.gnn_model import GNNEncoder
from sklearn.metrics import roc_auc_score, average_precision_score


# -----------------------------
# Custom MLP Decoder
# -----------------------------
class MLPDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # reduced dropout
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, edge_index):
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        return self.fc(h).squeeze(-1)


# -----------------------------
# Training function
# -----------------------------
def train(model, optimizer, data, device, epoch, neg_ratio=1.0, clip_value=1.0):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x.to(device), data.edge_index.to(device))

    pos_edge_index = data.edge_label_index.to(device).long()
    pos_edge_label = data.edge_label.to(device).float()

    # Positive loss
    pos_out = model.decoder(z, pos_edge_index).view(-1)
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_out, pos_edge_label)

    # Fixed smarter negative sampling
    num_neg = int(pos_edge_index.size(1) * neg_ratio)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg,
        method="sparse"
    ).to(device)

    neg_out = model.decoder(z, neg_edge_index).view(-1)
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        neg_out, torch.zeros_like(neg_out)
    )

    loss = pos_loss + neg_loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()

    print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f} | NegRatio: {neg_ratio:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    return loss.item()


# -----------------------------
# Testing function
# -----------------------------
@torch.no_grad()
def test(model, data, device, stage="Validation"):
    model.eval()
    z = model.encode(data.x.to(device), data.edge_index.to(device))

    edge_index = data.edge_label_index.to(device).long()
    edge_label = data.edge_label.to(device).float()

    pred = model.decoder(z, edge_index).view(-1).sigmoid()
    auc = roc_auc_score(edge_label.cpu(), pred.cpu())
    ap = average_precision_score(edge_label.cpu(), pred.cpu())
    print(f"⚡ {stage} -> AUC: {auc:.4f} | AP: {ap:.4f}")
    return auc, ap


# -----------------------------
# Main function
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyg_path", type=str, default="backend/app/data/pyg_graph_full.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--neg_ratio", type=float, default=1.0)  # fixed negative ratio
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.005)  # lower lr for stability
    parser.add_argument("--num_layers", type=int, default=2)  # configurable depth
    parser.add_argument("--dropout", type=float, default=0.2)  # reduced dropout
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load graph
    data = torch.load(args.pyg_path, map_location=device, weights_only=False)
    print(f"✅ Graph loaded! Nodes: {data.num_nodes}, Features: {data.num_node_features}, Edges: {data.num_edges}")

    # Train/val/test split
    print("🔹 Splitting data into train/val/test sets...")
    transform = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
    print(f" - Training edges: {train_data.edge_label_index.size(1)}")
    print(f" - Validation edges: {val_data.edge_label_index.size(1)}")
    print(f" - Test edges: {test_data.edge_label_index.size(1)}")

    # Build model
    print("🔹 Building GAE model...")
    encoder = GNNEncoder(
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        dropout=args.dropout,
        num_layers=args.num_layers
    )
    model = GAE(encoder, decoder=MLPDecoder(args.out_channels)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training loop
    print("\n🚀 Starting training...")
    best_val_auc = 0
    early_stop_counter = 0
    patience = 7

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_data, device, epoch, neg_ratio=args.neg_ratio, clip_value=args.clip_value)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            val_auc, val_ap = test(model, val_data, device, stage="Validation")
            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"⏹ Early stopping at epoch {epoch}. Best Validation AUC: {best_val_auc:.4f}")
                break

    print("\n🎯 Training completed!")

    # Test set evaluation
    print("\n🔹 Evaluating on test set...")
    test(model, test_data, device, stage="Test")

    # Save model + graph data
    model_save_path = "backend/app/data/gnn_trained_model.pt"
    data_save_path = "backend/app/data/graph_data.pt"

    torch.save(model.state_dict(), model_save_path)
    torch.save(data, data_save_path)

    print(f"💾 Model saved to {model_save_path}")
    print(f"💾 Graph data saved to {data_save_path}")


if __name__ == "__main__":
    main()
