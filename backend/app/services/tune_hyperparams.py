import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit


# -------------------------------------------------
# ✅ Try importing your real model
# -------------------------------------------------
try:
    from app.services.gnn_model import GAEModel
except Exception as e:
    print(f"[tuner] Could not import GAEModel — using fallback. Error: {e}")

    class GAEModel(nn.Module):
        """Fallback Graph Autoencoder."""
        def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.3):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.dropout = dropout
            self.decoder = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )

        def encode(self, x, edge_index):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        def decode(self, z, edge_index):
            src, dst = edge_index
            h = torch.cat([z[src], z[dst]], dim=-1)
            return self.decoder(h).view(-1)

        def forward(self, x, edge_index):
            return self.encode(x, edge_index)


# -------------------------------------------------
# ✅ Universal Edge Getter (handles all PyG versions)
# -------------------------------------------------
def get_edges(data, split="pos"):
    """
    Returns edge indices for given split ('pos' or 'neg').
    Supports both old and new PyG versions.
    """
    # New PyG: edge_label_index + edge_label
    if hasattr(data, "edge_label_index"):
        if split == "pos":
            mask = data.edge_label == 1
        else:
            mask = data.edge_label == 0
        return data.edge_label_index[:, mask]

    # Old PyG: pos_edge_label_index / neg_edge_label_index
    if split == "pos":
        if hasattr(data, "pos_edge_label_index"):
            return data.pos_edge_label_index
        elif hasattr(data, "pos_edge_index"):
            return data.pos_edge_index
    else:
        if hasattr(data, "neg_edge_label_index"):
            return data.neg_edge_label_index
        elif hasattr(data, "neg_edge_index"):
            return data.neg_edge_index

    raise AttributeError("Edge labels not found in this data split.")


# -------------------------------------------------
# ✅ Objective for Optuna tuning
# -------------------------------------------------
def objective(trial, data_splits, out_dir, epochs=50, device="cpu"):
    train_data, val_data, test_data = data_splits
    neg_ratio = trial.suggest_int("neg_ratio", 1, 5)
    hidden_channels = trial.suggest_int("hidden_channels", 16, 128)
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    in_channels = train_data.x.size(-1)
    model = GAEModel(in_channels, hidden_channels, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))

        pos_edges = get_edges(train_data, "pos").to(device)
        neg_edges_all = get_edges(train_data, "neg")
        if neg_edges_all.size(1) > neg_ratio:
            idx = torch.randint(0, neg_edges_all.size(1), (neg_ratio,))
            neg_edges = neg_edges_all[:, idx].to(device)
        else:
            neg_edges = neg_edges_all.to(device)

        pos_pred = model.decode(z, pos_edges)
        neg_pred = model.decode(z, neg_edges)

        loss = F.binary_cross_entropy(
            torch.cat([pos_pred, neg_pred]),
            torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        )
        loss.backward()
        optimizer.step()

    # Validation AUC
    model.eval()
    with torch.no_grad():
        z = model.encode(val_data.x.to(device), val_data.edge_index.to(device))
        pos_edges = get_edges(val_data, "pos").to(device)
        neg_edges = get_edges(val_data, "neg").to(device)
        pos_pred = model.decode(z, pos_edges)
        neg_pred = model.decode(z, neg_edges)

        y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu()
        y_pred = torch.cat([pos_pred, neg_pred]).cpu()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_pred)
    return auc


# -------------------------------------------------
# ✅ Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_path = args.data_path
    out_dir = os.path.join("backend", "app", "data")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[tuner] Loading data from {data_path}")
    data = torch.load(data_path, weights_only=False)

    print("[tuner] Performing RandomLinkSplit (low-memory safe)...")
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, (train_data, val_data, test_data), out_dir, args.epochs, args.device),
        n_trials=args.n_trials
    )

    print("\n✅ Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    print(f"   Best AUC: {study.best_value:.4f}")

    torch.save(study.best_params, os.path.join(out_dir, "best_hyperparams.pt"))
    print(f"[tuner] Saved best hyperparameters to {out_dir}/best_hyperparams.pt")


if __name__ == "__main__":
    main()
