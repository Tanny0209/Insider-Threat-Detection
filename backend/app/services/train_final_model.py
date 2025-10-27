import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch.serialization import add_safe_globals


# --------------------------------------------------
# 1️⃣ Encoder Definition
# --------------------------------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(GCNEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# --------------------------------------------------
# 2️⃣ Training function
# --------------------------------------------------
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
    loss.backward()
    optimizer.step()
    return float(loss)


# --------------------------------------------------
# 3️⃣ Evaluation helper
# --------------------------------------------------
@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


# --------------------------------------------------
# 4️⃣ Main training pipeline
# --------------------------------------------------
def train_final_model(data_path, hyperparams_path="backend/app/data/best_hyperparams.pt"):
    print("[trainer] Loading data and best hyperparameters...")

    # Safe loading
    add_safe_globals([Data])
    data = torch.load(data_path, weights_only=False)

    # Load hyperparameters (.json or .pt)
    if hyperparams_path.endswith(".json"):
        with open(hyperparams_path, "r") as f:
            best_hyperparams = json.load(f)
    else:
        best_hyperparams = torch.load(hyperparams_path)

    print("[trainer] Loaded best hyperparameters:")
    for k, v in best_hyperparams.items():
        print(f"   {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 5️⃣ Split dataset safely
    # --------------------------------------------------
    print("[trainer] Splitting data safely using RandomLinkSplit...")
    transform = RandomLinkSplit(is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(data)

    torch.save({
        "train": train_data,
        "val": val_data,
        "test": test_data
    }, "backend/app/data/split_graphs.pt")
    print("[trainer] Saving split data for consistent evaluation...")

    # --------------------------------------------------
    # 6️⃣ Model setup
    # --------------------------------------------------
    encoder = GCNEncoder(
        data.num_features,
        best_hyperparams["hidden_channels"],
        best_hyperparams["num_layers"],
        best_hyperparams["dropout"]
    )
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_hyperparams["lr"],
        weight_decay=best_hyperparams["weight_decay"]
    )

    # --------------------------------------------------
    # 7️⃣ Training loop
    # --------------------------------------------------
    print("[trainer] Starting final training...")
    best_val_auc = 0
    best_model_state = None
    patience = 25
    patience_counter = 0
    num_epochs = 200

    for epoch in range(1, num_epochs + 1):
        loss = train(model, optimizer, train_data)
        val_auc, _ = test(model, val_data)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d}: Loss {loss:.4f}, Val AUC {val_auc:.4f}")

        if patience_counter >= patience:
            print(f"[trainer] Early stopping at epoch {epoch} (best AUC={best_val_auc:.4f})")
            break

    # --------------------------------------------------
    # 8️⃣ Save model
    # --------------------------------------------------
    os.makedirs("backend/app/models", exist_ok=True)
    torch.save(best_model_state, "backend/app/models/final_gnn_model.pt")
    torch.save(model.encoder.state_dict(), "backend/app/models/final_encoder.pt")
    print("✅ [trainer] Final model saved to backend/app/models/final_gnn_model.pt")


# --------------------------------------------------
# 9️⃣ Entry point
# --------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to normalized graph_data .pt file")
    parser.add_argument("--hyperparams-path", type=str,
                        default="backend/app/data/best_hyperparams.pt")
    args = parser.parse_args()
    train_final_model(args.data_path, args.hyperparams_path)
