import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCNConv
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import json
import os
import numpy as np
from contextlib import nullcontext


# Simple GCN Encoder used for GAE
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def evaluate_model():
    print("[evaluator] Loading model and data...")

    data_path = "backend/app/data/graph_data_normalized.pt"
    model_path = "backend/app/data/final_gnn_model.pt"
    split_path = "backend/app/data/split_graphs.pt"
    output_metrics_path = "backend/app/data/test_metrics.json"

    # Allow safe torch_geometric classes
    import torch_geometric
    torch.serialization.add_safe_globals([
        torch_geometric.data.data.Data,
        torch_geometric.data.data.DataEdgeAttr
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data to infer dimensions
    full_data = torch.load(data_path, weights_only=False)
    in_channels = full_data.x.size(1)

    # Rebuild encoder and model exactly like training
    hidden_channels = 28
    num_layers = 4
    dropout = 0.3
    out_channels = hidden_channels

    encoder = GCNEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout)
    model = GAE(encoder).to(device)

    # ✅ Load model weights properly
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load split data safely
    with nullcontext():
        split_data = torch.load(split_path, weights_only=False)

    test_data = split_data["test"]

    print("[evaluator] Computing embeddings and link probabilities...")

    # Encode node features
    z = model.encode(test_data.x.to(device), test_data.edge_index.to(device))

    pos_edges = test_data.pos_edge_label_index
    neg_edges = test_data.neg_edge_label_index

    pos_pred = model.decoder(z, pos_edges).detach().cpu()
    neg_pred = model.decoder(z, neg_edges).detach().cpu()

    y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).numpy()
    y_pred = torch.cat([pos_pred, neg_pred]).numpy()

    auc = roc_auc_score(y_true, y_pred)

    # Find best F1 threshold
    thresholds = np.linspace(0.1, 0.9, 17)
    best_metrics = {}
    best_f1 = -1

    for t in thresholds:
        y_pred_bin = (y_pred > t).astype(int)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        rec = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        acc = accuracy_score(y_true, y_pred_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                "Threshold": round(float(t), 3),
                "AUC": float(auc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1": float(f1),
                "Accuracy": float(acc)
            }

    print("✅ [evaluator] Evaluation complete!")
    print(f"📊 Best Threshold: {best_metrics['Threshold']}")
    print(json.dumps(best_metrics, indent=4))

    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=4)

    print(f"[evaluator] Metrics saved to: {output_metrics_path}")


if __name__ == "__main__":
    evaluate_model()
