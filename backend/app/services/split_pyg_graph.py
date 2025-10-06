# backend/app/services/split_pyg_graph.py

import torch
from torch_geometric.transforms import RandomLinkSplit

def main():
    pyg_path = "backend/app/data/pyg_graph_full.pt"

    # Load the graph
    data = torch.load(pyg_path, map_location="cpu", weights_only=False)
    print("✅ Loaded PyG graph")
    print(f" - Nodes: {data.num_nodes}")
    print(f" - Node features: {data.num_node_features}")
    print(f" - Edges: {data.num_edges}")

    # Perform edge split (train/val/test)
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = transform(data)

    # Save splits
    torch.save(train_data, "backend/app/data/train_data.pt")
    torch.save(val_data, "backend/app/data/val_data.pt")
    torch.save(test_data, "backend/app/data/test_data.pt")

    print("✅ Train/Val/Test splits saved")
    print(f" - Train edges: {train_data.edge_index.shape[1]}")
    print(f" - Val edges: {val_data.edge_index.shape[1]}")
    print(f" - Test edges: {test_data.edge_index.shape[1]}")

if __name__ == "__main__":
    main()
