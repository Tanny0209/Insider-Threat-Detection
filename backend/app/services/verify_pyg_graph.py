# backend/app/services/verify_pyg_graph.py

import torch
from torch_geometric.data import Data  # Needed so torch.load knows about Data


def main():
    pyg_path = "backend/app/data/pyg_graph_full.pt"

    print("🔍 Loading PyG graph from:", pyg_path)

    # ✅ Safe load for PyTorch ≥2.6
    data: Data = torch.load(pyg_path, weights_only=False)

    print("\n=== Graph Summary ===")
    print(data)

    print("\n=== Key Properties ===")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of node features: {data.num_node_features}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of edge features: {data.num_edge_features}")

    if data.edge_index.shape[1] != data.edge_attr.shape[0]:
        print("\n⚠️ [WARNING] edge_index count does not match edge_attr rows!")
    else:
        print("\n✅ Edge index and attributes match correctly.")

    # Check if node features exist
    if hasattr(data, "x") and data.x is not None:
        print("\n=== Sample Node Features (first 5) ===")
        print(data.x[:5])
    else:
        print("\nℹ️ No node features found in this graph.")

    # Check if edge attributes exist
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        print("\n=== Sample Edges (edge_index + edge_attr, first 5) ===")
        print("edge_index:", data.edge_index[:, :5])
        print("edge_attr:", data.edge_attr[:5])
    else:
        print("\nℹ️ No edge attributes found in this graph.")


if __name__ == "__main__":
    main()
