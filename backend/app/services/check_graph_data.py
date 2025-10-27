import torch

print("[check] Loading graph_data.pt...")
data = torch.load("backend/app/data/graph_data.pt", weights_only=False)
print(data)

# Check what fields exist
print("\n[check] Data attributes:")
print(data.keys if hasattr(data, 'keys') else dir(data))

# Check feature stats
if hasattr(data, 'x'):
    print("\n[check] Node feature tensor:")
    print("Shape:", data.x.shape)
    print("Mean:", data.x.mean().item())
    print("Std:", data.x.std().item())

# Check edge info
if hasattr(data, 'edge_index'):
    print("\n[check] Edge index:")
    print("Shape:", data.edge_index.shape)
    print("Sample:", data.edge_index[:, :10])
    print("Num edges:", data.edge_index.size(1))
