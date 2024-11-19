import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

# Load the 14-bus OPFData FullTopology dataset training split and store it in the
# directory 'data'. Each record is a `torch_geometric.data.HeteroData`.
train_ds = OPFDataset('data', case_name='pglib_opf_case14_ieee', split='train')
# Batch and shuffle.
training_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

# A simple model to predict the generator active and reactive power outputs.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(-1, 16)
        self.conv2 = GraphConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
# Initialise the model.
# data.metadata() here refers to the PyG graph metadata, not the OPFData metadata.
data = train_ds[0]
model = to_hetero(Model(), data.metadata())

with torch.no_grad(): # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
    # Train with MSE loss for one epoch.
    # In reality we would need to account for AC-OPF constraints.
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

for data in training_loader:
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.mse_loss(out['generator'], data['generator'].y)
    print(f"Loss: {loss}")
    loss.backward()
    optimizer.step()
