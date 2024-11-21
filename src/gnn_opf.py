import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from utils.loss import compute_branch_powers

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

print(data['bus', 'ac_line', 'bus'].edge_index.size(1))

with torch.no_grad(): # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
    # Train with MSE loss for one epoch.
    # In reality we would need to account for AC-OPF constraints.
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

for data in training_loader:
    print(data)
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)

    loss_generator = F.mse_loss(out['generator'], data['generator'].y)
    loss_bus = F.mse_loss(out['bus'], data['bus'].y)

    loss_supervised = loss_bus + loss_generator
    branch_powers = compute_branch_powers(out, data, 'transformer')
    print(branch_powers[0])
    #print(out['generator'].shape)

    print(f"Loss: {loss_supervised}")
    loss_supervised.backward()
    optimizer.step()
    break
