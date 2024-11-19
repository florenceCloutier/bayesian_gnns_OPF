import torch
from torch_geometric.nn import to_hetero

from models.bayesian_gnn import BayesianGNN, custom_to_hetero
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

# Load the 14-bus OPFData FullTopology dataset training split and store it in the
# directory 'data'. Each record is a `torch_geometric.data.HeteroData`.
train_ds = OPFDataset("data", case_name="pglib_opf_case14_ieee", split="train")
# Batch and shuffle.
batch_size = 4
training_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# Initialise the model.
# data.metadata() here refers to the PyG graph metadata, not the OPFData metadata.
data = train_ds[0]
model = custom_to_hetero(BayesianGNN(in_channels=-1, hidden_channels=16, out_channels=2), data.metadata())

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
    # In reality we would need to account for AC-OPF constraints.
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

for data in training_loader:
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    kl_loss = model.kl_loss()
    ce_loss = criterion(out["generator"], data["generator"].y)
    loss = ce_loss + kl_loss / batch_size
    print(f"Loss: {loss}")

    loss.backward()
    optimizer.step()
