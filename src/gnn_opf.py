import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from utils.loss import compute_branch_powers, enforce_bound_constraints, power_balance_loss, flow_loss, voltage_angle_loss

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


"""
The loss_supervised function from the CANOS paper aggregates the L2 losses for the bus voltage,
the generator power, and the branch power between the predicted values and the targets.
"""
def compute_loss_supervised(out, data, branch_powers_ac_line, branch_powers_transformer):
    loss_generator = F.mse_loss(out['generator'], data['generator'].y)

    loss_bus = F.mse_loss(out['bus'], data['bus'].y)

    pf_pred_ac_line, qf_pred_ac_line, pt_pred_ac_line, qt_pred_ac_line = branch_powers_ac_line
    pf_pred_transformer, qf_pred_transformer, pt_pred_transformer, qt_pred_transformer = branch_powers_transformer

    assert 'edge_label' in data['bus', 'ac_line', 'bus'], "Edge label for AC lines is missing."
    edge_label_ac_line = data['bus', 'ac_line', 'bus'].edge_label
    pf_true_ac_line = edge_label_ac_line[:, 0]
    qf_true_ac_line = edge_label_ac_line[:, 1]
    pt_true_ac_line = edge_label_ac_line[:, 2]
    qt_true_ac_line = edge_label_ac_line[:, 3]

    assert 'edge_label' in data['bus', 'transformer', 'bus'], "Edge label for transformers is missing."
    edge_label_transformer = data['bus', 'transformer', 'bus'].edge_label
    pf_true_transformer = edge_label_transformer[:, 0]
    qf_true_transformer = edge_label_transformer[:, 1]
    pt_true_transformer = edge_label_transformer[:, 2]
    qt_true_transformer = edge_label_transformer[:, 3]

    loss_pf = F.mse_loss(pf_pred_ac_line, pf_true_ac_line) + F.mse_loss(pf_pred_transformer, pf_true_transformer)
    loss_qf = F.mse_loss(qf_pred_ac_line, qf_true_ac_line) + F.mse_loss(qf_pred_transformer, qf_true_transformer)
    loss_pt = F.mse_loss(pt_pred_ac_line, pt_true_ac_line) + F.mse_loss(pt_pred_transformer, pt_true_transformer)
    loss_qt = F.mse_loss(qt_pred_ac_line, qt_true_ac_line) + F.mse_loss(qt_pred_transformer, qt_true_transformer)

    total_loss = loss_generator + loss_bus + loss_pf + loss_qf + loss_pt + loss_qt

    return total_loss


def loss_constraints(out, data, branch_powers_ac_line, branch_powers_transformer):
    pb_loss = power_balance_loss(out, data, branch_powers_ac_line, branch_powers_transformer)
    branch_loss = flow_loss(data, branch_powers_ac_line, branch_powers_transformer)
    va_loss = voltage_angle_loss(out, data)

    return pb_loss + branch_loss + va_loss


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
    #print(data)
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)

    #print(out['bus'])

    # Bound constraints (6) and (7) from CANOS
    enforce_bound_constraints(out, data)
    
    branch_powers_ac_line = compute_branch_powers(out, data, 'ac_line')
    branch_powers_transformer = compute_branch_powers(out, data, 'transformer')



    loss_supervised = compute_loss_supervised(out, data, branch_powers_ac_line, branch_powers_transformer)
    # #print(out['generator'].shape)

    loss_constraint = loss_constraints(out, data, branch_powers_ac_line, branch_powers_transformer)

    loss_total = loss_supervised + 0.1*loss_constraint

    print(f"Loss: {loss_supervised}")
    loss_supervised.backward()
    optimizer.step()

