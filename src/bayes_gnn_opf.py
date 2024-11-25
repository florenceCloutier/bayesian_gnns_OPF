import torch
import wandb
from torch_geometric.nn import to_hetero
import torch.nn.functional as F
from torch.optim import Adam

from models.bayesian_gnn import BayesianGNN, custom_to_hetero
from utils.data.load import OPFDataModule
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

# These indices are based on OPF dataset paper: https://arxiv.org/pdf/2406.07234
BRANCH_FEATURE_INDICES = {
    'ac_line': {
        'angmin': 0,
        'angmax': 1,
        'b_fr': 2,
        'b_to': 3,
        'R_ij': 4,
        'X_ij': 5,
        'rate_a': 6,
    },
    'transformer': {
        'angmin': 0,
        'angmax': 1,
        'b_fr': 9,
        'b_to': 10,
        'R_ij': 2,
        'X_ij': 3,
        'rate_a': 4,
        'tap': 7,
        'shift': 8,
    }
}

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
    
    print(pf_pred_ac_line.shape)
    print(pf_true_ac_line.shape)
    
    pf_pred_ac_line_loss = F.mse_loss(pf_pred_ac_line, pf_true_ac_line)
    pf_pred_transformer_loss = F.mse_loss(pf_pred_transformer, pf_true_transformer)

    loss_pf = F.mse_loss(pf_pred_ac_line, pf_true_ac_line) + F.mse_loss(pf_pred_transformer, pf_true_transformer)
    loss_qf = F.mse_loss(qf_pred_ac_line, qf_true_ac_line) + F.mse_loss(qf_pred_transformer, qf_true_transformer)
    loss_pt = F.mse_loss(pt_pred_ac_line, pt_true_ac_line) + F.mse_loss(pt_pred_transformer, pt_true_transformer)
    loss_qt = F.mse_loss(qt_pred_ac_line, qt_true_ac_line) + F.mse_loss(qt_pred_transformer, qt_true_transformer)

    total_loss = loss_generator + loss_bus + loss_pf + loss_qf + loss_pt + loss_qt

    return total_loss

def compute_branch_powers(out, data, type):
    voltage_magnitude = out['bus'][:, 1] # |V|
    voltage_angle = out['bus'][:, 0] # theta in radians

    # Compute complex bus voltages
    V = voltage_magnitude * torch.exp(1j * voltage_angle)

    # extract edge attributes
    edge_index = data['bus', type, 'bus'].edge_index
    edge_attr = data['bus', type, 'bus'].edge_attr

    # Extract branch features 
    indices = BRANCH_FEATURE_INDICES[type]
    b_fr = edge_attr[:, indices['b_fr']]
    b_to = edge_attr[:, indices['b_to']]
    R_ij = edge_attr[:, indices['R_ij']]
    X_ij = edge_attr[:, indices['X_ij']]

    # unified tap ratio and shift initialization
    T_ij = torch.ones(edge_attr.shape[0], dtype=torch.complex64)
    shift_rad = torch.zeros(edge_attr.shape[0])

    if type == 'transformer':
      tap = edge_attr[:, indices['tap']]
      shift_rad = edge_attr[:, indices['shift']]
      # shift_rad = shift * (torch.pi / 180)   # I think shift is already in radians?
      T_ij = tap * torch.exp(1j * shift_rad)
    else:
       tap = torch.ones(edge_attr.shape[0])

    # Series admittance
    Z_ij = R_ij + 1j * X_ij
    Y_ij = 1 / Z_ij

    # Shunt admittance (not 100% about this part, need to double check averaging is right)
    # B_c = (b_fr + b_to) / 2 # Why are we averaging? Shouldnt we just be using b_fr
    Y_c = 1j * b_fr

    # From and to Buses
    from_bus = edge_index[0]
    to_bus = edge_index[1]

    # Complex voltages
    V_i = V[from_bus]
    V_j = V[to_bus]

    # Voltage magnitudes squared
    # V_i_abs_squared = torch.abs(V_i) ** 2 # Should we be getting these from our bus node vm outputs
    # V_j_abs_squared = torch.abs(V_j) ** 2

    # Voltage products
    Vi_Vj_conj = V_i * V_j.conj()
    Vi_conj_Vj = V_i.conj() * V_j

    # Equation 9 (not sure but i think for ac_line, we set Tij to 1)
    S_ij = (Y_ij + Y_c).conj() * ((voltage_magnitude[from_bus])**2 / tap**2) - Y_ij.conj() * (Vi_Vj_conj / T_ij)  # Should we just use tap instead of using .abs

    # Equation 10
    S_ji = (Y_ij + Y_c).conj() * (voltage_magnitude[to_bus])**2 - Y_ij.conj() * (Vi_conj_Vj / T_ij) # Why is it T_ij.conj() here

    pf = S_ij.real
    qf = S_ij.imag
    pt = S_ji.real
    qt = S_ji.imag

    return pf, qf, pt, qt

"""
Bound constraints (6)-(7) from CANOS
y = sigmoid(y) * (y_upper - y_lower) + y_lower
"""
def enforce_bound_constraints(out, data):
    vmin = data['bus'].x[:, 2]
    vmax = data['bus'].x[:, 3]
    pmin = data['generator'].x[:, 2]
    pmax = data['generator'].x[:, 3]
    qmin = data['generator'].x[:, 5]
    qmax = data['generator'].x[:, 6]

    out['bus'][:, 1] = torch.sigmoid(out['bus'][:, 1]) * (vmax - vmin) + vmin
    out['generator'][:, 0] = torch.sigmoid(out['generator'][:, 0]) * (pmax - pmin) + pmin
    out['generator'][:, 1] = torch.sigmoid(out['generator'][:, 1]) * (qmax - qmin) + qmin


# Define hyperparameters
learning_rate = 0.01
epochs = 100
kl_weight = 0.1  # Weight for KL divergence loss
# Load the 14-bus OPFData FullTopology dataset training split and store it in the
# directory 'data'. Each record is a `torch_geometric.data.HeteroData`.
train_ds = OPFDataset("/network/scratch/f/florence.cloutier", case_name="pglib_opf_case14_ieee", split="train")
# Batch and shuffle.
batch_size = 4
training_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# Initialise the model.
# data.metadata() here refers to the PyG graph metadata, not the OPFData metadata.
data = train_ds[0]
model = custom_to_hetero(BayesianGNN(in_channels=-1, hidden_channels=16, out_channels=2), data.metadata())

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        # In reality we would need to account for AC-OPF constraints.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()

    for batch in training_loader:
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x_dict, batch.edge_index_dict)
        
        enforce_bound_constraints(out, batch)
        
        branch_powers_ac_line = compute_branch_powers(out, data, 'ac_line')
        branch_powers_transformer = compute_branch_powers(out, data, 'transformer')
        
        pred_loss = compute_loss_supervised(out, batch, branch_powers_ac_line, branch_powers_transformer)
        kl_loss = model.kl_loss()
        loss = pred_loss + kl_weight * kl_loss

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()

    # Print epoch statistics
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")




# for data in training_loader:
#     optimizer.zero_grad()
#     out = model(data.x_dict, data.edge_index_dict)
#     kl_loss = model.kl_loss()
#     ce_loss = criterion(out["generator"], data["generator"].y)
#     loss = ce_loss + kl_loss / batch_size
#     print(f"Loss: {loss}")

#     loss.backward()
#     optimizer.step()


# def train_model():
#     wandb_logger = WandbLogger(entity="florence-cloutier-mila", project="PGM_project", offline=False)

#     torch.set_float32_matmul_precision("high")

#     torch.multiprocessing.set_sharing_strategy("file_system")
    
#     # Load OPF dataset
#     dataset = OPFDataModule('pglib_opf_case14_ieee', '/network/scratch/f/florence.cloutier')
    
#     # Create heterogeneous model
#     model = custom_to_hetero(BayesianGNN(in_channels=-1, hidden_channels=16, out_channels=2), dataset.train_dataset[0].metadata())
    
#     trainer = Trainer(
#         deterministic=True,
#         accelerator='gpu',
#         max_epochs=5,
#         gradient_clip_val=1.0,
#         logger=wandb_logger,
#     )

#     trainer.fit(model, dataset)

#     print('Heterogeneous GAT Model done')
    
# def main():
#     torch.set_float32_matmul_precision("high")

#     torch.multiprocessing.set_sharing_strategy("file_system")
    
#     train_model()

# if __name__ == "__main__":
#     wandb.init(entity= "florence-cloutier-mila", project="PGM_project")
#     main()
