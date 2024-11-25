import torch
import hydra
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from omegaconf import DictConfig
from models.bayesian_gnn import BayesianGNN, custom_to_hetero
from utils.metrics import compute_trmae, thermal_limit_violation, voltage_angle_difference, power_balance_violation
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

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

# Custom learning rate schedule function from CANOS paper
def learning_rate_schedule(step, warmup_steps, initial_learning_rate, peak_learning_rate, transition_steps, decay_rate, final_learning_rate):
    if step < warmup_steps:
        # Linear warm-up
        return initial_learning_rate + step * (peak_learning_rate - initial_learning_rate) / warmup_steps
    else:
        # Exponential decay
        decay_steps = (step - warmup_steps) // transition_steps
        decayed_lr = peak_learning_rate * (decay_rate ** decay_steps)
        return max(decayed_lr, final_learning_rate)

"""
Training loop
Hyperparameters from CANOS 
"""
def train_model(train_ds,
                train_log_interval=4,
                initial_learning_rate=0.0, 
                peak_learning_rate=2e-4, 
                final_learning_rate=5e-6, 
                warmup_steps=10000, 
                decay_rate=0.9, 
                transition_steps=4000,
                total_steps=600000,
                epochs=100, 
                kl_weight=0.1):
    
    
    # Batch and shuffle.
    batch_size = 4
    training_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Initialise the model.
    # data.metadata() here refers to the PyG graph metadata, not the OPFData metadata.
    data = train_ds[0]
    model = custom_to_hetero(BayesianGNN(in_channels=-1, hidden_channels=16, out_channels=2), data.metadata())

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: learning_rate_schedule(step, warmup_steps, initial_learning_rate, peak_learning_rate, transition_steps, decay_rate, final_learning_rate))

    # Training loop
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, data in enumerate(training_loader):
            if step >= total_steps:
                break  # Stop training after 600,000 steps

            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            
            enforce_bound_constraints(out, data) 
            branch_powers_ac_line = compute_branch_powers(out, data, 'ac_line')
            branch_powers_transformer = compute_branch_powers(out, data, 'transformer')
            
            loss_supervised = compute_loss_supervised(out, data, branch_powers_ac_line, branch_powers_transformer)
            kl_loss = model.kl_loss()
            loss = loss_supervised + kl_weight * kl_loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            scheduler.step()

            step += 1  # Increment global step
            
            # Log metrics at intervals
            if batch_idx % train_log_interval == 0:
                # TRMAE
                edge_label_ac_line = data['bus', 'ac_line', 'bus'].edge_label
                pf_true_ac_line = edge_label_ac_line[:, 0]
                qf_true_ac_line = edge_label_ac_line[:, 1]
                pt_true_ac_line = edge_label_ac_line[:, 2]
                qt_true_ac_line = edge_label_ac_line[:, 3]
                
                edge_label_transformer = data['bus', 'transformer', 'bus'].edge_label
                pf_true_transformer = edge_label_transformer[:, 0]
                qf_true_transformer = edge_label_transformer[:, 1]
                pt_true_transformer = edge_label_transformer[:, 2]
                qt_true_transformer = edge_label_transformer[:, 3]
                
                pf_pred_ac_line, qf_pred_ac_line, pt_pred_ac_line, qt_pred_ac_line = branch_powers_ac_line
                pf_pred_transformer, qf_pred_transformer, pt_pred_transformer, qt_pred_transformer = branch_powers_transformer
                
                # LR
                current_lr = optimizer.param_groups[0]['lr']
                
                # TRMAE
                trmae_to_bus_real_batch = compute_trmae(pt_true_ac_line, pt_pred_ac_line)
                trmae_to_bus_reactive_batch = compute_trmae(qt_true_ac_line, qt_pred_ac_line)
                trmae_from_bus_real_batch = compute_trmae(pf_true_ac_line, pf_pred_ac_line)
                trmae_from_bus_reactive_batch = compute_trmae(qf_true_ac_line, qf_pred_ac_line)
                
                trmae_generator_withdrawn_to_bus_real_batch = compute_trmae(pt_true_transformer, pt_pred_transformer)
                trmae_generator_withdrawn_to_bus_reactive_batch = compute_trmae(qt_true_transformer, qt_pred_transformer)
                trmae_generator_withdrawn_from_bus_real_batch = compute_trmae(pf_true_transformer, pf_pred_transformer)
                trmae_generator_withdrawn_from_bus_reactive_batch = compute_trmae(qf_true_transformer, qf_pred_transformer)
                
                # Voltage angles
                voltage_angle_bus = out['bus'][:, 0] # theta in radians
                indices_bus_angmin = BRANCH_FEATURE_INDICES['ac_line']['angmin']
                indices_bus_angmax = BRANCH_FEATURE_INDICES['ac_line']['angmax']
                edge_attr_ac_line = data['bus', 'ac_line', 'bus'].edge_attr
                bus_angmin = edge_attr_ac_line[:, indices_bus_angmin]
                bus_angmax = edge_attr_ac_line[:, indices_bus_angmax]

                voltage_angle_node = out['generator'][:, 0] # theta in radians
                indices_node_angmin = BRANCH_FEATURE_INDICES['transformer']['angmin']
                indices_node_angmax = BRANCH_FEATURE_INDICES['transformer']['angmax']
                edge_attr_transformer = data['bus', 'transformer', 'bus'].edge_attr
                node_angmin = edge_attr_transformer[:, indices_node_angmin]
                node_angmax = edge_attr_transformer[:, indices_node_angmax]
                
                # Thermal limits
                indices_ac_line = BRANCH_FEATURE_INDICES['ac_line']
                rate_a_ac_line = edge_attr_ac_line[:, indices_ac_line['rate_a']]
                indices_transformer = BRANCH_FEATURE_INDICES['transformer']
                rate_a_transformer = edge_attr_transformer[:, indices_transformer['rate_a']]
                
                # Power balance
                power_balance = power_balance_violation(pf_true_ac_line, pt_true_ac_line, pf_pred_ac_line, pt_pred_ac_line, pf_true_transformer, pt_true_transformer, pf_pred_transformer, pt_pred_transformer),
                
                metrics = {
                    'loss': loss.item(),
                    'lr': current_lr,
                    'trmae_to_bus_real_batch': trmae_to_bus_real_batch,
                    'trmae_to_bus_reactive_batch': trmae_to_bus_reactive_batch,
                    'trmae_from_bus_real_batch': trmae_from_bus_real_batch,
                    'trmae_from_bus_reactive_batch': trmae_from_bus_reactive_batch,
                    'trmae_bus_total': trmae_to_bus_real_batch + trmae_to_bus_reactive_batch + trmae_from_bus_real_batch + trmae_from_bus_reactive_batch,
                    'trmae_generator_withdrawn_to_bus_real_batch': trmae_generator_withdrawn_to_bus_real_batch,
                    'trmae_generator_withdrawn_to_bus_reactive_batch': trmae_generator_withdrawn_to_bus_reactive_batch,
                    'trmae_generator_withdrawn_from_bus_real_batch': trmae_generator_withdrawn_from_bus_real_batch,
                    'trmae_generator_withdrawn_from_bus_reactive_batch': trmae_generator_withdrawn_from_bus_reactive_batch,
                    'trma_generator_total': trmae_generator_withdrawn_to_bus_real_batch + trmae_generator_withdrawn_to_bus_reactive_batch + trmae_generator_withdrawn_from_bus_real_batch + trmae_generator_withdrawn_from_bus_reactive_batch,
                    'power_balance_violation_ac_line': power_balance[0][0],
                    'power_balance_violation_trans': power_balance[0][1],
                    'voltage_angle_difference_node': voltage_angle_difference(voltage_angle_node, [node_angmin, node_angmax]),
                    'voltage_angle_difference_bus': voltage_angle_difference(voltage_angle_bus, [bus_angmin, bus_angmax]),
                    'thermal_limits': thermal_limit_violation(pf_pred_ac_line, qf_pred_ac_line, pf_pred_transformer, qf_pred_transformer, rate_a_ac_line, rate_a_transformer),
                }
                wandb.log(metrics)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print(f"Training completed at step {step}/{total_steps}.")

@hydra.main(config_path='../cfgs', config_name='bayes_gnn_opf', version_base=None)  
def main(cfg: DictConfig):
    # Load the 14-bus OPFData FullTopology dataset training split and store it in the
    # directory 'data'. Each record is a `torch_geometric.data.HeteroData`.
    train_ds = OPFDataset(cfg.data_dir, case_name=cfg.case_name, split="train")
    train_model(train_ds)


if __name__ == "__main__":
    wandb.init(entity= "real-lab", project="PGM_bayes_gnn_opf")
    main()
