import torch
import wandb

from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from utils.loss import enforce_bound_constraints, compute_branch_powers, compute_loss_supervised, learning_rate_schedule, cost
from utils.metrics import compute_metrics


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
Training loop
Hyperparameters from CANOS 
"""
def train_eval_model(model,
                train_ds,
                eval_ds,
                constraints,
                lambdas,
                device,
                rho=0.01,
                train_log_interval=100,
                initial_learning_rate=0.0, 
                peak_learning_rate=2e-4, 
                final_learning_rate=5e-6, 
                warmup_steps=10000, 
                decay_rate=0.9, 
                transition_steps=4000,
                total_steps=600000,
                epochs=100, 
                batch_size=4
    ):
    
    
    # Batch and shuffle.
    training_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    
    data = train_ds[0]
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: learning_rate_schedule(step, warmup_steps, initial_learning_rate, peak_learning_rate, transition_steps, decay_rate, final_learning_rate))
        
    # Training loop
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        if step >= total_steps:
            break  # Stop training after 600,000 steps
        lambdas = learning_step(model, optimizer, training_loader, eval_loader, lambdas, constraints, train_log_interval, lr_scheduler, rho, step, device=device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print(f"Training completed at step {step}/{total_steps}.")


def learning_step(model, optimizer, data_loader, eval_loader, lambdas, constraints, train_log_interval, lr_scheduler, rho, step, device):
    for batch_idx, data in enumerate(data_loader):
        step += 1
        model.train()
        data = data.to(device)
        optimizer.zero_grad()

        # forward pass
        # out = model(data.x_dict, data.edge_index_dict)
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

        # Bound constraints (6) and (7) from CANOS
        enforce_bound_constraints(out, data)

        # compute branch powers
        branch_powers_ac_line = compute_branch_powers(out, data, 'ac_line', device)
        branch_powers_transformer = compute_branch_powers(out, data, 'transformer', device)

        # supervised loss
        l_supervised = compute_loss_supervised(out, data, branch_powers_ac_line, branch_powers_transformer)

        # constraint losses
        l_constraints = 0.0
        metrics = {}
        for name, constraint_fn in constraints.items():
            if name == "power_balance":
                violation = constraint_fn(out, data, branch_powers_ac_line, branch_powers_transformer, device)
                metrics['power_balance_loss'] = violation
            elif name == "flow":
                violation = constraint_fn(data, branch_powers_ac_line, branch_powers_transformer)
                metrics['flow_loss'] = violation
            else:
                violation = constraint_fn(out, data)
                metrics['voltage_angle_loss'] = violation
            l_constraints += lambdas[name] * violation
        
        # Total loss
        total_loss = l_supervised + 0.1 * l_constraints
        
        # Log metrics at intervals
        if batch_idx % train_log_interval == 0:
            metrics = compute_metrics(data, out, branch_powers_ac_line, branch_powers_transformer, optimizer, BRANCH_FEATURE_INDICES, "train")
            metrics['train_loss'] = total_loss.item()
            metrics["train_cost"] = cost(out, data)
            eval_metrics = evaluate_model(model, eval_loader, constraints, lambdas, optimizer, device)
            metrics.update(eval_metrics)
            wandb.log(metrics)
        
        
        # Backprop and optimization
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
    
    # compute the violation degrees
    model.eval()
    violation_degrees = {k: 0.0 for k in constraints.keys()}
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict)
        for name, constraint_fn in constraints.items():
            if name == "power_balance":
                violation = constraint_fn(out, data, branch_powers_ac_line, branch_powers_transformer, device)
            elif name == "flow":
                violation = constraint_fn(data, branch_powers_ac_line, branch_powers_transformer)
            else:
                violation = constraint_fn(out, data)
            violation_degrees[name] += violation

    # update lambdas
    for name in lambdas.keys():
        lambdas[name] = lambdas[name] + rho * (violation_degrees[name].detach()/len(data_loader)) # detach ensures no gradient tracking
    

    return lambdas


def evaluate_model(model, eval_loader, constraints, lambdas, optimizer, device):
    model.eval()
    metrics = [] # Logging metrics

    with torch.no_grad():
        for data in eval_loader:
            data = data.to(device)
            out = model(data.x_dict, data.edge_index_dict)
            enforce_bound_constraints(out, data)
            
            branch_powers_ac_line = compute_branch_powers(out, data, 'ac_line', device)
            branch_powers_transformer = compute_branch_powers(out, data, 'transformer', device)
            
            # Compute supervised and KL loss
            l_supervised = compute_loss_supervised(out, data, branch_powers_ac_line, branch_powers_transformer)
            
            # constraint losses
            violation_degrees = {}
            l_constraints = 0.0
            metrics = {}
            for name, constraint_fn in constraints.items():
                if name == "power_balance":
                    violation = constraint_fn(out, data, branch_powers_ac_line, branch_powers_transformer, device)
                    metrics['val_power_balance_loss'] = violation
                elif name == "flow":
                    violation = constraint_fn(data, branch_powers_ac_line, branch_powers_transformer)
                    metrics['val_flow_loss'] = violation
                else:
                    violation = constraint_fn(out, data)
                    metrics['val_voltage_angle_loss'] = violation
                violation_degrees[name] = violation
                l_constraints += lambdas[name] * violation
        
            # Total loss
            total_loss = l_supervised + 0.1 * l_constraints
            
            # Compute metrics
            metrics = compute_metrics(data, out, branch_powers_ac_line, branch_powers_transformer, optimizer, BRANCH_FEATURE_INDICES, "val")
            metrics['val_loss'] = total_loss.item()

    return metrics
            
            

