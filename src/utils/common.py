import torch
import wandb

from utils.loss import enforce_bound_constraints, compute_branch_powers, compute_loss_supervised, cost, BRANCH_FEATURE_INDICES
from utils.metrics import compute_metrics


"""
Training loop
Hyperparameters from CANOS 
"""
def train_eval_model(model, 
                training_loader,
                eval_loader,
                constraints,
                lambdas,
                device,
                rho=0.0001,
                train_log_interval=100,
                epochs=100
    ):
    
    with torch.no_grad(): # Initialize lazy modules.
        optimizer = torch.optim.Adam(model.parameters())

    # lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: learning_rate_schedule(step, warmup_steps, initial_learning_rate, peak_learning_rate, transition_steps, decay_rate, final_learning_rate))

    # training loop
    for _ in range(epochs):
        model.train()
        lambdas = learning_step(model, optimizer, training_loader, eval_loader, lambdas, constraints, rho, train_log_interval, device=device) #lr_scheduler, 

def learning_step(model, optimizer, data_loader, eval_loader, lambdas, constraints, rho, train_log_interval, device): # lr_scheduler, 
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # forward pass
        out = model(data.x_dict, data.edge_index_dict)

        # Bound constraints (6) and (7) from CANOS
        enforce_bound_constraints(out, data)

        # compute branch powers
        branch_powers_ac_line = compute_branch_powers(out, data, 'ac_line', device)
        branch_powers_transformer = compute_branch_powers(out, data, 'transformer', device)

        # supervised loss
        L_supervised = compute_loss_supervised(out, data, branch_powers_ac_line, branch_powers_transformer)

        # constraint losses
        violation_degrees = {}
        L_constraints = 0.0
        for name, constraint_fn in constraints.items():
            if name == "power_balance":
                violation = constraint_fn(out, data, branch_powers_ac_line, branch_powers_transformer, device)
            elif name == "flow":
                violation = constraint_fn(data, branch_powers_ac_line, branch_powers_transformer)
            else:
                violation = constraint_fn(out, data)
            violation_degrees[name] = violation
            L_constraints += lambdas[name] * violation
        
        # Total loss
        total_loss = L_supervised + 0.1 * L_constraints
        
        # Log metrics at intervals
        if batch_idx % train_log_interval == 0:
            metrics = compute_metrics(data, branch_powers_ac_line, branch_powers_transformer, optimizer, violation_degrees, "train")
            metrics['train_loss'] = total_loss.item()
            metrics['training_cost'] = cost(out, data).mean().item()
            
            eval_metrics = evaluate_model(model, eval_loader, constraints, lambdas, optimizer, device)
            metrics.update(eval_metrics)
            wandb.log(metrics)

        # Backprop and optimization
        total_loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        print(f"Total loss: {total_loss}")
        
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
            metrics = compute_metrics(data, branch_powers_ac_line, branch_powers_transformer, optimizer, violation_degrees, "val")
            metrics['val_loss'] = total_loss.item()
            metrics['val_cost'] = cost(out, data).mean().item()

    return metrics
            
            
