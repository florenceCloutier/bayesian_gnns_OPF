import torch
import wandb

from utils.loss import enforce_bound_constraints, compute_branch_powers, compute_loss_supervised, compute_loss_contraints, cost, BRANCH_FEATURE_INDICES
from utils.metrics import compute_metrics
from models.bayesian_gnn import HeteroBayesianGNN


"""
Training loop
Hyperparameters from CANOS 
"""
def train_eval_models(models, 
                training_loader,
                eval_loader,
                constraints,
                lambdas,
                device,
                checkpoint_path,
                rho=0.0001,
                beta=0.001,
                train_log_interval=100,
                checkpoint_interval=10000,
                epochs=100,
                num_samples=10,
                approx_method="variational_inference"
    ):
    
    optimizers = []
    for model in models:
        with torch.no_grad(): # Initialize lazy modules.
            optimizers.append(torch.optim.Adam(model.parameters()))

    # lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: learning_rate_schedule(step, warmup_steps, initial_learning_rate, peak_learning_rate, transition_steps, decay_rate, final_learning_rate))

    # training loop
    for _ in range(epochs):
        lambdas = learning_step(models, optimizers, num_samples, approx_method, training_loader, eval_loader, lambdas, constraints, rho, beta, train_log_interval, checkpoint_interval, checkpoint_path, device=device) #lr_scheduler, 

def learning_step(models, optimizers, num_samples, approx_method, data_loader, eval_loader, lambdas, constraints, rho, beta, train_log_interval, checkpoint_interval, checkpoint_path, device): # lr_scheduler, 
    for model in models:
        model.train()
    batch_size = data_loader.batch_size
    for batch_idx, data in enumerate(data_loader):
        ensemble_outs = []
        branch_powers_ac_line = []
        branch_powers_transformer = []
        L_supervised = []
        data = data.to(device)
        
        for model_idx, model in enumerate(models):
            optimizer = optimizers[model_idx]
            optimizer.zero_grad()

            # forward pass
            ensemble_outs.append(model(data.x_dict, data.edge_index_dict, data.edge_attr_dict))

            # Bound constraints (6) and (7) from CANOS
            enforce_bound_constraints(ensemble_outs[model_idx], data)

            # compute branch powers
            branch_powers_ac_line.append(compute_branch_powers(ensemble_outs[model_idx], data, 'ac_line', device))
            branch_powers_transformer.append(compute_branch_powers(ensemble_outs[model_idx], data, 'transformer', device))

            # supervised loss
            L_supervised.append(compute_loss_supervised(ensemble_outs[model_idx], data, branch_powers_ac_line[model_idx], branch_powers_transformer[model_idx]))

            # constraint losses
            violation_degrees, L_constraints = compute_loss_contraints(constraints, ensemble_outs[model_idx], data, branch_powers_ac_line[model_idx], branch_powers_transformer[model_idx], lambdas, device)
            
            # Total loss
            total_loss = L_supervised[model_idx] + 0.1 * L_constraints

            kl_loss = None
            if approx_method == "variational_inference":
                kl_loss = model.kl_loss()*beta / batch_size
                total_loss += kl_loss
            
            # Backprop and optimization
            total_loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            print(f"Total loss: {total_loss}")
        
        # Log metrics at intervals
        if batch_idx % train_log_interval == 0 and batch_idx != 0:
            ensemble_avg_out = {}
            for key in ensemble_outs[0].keys():
                stacked_tensors = torch.stack([d[key] for d in ensemble_outs], dim=0)
                ensemble_avg_out[key] = stacked_tensors.mean(dim=0)

            branch_powers_ac_line_ens = compute_branch_powers(ensemble_avg_out, data, 'ac_line', device)
            branch_powers_transformer_ens = compute_branch_powers(ensemble_avg_out, data, 'transformer', device)
            
            # supervised loss
            L_supervised = compute_loss_supervised(ensemble_avg_out, data, branch_powers_ac_line_ens, branch_powers_transformer_ens)

            # constraint losses
            violation_degrees, L_constraints = compute_loss_contraints(constraints, ensemble_avg_out, data, branch_powers_ac_line_ens, branch_powers_transformer_ens, lambdas, device)
                
            total_loss = L_supervised + 0.1 * L_constraints
            
            metrics = compute_metrics(data, branch_powers_ac_line_ens, branch_powers_transformer_ens, violation_degrees, "train")
            metrics['train_loss'] = total_loss.item()
            metrics['training_cost'] = cost(ensemble_avg_out, data).mean().item()
            if kl_loss is not None:
                metrics['val_kl_loss'] = kl_loss.item()
            
            eval_metrics = evaluate_models(models, eval_loader, constraints, beta, lambdas, device, num_samples, approx_method)
            metrics.update(eval_metrics)
            wandb.log(metrics)
        
        if batch_idx % checkpoint_interval == 0:
            model_init_kwargs = model.get_init_kwargs()
            model_state_dict = model.state_dict()
            torch.save(
                {
                    'model_state_dict': model_state_dict,
                    'model_init_kwargs': model_init_kwargs,
                },
                checkpoint_path
            )
        
    # compute the violation degrees
    model.eval()
    violation_degrees = {k: 0.0 for k in constraints.keys()}
    for batch_idx, data in enumerate(data_loader):
        ensemble_outs = []
        for model_idx, model in enumerate(models):
            data = data.to(device)
            ensemble_outs.append(model(data.x_dict, data.edge_index_dict))
        
        ensemble_avg_out = {}
        for key in ensemble_outs[0].keys():
            stacked_tensors = torch.stack([d[key] for d in ensemble_outs], dim=0)
            ensemble_avg_out[key] = stacked_tensors.mean(dim=0)
        
        branch_powers_ac_line_ens = compute_branch_powers(ensemble_avg_out, data, 'ac_line', device)
        branch_powers_transformer_ens = compute_branch_powers(ensemble_avg_out, data, 'transformer', device)
        
        individual_violations, _ = compute_loss_contraints(constraints, ensemble_avg_out, data, branch_powers_ac_line, branch_powers_transformer, lambdas, device)      

        for name, _ in constraints.items():  
            violation_degrees[name] += individual_violations[name]

    # update lambdas
    for name in lambdas.keys():
        lambdas[name] = lambdas[name] + rho * (violation_degrees[name].detach()/len(data_loader)) # detach ensures no gradient tracking

    return lambdas

def evaluate_models(models, eval_loader, constraints, beta, lambdas, device, num_samples, approx_method):
    for model in models:
        model.eval()
    batch_metrics = [] # Logging metrics
    batch_size = eval_loader.batch_size
    
    with torch.no_grad():
        for data in eval_loader:
            ensemble_outs = []
            branch_powers_ac_line = []
            branch_powers_transformer = []
            l_supervised = []
            l_constraints_arr = []

            kl_loss = None
            predictive_variance = None
            
            for model in models:
                data = data.to(device)
                
                if approx_method == "variational_inference":
                    out, predictive_variance = monte_carlo_integration(model, data, num_samples)
                    kl_loss = model.kl_loss()*beta / batch_size
                elif approx_method == "MC_dropout":
                    model.train()
                    out, predictive_variance = monte_carlo_integration(model, data, num_samples)
                    model.eval()
                else:
                    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
                    predictive_variance = None
                    
                ensemble_outs.append(out)
            
            ensemble_avg_out = {}
            for key in ensemble_outs[0].keys():
                stacked_tensors = torch.stack([d[key] for d in ensemble_outs], dim=0)
                ensemble_avg_out[key] = stacked_tensors.mean(dim=0)

            enforce_bound_constraints(ensemble_avg_out, data)
                
            branch_powers_ac_line = compute_branch_powers(ensemble_avg_out, data, 'ac_line', device)
            branch_powers_transformer = compute_branch_powers(ensemble_avg_out, data, 'transformer', device)
            
            # Compute supervised
            l_supervised = compute_loss_supervised(ensemble_avg_out, data, branch_powers_ac_line, branch_powers_transformer)
            
            # constraint losses
            violation_degrees = {}
            l_constraints = 0.0
            metrics = {}
            for name, constraint_fn in constraints.items():
                if name == "power_balance":
                    violation = constraint_fn(ensemble_avg_out, data, branch_powers_ac_line, branch_powers_transformer, device)
                    metrics['val_power_balance_loss'] = violation
                elif name == "flow":
                    violation = constraint_fn(data, branch_powers_ac_line, branch_powers_transformer)
                    metrics['val_flow_loss'] = violation
                else:
                    violation = constraint_fn(ensemble_avg_out, data)
                    metrics['val_voltage_angle_loss'] = violation
                violation_degrees[name] = violation
                l_constraints += lambdas[name] * violation
        
            l_constraints_arr.append(l_constraints)
            
            # Total loss
            total_loss = l_supervised + 0.1 * l_constraints
            if kl_loss:
                total_loss += kl_loss
            
            # Compute metrics
            metrics = compute_metrics(data, branch_powers_ac_line, branch_powers_transformer, violation_degrees, "val")
            metrics['val_loss'] = total_loss.item()
            metrics['val_cost'] = cost(ensemble_avg_out, data).mean().item()

            if predictive_variance:
                metrics.update({
                    f'predictive_variance_{key}': predictive_variance[key].mean().item()
                    for key in predictive_variance
                })
            
            if kl_loss is not None:
                metrics['val_kl_loss'] = kl_loss.item()
                
            batch_metrics.append(metrics)

    aggregated_metrics = {key: torch.tensor([m[key] for m in batch_metrics]).mean().item() for key in batch_metrics[0]}

    return aggregated_metrics
            
            

def monte_carlo_integration(model, data, num_samples=50):
    predictions = []

    for _ in range(num_samples):
        preds = model(data.x_dict, data.edge_index_dict)
        predictions.append(preds)
    
    aggregated_out = {key : torch.stack([pred[key] for pred in predictions]).mean(dim=0)
                      for key in predictions[0]}
    predictive_variance = {key: torch.stack([pred[key] for pred in predictions]).var(dim=0)
                           for key in predictions[0]}
    
    return aggregated_out, predictive_variance