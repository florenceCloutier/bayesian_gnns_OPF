import torch
import hydra
import wandb
import time
import numpy as np

from torch_geometric.nn import GraphConv
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from omegaconf import DictConfig

from utils.common import train_eval_models, monte_carlo_integration
from sklearn.metrics import r2_score, mean_squared_error
from utils.loss import flow_loss, voltage_angle_loss, power_balance_loss, compute_branch_powers
from models.CANOS import CANOS
from utils.loss import cost

def combine_all_batches(data_loader, device):
    all_batches = []
    for batch in data_loader:
        batch = batch.to(device)
        all_batches.append(batch)
    # Combine all batches into a single batch
    combined_batch = Batch.from_data_list(all_batches)
    return combined_batch

# Define evaluation metrics
def compute_cost(predictions, targets, cost_function):
    """Computes cost metric using a custom cost function."""
    return cost_function(predictions, targets).mean().item()

def evaluate_feasibility(predictions, batch, constraints, device):
    """Check if predictions satisfy all given constraints."""
    branch_powers_ac_line = compute_branch_powers(predictions, batch, 'ac_line', device)
    branch_powers_transformer = compute_branch_powers(predictions, batch, 'transformer', device)
            
    violation_degrees = {"power_balance": 0, "flow": 0, "voltage_angle": 0}
    feasible = {"power_balance": 0, "flow": 0, "voltage_angle": 0}
    for name, constraint_fn in constraints.items():
        if name == "power_balance":
            violation = constraint_fn(predictions, batch, branch_powers_ac_line, branch_powers_transformer, device)
        elif name == "flow":
            violation = constraint_fn(batch, branch_powers_ac_line, branch_powers_transformer)
        else:
            violation = constraint_fn(predictions, batch)
        if violation.item() == 0:
            feasible[name] += 1
        violation_degrees[name] += violation

    for name in constraints.keys():
        violation_degrees[name] /= 128 # Batch size
    return feasible, violation_degrees

def evaluate_model_test(model, data_loader, device, constraints, cost_function):
    """Evaluates the model on the test set."""
    model.eval()

    all_predictions = []
    all_targets = []
    inference_times = []
    feasibility_results = {"power_balance": 0, "flow": 0, "voltage_angle": 0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            
            start_time = time.time()
            predictions = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            inference_times.append(time.time() - start_time)

            all_predictions.append(predictions)
            all_targets.append(batch.y_dict)

            # Check feasibility
            feasible, violation_degrees = evaluate_feasibility(predictions, batch, constraints, device)
            feasibility_results = {key: feasibility_results[key] + feasible[key] for key in feasible}
            if batch_idx > 0:
                violation_degrees_results = {key: (violation_degrees_results[key] + violation_degrees[key]) / 2 for key in violation_degrees}
            else:
                violation_degrees_results = violation_degrees
            
    # Format predictions
    all_predictions_combined = {key: [] for key in all_targets[0].keys()}
    all_targets_combined = {key: [] for key in all_targets[0].keys()}
    # Concatenate predictions and targets for each key
    for batch_predictions, batch_targets in zip(all_predictions, all_targets):
        for key in batch_targets.keys():
            all_predictions_combined[key].append(batch_predictions[key])
            all_targets_combined[key].append(batch_targets[key])
    final_predictions = {key: torch.cat(tensors, dim=0) for key, tensors in all_predictions_combined.items()}
    final_targets = {key: torch.cat(tensors, dim=0) for key, tensors in all_targets_combined.items()}

    # Compute metrics
    for key in final_predictions.keys():
        r2 = r2_score(final_targets[key].cpu().numpy(), final_predictions[key].cpu().numpy())
        mse = mean_squared_error(final_targets[key].cpu().numpy(), final_predictions[key].cpu().numpy())
        print(f"{key} - R²: {r2}, MSE: {mse}")
   
    data = combine_all_batches(data_loader, device)
    
    # Cost
    cost = compute_cost(final_predictions, data, cost_function)
    
    # Inference time
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    # Variance test predictions
    variances = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            _, predictive_variance = monte_carlo_integration(model, batch)
            variances.append(predictive_variance)
    
    # Initialize an empty dictionary to hold sums and counts for each key
    variance_sums = {}
    counts = {}
    # Iterate through all dictionaries in the variances list
    for predictive_variance in variances:
        for key, value in predictive_variance.items():
            if key not in variance_sums:
                variance_sums[key] = value.mean()  # Initialize sum
                counts[key] = 1  # Initialize count
            else:
                variance_sums[key] += value.mean()  # Accumulate sum
                counts[key] += 1  # Increment count

    # Compute the mean for each key
    mean_variances = {key: variance_sums[key] / counts[key] for key in variance_sums}

    print('Num samples: ', data.num_graphs)
    print(f"Cost: {cost}")
    print(f"Avg inference time: {avg_inference_time}")
    print(f"Predictive variance on test set: ", mean_variances)
    print(f"Feasibility: {feasibility_results}")
    print(f"Violation degrees mean: {violation_degrees_results}")

# Load test dataset
def load_dataset(cfg):
    test_ds = OPFDataset(cfg.data_dir, case_name=cfg.case_name, split="test")
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return test_loader

# Load model from checkpoint
def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = checkpoint['model_init_kwargs']
    model_state_dict = checkpoint['model_state_dict']

    model = CANOS(**model_kwargs).to(device)
    model.load_state_dict(model_state_dict)
    return model

# Main function for metrics
def metrics(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_loader = load_dataset(cfg)

    # Load model
    model = load_model_from_checkpoint(cfg.checkpoint_path, device)

    # Define constraints and cost function
    constraints = {
        "voltage_angle": voltage_angle_loss,
        "flow": flow_loss,
        "power_balance": power_balance_loss,
    }

    # Evaluate model
    metrics = evaluate_model_test(model, test_loader, device, constraints, cost)

    # # Print results
    # print("Evaluation Metrics:")
    # for key, value in metrics.items():
    #     print(f"{key}: {value}")


@hydra.main(config_path="../cfgs", config_name="canos_mc_dropout", version_base=None)
def main(cfg: DictConfig):
    if cfg.evaluate_metrics: 
        metrics(cfg)
    else:
        # Load the 14-bus OPFData FullTopology dataset training split and store it in the
        # directory 'data'. Each record is a `torch_geometric.data.HeteroData`.
        train_ds = OPFDataset(cfg.data_dir, case_name=cfg.case_name, split="train")
        eval_ds = OPFDataset(cfg.data_dir, case_name=cfg.case_name, split="val")
        # Batch and shuffle.
        training_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=True)

        # Initialise the model.
        # data.metadata() here refers to the PyG graph metadata, not the OPFData metadata.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = train_ds[0]

        # initialize lambdas
        lambdas = {
            "voltage_angle": torch.tensor(0.0, requires_grad=False, device=device),
            "power_balance": torch.tensor(0.0, requires_grad=False, device=device),
            "flow": torch.tensor(0.0, requires_grad=False, device=device),
        }

        # constraints
        constraints = {"voltage_angle": voltage_angle_loss, "flow": flow_loss, "power_balance": power_balance_loss}

        models = []
        seed_start = 0
        use_dropout = cfg.approx_method == "MC_dropout"
        use_va = cfg.approx_method == "variational_inference"
        for i in range(cfg.num_models_ensemble_method):
            # Set unique random seed
            torch.manual_seed(seed_start + i)
            models.append(CANOS(
                in_channels=-1,
                hidden_size=cfg.hidden_size,
                out_channels=2,
                num_message_passing_steps=cfg.num_message_passing_steps,
                metadata=data.metadata(),
                use_dropout=use_dropout,
                use_va=use_va,
            ).to(device))
        
        train_eval_models(
            models,
            training_loader,
            eval_loader,
            constraints,
            lambdas,
            device,
            cfg.checkpoint_path,
            rho=cfg.rho,
            beta=cfg.beta,
            train_log_interval=cfg.train_log_interval, 
            epochs=cfg.epochs,
            num_samples=cfg.num_samples,
            approx_method=cfg.approx_method,
        )

if __name__ == "__main__":
    wandb.init(entity="real-lab", project="PGM_bayes_gnn_opf", name="canos_gnn")
    main()
