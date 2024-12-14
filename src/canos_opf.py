import torch
import hydra
import wandb
import time

from torch_geometric.nn import GraphConv
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from omegaconf import DictConfig

from utils.common import train_eval_models
from sklearn.metrics import r2_score, mean_squared_error
from utils.loss import flow_loss, voltage_angle_loss, power_balance_loss
from models.CANOS import CANOS

# Define evaluation metrics
def compute_cost(predictions, targets, cost_function):
    """Computes cost metric using a custom cost function."""
    return cost_function(predictions, targets)

def evaluate_feasibility(predictions, constraints):
    """Check if predictions satisfy all given constraints."""
    feasibility = {
        name: (loss_fn(predictions).item() <= 1e-6)  # Tolerance for constraint satisfaction
        for name, loss_fn in constraints.items()
    }
    return feasibility

def evaluate_model_test(model, data_loader, device, constraints, cost_function):
    """Evaluates the model on the test set."""
    model.eval()

    all_predictions = []
    all_targets = []
    inference_times = []
    feasibility_results = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Measure inference time
            start_time = time.time()
            predictions = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            inference_times.append(time.time() - start_time)

            # Collect predictions and targets
            all_predictions.append(predictions)
            all_targets.append(batch.y_dict)

            # Check feasibility
            feasibility_results.append(evaluate_feasibility(predictions, constraints))

    # Aggregate results
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    r2 = r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
    mse = mean_squared_error(targets.cpu().numpy(), predictions.cpu().numpy())
    cost = compute_cost(predictions, targets, cost_function)
    avg_inference_time = sum(inference_times) / len(inference_times)
    overall_feasibility = all(all(feas.values()) for feas in feasibility_results)

    return {
        "R2": r2,
        "MSE": mse,
        "Cost": cost,
        "Inference Time": avg_inference_time,
        "Feasibility": overall_feasibility,
    }

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

    # def custom_cost_function(predictions, targets):
    #     # Replace with your cost function logic
    #     return ((predictions - targets) ** 2).mean().item()

    # # Evaluate model
    # metrics = evaluate_model_test(model, test_loader, device, constraints, custom_cost_function)

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
