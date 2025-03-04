import torch
import hydra
import wandb

from omegaconf import DictConfig
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from models.bayesian_gnn import BayesianGNN, custom_to_hetero
from utils.common import train_eval_models
from utils.loss import voltage_angle_loss, flow_loss, power_balance_loss

@hydra.main(config_path='../cfgs', config_name='bayes_gnn_opf', version_base=None)  
def main(cfg: DictConfig):
    # Load the 14-bus OPFData FullTopology dataset training split and store it in the
    # directory 'data'. Each record is a `torch_geometric.data.HeteroData`.
    train_ds = OPFDataset(cfg.data_dir, case_name=cfg.case_name, split='train')
    eval_ds = OPFDataset(cfg.data_dir, case_name=cfg.case_name, split='val')
    # Batch and shuffle.
    training_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize lambdas
    lambdas = {
        "voltage_angle": torch.tensor(0.0, requires_grad=False, device=device),
        "power_balance": torch.tensor(0.0, requires_grad=False, device=device),
        "flow": torch.tensor(0.0, requires_grad=False, device=device),
    }
    
    # constraints
    constraints = {
        "voltage_angle": voltage_angle_loss,
        "flow": flow_loss,
        "power_balance": power_balance_loss
    }
    
    train_data = train_ds[0]

    # Initialize models for ensemble method (if no ensemble method, the is only 1 model)
    models = []
    seed_start = 0 
    for i in range(cfg.num_models_ensemble_method):
        # Set unique random seed
        torch.manual_seed(seed_start + i)
        models.append(custom_to_hetero(BayesianGNN(in_channels=-1, hidden_channels=16, out_channels=2, num_layers=cfg.hidden_dim), train_data.metadata()).to(device))
    
    train_eval_models(models,  
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
                     approx_method=cfg.approx_method)

if __name__ == "__main__":
    wandb.init(entity= "real-lab", project="PGM_bayes_gnn_opf", name="bayes_gnn")
    main()
