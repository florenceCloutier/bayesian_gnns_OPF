import torch
import hydra
import wandb

from omegaconf import DictConfig
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from models.bayesian_gnn import BayesianGNN, custom_to_hetero
from utils.common import train_eval_model
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
    model = custom_to_hetero(BayesianGNN(in_channels=-1, hidden_channels=16, out_channels=2, num_layers=cfg.hidden_dim), train_data.metadata())
    train_eval_model(model, 
                     training_loader, 
                     eval_loader, 
                     constraints, 
                     lambdas, 
                     device,
                     cfg.checkpoint_path,
                     rho=cfg.rho,
                     train_log_interval=cfg.train_log_interval, 
                     epochs=cfg.epochs,
                     batch_size=cfg.batch_size)

if __name__ == "__main__":
    wandb.init(entity= "real-lab", project="PGM_bayes_gnn_opf", name="bayes_gnn")
    main()
