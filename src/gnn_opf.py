import torch
import wandb
import hydra

from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig

from utils.common import train_eval_models
from utils.loss import power_balance_loss, flow_loss, voltage_angle_loss

# A simple model to predict the generator active and reactive power outputs.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(-1, 16)
        self.conv2 = GraphConv(16, 2)

    def forward(self, x, edge_index, edge_attr_dict=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

@hydra.main(config_path='../cfgs', config_name='gnn_opf', version_base=None)  
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
        torch.manual_seed(seed_start + i)
        models.append(to_hetero(Model(), train_data.metadata()).to(device))
    
    train_eval_models(models, 
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
    wandb.init(entity= "real-lab", project="PGM_bayes_gnn_opf", name="vanilla_gnn")
    main()
