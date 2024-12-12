import torch
import hydra
import wandb

from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.datasets import OPFDataset
from omegaconf import DictConfig

from utils.common import train_eval_model
from utils.loss import flow_loss, voltage_angle_loss, power_balance_loss
from models.CANOS import CANOS

# A simple model to predict the generator active and reactive power outputs.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(-1, 16)
        self.conv2 = GraphConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

@hydra.main(config_path='../cfgs', config_name='canos', version_base=None)  
def main(cfg: DictConfig):
    # Load the 14-bus OPFData FullTopology dataset training split and store it in the
    # directory 'data'. Each record is a `torch_geometric.data.HeteroData`.
    train_ds = OPFDataset(cfg.data_dir, case_name=cfg.case_name, split='train')
    eval_ds = OPFDataset(cfg.data_dir, case_name=cfg.case_name, split="val")
    
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
    constraints = {
        "voltage_angle": voltage_angle_loss,
        "flow": flow_loss,
        "power_balance": power_balance_loss
    }

    use_dropout = cfg.approx_method == "MC_dropout"
    use_va = cfg.approx_method == "variational_inference"

    model = CANOS(
        in_channels=-1,
        hidden_size=cfg.hidden_size,
        out_channels=2,
        num_message_passing_steps=cfg.num_message_passing_steps,
        metadata=data.metadata(),
        use_dropout=use_dropout,
        use_va=use_va,
    ).to(device)
    train_eval_model(
        model,
        train_ds,
        eval_ds,
        constraints,
        lambdas,
        device,
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        approx_method=cfg.approx_method,
    )


if __name__ == "__main__":
    wandb.init(entity="real-lab", project="PGM_bayes_gnn_opf", name="canos_gnn")
    main()
