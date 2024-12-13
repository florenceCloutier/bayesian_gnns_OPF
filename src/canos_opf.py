import torch
import hydra
import wandb

from torch_geometric.nn import GraphConv
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from omegaconf import DictConfig

from utils.common import train_eval_model
from utils.loss import flow_loss, voltage_angle_loss, power_balance_loss
from models.CANOS import CANOS


@hydra.main(config_path="../cfgs", config_name="canos", version_base=None)
def main(cfg: DictConfig):
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
        training_loader,
        eval_loader,
        constraints,
        lambdas,
        device,
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
