import click
from pathlib import Path

from typing import Dict, Any, Optional, List

from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import lightning as L
from lightning.pytorch import Trainer, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT

from fieldgnn.config import init_config, get_train_config, get_model_config, get_data_config, get_config
from fieldgnn.data.datamodule import FieldGNNDatamodule
from fieldgnn.modules.modules import TYPE_MODULE_MAP
from fieldgnn.modules.utils import MLP
from fieldgnn.modules.optimize import set_scheduler
from fieldgnn.utils.log import Log
from fieldgnn.utils.metric import metric_regression, metric_classification, metric_binary_classification


class FieldGNNLightningModule(L.LightningModule):
    def __init__(self, config: str, is_predict=False):
        """Initialize the FieldGNN Lightning module.

        Args:
            config: Path to configuration file
        """
        # L.LightningModule.__init__(self)
        super().__init__()
        # Log.__init__(self)

        self.is_predict = is_predict
        init_config(config, is_predict)
        self.data_config = get_data_config()
        self.model_config = get_model_config()
        self.train_config = get_train_config()

        # Initialize model flags
        self._init_params()

        # Initialize model and heads
        self.model = TYPE_MODULE_MAP[self.model_type]()
        self._init_loss()
        self._init_head()

        # Save hyperparameters
        if not is_predict:
            self.save_hyperparameters(get_config())

    def _init_params(self) -> None:
        self.enable_pes = self.train_config["enable_pes"]
        self.enable_task = self.train_config["enable_task"]
        if self.enable_task:
            self.task_name = self.train_config["task_name"]
            self.task_type = self.train_config['task_type']
        self.readout_node = self.train_config['readout_node']
        assert self.readout_node in ['vn', 'pn'], f'Unsupported readout node {self.readout_node}, support vn (virtual node), pn (physical node)'
        self.model_type = self.train_config["model"]
        self.model_config = self.model_config[self.model_type]

        assert (
            self.enable_pes
            or self.enable_task
        ), "at least one task of training (pes, mace, denoise, mae, task) must be true"


    def _init_loss(self) -> None:
        """Initialize loss functions."""
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _init_head(self) -> None:
        """Initialize task-specific heads."""
        if self.model_type == 'dimenet':
            hidden_channels = self.model_config['out_channels']
        elif self.model_type == 'cgcnn':
            hidden_channels = self.model_config['atom_fea_len']
        elif self.model_type == 'jmp':
            hidden_channels = self.model_config['hid_dim']
        else:
            hidden_channels = self.model_config["hidden_channels"]

        if self.task_type == 'Regression' or self.task_type == 'BinaryClassification':
            self.vn_head = MLP(
                dims=[hidden_channels] * self.train_config["head_layers"] + [2] # one for task, one for PES
            )
            self.pn_head = MLP(
                dims=[hidden_channels] * self.train_config["head_layers"] + [1] # 125 for MAE atom_type_prediction
            )
        else:
            raise ValueError(f"Unsupported task type {self.task_type}")

        self.readout = aggr_resolver(self.train_config["readout"])

    def _compute_loss(
        self,
        feature: Dict[str, Any],
        output: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute combined loss for all enabled tasks.

        Returns:
            Total loss value
        """
        loss = 0.0
        metric_dict = {}

        if self.enable_pes:
            pes_feature = batch["pes_feature"].reshape(-1)
            pes_output = output["pes_output"].reshape(-1)
            pes_feature = torch.tanh(pes_feature)
            # pes_mask = batch['pes_mask'].reshape(-1)
            # pes_feature = pes_feature[pes_mask]
            # pes_output = pes_output[pes_mask]
            pes_loss = self.mse_loss(pes_output, pes_feature)
            loss += pes_loss * self.train_config["pes_loss_alpha"]
            pes_metric = metric_regression(pes_feature, pes_output)
            metric_dict["pes/loss"] = pes_loss
            for k, v in pes_metric.items():
                metric_dict[f"pes/{k}"] = v

        if self.enable_task:
            if self.task_type == 'Regression':
                task_loss = self.mse_loss(
                    output[self.task_name].reshape(-1), batch[self.task_name].reshape(-1)
                )
                task_metric = metric_regression(
                    batch[self.task_name].reshape(-1), output[self.task_name].reshape(-1)
                )
            elif self.task_type == 'BinaryClassification':
                task_loss = self.bce_loss(
                    output[self.task_name].reshape(-1), batch[self.task_name].reshape(-1)
                )
                task_metric = metric_binary_classification(
                    batch[self.task_name].reshape(-1), output[self.task_name].reshape(-1), need_sigmoid=True
                )

            metric_dict[f"{self.task_name}/loss"] = task_loss
            for k, v in task_metric.items():
                metric_dict[f"{self.task_name}/{k}"] = v
            loss += task_loss

        return metric_dict, loss

    def forward(self, batch: Dict[str, Any], log_contribution: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Returns:
            Dictionary of model outputs
        """
        # only for testing the optimization config
        feature = self.model(batch)
        outputs = {}

        if self.enable_pes:
            outputs["pes_output"] = self.vn_head(feature["vn_feat"])[
                ..., 1
            ]

        if self.enable_task:
            if self.readout_node == 'vn':
                task_outputs = self.vn_head(feature["vn_feat"])[
                    ..., 0
                ].unsqueeze(-1)
                matid = batch['matid']
                if self.train_config['log_contribution'] or log_contribution:
                    contribution = dict()
                    for sample_idx, sample_id in enumerate(matid):
                        sample_vn_contribution = task_outputs[feature['vn_batch'] == sample_idx]
                        sample_vn_contribution = sample_vn_contribution.reshape(self.data_config['num_grids']).cpu().detach().numpy()
                        if not self.is_predict:
                            contri_dir = Path(self.logger.log_dir) / 'contribution'
                            contri_dir.mkdir(exist_ok=True)
                            np.save(contri_dir / f'{sample_id}.npy', sample_vn_contribution)
                        contribution[sample_id] = sample_vn_contribution
                    outputs[f'{self.task_name}_contribution'] = contribution
                outputs[self.task_name] = self.readout(
                    task_outputs, feature["vn_batch"]
                )
            elif self.readout_node == 'pn':
                if feature['pn_feat'].shape[-1] == 1:
                    task_outputs = feature['pn_feat']
                else:
                    task_outputs = self.pn_head(feature["pn_feat"])[
                        ..., 0
                    ].unsqueeze(-1)
                outputs[self.task_name] = self.readout(
                    task_outputs, feature["pn_batch"]
                )
                if self.train_config['log_contribution'] or log_contribution:
                    contribution = dict()
                    for sample_idx, sample_id in enumerate(matid):
                        sample_pn_contribution = task_outputs[feature['pn_batch'] == sample_idx]
                        if not self.is_predict:
                            contri_dir = Path(self.logger.log_dir) / 'contribution'
                            contri_dir.mkdir(exist_ok=True)
                            np.save(contri_dir / f'{sample_id}.npy', sample_pn_contribution)
                        contribution[sample_id] = sample_pn_contribution
                    outputs[f'{self.task_name}_contribution'] = contribution
            else:
                raise ValueError(f'Unsupported readout node {self.readout_node}, support vn (virtual node), pn (physical node)')

        return feature, outputs

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        """Training step with logging."""
        feature, outputs = self(batch)
        metric_dict, loss = self._compute_loss(feature, outputs, batch)

        # Log individual losses
        self.log_dict(
            {f"train_{k}": v for k, v in metric_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # 记录当前学习率（从优化器中获取）
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.best_val_metric = 1e5
        self.validation_outputs = []

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Validation step with logging."""
        feature, outputs = self(batch)
        metric_dict, loss = self._compute_loss(feature, outputs, batch)

        # Log individual losses
        self.log_dict(
            {f"val_{k}": v for k, v in metric_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        if self.enable_task:
            if self.task_type == 'Regression' or self.task_type == 'BinaryClassification':
                preds = outputs[self.task_name].reshape(-1)
                targets = batch[self.task_name].reshape(-1)

            self.validation_outputs.append({"matid": batch['matid'], "preds": preds, "targets": targets})
        else:
            # TODO: This will ABSOLUTELY result in error when enable task is false or other data need to be preserved in validation_step
            self.validation_outputs.append(None)

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        outputs = self.validation_outputs

        if not self.enable_task or not outputs or not self.trainer.datamodule:
            return

        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])

        matids = [m for x in outputs for m in x['matid']]

        # Use datamodule's evaluate method for consistent normalization
        if self.task_type == 'Regression':
            metric_dict = self.trainer.datamodule.metric_regression(
                targets, preds, split="val"
            )

            df = pd.DataFrame()
            df['matid'] = matids
            df['predict'] = (self.trainer.datamodule.denorm(preds)).cpu().detach().numpy()
            df['target'] = (self.trainer.datamodule.denorm(targets)).cpu().detach().numpy()
            df.to_csv(Path(self.logger.log_dir) / 'val_result.csv', index=False)

            if metric_dict['mae'] < self.best_val_metric:
                self.best_val_metric = metric_dict['mae']
                df.to_csv(Path(self.logger.log_dir) / 'best_mae_val_result.csv', index=False)

        elif self.task_type == 'BinaryClassification':
            metric_dict = metric_binary_classification(
                targets, preds, need_sigmoid=True
            )

            df = pd.DataFrame()
            df['matid'] = matids
            df['predict'] = F.sigmoid(preds).cpu().detach().numpy()
            df['target'] = targets.cpu().detach().numpy()
            df.to_csv(Path(self.logger.log_dir) / 'val_result.csv', index=False)

            if metric_dict['acc'] > self.best_val_metric:
                self.best_val_metric = metric_dict['acc']
                df.to_csv(Path(self.logger.log_dir) / 'best_acc_val_result.csv', index=False)

        self.log_dict(
            {f"val_end_{self.task_name}/{k}": v for k, v in metric_dict.items()},
            on_epoch=True,
            # prog_bar=True,
            logger=True,
        )
            
    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Validation step with logging."""
        feature, outputs = self(batch)
        metric_dict, loss = self._compute_loss(feature, outputs, batch)

        # Log individual losses
        self.log_dict(
            {f"test_{k}": v for k, v in metric_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        if self.enable_task:
            preds = outputs[self.task_name]
            targets = batch[self.task_name]

            self.test_outputs.append({"matid": batch['matid'], "preds": preds, "targets": targets})
        else:
            # TODO: This will ABSOLUTELY result in error when enable task is false or other data need to be preserved in validation_step
            self.test_outputs.append(None)

    def on_test_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        outputs = self.test_outputs

        if not self.enable_task or not outputs or not self.trainer.datamodule:
            return

        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        matids = [m for x in outputs for m in x['matid']]

        # Use datamodule's evaluate method for consistent normalization
        metric_dict = self.trainer.datamodule.metric_regression(
            targets, preds, split="test"
        )

        self.log_dict(
            {f"test_end_{self.task_name}/{k}": v for k, v in metric_dict.items()},
            on_epoch=True,
            # prog_bar=True,
            logger=True,
        )

        df = pd.DataFrame()
        df['matid'] = matids
        df['predict'] = (self.trainer.datamodule.denorm(preds)).cpu().detach().numpy()
        df['target'] = (self.trainer.datamodule.denorm(targets)).cpu().detach().numpy()
        df.to_csv(Path(self.logger.log_dir) / 'test_result.csv', index=False)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        return set_scheduler(self)
        # return torch.optim.AdamW(self.parameters(), 1e-3, weight_decay=0.0)

    def predict(self, graph: torch_geometric.data.Data, matid: str = "mat", log_contribution: bool = True):
        feature, outputs = self({
            "matid": [matid],
            "graph": [graph]
        }, log_contribution=log_contribution)
        pred_val = outputs[f'{self.task_name}'].reshape(-1)[0]
        if log_contribution:
            contribution = outputs[f'{self.task_name}_contribution'][matid]
            return pred_val, contribution
        else:
            return pred_val


@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", type=str, required=True, help="Path to config YAML file")
def train(config: str) -> None:
    """Main training CLI entry point."""
    init_config(config)
    train_config = get_train_config()

    module = FieldGNNLightningModule(config)
    datamodule = FieldGNNDatamodule()

    tb_logger = loggers.TensorBoardLogger(save_dir=train_config["log_dir"])

    trainer = Trainer(
        max_epochs=train_config["max_epochs"],
        accelerator=train_config["accelerator"],
        devices=train_config["device"],
        precision=train_config["precision"],
        strategy=train_config["strategy"],
        enable_progress_bar=True,
        logger=tb_logger,
        log_every_n_steps=train_config["log_every_n_steps"],
        gradient_clip_val=train_config["gradient_clip_val"],
        accumulate_grad_batches=train_config["accumulate_grad_batches"],
    )

    ckpt_path = train_config["ckpt_path"]
    if ckpt_path is not None:
        # module.load_state_dict(torch.load(ckpt_path, weights_only=False)['state_dict'], strict=True)
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(module, datamodule=datamodule)
    # trainer.test(module, datamodule=datamodule)

@cli.command()
def test(config: str) -> None:
    """Main training CLI entry point."""
    init_config(config)
    train_config = get_train_config()

    module = FieldGNNLightningModule(config)
    datamodule = FieldGNNDatamodule()

    tb_logger = loggers.TensorBoardLogger(save_dir=train_config["log_dir"])

    trainer = Trainer(
        max_epochs=train_config["max_epochs"],
        accelerator=train_config["accelerator"],
        devices=train_config["device"],
        precision=train_config["precision"],
        strategy=train_config["strategy"],
        enable_progress_bar=True,
        logger=tb_logger,
        log_every_n_steps=train_config["log_every_n_steps"],
        gradient_clip_val=train_config["gradient_clip_val"],
        accumulate_grad_batches=train_config["accumulate_grad_batches"],
    )

    ckpt_path = train_config["ckpt_path"]
    assert ckpt_path is not None, f'test must have ckpt_path'
    module.load_state_dict(torch.load(ckpt_path, weights_only=False)['state_dict'], strict=True)
    trainer.test(module, datamodule=datamodule)

@cli.command()
@click.option('--ckpt-path', type=str, required=True)
@click.option('--cif-path', type=str, required=True)
@click.option('--output-dir', type=str, default='.')
def predict(ckpt_path: str, cif_path: str, output_dir: str):
    from torch_geometric.data import Data
    from pymatgen.core.structure import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import read
    from fieldgnn.utils.chem import _make_supercell

    matid = Path(cif_path).stem
    ckpt = torch.load(ckpt_path, weights_only=False)

    """Main training CLI entry point."""
    config = ckpt['hyper_parameters']
    init_config(config, is_predict=True)
    train_config = get_train_config()
    data_config = get_data_config()

    atoms = read(str(cif_path))
    atoms = _make_supercell(atoms, cutoff=data_config['min_lat_len'])
    struct = AseAtomsAdaptor().get_structure(atoms)
    graph = Data(
        atomic_numbers=torch.tensor([site.specie.Z for site in struct], dtype=torch.long),
        pos=torch.from_numpy(np.stack([site.coords for site in struct])).to(torch.float),
        cell=torch.tensor(struct.lattice.matrix, dtype=torch.float),
        matid=matid,
    )

    module = FieldGNNLightningModule(config, is_predict=True)

    assert ckpt_path is not None, f'predict must have ckpt_path'
    module.load_state_dict(ckpt['state_dict'], strict=True)

    pred_val, contribution = module.predict(graph, matid=matid, log_contribution=True)

    output_dir = Path(output_dir)
    (output_dir / 'contribution').mkdir(exist_ok=True, parents=True)
    np.save(output_dir / 'contribution' / f'{matid}.npy', contribution)
    print("Matid: ", matid, 'Predicted Value:', pred_val)
    


if __name__ == "__main__":
    cli()
