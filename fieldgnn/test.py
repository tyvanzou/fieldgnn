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
from torch_geometric.data import Data
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import lightning as L
from lightning.pytorch import Trainer, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT

from fieldgnn.config import init_config, get_train_config, get_model_config, get_config
from fieldgnn.data.datamodule import FieldGNNDatamodule
from fieldgnn.modules.modules import TYPE_MODULE_MAP
from fieldgnn.modules.utils import MLP
from fieldgnn.modules.optimize import set_scheduler
from fieldgnn.utils.log import Log
from fieldgnn.utils.metric import metric_regression, metric_classification


VN_TASKS = ["ff", "mace", "task"]
PN_TASKS = ["denoise", "task"]


class FieldGNNLightningModule(L.LightningModule):
    def __init__(self, config: str):
        """Initialize the FieldGNN Lightning module.

        Args:
            config: Path to configuration file
        """
        L.LightningModule.__init__(self)
        # Log.__init__(self)

        init_config(config)
        self.model_config = get_model_config()
        self.train_config = get_train_config()

        # Initialize model flags
        self._init_params()

        # Initialize model and heads
        self.model = TYPE_MODULE_MAP[self.model_type]()
        self._init_loss()
        self._init_head()

        # Save hyperparameters
        self.save_hyperparameters(get_config())

    def _init_params(self) -> None:
        self.enable_ff = self.train_config["enable_ff"]
        self.enable_mace = self.train_config["enable_mace"]
        self.enable_denoise = self.train_config["enable_denoise"]
        self.enable_mae = self.train_config["enable_mae"]
        self.enable_task = self.train_config["enable_task"]
        if self.enable_task:
            self.task_name = self.train_config["task_name"]
        self.enable_vn = self.train_config["enable_vn"]  # only used for target task
        self.model_type = self.train_config["model"]
        self.model_config = self.model_config[self.model_type]

        assert (
            self.enable_ff
            or self.enable_mace
            or self.enable_task
            or self.enable_denoise
            or self.enable_mae
        ), "at least one task of training (ff, mace, denoise, mae, task) must be true"

        if self.enable_ff or self.enable_mace:
            assert (
                self.enable_vn
            ), f"enable_vn must be ture when ff or mace training is enable"

    def _init_loss(self) -> None:
        """Initialize loss functions."""
        self.mse_loss = nn.MSELoss()
        if self.enable_mae:
            self.ce_loss = nn.CrossEntropyLoss()

    def _init_head(self) -> None:
        """Initialize task-specific heads."""
        hidden_channels = self.model_config["hidden_channels"]

        self.vn_head = MLP(
            dims=[hidden_channels] * self.train_config["head_layers"] + [20]
        )
        # TODO: 125 is the total atom types, should be configurable
        self.pn_head = MLP(
            dims=[hidden_channels] * self.train_config["head_layers"] + [20 + 125] # 125 for MAE atom_type_prediction
        )

        # if self.enable_ff:
        #     self.ff_head = MLP(dims=[hidden_channels] * self.train_config['head_layers'] + [1])

        # if self.enable_mace:
        #     self.mace_head = MLP(dims=[hidden_channels] * self.train_config['head_layers'] + [1])

        # if self.enable_mae:
        #     num_classes = self.train_config.get("mae_num_classes", 125)
        #     self.mae_head = MLP(dims=[hidden_channels] * self.train_config['head_layers'] + [num_classes])

        # if self.enable_task:
        #     self.task_head = MLP(dims=[hidden_channels] * self.train_config['head_layers'] + [1])
        self.readout = aggr_resolver(self.train_config["readout"])

    def _perturb(self, data: Data, noise_std: float = 0.15) -> None:
        """Add Gaussian noise to atomic positions.

        Args:
            data: Input graph data
            noise_std: Standard deviation of Gaussian noise
        """
        noise = torch.randn_like(data.pos) * noise_std
        data.origin_pos = data.pos.clone()
        data.pos = data.pos + noise

    def _mask(
        self, data: Data, mask_atomic_number: int = 119, mask_rate: float = 0.10
    ) -> None:
        """Mask atoms by replacing their atomic numbers.

        Args:
            data: Input graph data
            mask_atomic_number: Atomic number to use for masking
            mask_rate: Fraction of atoms to mask
        """
        atomic_numbers = data.atomic_numbers
        num_atoms = len(atomic_numbers)
        num_to_mask = max(1, int(num_atoms * mask_rate))

        data.origin_atomic_numbers = atomic_numbers.clone()
        data.mask_indices = torch.randperm(num_atoms)[:num_to_mask]
        data.atomic_numbers[data.mask_indices] = mask_atomic_number

    def preprocess_graphs(self, graph_list: list[Data]) -> None:
        """Preprocess graphs based on enabled tasks."""
        if self.enable_denoise:
            for graph in graph_list:
                self._perturb(graph, self.train_config["noise_std"])
        if self.enable_mae:
            for graph in graph_list:
                self._mask(graph, mask_rate=self.train_config.get("mask_rate", 0.1))

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

        if self.enable_ff:
            ff_feature = batch["ff_feature"].reshape(-1)
            ff_output = output["ff_output"].reshape(-1)
            ff_feature = torch.tanh(ff_feature)
            # ff_mask = batch['ff_mask'].reshape(-1)
            # ff_feature = ff_feature[ff_mask]
            # ff_output = ff_output[ff_mask]
            ff_loss = self.mse_loss(ff_output, ff_feature)
            loss += ff_loss * self.train_config["ff_loss_alpha"]
            ff_metric = metric_regression(ff_feature, ff_output)
            metric_dict["ff/loss"] = ff_loss
            for k, v in ff_metric.items():
                metric_dict[f"ff/{k}"] = v

        if self.enable_mace:
            mace_feature = batch["mace_feature"].reshape(-1)
            mace_output = output["mace_output"].reshape(-1)
            mace_mask = batch["mace_mask"].reshape(-1)
            mace_feature = mace_feature[mace_mask]
            mace_output = mace_output[mace_mask]
            mace_loss = self.mse_loss(mace_output, mace_feature)
            loss += mace_loss
            mace_metric = metric_regression(mace_feature, mace_output)
            metric_dict["mace/loss"] = mace_loss
            for k, v in mace_metric.items():
                metric_dict[f"mace/{k}"] = v

        if self.enable_denoise:
            graph_data = batch["collated_graph"]
            denoise_label = (
                graph_data["pos"][
                    graph_data["atomic_numbers"] != self.train_config["VNODE_Z"]
                ]
                - graph_data["origin_pos"]
            )
            denoise_pred = feature["pn_force"]
            denoise_loss = self.mse_loss(
                denoise_label.reshape(-1),
                denoise_pred.reshape(-1),
            )
            loss += denoise_loss
            denoise_metric = metric_regression(
                denoise_label.reshape(-1),
                denoise_pred.reshape(-1),
            )
            metric_dict["denoise/loss"] = denoise_loss
            for k, v in denoise_metric.items():
                metric_dict[f"denoise/{k}"] = v

        if self.enable_mae:
            graph_data = batch["collated_graph"]
            mae_pred = output["atom_type_pred"][graph_data.mask_indices]
            mae_label = graph_data["origin_atomic_numbers"][graph_data.mask_indices]
            mae_loss = self.ce_loss(
                mae_pred,
                mae_label,
            )
            loss += mae_loss
            # Use datamodule's evaluate method for consistent normalization
            mae_metric = metric_classification(
                mae_label,
                mae_pred,
            )
            metric_dict["mae/loss"] = mae_loss
            for k, v in mae_metric.items():
                metric_dict[f"mae/{k}"] = v

        if self.enable_task:
            # TODO: always use pn_pred
            task_loss = self.mse_loss(
                output[self.task_name].reshape(-1), batch[self.task_name].reshape(-1)
            )
            task_metric = metric_regression(
                batch[self.task_name].reshape(-1), output[self.task_name].reshape(-1)
            )
            metric_dict[f"{self.task_name}/loss"] = task_loss
            for k, v in task_metric.items():
                metric_dict[f"{self.task_name}/{k}"] = v
            loss += task_loss

        return metric_dict, loss

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Returns:
            Dictionary of model outputs
        """
        self.preprocess_graphs(batch["graph"])
        # only for testing the optimization config
        feature = self.model(batch, return_force=self.enable_denoise)
        outputs = {}

        if self.enable_ff:
            outputs["ff_output"] = self.vn_head(feature["vn_feat"])[
                ..., VN_TASKS.index("ff")
            ]

        if self.enable_mace:
            outputs["mace_output"] = self.vn_head(feature["vn_feat"])[
                ..., VN_TASKS.index("mace")
            ]

        if self.enable_mae:
            outputs["atom_type_pred"] = self.pn_head(feature["pn_feat"])[
                ..., -125:
            ]

        if self.enable_task:
            if self.enable_vn:
                task_outputs = self.vn_head(feature["vn_feat"])[
                    ..., VN_TASKS.index("task")
                ].unsqueeze(-1)
                outputs[self.task_name] = self.readout(
                    task_outputs, feature["vn_batch"]
                )
            else:
                task_outputs = self.pn_head(feature["pn_feat"])[
                    ..., PN_TASKS.index("task")
                ].unsqueeze(-1)
                outputs[self.task_name] = self.readout(
                    task_outputs, feature["pn_batch"]
                )

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
        self.best_val_mape = 1e5
        self.validation_outputs = []

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Validation step with logging."""
        if self.enable_denoise:
            with torch.enable_grad():
                feature, outputs = self(batch)
                metric_dict, loss = self._compute_loss(feature, outputs, batch)
        else:
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
            preds = outputs[self.task_name]
            targets = batch[self.task_name]

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
        metric_dict = self.trainer.datamodule.metric_regression(
            targets, preds, split="val"
        )

        self.log_dict(
            {f"val_end_{self.task_name}/{k}": v for k, v in metric_dict.items()},
            on_epoch=True,
            # prog_bar=True,
            logger=True,
        )

        df = pd.DataFrame()
        df['matid'] = matids
        df['predict'] = (self.trainer.datamodule.denorm(preds)).cpu().detach().numpy()
        df['target'] = (self.trainer.datamodule.denorm(targets)).cpu().detach().numpy()
        df.to_csv(Path(self.logger.log_dir) / 'val_result.csv', index=False)

        if metric_dict['mae'] < self.best_val_mape:
            self.best_val_mape = metric_dict['mae']
            df.to_csv(Path(self.logger.log_dir) / 'best_mape_val_result.csv', index=False)
            
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
        # return torch.optim.AdamW(self.parameters(), 1e-4, weight_decay=0.01)


@click.command()
@click.option("--config", type=str, required=True, help="Path to config YAML file")
def cli(config: str) -> None:
    """Main training CLI entry point."""
    init_config(config)
    train_config = get_train_config()

    module = FieldGNNLightningModule(config)
    datamodule = FieldGNNDatamodule()

    tb_logger = loggers.TensorBoardLogger(save_dir=train_config["lightning_log_dir"])

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
        module.load_state_dict(torch.load(ckpt_path, weights_only=False)['state_dict'], strict=True)
    # trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    cli()
