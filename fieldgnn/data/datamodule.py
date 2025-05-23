import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from einops import rearrange

from fieldgnn.config import get_data_config, get_train_config
from fieldgnn.utils.log import Log
from fieldgnn.utils.metric import metric_classification, metric_regression

from typing import Union, Dict, List, Any, Literal
import numpy.typing as npt


class FieldGNNDataset(Dataset, Log):
    """
    Custom PyTorch Dataset for loading FieldGNN (Crystal Chemistry Knowledge Database) data.
    Supports loading different types of features (FF and MACE) for crystal structures.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        mean: Dict[str, float] = None,
        std: Dict[str, float] = None,
    ):
        """
        Initialize the dataset.

        Args:
            split: Dataset split to load ('train', 'val', or 'test')
            mean: Precomputed mean for normalization (if None, will compute from this dataset)
            std: Precomputed std for normalization (if None, will compute from this dataset)
        """
        Dataset.__init__(self)
        Log.__init__(self)

        self.split = split
        assert self.split in [
            "train",
            "val",
            "test",
        ], "Split must be 'train', 'val', or 'test'"
        self.data_config = get_data_config()
        self.train_config = get_train_config()

        self._init_params()
        self._filter_df()
        self._set_statistic(mean, std)

    def _init_params(self) -> None:
        """Initialize dataset parameters from config."""
        self.root_dir = self.data_config["root_dir"]
        self.cif_dir = self.data_config["cif_dir"]
        self.matid_df = self.data_config[f"{self.split}_matid_df"]

        # Feature configuration
        self.enable_pes = self.train_config["enable_pes"]
        self.enable_task = self.train_config["enable_task"]
        if self.enable_task:
            self.task_name = self.train_config["task_name"]
            self.task_type = self.train_config['task_type']
            assert self.task_type in ['Regression', 'BinaryClassification'], 'Unsupported task type'
        self.pes_dir = self.data_config["pes_dir"]
        self.graph_dir = self.data_config["graph_dir"]
        self.num_grids = self.data_config["num_grids"]

    def _load_pes_feature(self, matid: str) -> Dict[str, torch.Tensor]:
        """
        Load force field (FF) features for a given material ID.

        Args:
            matid: Material ID to load features for

        Returns:
            Dictionary containing FF features if enabled and available, else empty dict
        """
        try:
            pes_feature = np.load(self.pes_dir / f"{matid}.npy")
            pes_feature = np.transpose(pes_feature, (2, 1, 0))

            pes_num_grid = self.data_config['pes_num_grids'][0]
            num_grid = self.data_config['num_grids'][0]
            assert pes_num_grid % num_grid == 0
            patch_size = pes_num_grid // num_grid
            pes_feature = rearrange(pes_feature, '(h1 h2) (w1 w2) (d1 d2) -> h1 w1 d1 (h2 w2 d2)', 
                                    h1=num_grid, w1=num_grid, d1=num_grid, h2=patch_size, w2=patch_size, d2=patch_size)
            pes_feature = pes_feature[..., 0]

            pes_mask = np.logical_and(pes_feature > self.train_config['pes_min_energy'], pes_feature < self.train_config['pes_max_energy'])
            pes_feature /= self.train_config["pes_energy_norm"]
            pes_feature = torch.from_numpy(pes_feature).float()
            pes_mask = torch.from_numpy(pes_mask)
            # we calcualte ff energy in order c, b, a, nees to transpose. see fieldgnn.data.libs.griday for detail
            return {"pes_feature": pes_feature, "pes_mask": pes_mask}
        except Exception as e:
            self.log(f"Load pes feature failed for {matid}: {e}", "error")

    def _load_graph(self, matid: str):
        try:
            graph = torch.load(self.graph_dir / f"{matid}.pt")
            return {"graph": graph}
        except Exception as e:
            self.log(f"Load graph (sampling points) failed for {matid}: {e}", "error")
            return None

    def _load_label(self, idx: int):
        matid = self.matid_df.iloc[idx]["matid"]

        try:
            if self.task_type == 'Regression':
                label = self.matid_df.iloc[idx][self.task_name]
                if hasattr(self, "_mean") and hasattr(self, "_std"):
                    label = (label - self._mean[self.task_name]) / self._std[self.task_name]
                return {self.task_name: torch.tensor([label]).float()}
            elif self.task_type == 'BinaryClassification':
                label = self.matid_df.iloc[idx][self.task_name]
                return {self.task_name: torch.tensor([label]).float()}
        except Exception as e:
            self.log(f"Load label of {self.task_name} failed for {matid}: {e}", "error")
            return None

    def _filter_df(self) -> None:
        """
        Filter the material ID dataframe to only include materials with all required features.
        Logs errors for materials missing required features.
        """
        items = []

        olen = len(self.matid_df)
        self.matid_df = self.matid_df.dropna(
            subset=[self.task_name] if self.enable_task else []
        )
        self.log(f"Total [{len(self.matid_df)}/{olen}] mats with label")

        for item in self.matid_df.iloc:
            matid = item["matid"]
            correct = True

            # Check if required features exist
            if not (self.graph_dir / f"{matid}.pt").exists():
                correct = False
                self.log(f"{matid} missing graph", "error")
            graph = self._load_graph(matid)['graph']
            if len(graph.atomic_numbers) > 1000:
                continue
            if self.enable_pes and not (self.pes_dir / f"{matid}.npy").exists():
                correct = False
                self.log(f"FF feature enabled but {matid} missing FF feature", "error")
            # if self.enable_task and np.isnan(item[self.task_name]):
            #     correct = False
            #     self.log(
            #         f"enable task is true but {matid}'s {self.task_name} is nan. Please check the label (csv) file"
            #     )

            if correct:
                items.append(item)

        self.log(f"Total [{len(items)}/{len(self.matid_df)}] mats found")
        self.matid_df = pd.DataFrame(items, columns=self.matid_df.columns)

    def _set_statistic(
        self, mean: Dict[str, float] = None, std: Dict[str, float] = None
    ):
        if self.enable_task:
            if mean is not None and std is not None:
                # Use provided statistics
                self._mean = mean
                self._std = std
            elif self.split == "train":
                # Compute statistics from training set
                self._mean = {self.task_name: self.matid_df[self.task_name].mean()}
                self._std = {self.task_name: self.matid_df[self.task_name].std()}
            else:
                raise ValueError(
                    "Normalization statistics must be provided for validation and test sets"
                )

    def metric_regression(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth using multiple metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary containing evaluation metrics
        """
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        # Denormalize if task is enabled and statistics are available
        if self.enable_task and hasattr(self, "_mean") and hasattr(self, "_std"):
            return metric_regression(
                y_true, y_pred, self._mean[self.task_name], self._std[self.task_name]
            )

        return metric_regression(y_true, y_pred)

    def denorm(
        self, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        if self.enable_task and hasattr(self, "_mean") and hasattr(self, "_std"):
            return y_pred * self._std[self.task_name] + self._mean[self.task_name]

        return y_pred

    def metric_classification(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth using multiple metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary containing evaluation metrics
        """
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        return metric_classification(y_true, y_pred)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing material ID and all requested features
        """
        matid = self.matid_df.iloc[idx]["matid"]

        ret = {"matid": matid}
        ret.update(self._load_graph(matid))
        if self.enable_pes:
            ret.update(self._load_pes_feature(matid))
        if self.enable_task:
            ret.update(self._load_label(idx))

        return ret

    def collate_fn(
        self, data_list: List[Dict[str, Union[str, torch.Tensor]]]
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        """
        Custom collate function for DataLoader to properly batch samples.

        Args:
            data_list: List of samples to collate

        Returns:
            Dictionary of batched data
        """
        list_keys = ["matid", "graph"]
        stack_keys = []
        if self.enable_pes:
            stack_keys.append("pes_feature")
            stack_keys.append("pes_mask")
        concat_keys = []
        if self.enable_task:
            concat_keys.append(self.task_name)

        ret = {}
        for key in list_keys:
            ret[key] = [d[key] for d in data_list]
        for key in stack_keys:
            ret[key] = torch.stack([d[key] for d in data_list])
        for key in concat_keys:
            ret[key] = torch.cat([d[key] for d in data_list])

        return ret

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.matid_df)


class FieldGNNDatamodule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for FieldGNN dataset.
    Handles dataset initialization and DataLoader creation for all splits.
    """

    def __init__(self):
        super().__init__()
        self.data_config = get_data_config()
        self.train_config = get_train_config()
        self.datasets = {}
        self._mean = None
        self._std = None

    def setup(self, stage: str = None) -> None:
        """
        Initialize datasets for each split.

        Args:
            stage: Optional stage ('fit', 'validate', 'test', or None)
        """
        # First setup training set to compute normalization statistics
        self.datasets["train"] = FieldGNNDataset("train")

        if self.train_config["enable_task"]:
            self._mean = self.datasets["train"]._mean
            self._std = self.datasets["train"]._std

        # Then setup validation and test sets using the same statistics
        self.datasets["val"] = FieldGNNDataset("val", self._mean, self._std)
        self.datasets["test"] = FieldGNNDataset("test", self._mean, self._std)

    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training data."""
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.train_config["batch_size"],
            num_workers=self.train_config["num_workers"],
            collate_fn=self.datasets["train"].collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create DataLoader for validation data."""
        return DataLoader(
            dataset=self.datasets["val"],
            batch_size=self.train_config["batch_size"],
            num_workers=self.train_config["num_workers"],
            collate_fn=self.datasets["val"].collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create DataLoader for test data."""
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.train_config["batch_size"],
            num_workers=self.train_config["num_workers"],
            collate_fn=self.datasets["test"].collate_fn,
        )

    def metric_regression(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = "test"
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth using dataset's evaluation method.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            split: Which dataset split to use for evaluation ('train', 'val', or 'test')

        Returns:
            Dictionary containing evaluation metrics
        """
        if split not in self.datasets:
            raise ValueError(f"Split {split} not found in datasets")

        return self.datasets[split].metric_regression(y_true, y_pred)

    def denorm(
        self, y_pred: torch.Tensor, split: str = "test"
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth using dataset's evaluation method.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            split: Which dataset split to use for evaluation ('train', 'val', or 'test')

        Returns:
            Dictionary containing evaluation metrics
        """
        if split not in self.datasets:
            raise ValueError(f"Split {split} not found in datasets")

        return self.datasets[split].denorm(y_pred)

    def metric_classification(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth using dataset's evaluation method.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary containing evaluation metrics
        """
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        return metric_classification(y_true, y_pred)
