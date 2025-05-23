import yaml
import logging
from pathlib import Path
import pandas as pd
from typing import Union, Dict, Any, Optional, overload
from pprint import pprint as pp


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)


# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)

# 获取当前模块所在目录
cur_dir = Path(__file__).parent

# 模块级全局变量存储配置
_config: Dict[str, Any] = None

IS_PREDICT: bool = None


def init_config(config: Union[str, Path, Dict[str, Any]], is_predict: bool = False) -> Dict[str, Any]:
    """Load and merge default config with user config.

    Args:
        config: Path to user configuration file or config dict

    Returns:
        Merged configuration dictionary

    Raises:
        ValueError: If config type is invalid
        FileNotFoundError: If config file not found
    """
    global _config
    global IS_PREDICT
    IS_PREDICT = is_predict

    # Load default config from package
    default_config_path = cur_dir / "config.yaml"
    try:
        with default_config_path.open("rb") as f:
            default_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        default_config = {
            "logger": {
                "log_dir": "./logs",
                "level": "INFO",
                "format": "%(asctime)s | %(levelname)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        }

    if isinstance(config, (str, Path)):
        # Load user config from file
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("rb") as f:
            user_config = yaml.safe_load(f) or {}
    elif isinstance(config, Dict):
        user_config = config
    else:
        raise ValueError(
            f"Invalid config type {type(config)}, expected str, Path or Dict"
        )

    # 深度合并配置（避免浅层update覆盖整个嵌套字典）
    def deep_update(d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if isinstance(v, Dict) and k in d and isinstance(d[k], Dict):
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        return d

    merged_config = deep_update(default_config.copy(), user_config)
    _config = merged_config

    # print(f"Init config success: ")
    # pp(_config)


def get_config():
    if _config is None:
        raise RuntimeError("Logger config not initialized. Call init_config() first.")
    return _config.copy()


def get_log_config() -> Dict[str, Any]:
    """Get logger configuration.

    Returns:
        Current logger configuration dictionary

    Raises:
        RuntimeError: If config not initialized
    """
    config = get_config()
    if config.get("logger") is None:
        raise ValueError(
            "config object has no `logger` config, please check your config"
        )
    return config["logger"].copy()


def get_data_config() -> Dict[str, Any]:
    """Get logger configuration.

    Returns:
        Current logger configuration dictionary

    Raises:
        RuntimeError: If config not initialized
    """
    config = get_config()
    if config.get("data") is None:
        raise ValueError("config object has no `data` config, please check your config")
    data_config = config["data"].copy()

    # folders
    data_config["root_dir"] = Path(data_config["root_dir"])
    if not data_config["root_dir"].exists():
        raise ValueError(f"data root_dir {root_dir} not exists")
    data_config["cif_dir"] = data_config["root_dir"] / data_config["cif_folder"]
    if not data_config["cif_dir"].exists():
        raise ValueError(f"data cif_dir {cif_dir} not exists")
    for folder in ["pes", "pes_tmp", "graph", "pes_graph", "grid", "atomgrid"]:
        data_config[f"{folder}_dir"] = (
            data_config["root_dir"] / data_config[f"{folder}_folder"]
        )
        data_config[f"{folder}_dir"].mkdir(exist_ok=True)

    global IS_PREDICT
    if not IS_PREDICT:
        init_matid_csv(data_config)
    return data_config


def init_matid_csv(data_config):
    """Load and validate data configuration with priority splitting rules.

    Rules:
    1. If train/val/test_matid_csv are ALL specified:
       - Use these files directly (ignore matid_csv even if present)
    2. Otherwise:
       - matid_csv MUST be specified
       - train/val/test_matid_csv must NOT be specified (all empty)
       - Will split matid_csv according to train_ratio and val_ratio

    Args:
        data_config: Dictionary containing data configuration parameters

    Returns:
        Updated data_config with loaded DataFrames
    """
    # Check which configuration we have
    all_splits_specified = all(
        f"{split}_matid_csv" in data_config for split in ["train", "val", "test"]
    )
    any_split_specified = any(
        f"{split}_matid_csv" in data_config for split in ["train", "val", "test"]
    )

    # Case 1: All splits are specified (highest priority)
    if all_splits_specified:
        # Load each split
        for split in ["train", "val", "test"]:
            split_csv = data_config[f"{split}_matid_csv"]
            if not split_csv.endswith(".csv"):
                split_csv += ".csv"
            split_path = data_config["root_dir"] / split_csv
            if not split_path.is_file():
                raise ValueError(f"{split}_matid_csv {split_path} is not a file")
            data_config[f"{split}_matid_df"] = pd.read_csv(split_path, dtype={"matid": str})
            print(f"Loaded {split} split from {split_path}")
        data_config["matid_df"] = pd.concat(
            [data_config[f"{split}_matid_df"] for split in ["train", "val", "test"]]
        )

    # Case 2: Not all splits are specified
    else:
        # Check matid_csv is specified
        if "matid_csv" not in data_config or data_config["matid_csv"] is None:
            raise ValueError(
                "When not all split CSVs are specified, matid_csv must be provided and "
                "no partial split CSVs should be specified"
            )

        # Check no partial splits are specified
        if any_split_specified:
            specified_splits = [
                split
                for split in ["train", "val", "test"]
                if f"{split}_matid_csv" in data_config
            ]
            raise ValueError(
                f"Partial split CSVs are not allowed. Found specified: {', '.join(specified_splits)}_matid_csv. "
                "Either specify ALL split CSVs or ONLY matid_csv."
            )

        # Process matid_csv
        matid_csv = data_config["matid_csv"]
        if not matid_csv.endswith(".csv"):
            matid_csv += ".csv"
        csv_path = data_config["root_dir"] / matid_csv
        if not csv_path.is_file():
            raise ValueError(f"matid_csv {csv_path} is not a file")

        # Load and split the data
        df = pd.read_csv(csv_path, dtype={"matid": str})
        data_config["matid_df"] = df

        if all(
            [
                (data_config["root_dir"] / f"{csv_path.stem}.{split}.csv").exists()
                for split in ["train", "val", "test"]
            ]
        ):
            print(f"{csv_path.stem} has already been splitted, directly load")
            for split in ["train", "val", "test"]:
                data_config[f"{split}_matid_df"] = pd.read_csv(
                    data_config["root_dir"] / f"{csv_path.stem}.{split}.csv", dtype={"matid": str}
                ).dropna(subset=['matid'])
            return

        # Get split ratios (with defaults)
        train_ratio = data_config.get("train_ratio", 0.7)
        val_ratio = data_config.get("val_ratio", 0.15)

        # Validate ratios
        if not (
            0 < train_ratio < 1 and 0 < val_ratio < 1 and (train_ratio + val_ratio) < 1
        ):
            raise ValueError(
                f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
                "All ratios must be between 0 and 1 and their sum must be less than 1."
            )

        # Shuffle and split
        df = df.sample(frac=1).reset_index(drop=True).dropna(subset=['matid'])
        train_end = int(len(df) * train_ratio)
        val_end = train_end + int(len(df) * val_ratio)

        # Create splits
        splits = {
            "train": df.iloc[:train_end],
            "val": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:],
        }

        # Save splits to CSV and store in config
        base_name = csv_path.stem
        for split_name, split_df in splits.items():
            split_path = csv_path.parent / f"{base_name}.{split_name}.csv"
            split_df.to_csv(split_path, index=False)
            data_config[f"{split_name}_matid_df"] = split_df
            print(
                f"Created and saved {split_name} split ({len(split_df)} samples) to {split_path}"
            )

    return data_config


def get_train_config():
    return get_config()["train"]


def get_model_config():
    return get_config()["model"]


def get_optimize_config():
    return get_config()["optimize"]
