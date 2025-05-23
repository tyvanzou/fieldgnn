import math
from pathlib import Path
from typing import Dict, Any, List
import multiprocessing as mp
import json
import warnings

import click
import numpy as np
import torch
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from ase.io import read
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from fieldgnn.utils.chem import _make_supercell
from fieldgnn.utils.log import get_logger
from fieldgnn.utils.radius_graph import radius_graph_pbc
from fieldgnn.utils.ocp import get_pbc_distances
from fieldgnn.config import get_data_config, init_config


class GraphPreprocessor:
    """Handles Graph data preprocessing and caching."""

    def __init__(self, config: Dict[str, Any], device):
        self.logger = get_logger(filename="build_edge")

        # Get parameters from config
        self.override = config.get("override", False)

        # Setup directories
        self.cif_dir = config["cif_dir"]
        self.graph_dir = config["graph_dir"]

        self.device = device

        # Load target values
        self.df = config["matid_df"]

    def process_material(self, matid: str) -> bool:
        """Process and save data for a single material."""
        output_path = self.graph_dir / f"{matid}.pt"

        if output_path.exists() and not self.override:
            data = torch.load(output_path, weights_only=False, map_location="cpu")
            if data.get("edge_index") is not None:
                self.logger.info(
                    f"Skipping {matid} - {self.graph_dir / f'{matid}.pt'} edge already processed"
                )
                return True

        def process(device, cutoff=None, max_neighbors=None):
            if not (self.graph_dir / f"{matid}.pt").exists():
                self.logger.error(
                    f"Failed to load graph from {self.graph_dir / f'{matid}.pt'}, Please process graph first"
                )
                return False

            data = torch.load(
                self.graph_dir / f"{matid}.pt", weights_only=False, map_location=device
            )

            data.natoms = torch.tensor([len(data.atomic_numbers)], device=device)
            data.cell = data.cell.reshape(
                1, 3, 3
            )  # for collate we unsqueeze one dimension

            # if cutoff is None:
            #     cutoff = 19
            # if max_neighbors is None:
            #     # max_neighbors = 12
            #     natoms = len(data.atomic_numbers)
            #     if natoms > 300:
            #         max_neighbors = 5
            #     elif natoms > 200:
            #         max_neighbors = 12
            #     else:
            #         max_neighbors = 20
            cutoff = 19
            max_neighbors = 30

            # natoms = len(data.atomic_numbers)
            # if natoms < 768:
            #     return True

            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data,
                radius=cutoff,
                max_num_neighbors_threshold=max_neighbors,
                pbc=[True, True, True],
            )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index: torch.Tensor = out["edge_index"]
            edge_dist: torch.Tensor = out["distances"]
            edge_vec: torch.Tensor = out["distance_vec"]

            data["edge_index"] = edge_index
            data["edge_dist"] = edge_dist
            data["edge_vec"] = edge_vec
            data["cell_offsets"] = cell_offsets

            torch.save(data.to("cpu"), self.graph_dir / f"{matid}.pt")

        try:
            process(self.device)
        except torch.cuda.OutOfMemoryError as e:
            self.logger.info(
                f"Failed to process {matid}: {str(e)}, reduce cutoff to 8, max_neighbors to 5"
            )
            try:
                process(self.device, 19, 30)
            except torch.cuda.OutOfMemoryError as e:
                self.logger.info(
                    f"Failed to process {matid}: {str(e)}, use cpu instaed"
                )
                try:
                    process("cpu", 19, 30)
                except Exception as e:
                    self.logger.error(
                        f"Failed to process {matid}: {str(e)}", exc_info=True
                    )
                    return False
            except Exception as e:
                self.logger.error(f"Failed to process {matid}: {str(e)}", exc_info=True)
                return False
        except Exception as e:
            self.logger.error(f"Failed to process {matid}: {str(e)}", exc_info=True)
            return False


def process_chunk(matids: List[str], config: str, data_config: Dict[str, Any], device):
    """Process a chunk of materials with a single preprocessor instance."""
    init_config(config)
    preprocessor = GraphPreprocessor(data_config, device)
    for matid in tqdm(matids):
        preprocessor.process_material(matid)


def build_edge(config: str, devices: List[str]):
    """Build Graph data for all materials using multiprocessing."""
    logger = get_logger(filename="build_edge")
    logger.info("Starting Graph data preprocessing")

    try:
        init_config(config)
        data_config = get_data_config()

        # Get list of materials to process
        matid_df = data_config["matid_df"]
        matids = matid_df["matid"].tolist()

        # Setup multiprocessing
        chunk_size = (len(matids) + len(devices) - 1) // len(devices)
        matid_chunks = [
            matids[i : i + chunk_size] for i in range(0, len(matids), chunk_size)
        ]
        assert len(matid_chunks) == len(devices)

        processes = []
        for device, chunk in zip(devices, matid_chunks):
            p = mp.Process(
                target=process_chunk, args=(chunk, config, data_config, device)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        logger.info("Graph preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Graph preprocessing failed: {str(e)}", exc_info=True)
        raise


@click.command()
@click.option("--config", type=str, required=True, help="Path to config file")
@click.option(
    "--devices",
    type=str,
    required=True,
    help="Device to calculate edge_index, separated by comma",
)
def cli(config: str, devices: str):
    """Command line interface for Graph preprocessing."""
    try:
        mp.set_start_method("spawn")
        init_config(config)
        logger = get_logger(filename="build_edge")
        logger.info("Starting Graph preprocessing CLI")
        devices = devices.split(",")
        build_edge(config, devices)
        logger.info("Graph preprocessing CLI completed")
    except Exception as e:
        logger.critical(f"Graph CLI failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    cli()
