import math
from pathlib import Path
from typing import Dict, Any, List
import multiprocessing as mp

import click
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read

from fieldgnn.utils.log import get_logger
from fieldgnn.utils.chem import _make_supercell
from fieldgnn.config import get_data_config, init_config

def _generate_uniform_grid(structure: Structure, num_grids: List[int]) -> np.ndarray:
    """Generate uniform grid points in the unit cell"""
    logger = get_logger(filename="build_graph")
    logger.debug(f"Generating uniform grid with dimensions {num_grids}")
    try:
        grid = np.mgrid[
            0:1:1/num_grids[0],
            0:1:1/num_grids[1],
            0:1:1/num_grids[2]
        ].reshape(3, -1).T
        result = structure.lattice.get_cartesian_coords(grid)
        logger.debug(f"Generated {len(result)} grid points")
        return result
    except Exception as e:
        logger.error(f"Error generating uniform grid: {str(e)}")
        raise

def build_graph_i(matid: str, data_config: Dict[str, Any]) -> bool:
    """Build graph data for a single material"""
    logger = get_logger(filename="build_graph")
    logger.info(f"Starting graph construction for {matid}")

    override = data_config['override']
    
    # Setup paths
    graph_path = data_config['graph_dir'] / f"{matid}.pt"
    grid_path = data_config['grid_dir'] / f"{matid}.npy"
    
    if graph_path.exists() and grid_path.exists() and not override:
        logger.info(f"{matid} graph and grid already exist, skipping")
        return True

    # 1. Build supercell and check constraints
    cif_path = data_config['cif_dir'] / f"{matid}.cif"
    logger.info(f"Processing CIF file: {cif_path}")
    try:
        atoms = read(str(cif_path))
        logger.debug(f"Original structure has {len(atoms)} atoms")
        atoms = _make_supercell(atoms, cutoff=data_config['min_lat_len'])
        logger.debug(f"Supercell has {len(atoms)} atoms after expansion")
        
        # Check supercell constraints
        cell_params = atoms.cell.cellpar()
        logger.debug(f"Cell parameters: {cell_params}")
        for l in cell_params[:3]:
            if data_config['max_lat_len'] and l > data_config['max_lat_len']:
                logger.warning(f"{matid} failed: supercell length {l} exceeds max_length {data_config['max_lat_len']}")
                return False
        if data_config['max_num_atoms'] and len(atoms) > data_config['max_num_atoms']:
            logger.warning(f"{matid} failed: {len(atoms)} atoms exceeds max_num_atoms {data_config['max_num_atoms']}")
            return False
            
        struct = AseAtomsAdaptor().get_structure(atoms)
        logger.debug(f"Successfully converted to pymatgen structure")
    except Exception as e:
        logger.error(f"{matid} failed when building supercell: {e}", exc_info=True)
        return False

    try:
        # Save graph data
        data = Data(
            atomic_numbers=torch.tensor([site.specie.Z for site in struct], dtype=torch.long),
            pos=torch.from_numpy(np.stack([site.coords for site in struct])).to(torch.float),
            cell=torch.tensor(struct.lattice.matrix, dtype=torch.float),
            matid=matid,
        )
        torch.save(data, graph_path)
        logger.info(f"Saved graph data to {graph_path}")

        # Generate and save grid points
        pos = _generate_uniform_grid(struct, data_config['num_grids'])
        pos_shape = data_config['num_grids']
        logger.debug(f"Generated {len(pos)} grid points with shape {pos_shape}")
        np.save(grid_path, pos.reshape((*pos_shape, 3)))
        logger.info(f"Saved grid points to {grid_path}")

    except Exception as e:
        logger.error(f"Error creating/saving graph or grid data: {e}", exc_info=True)
        return False

    logger.info(f"Successfully completed graph construction for {matid}")
    return True

def process_matids(matids: List[str], config: str, data_config: Dict[str, Any]):
    """Process a batch of matids"""
    try:
        init_config(config)
        logger = get_logger(filename="build_graph")
        logger.info(f"Starting process with {len(matids)} materials")
        
        torch.set_default_dtype(torch.double)
        logger.debug("Set default dtype to double")
        
        for matid in tqdm(matids, desc="Processing materials"):
            try:
                logger.info(f"Processing material {matid}")
                success = build_graph_i(
                    matid=matid,
                    data_config=data_config
                )
                if not success:
                    logger.warning(f"Failed to process {matid}")
            except Exception as e:
                logger.error(f"Error processing {matid}: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Fatal error in process_matids: {e}", exc_info=True)
        raise

def build_graph(config: str):
    """Build graph data for all materials in dataset using multiprocessing."""
    logger = get_logger(filename="build_graph")
    logger.info("Starting graph construction")
    
    try:
        init_config(config)
        data_config = get_data_config()
        matids = data_config['matid_df']['matid'].tolist()
        num_process = data_config.get('num_process', 1)
        logger.info(f"Loaded configuration, found {len(matids)} materials to process with {num_process} processes")

        # Split matids into chunks for each process
        chunk_size = (len(matids) + num_process - 1) // num_process
        matid_chunks = [matids[i:i + chunk_size] for i in range(0, len(matids), chunk_size)]
        logger.info(f"Split materials into {len(matid_chunks)} chunks")

        processes = []
        for i, chunk in enumerate(matid_chunks):
            p = mp.Process(
                target=process_matids,
                args=(chunk, config, data_config.copy())
            )
            p.start()
            processes.append(p)
            logger.debug(f"Started process {i+1} (PID: {p.pid}) with {len(chunk)} materials")

        for p in processes:
            p.join()
            logger.debug(f"Process {p.pid} completed")

        logger.info("All processes completed")
    except Exception as e:
        logger.error(f"Error in build_graph: {e}", exc_info=True)
        raise

@click.command()
@click.option('--config', type=str, required=True, help="Path to config file")
def cli(config: str):
    """Command line interface for building graph data."""
    try:
        # Set multiprocessing start method
        mp.set_start_method('spawn')
        init_config(config)
        logger = get_logger(filename="build_graph")
        logger.info("Starting CLI execution")
        logger.debug("Set multiprocessing start method to 'spawn'")
        build_graph(config)
        logger.info("CLI execution completed successfully")
    except Exception as e:
        logger.critical(f"CLI execution failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    cli()
