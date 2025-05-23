from functools import partial

import math
import numpy as np
from ase.io import read
from multiprocessing import Pool
from tqdm import tqdm
import click

from fieldgnn.utils.chem import _make_supercell
from fieldgnn.utils.log import get_logger
from fieldgnn.config import get_data_config, init_config

from typing import Dict, Any, List


def build_atomgrid_i(matid: str, data_config: Dict[str, Any]) -> None:
    """Build atom grid for a single material.
    
    Args:
        matid: Material ID
        data_config: Configuration dictionary containing all necessary parameters
    """
    logger = get_logger('build_atomgrid')

    cif_dir = data_config['cif_dir']
    pes_num_grids = data_config['pes_num_grids']
    pes_min_lat_len = data_config['pes_min_lat_len']
    atomgrid_dir = data_config['atomgrid_dir']
    
    # Validate input parameters
    assert len(pes_num_grids) == 3, logger.error(
        f'PES num_grids must have length of 3, got {len(pes_num_grids)}'
    )
    
    try:
        # Read CIF file and create supercell
        cif_path = cif_dir / f'{matid}.cif'
        atoms = read(cif_path)
        atoms = _make_supercell(atoms, cutoff=pes_min_lat_len)
        
        # Initialize atom grid
        atomgrid = np.zeros(pes_num_grids)
        frac_positions = atoms.get_scaled_positions()
        atomic_numbers = atoms.get_atomic_numbers()

        # Count atoms in each grid cell
        for frac_coords, _ in zip(frac_positions, atomic_numbers):
            frac_ints = [
                math.floor(frac_coords[i] * pes_num_grids[i]) 
                for i in range(3)
            ]
            atomgrid[frac_ints[0], frac_ints[1], frac_ints[2]] += 1

        # Save the atom grid
        output_path = atomgrid_dir / f"{matid}.npy"
        np.save(output_path.absolute(), atomgrid)
        
    except Exception as e:
        logger.error(f"Failed to process {matid}: {str(e)}")
        raise


def build_atomgrid() -> None:
    """Main function to build atom grids for all materials in parallel."""
    logger = get_logger('build_atomgrid')
    data_config = get_data_config()

    cif_dir = data_config['cif_dir']
    matid_df = data_config['matid_df']
    matids = list(matid_df['matid'])
    process_num = data_config['num_process']

    logger.info(f"Building atom grid for {len(matids)} materials from {cif_dir}")
    
    try:
        # Create a pool of workers
        with Pool(processes=process_num) as pool:
            # Prepare arguments for each task
            tasks = [matid for matid in matids]
            
            # Process in parallel with progress bar
            for _ in tqdm(
                pool.imap_unordered(partial(build_atomgrid_i, data_config=data_config), tasks),
                total=len(matids),
                desc="Processing materials"
            ):
                pass
                
        logger.info("Successfully built all atom grids")
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        raise


@click.command()
@click.option(
    "--config", 
    type=str, 
    required=True,
    help="Path to configuration file"
)
def cli(config: str) -> None:
    """Command line interface for building atom grids.
    
    Args:
        config: Path to configuration file
    """
    init_config(config)  # Pass config file path to init_config
    build_atomgrid()


if __name__ == "__main__":
    cli()
