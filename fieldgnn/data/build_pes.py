# This code are based on MOFTransformer (https://www.nature.com/articles/s42256-023-00628-2), code can be found in https://github.com/hspark1212/MOFTransformer.

import os
import math
import subprocess
import shutil
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import Data

from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cssr import Cssr

from ase.io import read

from fieldgnn.utils.log import get_logger
from fieldgnn.utils.chem import _make_supercell
from fieldgnn.config import get_data_config, init_config

import click
import multiprocessing


cur_dir = Path(__file__).parent

GRIDAY_PATH = os.path.join(cur_dir, "libs/GRIDAY/scripts/grid_gen")
FF_PATH = os.path.join(cur_dir, "libs/GRIDAY/FF")


def _calculate_scaling_matrix_for_orthogonal_supercell(cell_matrix, eps=0.01):
    """
    cell_matrix: contains lattice vector as column vectors.
                 e.g. cell_matrix[:, 0] = a.
    eps: when value < eps, the value is assumed as zero.
    """
    logger = get_logger(filename="build_pes")
    logger.debug("Calculating scaling matrix for orthogonal supercell")

    try:
        inv = np.linalg.inv(cell_matrix)
        logger.debug("Inverted cell matrix successfully")

        # Get minimum absolute values of each row.
        abs_inv = np.abs(inv)
        mat = np.where(abs_inv < eps, np.full_like(abs_inv, 1e30), abs_inv)
        min_values = np.min(mat, axis=1)
        logger.debug(f"Minimum values: {min_values}")

        # Normalize each row with minimum absolute value of each row.
        normed_inv = inv / min_values[:, np.newaxis]

        # Calculate scaling_matrix.
        # New cell = np.dot(scaling_matrix, cell_matrix).
        scaling_matrix = np.around(normed_inv).astype(np.int32)
        logger.debug(f"Calculated scaling matrix: {scaling_matrix}")

        return scaling_matrix
    except Exception as e:
        logger.error(f"Error calculating scaling matrix: {str(e)}", exc_info=True)
        raise


def _calculate_energy_grid_i(atoms, matid, pes_num_grids, root_dataset, eg_logger):
    # Before 1.1.1 version : num_grid = [str(round(cell)) for cell in structure.lattice.abc]
    # After 1.1.1 version : num_grid = [30, 30, 30]
    global GRIDAY_PATH, FF_PATH

    eg_logger.info(f"Starting energy grid calculation for {matid}")
    eg_file = os.path.join(root_dataset, matid)
    tmp_file = os.path.join(root_dataset, f"{matid}.cssr")

    try:
        structure = AseAtomsAdaptor().get_structure(atoms)
        Cssr(structure).write_file(tmp_file)
        if not os.path.exists(tmp_file):
            eg_logger.error(f"{matid} cssr write failed - file not created")
            return False

        eg_logger.debug(f"Running GRIDAY with grids: {pes_num_grids}")
        proc = subprocess.Popen(
            [
                GRIDAY_PATH,
                *[str(num_grid) for num_grid in pes_num_grids],
                f"{FF_PATH}/UFF_Type.def",
                f"{FF_PATH}/UFF_FF.def",
                tmp_file,
                eg_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate()

        if proc.returncode != 0:
            eg_logger.error(f"GRIDAY process failed with return code {proc.returncode}")
    except Exception as e:
        eg_logger.error(f"Error in energy grid calculation: {str(e)}", exc_info=True)
        return False
    finally:
        # remove temp_file
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
                eg_logger.debug("Temporary CSSR file removed")
            except Exception as e:
                eg_logger.warning(f"Failed to remove temp file {tmp_file}: {str(e)}")

    if err:
        eg_logger.error(f"{matid} energy grid failed: {err.decode().strip()}")
        return False
    else:
        eg_logger.info(f"{matid} energy grid calculation completed successfully")
        return True


def calculate_energy_grid_i(matid: str):
    logger = get_logger(filename="build_pes")
    logger.info(f"Starting energy grid processing for {matid}")

    data_config = get_data_config()
    cif_path = data_config["cif_dir"] / f"{matid}.cif"
    root_dir = data_config["root_dir"] / data_config["pes_tmp_folder"]
    p_griddata = root_dir / f"{matid}.griddata"

    # Grid data and Graph data already exists
    if p_griddata.exists() and not data_config["override"]:
        logger.info(f"{matid} energy grid already exists - skipping")
        return True

    # valid cif check
    try:
        logger.debug(f"Validating CIF file: {cif_path}")
        CifParser(cif_path).get_structures()
    except ValueError as e:
        logger.error(f"{matid} failed - invalid CIF: {str(e)}")
        return False

    # read cif by ASE
    try:
        logger.debug(f"Reading CIF file with ASE")
        atoms = read(str(cif_path))
        logger.debug(f"Read structure with {len(atoms)} atoms")
    except Exception as e:
        logger.error(f"{matid} failed - ASE read error: {str(e)}", exc_info=True)
        return False

    # 1. get supercell
    try:
        logger.debug(f"Creating supercell with cutoff {data_config['pes_min_lat_len']}")
        atoms = _make_supercell(atoms, cutoff=data_config["pes_min_lat_len"])  # radius = 8
        cell_params = atoms.cell.cellpar()
        logger.debug(
            f"Supercell created with {len(atoms)} atoms, cell params: {cell_params[:3]}"
        )

        for l in cell_params[:3]:
            if data_config["max_lat_len"] and l > data_config["max_lat_len"]:
                logger.error(
                    f"{matid} failed - supercell length {l} exceeds max_length {data_config['max_lat_len']}"
                )
                return False

        if data_config["max_num_atoms"] and len(atoms) > data_config["max_num_atoms"]:
            logger.error(
                f"{matid} failed - {len(atoms)} atoms exceeds max_num_atoms {data_config['max_num_atoms']}"
            )
            return False
    except Exception as e:
        logger.error(
            f"{matid} failed during supercell creation: {str(e)}", exc_info=True
        )
        return False

    # 3. calculate energy grid
    pes_num_grids = data_config["pes_num_grids"]
    logger.info(f"Calculating energy grid with dimensions {pes_num_grids}")
    eg_success = _calculate_energy_grid_i(atoms, matid, pes_num_grids, root_dir, logger)

    if eg_success:
        logger.info(f"{matid} succeeded - saving results")
        try:
            # save cif files
            save_cif_path = root_dir / f"{matid}.cif"
            atoms.write(filename=save_cif_path)
            logger.debug(f"Saved supercell CIF to {save_cif_path}")

            graph_path = data_config["pes_graph_dir"] / f"{matid}.pt"
            struct = AseAtomsAdaptor().get_structure(atoms)
            data = Data(
                atomic_numbers=torch.tensor(
                    [site.specie.Z for site in struct], dtype=torch.long
                ),
                pos=torch.from_numpy(np.stack([site.coords for site in struct])).to(
                    torch.float
                ),
                cell=torch.tensor(struct.lattice.matrix, dtype=torch.float),
                matid=matid,
            )
            torch.save(data, graph_path)

            # save grid data
            griddata = np.fromfile(p_griddata.absolute(), dtype=np.float32).reshape(
                pes_num_grids
            )
            output_path = data_config["pes_dir"] / f"{matid}.npy"
            np.save(output_path, griddata)
            logger.info(f"Saved energy grid data to {output_path}")
        except Exception as e:
            logger.error(
                f"{matid} failed when saving energy data: {str(e)}", exc_info=True
            )
            return False

        return True
    else:
        logger.error(f"{matid} failed during energy grid calculation")
        return False


def calculate_energy_grid():
    logger = get_logger(filename="build_pes")
    data_config = get_data_config()
    matid_df = data_config["matid_df"]
    total_materials = len(matid_df["matid"])

    logger.info(f"Starting energy grid calculation for {total_materials} materials")

    # Using sequential processing (commented out multiprocessing for better logging)
    success_count = 0
    for matid in tqdm(matid_df["matid"], desc="Calculating FF"):
        try:
            if calculate_energy_grid_i(matid):
                success_count += 1
        except Exception as e:
            logger.error(
                f"Unexpected error processing {matid}: {str(e)}", exc_info=True
            )

    logger.info(
        f"Energy grid calculation completed. Success: {success_count}/{total_materials}"
    )

    # Original multiprocessing version (commented out)
    # with multiprocessing.Pool(processes=data_config['num_process']) as pool:
    #     list(tqdm(
    #         pool.imap(calculate_energy_grid_i, matid_df['matid']), total=len(matid_df), desc='calcualte ff'
    #     ))


@click.command()
@click.option("--config", type=str)
def cli(config: str):
    try:
        init_config(config)
        logger = get_logger(filename="build_pes")
        logger.info("Starting energy grid calculation CLI")
        calculate_energy_grid()
        logger.info("Energy grid calculation completed successfully")
    except Exception as e:
        logger.critical(f"Energy grid calculation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    cli()
