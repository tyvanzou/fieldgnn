import shutil
from pathlib import Path
from subprocess import Popen, DEVNULL
from tqdm import tqdm
import json
from multiprocessing import Pool
import argparse
import click
from functools import partial

# ZEO++ config
ZEO_PATH = "/localData/user081901/libs/zeo++-0.3/network" # see Zeo++ website
radius = "0.5"
chan_radius = radius
probe_radius = radius
num_sample = "50000"


def process(cifid: str, id_prop_path: Path, cif_dir: Path, ZEO_PATH: str, chan_radius: str, probe_radius: str, num_sample: str):
    def get_topo(cifid: str):
        try:
            return cifid.split("+")[1]
        except:
            return None

    def get_vf(cifid: str, cif_dir: Path, ZEO_PATH: str, chan_radius: str, probe_radius: str, num_sample: str):
        try:
            cifpath = cif_dir / f"{cifid}.cif"
            Path('./tmp').mkdir(exist_ok=True)
            shutil.copy2(cifpath, "./tmp")
            process = Popen(
                [
                    ZEO_PATH,
                    "-ha",
                    "MED",
                    "-vol",
                    chan_radius,
                    probe_radius,
                    num_sample,
                    f"./tmp/{cifid}.cif",
                ],
                stdout=DEVNULL,
                stderr=DEVNULL,
            )
            process.wait()
            vf = None
            with Path(f"./tmp/{cifid}.vol").open("r") as f:
                line = f.readline()
                idx_a = line.index("AV_Volume_fraction:") + len("AV_Volume_fraction:")
                idx_b = line.index("AV_cm^3/g:")
                vf = line[idx_a:idx_b]
                vf = float(vf)
            Path(f"./tmp/{cifid}.cif").unlink()
            Path(f"./tmp/{cifid}.vol").unlink()
        except:
            return None
        return vf
        
    vf = get_vf(cifid, cif_dir, ZEO_PATH, chan_radius, probe_radius, num_sample)
    topo = get_topo(cifid)
    # if vf is None or topo is None:
    #     return
    with id_prop_path.open('a') as f:
        f.write(f"{cifid},{vf},{topo}")
        f.write("\n")


@click.command()
@click.option('--root-dir', '-R', type=str)
@click.option('--cif-dir', '-C', type=str, default='cif')
@click.option('--prop-dir', '-P', type=str, default='.')
@click.option('--prop-name', type=str, default='vftopo')
@click.option('--n-process', '-N', type=int, default=1)
def buildVFTopo(root_dir, cif_dir, prop_dir, prop_name, n_process):
    root_dir = Path(root_dir)
    cif_dir = root_dir / cif_dir
    id_prop_dir = root_dir / prop_dir
    id_prop_path = id_prop_dir / f'{prop_name}.csv'

    cifids = [file.stem for file in cif_dir.iterdir() if file.suffix == '.cif']


    with id_prop_path.open('w') as f:
        f.write("cifid,voidfraction,topo")
        f.write("\n")
    with Pool(n_process) as pool:
        result = list(tqdm(pool.imap_unordered(partial(process, id_prop_path=id_prop_path, cif_dir=cif_dir, ZEO_PATH=ZEO_PATH, chan_radius=chan_radius, probe_radius=probe_radius, num_sample=num_sample), cifids), total=len(cifids)))


if __name__ == '__main__':
    buildVFTopo()
