# MOFTransformer version 2.0.0
import subprocess
from pathlib import Path
import click

from .build_ff import GRIDAY_PATH
from fieldgnn.utils.log import title, err, start, end


class InstallationError(Exception):
    def __init__(self, error_message=None):
        self.error_message = error_message

    def __str__(self):
        if self.error_message:
            return f"Installation Error : {self.error_message}"
        return "Installation Error"

@click.group()
def cli():
    pass


@cli.command()
def install_make():
    title("INSTALL MAKE")
    start("Downloading gcc=9.5.0")
    ps = subprocess.run("conda install -c conda-forge gcc=9.5.0 -y".split())
    if ps.returncode:
        raise InstallationError(ps.stderr)
    else:
        end("Successfully download")
    start("Downloading gxx=9.5.0")
    ps = subprocess.run("conda install -c conda-forge gxx=9.5.0 -y".split())
    if ps.returncode:
        raise InstallationError(ps.stderr)
    else:
        end("Successfully download")
    start("Downloading make=4.2.1")
    ps = subprocess.run("conda install -c anaconda make=4.2.1 -y".split())
    if ps.returncode:
        raise InstallationError(ps.stderr)
    else:
        end("Successfully download")
    title("INSTALL MAKE END")


@cli.command()
def make_griday():
    title("MAKEING GRIDAY")
    dir_griday = Path(GRIDAY_PATH).parent.parent
    if not dir_griday.exists():
        raise InstallationError(f"Invalid path specified : {dir_griday}")
    start("Makeing GRIDAY")
    ps = subprocess.run(["make"], cwd=dir_griday)
    if ps.returncode:
        raise InstallationError(ps.stderr)
    ps = subprocess.run(["make"], cwd=dir_griday / "scripts")
    if ps.returncode:
        raise InstallationError(ps.stderr)
    end("Successfully make")
    if not Path(GRIDAY_PATH).exists():
        raise InstallationError(f"GRIDAY is not installed. Please try again.")

    start("Checking GIRDAY")
    ps = subprocess.run(
        [str(GRIDAY_PATH)],
        cwd=dir_griday,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if (
        ps.stderr
        == b"./make_egrid spacing atom_type force_field input_cssr egrid_stem\n"
    ):
        end(f"GRIDAY is installed to {dir_griday}")
    else:
        err(ps.stdout, ps.stderr)
        err(
            f"GRIDAY does not installed correctly. Please uninstall griday and re-install."
        )
    title("MAKE GRIDAY END")

@cli.command()
def uninstall_griday():
    dir_griday = Path(GRIDAY_PATH).parent.parent
    if not dir_griday.exists():
        raise InstallationError(f"Invalid path specified : {dir_griday}")
    title("UNINSTALLING GRIDAY")
    ps = subprocess.run(["make", "clean"], cwd=dir_griday)
    if ps.returncode:
        raise InstallationError(ps.stderr)
    ps = subprocess.run(["make", "clean"], cwd=dir_griday / "scripts")
    if ps.returncode:
        raise InstallationError(ps.stderr)

    if not Path(GRIDAY_PATH).exists():
        end(f"GRIDAY is uninstalled")
    else:
        raise InstallationError()
    title("UNINSTALL GRIDAY END")


if __name__ == '__main__':
    cli()
