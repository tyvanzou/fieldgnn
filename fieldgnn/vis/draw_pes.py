import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import click
from pathlib import Path
import torch
from pymatgen.core.lattice import Lattice
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)


@click.command()
@click.option(
    "--root-dir",
    "-R",
    type=str,
    required=True,
    help="Root directory containing graph/ and irfeat/ folders",
)
@click.option("--matid", "-M", type=str, required=True, help="Material ID")
@click.option(
    "--output-dir", "-O", type=str, required=True, help="Output directory for plots"
)
@click.option(
    "--plane",
    "-P",
    type=click.Choice(["x", "y", "z"]),
    default="z",
    help="Plane to sample (x, y, or z)",
)
@click.option(
    "--fixed-coord",
    type=float,
    required=True,
    help="Coordinate value (Å) for PES slice along the selected plane",
)
@click.option(
    "--tolerance", type=float, default=0.5, help="Tolerance around fixed coordinate (Å)"
)
@click.option(
    "--sample-points",
    "-N",
    type=int,
    default=50000,
    help="Max points to use for interpolation",
)
@click.option(
    "--max-energy", type=float, default=200, help="Maximum energy value to display"
)
@click.option(
    "--min-energy", type=float, default=-200, help="Minimum energy value to display"
)
@click.option(
    "--grid-density", type=int, default=200, help="Density of interpolation grid"
)
def main(
    root_dir,
    matid,
    output_dir,
    plane,
    fixed_coord,
    tolerance,
    sample_points,
    min_energy,
    max_energy,
    grid_density,
):
    # Setup paths and directories
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load pos
    pos_dir_name = 'grid'
    pos_dir = root_dir / pos_dir_name
    pos = np.load(pos_dir / f'{matid}.npy') # [num_grid_a, num_grid_b, num_grid_c, 6] 3 * cart_coords & 3 * frac_coords
    cart_coords = pos[..., :3]  # Extract Cartesian coordinates

    # 1. Load lattice information
    graph_dir_name = 'graph'
    graph_path = root_dir / graph_dir_name / f"{matid}.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    graph_data = torch.load(graph_path, weights_only=False)
    lattice = Lattice(graph_data.cell.numpy())

    # 2. Load energy grid data
    irfeat_files = list(root_dir.glob(f"ff/{matid}.npy"))
    if not irfeat_files:
        raise FileNotFoundError(f"No irfeat file found for {matid} in {root_dir}")
    grid_data = np.load(irfeat_files[0])  # shape: (num_grid_a, num_grid_b, num_grid_c)
    print("GRID DATA SHAPE: ", grid_data.shape)
    grid_data = np.transpose(grid_data, (2, 1, 0))

    # 3. Get absolute coordinates from pos array
    abs_coords = cart_coords.reshape(-1, 3)  # Flatten to (num_grid_a*num_grid_b*num_grid_c, 3)
    energies = grid_data.ravel()  # shape: (N^3,)

    # Determine which axis to filter based on the selected plane
    if plane == "x":
        axis_idx = 0
        plot_axes = (1, 2)  # y and z for plotting
    elif plane == "y":
        axis_idx = 1
        plot_axes = (0, 2)  # x and z for plotting
    else:  # z
        axis_idx = 2
        plot_axes = (0, 1)  # x and y for plotting

    # Apply all filters
    valid_idx = np.logical_and.reduce(
        [
            energies >= min_energy,
            energies <= max_energy,
            np.abs(abs_coords[:, axis_idx] - fixed_coord) < tolerance,
        ]
    )

    filtered_coords = abs_coords[valid_idx]
    filtered_energies = energies[valid_idx]

    print(
        f"Found {len(filtered_energies)} points within {plane}={fixed_coord:.2f}±{tolerance:.2f}Å and energy range"
    )

    # 5. Downsample if needed
    if len(filtered_energies) > sample_points:
        idx = np.random.choice(
            len(filtered_energies), size=sample_points, replace=False
        )
        filtered_coords = filtered_coords[idx]
        filtered_energies = filtered_energies[idx]
        print(f"Downsampled to {sample_points} points for interpolation")

    # 6. Create interpolation grid
    xi = np.linspace(
        filtered_coords[:, plot_axes[0]].min(),
        filtered_coords[:, plot_axes[0]].max(),
        grid_density,
    )
    yi = np.linspace(
        filtered_coords[:, plot_axes[1]].min(),
        filtered_coords[:, plot_axes[1]].max(),
        grid_density,
    )
    xi, yi = np.meshgrid(xi, yi)

    # 7. Interpolate energy surface
    zi = griddata(
        (filtered_coords[:, plot_axes[0]], filtered_coords[:, plot_axes[1]]),
        filtered_energies,
        (xi, yi),
        method="cubic",
        fill_value=np.nan,
    )

    # 8. Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surf = ax.plot_surface(
        xi,
        yi,
        zi,
        cmap="coolwarm",
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
        vmin=min_energy,
        vmax=max_energy,
    )

    # Plot lattice vectors
    origin = np.zeros(3)
    vectors = lattice.matrix
    
    # Draw lattice vectors
    for i, vec in enumerate(vectors):
        a = Arrow3D(
            [origin[0], vec[0]],
            [origin[1], vec[1]],
            [origin[2], vec[2]],
            mutation_scale=15,
            lw=2,
            arrowstyle="-|>",
            color=f"C{i}",
            label=f"Lattice vector {i+1}",
        )
        ax.add_artist(a)
    
    # Draw unit cell edges
    points = [
        origin,
        vectors[0],
        vectors[1],
        vectors[2],
        vectors[0] + vectors[1],
        vectors[0] + vectors[2],
        vectors[1] + vectors[2],
        vectors[0] + vectors[1] + vectors[2]
    ]
    
    # Connect the points to form the unit cell
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7)
    ]
    
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            'k-', linewidth=1, alpha=0.5
        )

    # Add colorbar
    cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label("Adsorption Energy (eV)", rotation=270, labelpad=20)

    # Set labels and title
    axis_labels = ["X (Å)", "Y (Å)", "Z (Å)"]
    ax.set_xlabel(axis_labels[plot_axes[0]], labelpad=10)
    ax.set_ylabel(axis_labels[plot_axes[1]], labelpad=10)
    ax.set_zlabel("Energy (eV)", labelpad=10)
    ax.set_title(
        f"Potential Energy Surface at {plane.upper()} = {fixed_coord:.2f} ± {tolerance:.2f} Å\n"
        f"{matid} | Lattice: {lattice.abc[0]:.2f}×{lattice.abc[1]:.2f}×{lattice.abc[2]:.2f} Å"
    )

    # Add legend
    ax.legend()

    # Adjust view
    ax.view_init(elev=90, azim=0)
    ax.set_box_aspect([1, 1, 0.5])  # Adjust z-axis scaling

    # 9. Save figure
    output_path = output_dir / f"{matid}_pes_{plane}{fixed_coord:.2f}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"PES plot saved to {output_path}")


if __name__ == "__main__":
    main()
