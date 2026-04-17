"""
Generate sample VTU files for testing apeGmshViewer.

Creates:
  1. A simple 2D quad mesh with displacement + stress fields
  2. A 3D cantilever beam with modal results (PVD time-series)

Run:  python examples/demo_generate_vtu.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# apeGmsh.viz.VTKExport is a pure-numpy writer with no live Gmsh
# dependency at runtime, so we can import it directly now that the
# package is installed as a library.
from apeGmsh.viz.VTKExport import (
    VTK_HEXAHEDRON,
    VTK_QUAD,
    write_vtu,
    write_vtu_series,
)


def make_2d_plate(output_dir: Path) -> Path:
    """Create a 2D plate mesh with displacement and stress results."""
    # 10 x 5 plate, meshed with quads
    nx, ny = 20, 10
    Lx, Ly = 10.0, 5.0

    # Nodes
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.zeros((nx + 1) * (ny + 1)),
    ])

    # Elements (quads)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            elems.append([n0, n1, n2, n3])
    connectivity = np.array(elems)

    # Fake displacement field: cantilever bending
    x = coords[:, 0]
    y = coords[:, 1]
    ux = np.zeros_like(x)
    uy = -0.001 * x**2 * (3 * Lx - x) / (6 * Lx)  # cubic bending
    uz = np.zeros_like(x)
    displacement = np.column_stack([ux, uy, uz])

    # Fake stress field: linear bending stress
    elem_cx = np.mean(coords[connectivity, 0], axis=1)
    elem_cy = np.mean(coords[connectivity, 1], axis=1)
    sigma_xx = 100.0 * (elem_cy - Ly / 2) / (Ly / 2) * (elem_cx / Lx)

    # Von Mises (simplified for demo)
    von_mises = np.abs(sigma_xx)

    filepath = output_dir / "plate_results.vtu"
    write_vtu(
        filepath,
        coords,
        connectivity,
        vtk_cell_type=VTK_QUAD,
        point_data={
            "Displacement": displacement,
            "Displacement_Y": uy,
        },
        cell_data={
            "Stress_XX": sigma_xx,
            "VonMises": von_mises,
        },
    )
    print(f"  Created: {filepath}")
    return filepath


def make_3d_beam_modes(output_dir: Path) -> Path:
    """Create a 3D beam with modal analysis results (PVD series)."""
    # Simple beam 10 x 1 x 1
    nx, ny, nz = 20, 3, 3
    Lx, Ly, Lz = 10.0, 1.0, 1.0

    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    zs = np.linspace(0, Lz, nz + 1)

    coords = []
    for z in zs:
        for y in ys:
            for x in xs:
                coords.append([x, y, z])
    coords = np.array(coords)

    # Hex elements
    elems = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i
                n1 = n0 + 1
                n2 = n0 + (nx + 1) + 1
                n3 = n0 + (nx + 1)
                n4 = n0 + (ny + 1) * (nx + 1)
                n5 = n4 + 1
                n6 = n4 + (nx + 1) + 1
                n7 = n4 + (nx + 1)
                elems.append([n0, n1, n2, n3, n4, n5, n6, n7])
    connectivity = np.array(elems)

    # Generate 3 fake mode shapes
    x = coords[:, 0]
    modes = []
    freqs = [2.5, 15.7, 43.8]  # Hz

    for m_idx in range(3):
        n = m_idx + 1
        # Beam bending mode shape: sin(n*pi*x/L)
        phi_y = np.sin(n * np.pi * x / Lx)
        mode_shape = np.column_stack([
            np.zeros_like(x),
            phi_y,
            np.zeros_like(x),
        ])
        modes.append(mode_shape)

    # Write as PVD time series
    steps = []
    for mode_shape, freq in zip(modes, freqs):
        mag = np.linalg.norm(mode_shape, axis=1)
        steps.append({
            "time": freq,
            "point_data": {
                "ModeShape": mode_shape,
                "Magnitude": mag,
            },
        })

    base = output_dir / "beam_modes"
    files = write_vtu_series(
        str(base),
        coords,
        connectivity,
        vtk_cell_type=VTK_HEXAHEDRON,
        steps=steps,
    )
    print(f"  Created: {files[-1]} + {len(files)-1} VTU steps")
    return files[-1]  # Return PVD path


def main():
    output_dir = Path(__file__).resolve().parent.parent / "demo_data"
    output_dir.mkdir(exist_ok=True)

    print("Generating sample VTU files for apeGmshViewer...")
    make_2d_plate(output_dir)
    make_3d_beam_modes(output_dir)
    print(f"\nAll files saved to: {output_dir}")
    print(f"\nLaunch the viewer:")
    print(f"  python -m apeGmshViewer {output_dir / 'plate_results.vtu'}")


if __name__ == "__main__":
    main()
