"""
Quick test: load frame_2D.dxf via pyGmsh and verify layers → physical groups.

Run from the pyGmsh root:
    python acad/test_dxf_import.py
"""
from pathlib import Path
import sys

# Ensure the local source is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apeGmsh import apeGmsh

dxf_path = Path(__file__).with_name("frame_2D.dxf")

with apeGmsh(verbose=True) as g:
    layers = g.model.io.load_dxf(dxf_path)

    print("\n--- Layer summary ---")
    for name, dim_tags in layers.items():
        for dim, tags in dim_tags.items():
            print(f"  Layer '{name}'  dim={dim}  entities={tags}")

    # Verify expected layers exist
    expected_layers = {"C80x80", "C40x40", "V30x50"}
    found = set(layers.keys())
    assert expected_layers.issubset(found), (
        f"Missing layers: {expected_layers - found}"
    )

    # Verify physical groups were created
    import gmsh
    pgs = gmsh.model.getPhysicalGroups(-1)
    print(f"\n--- Physical groups ({len(pgs)}) ---")
    for dim, pg_tag in pgs:
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag)
        print(f"  dim={dim}  tag={pg_tag}  name='{name}'  entities={list(ents)}")

    # Generate mesh to make sure geometry is valid
    g.mesh.generate(1)
    print("\n1D mesh generated successfully.")

    # Optional: open GUI to visually inspect
    # g.model.gui()

print("\nAll checks passed!")
