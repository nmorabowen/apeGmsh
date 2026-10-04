"""
Minimal test: just create the geometry and mesh.
"""

import sys
print("Test script starting...", file=sys.stderr)
sys.stderr.flush()

print("Importing apeGmsh...", file=sys.stderr)
sys.stderr.flush()
from apeGmsh import apeGmsh

print("Creating apeGmsh context...", file=sys.stderr)
sys.stderr.flush()

try:
    with apeGmsh(model_name="test_truss", save_to="out_f/test.h5") as g:
        print("Context opened successfully", file=sys.stderr)
        sys.stderr.flush()

        # Create two simple points
        p1 = g.model.geometry.add_point(0, 0, 0, label="p1")
        p2 = g.model.geometry.add_point(1, 0, 0, label="p2")
        print("Points created", file=sys.stderr)
        sys.stderr.flush()

        # Create a line
        line = g.model.geometry.add_line(p1, p2, label="line1")
        print("Line created", file=sys.stderr)
        sys.stderr.flush()

        # Physical group
        g.physical.add(1, [line], name="elements")
        print("Physical group added", file=sys.stderr)
        sys.stderr.flush()

        # Mesh
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=1)
        print("Mesh generated", file=sys.stderr)
        sys.stderr.flush()

        # Get FEMData
        fem = g.mesh.queries.get_fem_data(dim=1)
        print("FEMData obtained", file=sys.stderr)
        print(fem.info.summary(), file=sys.stderr)
        sys.stderr.flush()

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)

print("Test completed successfully", file=sys.stderr)
sys.stderr.flush()
