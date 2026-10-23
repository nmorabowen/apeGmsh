"""Write ``apeGmshViewer/fixtures/loads3d.h5``: a 3-D frame + slab whose
neutral ``/loads`` zone comes from apeGmsh's own resolver and writer.

This generator imports apeGmsh, so it lives here, outside ``apeGmshViewer/``:
the app never imports or runs apeGmsh (ADR 0112 D4,
``scripts/check_app_wall.py``). Run it from the repository root with the
venv interpreter and the checkout's ``src``:

    $env:PYTHONPATH = "src"
    python scripts/make_viewer_loads_fixtures.py

The point of the file is the writer's real layout (``element_id``,
``load_type``, ``params_json``; ``node_id``, ``force_xyz``), which a
hand-written table had got wrong (#2385 review, finding 4). The session
writes the file at ``save_to`` as it closes, with a ``loads3d.geometry.h5``
sibling the tests do not read; this script deletes the sibling.

Geometry (metres, one element per line):

* ``Column``: (0, 0, 0) -> (0, 0, 3), along +Z;
* ``Girder``: (0, 0, 3) -> (4, 0, 3), along +X;
* ``Slab``: the rectangle x in [0, 4], y in [2, 4], z = 3.

Cases, all ``target_form="element"`` except ``point``:

* ``dead``: 10 down the girder (``direction="z"``, magnitude -10, so the
  record is the GLOBAL ``wz=-10``), and pressure 7 on the slab;
* ``wind``: 5 along +X on the column (global ``wx=5``);
* ``trap``: a trapezoid 10 -> 20 down the girder over x/L in [0.25, 0.75];
* ``self``: gravity on the slab, a ``bodyForce`` record the viewer cannot
  draw (it must say so);
* ``wind_slab``: a traction (2, 0, 0) on the slab, a ``surfacePressure``
  with ``normal: false`` the viewer cannot draw either;
* ``point``: a nodal force of 3 along +Y at the girder's far end, and a
  prescribed ``uz = -0.005`` there (``/loads/sp/point``).
"""

from __future__ import annotations

from pathlib import Path

from apeGmsh import apeGmsh

FIXTURES = Path(__file__).resolve().parents[1] / "apeGmshViewer" / "fixtures"


def main() -> None:
    out = FIXTURES / "loads3d.h5"
    with apeGmsh(model_name="loads3d", save_to=out, verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(0.0, 0.0, 3.0)
        p2 = geo.add_point(4.0, 0.0, 3.0)
        column = geo.add_line(p0, p1)
        girder = geo.add_line(p1, p2)
        slab = geo.add_rectangle(0.0, 2.0, 3.0, 4.0, 2.0)
        g.model.sync()
        g.physical.add(1, [column], name="Column")
        g.physical.add(1, [girder], name="Girder")
        g.physical.add(2, [slab], name="Slab")
        g.physical.add(0, [p2], name="Tip")
        with g.loads.case("dead"):
            g.loads.line("Girder", magnitude=-10.0, direction="z", target_form="element")
            g.loads.surface.pressure("Slab", 7.0, target_form="element")
        with g.loads.case("wind"):
            g.loads.line("Column", magnitude=5.0, direction="x", target_form="element")
        with g.loads.case("trap"):
            g.loads.line("Girder", magnitude=(10.0, 20.0), direction=(0.0, 0.0, -1.0),
                         span=(0.25, 0.75), target_form="element")
        with g.loads.case("self"):
            g.loads.gravity("Slab", g=(0.0, 0.0, -9.81), density=2.5, target_form="element")
        with g.loads.case("wind_slab"):
            g.loads.surface.traction("Slab", (2.0, 0.0, 0.0), target_form="element")
        with g.loads.case("point"):
            g.loads.point.force("Tip", force=(0.0, 3.0, 0.0))
        with g.displacements.case("point"):
            g.displacements.point("Tip", dofs=[0, 0, 1], values=[0.0, 0.0, -0.005])
        g.mesh.sizing.set_global_size(10.0)
        g.mesh.generation.generate(2)
    # The session writes ``out`` (and a geometry sibling) as it closes.
    out.with_name("loads3d.geometry.h5").unlink(missing_ok=True)
    print(out)


if __name__ == "__main__":
    main()
