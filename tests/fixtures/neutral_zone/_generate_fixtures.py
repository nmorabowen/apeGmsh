"""Regenerate the committed neutral-zone golden ``box.h5``.

The golden is the **conformance artifact** for the published neutral-zone
layout (``docs/design/model-h5-neutral-zone.md``): an external consumer
that cannot import apeGmsh — the apeWorkbench browser reader is the first
— implements against the page and checks itself against this file.

Run with the project environment (needs ``gmsh``)::

    python tests/fixtures/neutral_zone/_generate_fixtures.py

The model is deliberately tiny and **fully deterministic**: a unit cube
meshed transfinitely at ``n=3`` is exactly 27 nodes and 8 hexahedra, so
the committed bytes only move when the emitter actually changes.  Only
``/meta@created_iso`` varies between runs.

Fixtures:

  * ``box.h5`` — unit cube, transfinite ``n=3`` (27 nodes / 8 ``hex8``),
    written by ``FEMData.to_h5`` (broker-only — no ``/opensees/`` zone).
    Covers every group the note publishes:

      - ``/meta`` (+ the ``/meta/lineage`` sub-group),
      - ``/nodes`` (ids, coords, module_label),
      - ``/elements/hex8`` (ids, connectivity, module_label),
      - ``/physical_groups`` with **both** sides populated and two
        entries: ``Body`` (dim 3, carries ``element_ids``) and ``Base``
        (dim 2, whose ``element_ids`` deliberately reference quad faces
        that are NOT in any ``/elements/{type}`` group — the dangling-id
        case a consumer must tolerate),
      - ``/labels`` from the geometry label ``box``,
      - ``/mesh_selections/base_nodes``, a flat node-level selection.

    Deliberately absent (absence is itself part of the contract): the
    optional ``/nodes/ndf`` and ``/nodes/provenance`` datasets, and every
    record group (``/loads``, ``/masses``, ``/constraints``, …) — this
    model declares none.
"""
from __future__ import annotations

import os

import gmsh

from apeGmsh import apeGmsh


HERE = os.path.dirname(os.path.abspath(__file__))


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-6) -> int:
    """Return the boundary face of ``volume_tag`` whose centroid sits at z."""
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of volume {volume_tag} at z={z}")


def _box(path: str) -> None:
    g = apeGmsh(model_name="neutral_box", verbose=False)
    g.begin()
    try:
        vol = g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                       label="box")
        g.model.sync()
        base = _face_at_z(vol, 0.0)

        # Two PGs: a volume (element_ids land in /elements/hex8) and a
        # face (element_ids reference quads that dim=3 extraction drops).
        g.physical.add_volume("box", name="Body")
        g.physical.add(2, [base], name="Base")

        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)

        # A flat node-level mesh selection (the z = 0 corner nodes).
        g.mesh_selection.select().in_box(
            (-0.01, -0.01, -0.01), (1.01, 1.01, 0.01),
        ).save_as("base_nodes")

        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()

    fem.to_h5(path)


FIXTURES = [
    ("box.h5", _box),
]


def main() -> None:
    for name, build in FIXTURES:
        path = os.path.join(HERE, name)
        build(path)
        size = os.path.getsize(path)
        print(f"wrote {name}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
