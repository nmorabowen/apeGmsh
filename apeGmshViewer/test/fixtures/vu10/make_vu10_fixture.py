"""Write the VU10 variant pair (ADR 0132 D8, D9; #2694). Synthetic values.

    $env:PYTHONPATH = "src"
    & C:\\Users\\nmb\\venv\\opensees_venv\\Scripts\\python.exe apeGmshViewer/test/fixtures/vu10/make_vu10_fixture.py

run from the repository root, with ``PYTHONPATH`` naming this checkout's ``src``
(the editable install may point at another checkout). It writes, beside this
file, one study of two variants of one model that differ in one parameter, the
plate's Young's modulus:

* ``plate_uncracked.h5`` and ``plate_uncracked.ladruno``: E = 30e9 (Ie/Ig 1.0);
* ``cracked/plate_cracked.h5`` and ``cracked/plate_cracked.ladruno``:
  E = 10.5e9 (Ie/Ig 0.35), in a subfolder (D8 allows one);
* ``study.variants.json``: the manifest, written by VU9a-py's
  ``apeGmsh.results.user.write_variants`` (#2806) to
  ``architecture/schemas/variants.schema.json``.

Each model is composed by the bridge (``apeSees(fem)``, ``ops.h5``) from one
gmsh mesh: a 2 x 1 rectangle of 4 x 2 ``FourNodeQuad`` (``Panel``) at z = 0,
declared ``ndm=2``. The two meshes are the same, so the two files have one
``fem_hash`` and two ``model_hash`` values (the material differs).

The ``.ladruno`` values are **synthetic**: no engine solved them
(``PROVENANCE@opensees_backend`` is ``"unknown"``). Each is written by
``write_nodal_history`` and stamped against its own model. The closed form,
with ``s = 0.01 (k + 1)`` at step ``k`` of 3 and ``r`` the variant's Ie/Ig::

    Ux = -s x y / r,   Uy = 0.5 s x^2 / r

so the difference Uncracked minus Cracked at a node is
``u(r=1) - u(r=0.35) = u(r=1) (1 - 1 / 0.35)``, the oracle of
``test/variants/difference.test.ts``.
"""

from __future__ import annotations

from pathlib import Path

import gmsh
import numpy as np

from apeGmsh import apeGmsh

HERE = Path(__file__).resolve().parent
STEPS = 3
VARIANTS = (
    # label, path relative to the manifest, Ie/Ig, E
    ("Uncracked", "plate_uncracked.h5", 1.0, 30e9),
    ("Cracked", "cracked/plate_cracked.h5", 0.35, 30e9 * 0.35),
)


def displacement(xyz: np.ndarray, k: int, ratio: float) -> np.ndarray:
    """``DATA[n x 2]`` of step ``k``: the closed form above."""
    s = 0.01 * (k + 1)
    x, y = xyz[:, 0], xyz[:, 1]
    return np.stack([-s * x * y / ratio, 0.5 * s * x * x / ratio], axis=-1)


def mesh(tmp: Path):
    with apeGmsh(model_name="plate", verbose=False, units="N_m_s", save_to=tmp / "plate.session.h5") as g:
        p0 = gmsh.model.geo.addPoint(0, 0, 0)
        edge = gmsh.model.geo.extrude([(0, p0)], 2, 0, 0, numElements=[4])
        surf = gmsh.model.geo.extrude([edge[1]], 0, 1, 0, numElements=[2], recombine=True)
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(2, [surf[1][1]], name="Panel")
        g.mesh.generation.generate(2)
        return g.mesh.queries.get_fem_data(remove_orphans=True)


def main() -> None:
    import tempfile

    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.opensees import apeSees
    from apeGmsh.results.readers._ladruno_writer import write_nodal_history
    from apeGmsh.results.user import VariantDef, read_variants, write_variants

    with tempfile.TemporaryDirectory() as tmp:
        fem = mesh(Path(tmp))
    rows = []
    for label, rel, ratio, E in VARIANTS:
        model_path = HERE / rel
        model_path.parent.mkdir(parents=True, exist_ok=True)
        ops = apeSees(fem)
        ops.model(ndm=2, ndf=2)
        plate = ops.nDMaterial.ElasticIsotropic(E=E, nu=0.2, rho=2400.0, name="plate")
        ops.element.FourNodeQuad(pg="Panel", thickness=0.2, material=plate)
        ops.h5(str(model_path))
        back = FEMData.from_h5(str(model_path))
        ids = np.asarray(back.nodes.ids, dtype=np.int64)
        xyz = np.asarray(back.nodes.coords, dtype=np.float64).reshape(-1, 3)
        disp = np.stack([displacement(xyz, k, ratio) for k in range(STEPS)])
        write_nodal_history(
            model_path.with_suffix(".ladruno"), model_h5=model_path, node_ids=ids, coords=xyz, ndm=2,
            kind="static", time=(np.arange(STEPS) + 1.0) / STEPS, results={"DISPLACEMENT": disp},
            source="viewer-fixture", opensees_backend="unknown",
        )
        rows.append(VariantDef(label=label, path=rel, params={"state": label.lower(), "Ie/Ig": ratio, "E": E}))
        print(f"wrote {rel} and its .ladruno: {ids.size} nodes, E = {E:g}")
    manifest = HERE / "study.variants.json"
    manifest.unlink(missing_ok=True)
    write_variants(manifest, rows)
    print(f"wrote {manifest.name}: {[r.label for r in read_variants(manifest).variants]}")


if __name__ == "__main__":
    main()
