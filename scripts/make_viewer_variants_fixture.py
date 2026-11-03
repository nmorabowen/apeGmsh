"""Write the VU10 variant pair (ADR 0132 D8, D9; #2694). Synthetic values.

    $env:PYTHONPATH = "src"
    & C:\\Users\\nmb\\venv\\opensees_venv\\Scripts\\python.exe scripts/make_viewer_variants_fixture.py

run from the repository root, with ``PYTHONPATH`` naming this checkout's ``src``
(the editable install may point at another checkout). It lives here, not in the
app tree, because it imports apeGmsh (ADR 0112 D4, the App wall). It writes, into
``apeGmshViewer/test/fixtures/vu10/``, one study of two variants of one model
that differ in one parameter, the plate's Young's modulus:

* ``plate_uncracked.h5`` and ``plate_uncracked.ladruno``: E = 30e9 (Ie/Ig 1.0);
* ``cracked/plate_cracked.h5`` and ``cracked/plate_cracked.ladruno``:
  E = 10.5e9 (Ie/Ig 0.35), in a subfolder (D8 allows one);
* ``study.variants.json``: the manifest, written by VU9a-py's
  ``apeGmsh.results.user.write_variants`` (#2806) to
  ``architecture/schemas/variants.schema.json``;
* ``codes.golden.json``: Python's ``read_variants`` verdict (the rows that
  stand, each refused row's code, or the file's code) on the schema's example
  corpus and on one case per code it does not reach. Python's codes are the
  canonical ones (#2806); ``apeGmshViewer/test/variants/manifest.test.ts``
  holds the app's reader to them. ``--codes-only`` writes this file alone.

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
``apeGmshViewer/test/variants/difference.test.ts``.
"""

from __future__ import annotations

import json
from pathlib import Path

import gmsh
import numpy as np

from apeGmsh import apeGmsh

HERE = Path(__file__).resolve().parents[1] / "apeGmshViewer" / "test" / "fixtures" / "vu10"
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


# -- The refusal codes, by Python (#2806, the canonical reader) -----------------------

#: The files every code case lays out beside its manifest (empty files suffice: a
#: reader judges the manifest and the disk, never a model's content).
CODE_FILES = ("slab_uncracked.h5", "slab_cracked.h5", "bischoff/slab_pass2.h5", "slab_cracked.user.h5")
EXAMPLES = Path(__file__).resolve().parents[1] / "architecture" / "schemas" / "examples"


def _row(label: object, path: object = "slab_uncracked.h5", params: object = None) -> dict:
    return {"label": label, "path": path, "params": {} if params is None else params}


def _doc(rows: list, **top: object) -> str:
    return json.dumps({"format": "apegmsh-variants", "version": "1.0.0", "variants": rows, **top})


def code_cases() -> list[tuple[str, str]]:
    """``(name, manifest text)``: the schema corpus, then one case per code the corpus does not reach."""
    cases = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(EXAMPLES.glob("*.json"))]
    cases += [(f"invalid/{p.name}", p.read_text(encoding="utf-8")) for p in sorted((EXAMPLES / "invalid").glob("*.json"))]
    ok = _row("Uncracked")
    cases += [
        ("row-not-object", _doc([1, ok])),
        ("row-unknown-key", _doc([{**_row("A"), "colour": "#fff"}, _row("B", "slab_cracked.h5")])),
        ("row-no-label", _doc([{"path": "slab_uncracked.h5", "params": {}}, _row("B", "slab_cracked.h5")])),
        ("row-params-list", _doc([_row("A", params=[1]), _row("B", "slab_cracked.h5")])),
        ("row-params-key-empty", _doc([_row("A", params={"": 1}), _row("B", "slab_cracked.h5")])),
        ("row-label-not-nfc", _doc([_row("Cafe\u0301"), _row("B", "slab_cracked.h5")])),
        ("duplicate-label-one-bad", _doc([_row("A"), _row("A", "C:/x.h5")])),
        ("duplicate-label", _doc([_row("A"), _row("A", "slab_cracked.h5"), _row("B", "bischoff/slab_pass2.h5")])),
        ("duplicate-path", _doc([_row("A"), _row("B"), _row("C", "slab_cracked.h5")])),
        ("missing", _doc([_row("A"), _row("B", "nope.h5")])),
        ("top-unknown-key", _doc([ok], extra=1)),
        ("variants-empty", _doc([])),
        ("variants-not-list", json.dumps({"format": "apegmsh-variants", "version": "1.0.0", "variants": {}})),
        ("version-major", _doc([ok]).replace('"1.0.0"', '"2.0.0"')),
        ("version-old-major", _doc([ok]).replace('"1.0.0"', '"0.9.0"')),
        ("version-two-parts", _doc([ok]).replace('"1.0.0"', '"1.0"')),
        ("format-wrong", _doc([ok]).replace("apegmsh-variants", "apegmsh-variant")),
        ("root-list", "[" + _doc([ok]) + "]"),
        ("not-json", "{ not json"),
        ("nan-token", _doc([_row("A", params={"x": 1})]).replace('"x": 1', '"x": NaN')),
    ]
    return cases


def codes() -> None:
    """Write ``codes.golden.json``: Python's verdict on each case, which the app's reader must match."""
    import tempfile

    from apeGmsh.results.user import VariantsRefused, read_variants

    out = []
    for name, text in code_cases():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for f in CODE_FILES:
                (root / f).parent.mkdir(parents=True, exist_ok=True)
                (root / f).write_bytes(b"")
            m = root / "s.variants.json"
            m.write_text(text, encoding="utf-8")
            try:
                r = read_variants(m)
                verdict: dict = {"rows": [v.row for v in r.variants], "refused": [[x.row, x.reason] for x in r.refused]}
            except VariantsRefused as exc:
                verdict = {"file": exc.reason}
        out.append({"name": name, "text": text, "python": verdict})
    target = HERE / "codes.golden.json"
    target.write_text(json.dumps({"files": list(CODE_FILES), "cases": out}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {target.name}: {len(out)} cases")


if __name__ == "__main__":
    import sys

    if "--codes-only" not in sys.argv:
        main()
    codes()
