"""Write the app's ``/entities`` fixtures (VU8 #2692, viewer half).

``python tests/results/user/make_app_entities_fixtures.py`` writes, into
``apeGmshViewer/test/reader/fixtures/entities/``:

* ``frame_cut.user.h5``: a skewed section cut through the STKO elastic frame
  (``fixtures/stko_elastic_frame``) and, on it, the resultant the cut kernel
  (``STKO_to_python.cuts``) computes for every step of the run, about the
  cut's reference;
* ``rows.user.h5``: two cuts, a storey and a point set, then edited with h5py
  so a cut is unnamed, a normal is not of unit length and a host element is
  below -1, with three ``ON_ENTITIES`` results on them;
* ``goldens.json``: what Python reads from both files (:func:`goldens`), and
  the kernel's resultant computed from the original cut, never from the file.

The app reads the ``.user.h5`` files and must equal ``goldens.json``
(``apeGmshViewer/test/reader/userEntities.test.ts``);
``test_user_entities_app_fixtures.py`` holds the committed goldens to the
Python reader and the kernel, so neither side can drift alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:  # run as a script from the repository root
    sys.path.insert(0, str(REPO))

from apeGmsh.cuts import SectionCutDef  # noqa: E402
from apeGmsh.results.user import (  # noqa: E402
    CutEntity,
    Entities,
    PointSet,
    StateAddress,
    Storey,
    UserFile,
    UserResult,
    write_user_file,
)

OUT = REPO / "apeGmshViewer" / "test" / "reader" / "fixtures" / "entities"
FRAME = Path(__file__).resolve().parent / "fixtures" / "stko_elastic_frame"
FEM_HASH = "fedcba9876543210fedcba9876543210"
MODEL_HASH = "0123456789abcdef0123456789abcdef"
COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

#: The cut of ``frame_cut.user.h5``: skewed, through both columns, off mid-height.
FRAME_CUT = dict(plane_point=(2500.0, 0.0, 1500.0), plane_normal=(0.3, -0.2, 1.0), label="frame cut")
FRAME_REFERENCE = (123.0, -45.0, 1500.0)


def _model(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        lineage = f.create_group("meta/lineage")
        lineage.attrs["fem_hash"] = FEM_HASH
        lineage.attrs["model_hash"] = MODEL_HASH
    return path


def kernel_resultant() -> tuple[SectionCutDef, np.ndarray, np.ndarray]:
    """The original def, and the kernel's ``F`` and ``M`` about the reference, ``(S, 3)`` each."""
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(FRAME), "results", verbose=False)
    eids = tuple(int(e) for e in ds.elements_info["dataframe"]["element_id"])
    cut = SectionCutDef(element_ids=eids, **FRAME_CUT)
    r = ds.section_cut(spec=cut.to_spec(), model_stage="MODEL_STAGE[1]")
    return cut, np.asarray(r.F, dtype="f8"), np.asarray(r.moment_about(FRAME_REFERENCE), dtype="f8")


def write_frame_cut(path: Path, model: Path) -> None:
    cut, force, moment = kernel_resultant()
    states = tuple(StateAddress("stage", "stage_0", k, 0) for k in range(force.shape[0]))
    result = UserResult(
        name="frame cut resultant", target="entities", basis="model", linear=True, states=states,
        components=COMPONENTS, data=np.concatenate([force, moment], axis=1)[:, None, :],
        entity_kind=("cut",), entity_row=np.array([0], dtype="i4"))
    write_user_file(path, model, results=[result],
                    entities=Entities(cuts=(cut.to_entity(reference=FRAME_REFERENCE),)))


def _rows_entities() -> Entities:
    return Entities(
        cuts=(
            CutEntity(name="pier P1 base", point=(0.0, 0.0, 3.0), normal=(0.0, 0.0, 1.0),
                      reference=(0.5, 0.0, 3.0), element_ids=np.array([101, 102], dtype="i4")),
            CutEntity(name="unnamed later", point=(2.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                      reference=(2.0, 1.0, 1.5), element_ids=np.array([7], dtype="i4"),
                      side="negative",
                      polygon=np.array([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 3.0]])),
            CutEntity(name="Delta_after (L/480)", point=(0.0, 1.0, 0.0), normal=(0.0, 1.0, 0.0),
                      reference=(0.0, 1.0, 0.0), element_ids=np.array([3, 4, 5], dtype="i4")),
        ),
        storeys=(Storey(name="L1", level=3.0, reference=(1.0, 1.0, 3.0),
                        node_ids=np.array([5, 6, 7, 8], dtype="i4")),),
        point_sets=(
            PointSet(name="plan points", xyz=np.array([[0.0, 0.0, 3.0], [1.0, 2.0, 3.0]]),
                     host_element=np.array([-1, 102], dtype="i4")),
            PointSet(name="drift points", xyz=np.array([[2.0, 2.0, 3.0]])),
        ),
    )


def _on(name: str, kind: str, row: int) -> UserResult:
    return UserResult(name=name, target="entities", basis="model", components=("v",),
                      data=np.array([[[0.25]]]), entity_kind=(kind,),
                      entity_row=np.array([row], dtype="i4"))


def write_rows(path: Path, model: Path) -> None:
    write_user_file(path, model, entities=_rows_entities(), results=[
        _on("on the storey", "storey", 0),
        _on("on the unnamed cut", "cut", 1),
        _on("on a row not held", "point_set", 1),
        _on("on the second point set", "point_set", 1),
    ])
    with h5py.File(path, "r+") as f:  # rows a writer refuses; the reader refuses each alone
        f["entities/cuts/name"][1] = ""
        f["entities/cuts/normal"][0] = (0.0, 0.0, 2.0)
        f["entities/point_sets/host_element"][0] = -5
        f["results/r2/RESULTS/ON_ENTITIES/values/ENTITY_ROW"][0] = 7


def _vec(a: Any) -> list[float]:
    return [float(x) for x in np.asarray(a, dtype="f8").ravel()]


def _entities(e: Entities) -> dict[str, Any]:
    return {
        "cuts": [{"name": c.name, "side": c.side, "point": _vec(c.point), "normal": _vec(c.normal),
                  "reference": _vec(c.reference), "elementIds": [int(x) for x in c.element_ids],
                  "polygon": [_vec(v) for v in c.polygon]} for c in e.cuts],
        "storeys": [{"name": s.name, "level": float(s.level), "reference": _vec(s.reference),
                     "nodeIds": [int(x) for x in s.node_ids]} for s in e.storeys],
        "pointSets": [{"name": p.name, "xyz": [_vec(v) for v in p.xyz],
                       "hostElement": [int(x) for x in p.host_element]} for p in e.point_sets],
        "refused": [[r.kind, r.row, r.reason] for r in e.refused],
    }


def goldens(out: Path = OUT) -> dict[str, Any]:
    """What Python reads from the two files, and the kernel's resultant from the original cut."""
    doc: dict[str, Any] = {}
    for stem in ("frame_cut", "rows"):
        uf = UserFile(out / f"{stem}.user.h5")
        index = uf.results()
        doc[stem] = {
            "entities": _entities(uf.entities()),
            "entries": [e.name for e in index.entries],
            "refused": [[r.name, r.reason] for r in index.refused],
        }
    cut, force, moment = kernel_resultant()
    back = UserFile(out / "frame_cut.user.h5").read_result("frame cut resultant")
    doc["frame_cut"]["kernel"] = {
        "def": {"point": list(cut.plane_point), "normal": list(cut.plane_normal),
                "elementIds": list(cut.element_ids), "side": cut.side},
        "reference": list(FRAME_REFERENCE),
        "components": list(COMPONENTS),
        "states": [[s.kind, s.source, s.step, s.mode] for s in back.states],
        "F": force.tolist(),
        "M": moment.tolist(),
    }
    return doc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_model.h5"
    try:
        model = _model(tmp)
        for stem, write in (("frame_cut", write_frame_cut), ("rows", write_rows)):
            path = OUT / f"{stem}.user.h5"
            path.unlink(missing_ok=True)
            write(path, model)
    finally:
        tmp.unlink(missing_ok=True)
    text = json.dumps(goldens(), indent=1) + "\n"
    (OUT / "goldens.json").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
