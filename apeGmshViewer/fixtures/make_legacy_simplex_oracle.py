"""Write ``legacy_simplex.oracle.json``: Python's node values for the legacy
simplex fixtures (#2274, the viewer side of #2239 / PR #2273).

``legacy_simplex_capture_tet.results.h5`` and
``legacy_simplex_transcoded_tri.results.h5`` are ``capture_tet.h5`` and
``transcoded_tri.h5`` of ``tests/fixtures/native_legacy_simplex/`` (PR #2273),
copied byte for byte: native files written by apeGmsh's own writers at
31ba6a63c, before #2238 moved the simplex Gauss tables to gmsh coordinates
(that directory's ``generate.py`` says how). Their ``_natural_coords`` hold
the old node-0-first rows. Every element is loaded with ``u_i = x_i^2 / 2``,
so the strain it records at a Gauss point is that point's world position.

For every multi-point Gauss group this reads the slab through apeGmsh's
``NativeReader`` (which maps the old rows, #2273) and carries it to the
corners with ``extrapolate_gauss_slab_to_nodes`` on a one-element gmsh-coded
stand-in: the corners are nodes 1..nc at ``CORNERS``, as the app's test
builds them. Run it from the repository root with an apeGmsh that has #2273::

    python apeGmshViewer/fixtures/make_legacy_simplex_oracle.py

It imports apeGmsh from this checkout's ``src`` and refuses to run where
``_SIMPLEX_GP_ROWS`` (#2273) is absent: without it the reader returns the old
rows and the oracle would be the rotated values the app must not draw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import apeGmsh  # noqa: E402

assert Path(apeGmsh.__file__).resolve().is_relative_to(ROOT / "src"), apeGmsh.__file__

from apeGmsh.results._gauss_extrapolation import (  # noqa: E402
    extrapolate_gauss_slab_to_nodes,
)
from apeGmsh.results.readers import _native  # noqa: E402

assert hasattr(_native, "_SIMPLEX_GP_ROWS"), "this apeGmsh has no #2273 legacy-row mapping"

#: The geometry of every fixture element (generate.py in #2273's directory).
CORNERS = {
    "tri": [[0.1, 0.2], [2.3, 0.4], [0.6, 1.8]],
    "tet": [[0.1, 0.2, 0.0], [2.3, 0.1, 0.4], [0.5, 1.7, 0.2], [0.3, 0.4, 1.9]],
}
FILES = {
    "legacy_simplex_capture_tet.results.h5": ("tet", 11, 10, ("strain_xx", "strain_yy", "strain_zz")),
    "legacy_simplex_transcoded_tri.results.h5": ("tri", 9, 6, ("strain_xx", "strain_yy")),
}


def _fem(stem: str, eid: int, code: int, npe: int) -> SimpleNamespace:
    """One gmsh-coded element with nodes 1..npe, its corners at CORNERS."""
    corners = np.asarray(CORNERS[stem])
    xyz = np.zeros((npe, 3))
    xyz[: corners.shape[0], : corners.shape[1]] = corners
    tags = np.arange(1, npe + 1, dtype=np.int64)

    class _Group:
        element_type = SimpleNamespace(code=code, dim=corners.shape[1], npe=npe, order=2, gmsh_name="gmsh")
        connectivity = tags[None, :]
        ids = np.array([eid])

        def __len__(self) -> int:
            return 1

    return SimpleNamespace(nodes=SimpleNamespace(ids=tags, coords=xyz), elements=[_Group()])


def main() -> None:
    out: dict = {}
    for name, (stem, code, npe, comps) in FILES.items():
        with _native.NativeReader(HERE / name) as r:
            sid = r.stages()[0].id
            per: dict = {}
            for comp in comps:
                slab = r.read_gauss(sid, comp)
                for eid in np.unique(np.asarray(slab.element_index)):
                    sel = np.asarray(slab.element_index) == eid
                    if int(sel.sum()) < 2:
                        continue
                    one = r.read_gauss(sid, comp, element_ids=np.array([eid]))
                    node_ids, nodal = extrapolate_gauss_slab_to_nodes(one, _fem(stem, int(eid), code, npe))
                    per.setdefault(str(int(eid)), {"node_ids": [int(n) for n in node_ids]})[comp] = [
                        float(v) for v in nodal[-1]
                    ]
            out[name] = per
    target = HERE / "legacy_simplex.oracle.json"
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", target)


if __name__ == "__main__":
    main()
