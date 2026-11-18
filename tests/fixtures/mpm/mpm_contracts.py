"""WP-232a: the executable spec of the two MPM contracts.

``validate_particle_set(path)`` checks Contract 1 (the particle set apeGmsh writes and
``mpm particles -file`` reads); ``validate_particle_results(path)`` checks Contract 2 (the
particle group inside a ``.ladruno``, FORMAT_VERSION 1, FORMAT_MINOR_VERSION 2) through the
fork's own format validator, ``Ladruno_scripts/ladruno_recorder_tests/ladruno_format.py``.

Both return a list of problems; empty means conformant. Every message starts with the code
of the rule it breaks (``C1-...`` / ``C2-...``), as listed in
``Ladruno_implementation/232a_mpm_contracts.md``. Pure h5py + numpy; no OpenSees.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
_FMT_DIR = REPO / "Ladruno_scripts" / "ladruno_recorder_tests"
if str(_FMT_DIR) not in sys.path:
    sys.path.insert(0, str(_FMT_DIR))

import ladruno_format as lf  # noqa: E402  (the fork's .ladruno executable spec)

FORMAT = "ladruno-mpm-particles"
FORMAT_VERSION = 1
#: Voigt order of STRESS: 3D [xx, yy, zz, xy, yz, zx]; plane strain [xx, yy, xy] (232b
#: mpm_interface.md section 1, the hypo kernel's order). In 2D sigma_33 is STRESS33.
VOIGT = {2: "11,22,12", 3: "11,22,33,12,23,13"}
NSTRESS = {2: 3, 3: 6}

#: dataset -> (dtype, second dimension: None | "ndm" | "stress", required)
PARTICLE_DATASETS = {
    "ID": ("<i8", None, True),
    "POSITION": ("<f8", "ndm", True),
    "VOLUME": ("<f8", None, True),
    "MASS": ("<f8", None, False),
    "DENSITY": ("<f8", None, False),
    "GIMP_HALF_LENGTH": ("<f8", "ndm", True),
    "VELOCITY": ("<f8", "ndm", False),
    "STRESS": ("<f8", "stress", False),
    "STRESS33": ("<f8", None, False),
    "BODY_ID": ("<i4", None, True),
    "MATERIAL_TAG": ("<i4", None, True),
    "FLAGS": ("<u4", None, False),
}
ROOT_STRINGS = ("FORMAT", "GENERATOR", "SET_ID", "UNIT_LENGTH", "UNIT_FORCE", "UNIT_TIME",
                "MASS_SOURCE", "STRESS_SIGN", "VOIGT_ORDER")


def _a(obj, name):
    """An attribute in any of the accepted encodings (scalar or 1-element array; fixed or
    variable-length string), as a Python scalar / str / ndarray. KeyError when absent."""
    return lf._attr(obj, name)


def validate_particle_set(path: str, warnings: list | None = None) -> list[str]:
    """Contract 1 (232a §1). Returns the problems; ``warnings`` collects non-rules."""
    problems: list[str] = []
    err = problems.append
    warn = warnings.append if warnings is not None else (lambda _m: None)

    with h5py.File(path, "r") as f:
        root = {k: _a(f, k) for k in f.attrs}
        for k in ROOT_STRINGS + ("FORMAT_VERSION", "NDM", "NUM_PARTICLES"):
            if k not in root:
                err(f"C1-REQ: root attribute {k} missing")
        if root.get("FORMAT") not in (None, FORMAT):
            err(f"C1-FMT: FORMAT {root.get('FORMAT')!r} != {FORMAT!r}")
        if "FORMAT_VERSION" in root and int(root["FORMAT_VERSION"]) != FORMAT_VERSION:
            err(f"C1-VER: FORMAT_VERSION {root['FORMAT_VERSION']} is not {FORMAT_VERSION}")
        if "GENERATOR" in root and not root["GENERATOR"]:
            err("C1-GEN: GENERATOR is empty")
        if "SET_ID" in root and not lf.UUID4.match(str(root["SET_ID"])):
            err(f"C1-SETID: SET_ID {root['SET_ID']!r} is not a canonical uuid4")
        for u in ("UNIT_LENGTH", "UNIT_FORCE", "UNIT_TIME"):
            if u in root and not root[u]:
                err(f"C1-UNIT: {u} is empty")
        if "STRESS_SIGN" in root and root["STRESS_SIGN"] != "tension-positive":
            err(f"C1-SIGN: STRESS_SIGN {root['STRESS_SIGN']!r} != 'tension-positive'")

        for k, width in (("FORMAT_VERSION", 4), ("NDM", 4), ("NUM_PARTICLES", 8)):
            if k in root and not lf.int_attr_ok(f, k, width):
                err(f"C1-DTYPE: root attribute {k} must be int{8 * width}")
        ndm = int(root.get("NDM", 0))
        if ndm not in (2, 3):
            err(f"C1-NDM: NDM {root.get('NDM')!r} must be 2 or 3")
            return problems
        if "VOIGT_ORDER" in root and root["VOIGT_ORDER"] != VOIGT[ndm]:
            err(f"C1-VOIGT: VOIGT_ORDER {root['VOIGT_ORDER']!r} != {VOIGT[ndm]!r} for NDM={ndm}")
        thk = root.get("PLANE_STRAIN_THICKNESS")
        if ndm == 2 and (thk is None or not np.isfinite(thk) or thk <= 0):
            err("C1-THK: PLANE_STRAIN_THICKNESS must be finite and > 0 when NDM = 2")
        if ndm == 3 and thk is not None:
            err("C1-THK: PLANE_STRAIN_THICKNESS present when NDM = 3")
        if ndm == 3 and "PARTICLES" in f and "STRESS33" in f["PARTICLES"]:
            err("C1-STRESS33: STRESS33 is plane-strain only; in 3D sigma_33 is in STRESS")
        if root.get("GENERATOR") == "apeGmsh":
            prov = f["PROVENANCE"] if "PROVENANCE" in f else None
            need = ("apegmsh_version", "apegmsh_commit", "session_id")
            lacking = [k for k in need if prov is None or k not in prov.attrs]
            if lacking:
                err(f"C1-PROV: PROVENANCE lacks {lacking} (required when GENERATOR = apeGmsh)")

        # GRID
        lo = hi = h = None
        if "GRID" not in f:
            err("C1-REQ: group GRID missing")
        else:
            lo, hi = lf.check_grid(f["GRID"], ndm, "GRID", err, "C1-GRID", "C1-BASIS")
            if "NUM_CELLS" in f["GRID"].attrs and not lf.int_attr_ok(f["GRID"], "NUM_CELLS"):
                err("C1-GRID: GRID/NUM_CELLS must be int32")
            if lo is not None:
                h = np.atleast_1d(np.asarray(_a(f["GRID"], "CELL_SIZE"), dtype=float))

        # BODIES
        body_ids: set[int] = set()
        if "BODIES" not in f or "ID" not in f["BODIES"] or "NAME" not in f["BODIES"]:
            err("C1-REQ: BODIES/ID and BODIES/NAME are required")
        else:
            body_ids = lf.check_bodies(f["BODIES"], "BODIES", err, "C1-BODY")

        # PARTICLES
        if "PARTICLES" not in f:
            err("C1-REQ: group PARTICLES missing")
            return problems
        P = f["PARTICLES"]
        n = int(root.get("NUM_PARTICLES", -1))
        if n < 1:
            err(f"C1-N: NUM_PARTICLES {n} must be >= 1")
        for name in P:
            if name not in PARTICLE_DATASETS:
                err(f"C1-UNKNOWN: PARTICLES/{name} is not a contract-1 field")
        arrays: dict[str, np.ndarray] = {}
        for name, (dt, second, required) in PARTICLE_DATASETS.items():
            if name not in P:
                if required:
                    err(f"C1-REQ: PARTICLES/{name} missing")
                continue
            ds = P[name]
            if not isinstance(ds, h5py.Dataset):
                err(f"C1-SHAPE: PARTICLES/{name} is not a dataset")
                continue
            if ds.dtype != np.dtype(dt):
                err(f"C1-DTYPE: PARTICLES/{name} dtype {ds.dtype} != {np.dtype(dt)}")
            want = (n,) if second is None else (n, ndm if second == "ndm" else NSTRESS[ndm])
            if ds.shape[:1] != (n,):
                err(f"C1-N: PARTICLES/{name} has {ds.shape[0] if ds.ndim else 0} rows, "
                    f"NUM_PARTICLES is {n}")
            elif ds.shape != want:
                err(f"C1-SHAPE: PARTICLES/{name} shape {ds.shape} != {want}")
            lf.check_filters(ds, f"PARTICLES/{name}", err, "C1-FILTER")
            arrays[name] = np.asarray(ds[...])

        src = root.get("MASS_SOURCE")
        if src not in ("mass", "density"):
            err(f"C1-MASS: MASS_SOURCE {src!r} must be 'mass' or 'density'")
        else:
            present = {k for k in ("MASS", "DENSITY") if k in P}
            if present != {src.upper()}:
                err(f"C1-MASS: MASS_SOURCE={src} needs exactly PARTICLES/{src.upper()}, "
                    f"found {sorted(present)}")

        ids = arrays.get("ID")
        if ids is not None and ids.ndim == 1:
            if np.any(np.diff(ids) <= 0):
                err("C1-ID: PARTICLES/ID not strictly increasing (a repeated or unsorted id)")
            if ids.size and (ids.min() < 0 or ids.max() > lf.MAX_SAFE_ID):
                err("C1-ID: PARTICLES/ID outside [0, 2^53 - 1]")
        for name in ("POSITION", "VOLUME", "MASS", "DENSITY", "GIMP_HALF_LENGTH",
                     "VELOCITY", "STRESS", "STRESS33"):
            v = arrays.get(name)
            if v is not None and v.dtype.kind == "f" and not np.all(np.isfinite(v)):
                err(f"C1-FINITE: PARTICLES/{name} has non-finite values")
        vol = arrays.get("VOLUME")
        if vol is not None and np.any(vol <= 0):
            err("C1-VOL: PARTICLES/VOLUME must be > 0 (zero or negative is refused)")
        for name in ("MASS", "DENSITY"):
            v = arrays.get(name)
            if v is not None and np.any(v <= 0):
                err(f"C1-MASS: PARTICLES/{name} must be > 0")
        lp = arrays.get("GIMP_HALF_LENGTH")
        if lp is not None and lp.shape == (n, ndm) and h is not None:
            if np.any(lp <= 0) or np.any(lp > h / 2 * (1 + 1e-12)):
                err("C1-GIMP: GIMP_HALF_LENGTH must satisfy 0 < l <= CELL_SIZE/2 per axis")
        x = arrays.get("POSITION")
        if (x is not None and lp is not None and lo is not None
                and x.shape == lp.shape == (n, ndm)):
            tol = 1e-9 * float(np.max(hi - lo))
            if np.any(x - lp < lo - tol) or np.any(x + lp > hi + tol):
                err("C1-POS: a particle or its GIMP domain lies outside the grid box")
        bid = arrays.get("BODY_ID")
        if bid is not None and body_ids and not set(bid.tolist()) <= body_ids:
            err("C1-BODYREF: PARTICLES/BODY_ID names a body not in BODIES/ID")
        mat = arrays.get("MATERIAL_TAG")
        if mat is not None and np.any(mat <= 0):
            err("C1-MAT: PARTICLES/MATERIAL_TAG must be > 0")
        flags = arrays.get("FLAGS")
        if flags is not None and np.any(flags != 0):
            err("C1-FLAGS: FLAGS has a set bit; contract-1 v1 defines none")

        # not a rule: lattice seeding fills the GIMP domain exactly (232a §1.4)
        if vol is not None and lp is not None and lp.shape == (n, ndm) and vol.shape == (n,):
            dom = np.prod(2.0 * lp, axis=1) * (thk if ndm == 2 and thk else 1.0)
            if not np.allclose(dom, vol, rtol=1e-9):
                warn("VOLUME differs from the GIMP domain volume (fine for non-lattice seeding)")
    return problems


def validate_particle_results(path: str) -> list[str]:
    """Contract 2 (232a §2): the whole .ladruno through ``ladruno_format.validate`` (v1
    invariants + the particle rules), and the file must hold a particle group at all."""
    problems = lf.validate(path)
    with lf.LadrunoReader(path) as r:
        if not r.particle_stages():
            problems.append("C2-PAIR: no MODEL_STAGE holds a particle group")
    return problems


class TornStageError(ValueError):
    """A particle stage whose OFFSETS decrease before the last entry: corruption, not a
    torn append (232a §2.5). The reader refuses the stage."""


def committed_outputs(src, stage: str | None = None) -> int:
    """The number of committed outputs of a particle stage, by the crash rule (232a §2.5).

    ``src`` is a path to a ``.ladruno`` (then ``stage`` names the MODEL_STAGE, or the file
    must hold exactly one particle stage), a ``MODEL_STAGE[...]`` group, or its
    ``RESULTS/ON_PARTICLES`` group. Outputs ``0 .. n-1`` are readable; rows beyond
    ``OFFSETS[n]`` are ignored. Raises ``TornStageError`` on a decrease before the last
    ``OFFSETS`` entry. A closed, conformant stage returns ``len(STEP)``.
    """
    if isinstance(src, (str, Path)):
        with h5py.File(src, "r") as f:
            if stage is None:
                stages = [s for s in f if s.startswith("MODEL_STAGE[")
                          and f"{s}/RESULTS/ON_PARTICLES" in f]
                if len(stages) != 1:
                    raise ValueError(f"{src}: name the stage; particle stages: {stages}")
                stage = stages[0]
            return committed_outputs(f[stage])
    op = src["RESULTS/ON_PARTICLES"] if "RESULTS/ON_PARTICLES" in src else src

    # step 1: the four index datasets as they stand on disk
    offs = np.asarray(op["OFFSETS"][...], dtype=np.int64).reshape(-1)
    n_step = op["STEP"].shape[0]
    n_time = op["TIME"].shape[0]
    if offs.size == 0:
        return 0
    if offs[0] != 0:
        raise TornStageError(f"{op.name}: OFFSETS[0] = {offs[0]}, not 0")
    # step 2: a LAST entry below its predecessor is a torn extension (the HDF5 fill
    # value 0): drop the in-flight output it would have committed
    if offs.size >= 2 and offs[-1] < offs[-2]:
        offs = offs[:-1]
    # step 3: any remaining decrease is corruption: refuse the stage
    if np.any(np.diff(offs) < 0):
        raise TornStageError(f"{op.name}: OFFSETS decrease before the last entry")
    # step 4: every axis must hold the output
    n = min(n_step, n_time, offs.size - 1)
    # step 5: the output's rows must exist in ID and in every field's DATA
    bound = op["ID"].shape[0]
    for name in op:
        item = op[name]
        if isinstance(item, h5py.Group) and "DATA" in item:
            bound = min(bound, item["DATA"].shape[0])
    while n > 0 and offs[n] > bound:
        n -= 1
    return int(n)
