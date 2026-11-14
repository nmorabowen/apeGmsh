"""The ``.ladruno`` particle group (ADR 0134 D3 and D4).

The fork's contract ``Ladruno_implementation/232a_mpm_contracts.md`` §2
adds, at ``INFO/FORMAT_MINOR_VERSION >= 2``, a particle group to a
``MODEL_STAGE``: per-stage metadata in ``MODEL/PARTICLES`` and the
outputs in ``RESULTS/ON_PARTICLES``. Outputs are ragged (particles are
removed or added during a run), so the file stores them concatenated
with an offset table, and :class:`~apeGmsh.results._slabs.ParticleSlab`
keeps that layout.

What this module accepts and refuses, by the contract:

* the fork's attribute encodings: a scalar stored as a 1-element array,
  fixed-length ``NULLTERM`` strings, and an empty string written as an
  absent attribute (§0);
* exact dtypes, never a silent cast (§0); ``int64`` ids within
  ``±(2**53 - 1)`` (§0);
* a particle group in a minor-1 file (C2-MINOR), a half group
  (C2-PAIR) and a part file with particles (C2-PART) are refused when
  the file opens;
* a file a crash left open reads its committed outputs (§2.5's crash
  rule): ``T' = min(len(STEP), len(TIME), len(OFFSETS) - 1)``, then only
  the outputs whose ``OFFSETS[k + 1]`` fits in ``len(ID)`` and in every
  field's ``len(DATA)``. The writer appends ``OFFSETS`` last, so only
  its last entry can be torn: a last entry smaller than its predecessor
  (an extension still holding its fill) drops that output, and a
  decrease anywhere earlier refuses the stage (C2-OFF). No full-file
  shape is trusted, and nothing depends on the chunk shape: reads are
  hyperslabs.
* a stage the contract refuses is never cached or listed as readable:
  :meth:`ParticleStage.validate` runs before the reader hands it out.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
from numpy import ndarray

from .._slabs import ParticleSlab

if TYPE_CHECKING:
    import h5py


#: The first ``INFO/FORMAT_MINOR_VERSION`` that may hold a particle group.
FORMAT_MINOR_VERSION_PARTICLES = 2

#: ``int64`` ids a browser reads exactly (contract §0).
SAFE_INTEGER = 2 ** 53 - 1

_STAGE_PREFIX = "MODEL_STAGE["
_PART_FILE_RE = re.compile(r"\.part-\d+\.ladruno$", re.IGNORECASE)
_INDEX_DATASETS = ("STEP", "TIME", "OFFSETS", "ID")


class ParticleFormatError(ValueError):
    """A ``.ladruno`` particle group the fork's contract refuses.

    The message names the file, the stage and the contract rule code
    (``C2-…``, contract 232a §2).
    """


class ParticleOutputsTruncatedWarning(UserWarning):
    """A particle stage holds rows past its last committed output.

    The file was not closed (a killed run): every committed output
    reads, and the partial output beyond them is ignored (contract
    §2.5's crash rule).
    """


# ---------------------------------------------------------------------
# Attribute decoding (contract §0, "Attribute encodings")
# ---------------------------------------------------------------------

def _text(value: Any) -> str:
    """A string attribute or dataset entry as ``str``: fixed or variable
    length, bytes or str, scalar or 1-element array; NULs stripped."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"expected one string, got shape {value.shape}")
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    return str(value).rstrip("\x00")


def _attr_text(attrs: Any, name: str, where: str, *, required: bool) -> str:
    """A string attribute; absent reads as ``""`` unless ``required``."""
    if name not in attrs:
        if required:
            raise ParticleFormatError(
                f"{where}: attribute {name!r} is missing (C2-ATTR)."
            )
        return ""
    return _text(attrs[name])


def _attr_number(attrs: Any, name: str, where: str) -> Any:
    """A required numeric scalar attribute, scalar or 1-element array."""
    if name not in attrs:
        raise ParticleFormatError(
            f"{where}: attribute {name!r} is missing (C2-ATTR)."
        )
    v = np.asarray(attrs[name])
    if v.size != 1:
        raise ParticleFormatError(
            f"{where}: attribute {name!r} has shape {v.shape}; "
            "a scalar or a 1-element array is expected."
        )
    return v.reshape(-1)[0]


def _attr_array(attrs: Any, name: str, where: str, dtype: str) -> ndarray:
    if name not in attrs:
        raise ParticleFormatError(
            f"{where}: attribute {name!r} is missing (C2-GRID)."
        )
    v = np.asarray(attrs[name])
    if v.dtype != np.dtype(dtype):
        raise ParticleFormatError(
            f"{where}: attribute {name!r} is {v.dtype}, the contract "
            f"says {np.dtype(dtype)} (C2-GRID)."
        )
    return v.reshape(-1).copy()


def _dataset(group: "h5py.Group", name: str, where: str,
             dtypes: Sequence[str], ndim: int) -> "h5py.Dataset":
    """``group[name]`` with an exact dtype and rank, or a refusal."""
    import h5py

    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise ParticleFormatError(
            f"{where}: dataset {name!r} is missing."
        )
    ds = group[name]
    if not any(ds.dtype == np.dtype(d) for d in dtypes):
        raise ParticleFormatError(
            f"{where}/{name} is {ds.dtype}; the contract says "
            f"{' or '.join(str(np.dtype(d)) for d in dtypes)}, and a "
            "reader never casts silently."
        )
    if ds.ndim != ndim:
        raise ParticleFormatError(
            f"{where}/{name} has rank {ds.ndim}, expected {ndim}."
        )
    return ds


def _require_safe_ids(ids: ndarray, where: str) -> None:
    if ids.size and (int(ids.max()) > SAFE_INTEGER
                     or int(ids.min()) < -SAFE_INTEGER):
        raise ParticleFormatError(
            f"{where}: a particle id is outside ±(2**53 - 1), so it is "
            "not a safe integer (contract §0)."
        )


def require_particle_reader(reader: Any, where: str) -> Any:
    """``reader`` when it is a one-file ``LadrunoReader``, the only reader
    with particle results (ADR 0134 D3); otherwise a refusal by name."""
    from ._ladruno import LadrunoReader

    if not isinstance(reader, LadrunoReader):
        raise ValueError(
            f"results.particles reads the particle group of a single "
            f".ladruno file (ADR 0134 D3); these Results read {where} "
            f"through {type(reader).__name__}, which carries no particles "
            f"(a native or MPCO file, a partitioned .ladruno set, or a "
            f"to_units view)."
        )
    return reader


# ---------------------------------------------------------------------
# File-level scan (run when a LadrunoReader opens)
# ---------------------------------------------------------------------

def scan_particle_stages(h5: "h5py.File", path: Path) -> frozenset[str]:
    """The ``MODEL_STAGE`` group names that hold a particle group.

    Refuses a half group (C2-PAIR), a particle group in a file below
    minor 2 (C2-MINOR), and a part file with particles (C2-PART). A file
    without particles returns an empty set and is not read further.
    """
    names = []
    for name in h5:
        if not name.startswith(_STAGE_PREFIX):
            continue
        grp = h5[name]
        has_model = "MODEL" in grp and "PARTICLES" in grp["MODEL"]
        has_results = "RESULTS" in grp and "ON_PARTICLES" in grp["RESULTS"]
        if has_model != has_results:
            present, absent = (
                ("MODEL/PARTICLES", "RESULTS/ON_PARTICLES") if has_model
                else ("RESULTS/ON_PARTICLES", "MODEL/PARTICLES")
            )
            raise ParticleFormatError(
                f"{path.name} {name}: {present} is present but {absent} "
                "is not; the two are present together or not at all "
                "(C2-PAIR)."
            )
        if has_model:
            names.append(name)
    if not names:
        return frozenset()
    info = h5["INFO"]
    minor = int(_attr_number(info.attrs, "FORMAT_MINOR_VERSION",
                             f"{path.name} INFO")) \
        if "FORMAT_MINOR_VERSION" in info.attrs else 0
    if minor < FORMAT_MINOR_VERSION_PARTICLES:
        raise ParticleFormatError(
            f"{path.name}: {', '.join(sorted(names))} hold a particle "
            f"group, but INFO/FORMAT_MINOR_VERSION is {minor}; a file "
            f"with particles writes minor >= "
            f"{FORMAT_MINOR_VERSION_PARTICLES} (C2-MINOR)."
        )
    partitioned = (
        int(_attr_number(info.attrs, "PARTITIONED", f"{path.name} INFO"))
        if "PARTITIONED" in info.attrs else 0
    )
    if partitioned == 1 or _PART_FILE_RE.search(path.name):
        raise ParticleFormatError(
            f"{path.name} is a part file of a partitioned run and holds "
            f"particles ({', '.join(sorted(names))}). The MPM driver is "
            "serial, and a part file with particles is refused (C2-PART, "
            "ADR 0134 D3)."
        )
    return frozenset(names)


# ---------------------------------------------------------------------
# Per-stage metadata records
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ParticleGrid:
    """``MODEL/PARTICLES/GRID``: the background grid of a stage.

    Nodes sit at ``origin + i * cell_size`` for ``i = 0 … num_cells``.
    """

    origin: ndarray        # (ndm,) float64
    cell_size: ndarray     # (ndm,) float64
    num_cells: ndarray     # (ndm,) int32
    basis: str             # "gimp" | "linear"


@dataclass(frozen=True)
class ParticleRegistry:
    """The stage registry: one row per particle in any output of the stage."""

    ids: ndarray            # (M,) int64, strictly increasing
    body_ids: ndarray       # (M,) int32
    material_tags: ndarray  # (M,) int32
    mass: ndarray           # (M,) float64
    bodies: "dict[int, str]"  # body id -> body name

    def body_names(self) -> ndarray:
        """``(M,)`` the body name of each registry row."""
        return np.array([self.bodies[int(b)] for b in self.body_ids],
                        dtype=object)


@dataclass(frozen=True)
class ParticleStageInfo:
    """``MODEL/PARTICLES`` attributes and the stage's epoch."""

    mode: str                   # "explicit" | "implicit" | "quasi-static"
    epoch: int                  # PARTICLE_EPOCH
    pattern_tag: int            # -1 in mode A
    units: "dict[str, str]"     # UNIT_LENGTH / UNIT_FORCE / UNIT_TIME [/ UNIT_SYSTEM]
    plane_strain_thickness: Optional[float]   # None in 3D
    source_set_id: str          # "" when no particle set was loaded


@dataclass(frozen=True)
class ParticleField:
    """One ``ON_PARTICLES/<FIELD>`` group's attributes."""

    name: str
    components: tuple[str, ...]
    display_name: str
    dimension: str              # "" when absent
    description: str            # "" when absent
    data_type: int


# ---------------------------------------------------------------------
# One stage's particle group
# ---------------------------------------------------------------------

class ParticleStage:
    """Reads one stage's ``MODEL/PARTICLES`` and ``RESULTS/ON_PARTICLES``.

    Metadata is read on first use and cached; output values are read on
    each :meth:`read`, by hyperslab over the committed rows only.
    """

    def __init__(self, stage_grp: "h5py.Group", where: str) -> None:
        self._stage = stage_grp
        self._model = stage_grp["MODEL"]["PARTICLES"]
        self._results = stage_grp["RESULTS"]["ON_PARTICLES"]
        self._where = where
        self._registry: Optional[ParticleRegistry] = None
        self._fields: "Optional[dict[str, ParticleField]]" = None
        self._committed: "Optional[tuple[ndarray, ndarray, ndarray]]" = None

    def validate(self) -> None:
        """Read every metadata record and apply the crash rule once.

        The reader calls it before it hands the stage out, so a stage
        the contract refuses raises on every access and is never listed,
        or read in part, as if it were readable.
        """
        self.info()
        self.grid()
        self.registry()
        self.fields()
        self.committed()

    # -- metadata -----------------------------------------------------

    def info(self) -> ParticleStageInfo:
        where = f"{self._where} MODEL/PARTICLES"
        attrs = self._model.attrs
        units = {
            k: _attr_text(attrs, k, where, required=True)
            for k in ("UNIT_LENGTH", "UNIT_FORCE", "UNIT_TIME")
        }
        system = _attr_text(attrs, "UNIT_SYSTEM", where, required=False)
        if system:
            units["UNIT_SYSTEM"] = system
        thickness = (
            float(_attr_number(attrs, "PLANE_STRAIN_THICKNESS", where))
            if "PLANE_STRAIN_THICKNESS" in attrs else None
        )
        return ParticleStageInfo(
            mode=_attr_text(attrs, "MODE", where, required=True),
            epoch=int(_attr_number(self._stage.attrs, "PARTICLE_EPOCH",
                                   self._where)),
            pattern_tag=int(_attr_number(attrs, "PATTERN_TAG", where)),
            units=units,
            plane_strain_thickness=thickness,
            source_set_id=_attr_text(attrs, "SOURCE_SET_ID", where,
                                     required=False),
        )

    def grid(self) -> ParticleGrid:
        where = f"{self._where} MODEL/PARTICLES/GRID"
        if "GRID" not in self._model:
            raise ParticleFormatError(f"{where} is missing (C2-GRID).")
        attrs = self._model["GRID"].attrs
        return ParticleGrid(
            origin=_attr_array(attrs, "ORIGIN", where, "<f8"),
            cell_size=_attr_array(attrs, "CELL_SIZE", where, "<f8"),
            num_cells=_attr_array(attrs, "NUM_CELLS", where, "<i4"),
            basis=_attr_text(attrs, "BASIS", where, required=True),
        )

    def bodies(self) -> "dict[int, str]":
        """``{body id: body name}`` from ``MODEL/PARTICLES/BODIES``."""
        where = f"{self._where} MODEL/PARTICLES/BODIES"
        if "BODIES" not in self._model:
            raise ParticleFormatError(f"{where} is missing (C2-BODY).")
        grp = self._model["BODIES"]
        ids = _dataset(grp, "ID", where, ("<i4",), 1)[...]
        if "NAME" not in grp:
            raise ParticleFormatError(f"{where}: dataset 'NAME' is missing.")
        raw = np.asarray(grp["NAME"][...]).reshape(-1)
        if raw.size != ids.size:
            raise ParticleFormatError(
                f"{where}: {ids.size} ids but {raw.size} names (C2-BODY)."
            )
        return {int(i): _text(n) for i, n in zip(ids, raw)}

    def registry(self) -> ParticleRegistry:
        if self._registry is not None:
            return self._registry
        where = f"{self._where} MODEL/PARTICLES"
        ids = np.asarray(_dataset(self._model, "ID", where, ("<i8",), 1)[...])
        body = np.asarray(
            _dataset(self._model, "BODY_ID", where, ("<i4",), 1)[...])
        mat = np.asarray(
            _dataset(self._model, "MATERIAL_TAG", where, ("<i4",), 1)[...])
        mass = np.asarray(
            _dataset(self._model, "MASS", where, ("<f8",), 1)[...])
        # The registry is appended to during a stage; a crash can leave
        # its columns at different lengths. Rows past the shortest column
        # are not committed.
        m = min(ids.size, body.size, mat.size, mass.size)
        ids, body, mat, mass = ids[:m], body[:m], mat[:m], mass[:m]
        if m > 1 and not bool(np.all(np.diff(ids) > 0)):
            raise ParticleFormatError(
                f"{where}/ID is not strictly increasing (C2-REG)."
            )
        _require_safe_ids(ids, f"{where}/ID")
        bodies = self.bodies()
        unknown = sorted(set(np.unique(body).tolist()) - set(bodies))
        if unknown:
            raise ParticleFormatError(
                f"{where}/BODY_ID holds {unknown}, which BODIES/ID does "
                "not name (C2-REG)."
            )
        self._registry = ParticleRegistry(
            ids=ids, body_ids=body, material_tags=mat, mass=mass,
            bodies=bodies,
        )
        return self._registry

    def fields(self) -> "dict[str, ParticleField]":
        """Every ``ON_PARTICLES/<FIELD>`` group, by field name."""
        if self._fields is not None:
            return self._fields
        import h5py

        out: "dict[str, ParticleField]" = {}
        for name in self._results:
            item = self._results[name]
            if not isinstance(item, h5py.Group):
                if name in _INDEX_DATASETS:
                    continue
                raise ParticleFormatError(
                    f"{self._where} RESULTS/ON_PARTICLES/{name} is a "
                    "dataset that is not one of the four index datasets "
                    f"{_INDEX_DATASETS} (C2-UNKNOWN)."
                )
            where = f"{self._where} RESULTS/ON_PARTICLES/{name}"
            labels = tuple(
                _attr_text(item.attrs, "COMPONENTS", where,
                           required=True).split(",")
            )
            if any(not c for c in labels) or len(set(labels)) != len(labels):
                raise ParticleFormatError(
                    f"{where}: COMPONENTS {labels} has an empty or a "
                    "repeated label (C2-COMP)."
                )
            data = _dataset(item, "DATA", where, ("<f8", "<f4"), 2)
            if data.shape[1] != len(labels):
                raise ParticleFormatError(
                    f"{where}/DATA has {data.shape[1]} columns but "
                    f"COMPONENTS names {len(labels)} (C2-COMP)."
                )
            out[name] = ParticleField(
                name=name,
                components=labels,
                display_name=_attr_text(item.attrs, "DISPLAY_NAME", where,
                                        required=True),
                dimension=_attr_text(item.attrs, "DIMENSION", where,
                                     required=False),
                description=_attr_text(item.attrs, "DESCRIPTION", where,
                                       required=False),
                data_type=int(_attr_number(item.attrs, "DATA_TYPE", where)),
            )
        if "POSITION" not in out:
            raise ParticleFormatError(
                f"{self._where} RESULTS/ON_PARTICLES has no POSITION "
                "field (C2-POS)."
            )
        self._fields = out
        return out

    def components(self) -> list[str]:
        """Every component as the file spells it, ``"<FIELD>/<label>"``."""
        return [
            f"{f.name}/{c}" for f in self.fields().values()
            for c in f.components
        ]

    # -- the crash rule (contract §2.5) -------------------------------

    def committed(self) -> "tuple[ndarray, ndarray, ndarray]":
        """``(offsets (T+1,), step (T,), time (T,))`` of the committed outputs."""
        if self._committed is not None:
            return self._committed
        where = f"{self._where} RESULTS/ON_PARTICLES"
        res = self._results
        step_ds = _dataset(res, "STEP", where, ("<i4",), 1)
        time_ds = _dataset(res, "TIME", where, ("<f8",), 1)
        off_ds = _dataset(res, "OFFSETS", where, ("<i8",), 1)
        id_ds = _dataset(res, "ID", where, ("<i8",), 1)
        reduction = _attr_text(res.attrs, "PARTITION_REDUCTION", where,
                               required=True)
        if reduction != "NONE":
            raise ParticleFormatError(
                f"{where}@PARTITION_REDUCTION is {reduction!r}; a serial "
                "particle group writes 'NONE' (C2-RED)."
            )
        fields = self.fields()
        # The fork's numbered steps (232a §2.5; its reference is
        # tests/mpm_contracts.py::committed_outputs), over OFFSETS as it
        # stands on disk (T + 1 int64 entries, a small read).
        offsets = np.asarray(off_ds[...], dtype=np.int64).reshape(-1)
        if offsets.size == 0:
            offsets = np.zeros(1, dtype=np.int64)
        # (1) the first entry is 0.
        if offsets[0] != 0:
            raise ParticleFormatError(
                f"{where}/OFFSETS[0] is {int(offsets[0])}, not 0 (C2-OFF)."
            )
        # (2) a last entry below its predecessor is a torn extension
        # (the HDF5 fill, 0): drop the in-flight output it would commit.
        if offsets.size >= 2 and offsets[-1] < offsets[-2]:
            offsets = offsets[:-1]
        # (3) any remaining decrease is corruption: refuse the stage.
        if bool(np.any(np.diff(offsets) < 0)):
            raise ParticleFormatError(
                f"{where}/OFFSETS decreases before its last entry "
                "(C2-OFF)."
            )
        n_out = min(len(step_ds), len(time_ds), offsets.size - 1)
        offsets = offsets[: n_out + 1]
        rows = min(
            [len(id_ds)]
            + [len(res[name]["DATA"]) for name in fields]
        )
        kept = int(np.searchsorted(offsets[1:], rows, side="right"))
        offsets = offsets[: kept + 1]
        step = np.asarray(step_ds[:kept], dtype=np.int32)
        time = np.asarray(time_ds[:kept], dtype=np.float64)
        if kept > 1 and not bool(np.all(np.diff(step) > 0)):
            raise ParticleFormatError(
                f"{where}/STEP is not strictly increasing over its "
                f"committed outputs ({step.tolist()}) (C2-AXES)."
            )
        if not bool(np.all(np.isfinite(time))):
            raise ParticleFormatError(
                f"{where}/TIME holds a non-finite value (C2-AXES)."
            )
        tail = (
            len(step_ds) > kept or len(time_ds) > kept
            or len(off_ds) > kept + 1 or len(id_ds) > offsets[-1]
            or any(len(res[n]["DATA"]) > offsets[-1] for n in fields)
        )
        if tail:
            warnings.warn(
                f"{where}: the file holds rows past its last committed "
                f"output (a run that did not close it); {kept} output(s) "
                f"read, the partial tail is ignored (contract §2.5).",
                ParticleOutputsTruncatedWarning,
                stacklevel=3,
            )
        self._committed = (offsets, step, time)
        return self._committed

    # -- values -------------------------------------------------------

    def _resolve_component(self, component: str) -> "tuple[ParticleField, list[int]]":
        fields = self.fields()
        name, sep, label = component.partition("/")
        field = fields.get(name)
        if field is not None and not sep:
            return field, list(range(len(field.components)))
        if field is not None and label in field.components:
            return field, [field.components.index(label)]
        raise KeyError(
            f"{self._where}: no particle component {component!r}. "
            f"Fields: {sorted(fields)}; components: {self.components()}."
        )

    def read(
        self,
        component: str,
        *,
        ids: "Optional[ndarray]" = None,
        outputs: "Optional[ndarray]" = None,
    ) -> ParticleSlab:
        """One field (``"STRESS"``) or one component (``"STRESS/sigma12"``).

        ``ids`` keeps those particles (a sorted ``int64`` array of
        registry ids); ``outputs`` keeps those output indices, in order.
        """
        field, cols = self._resolve_component(component)
        offsets, step, time = self.committed()
        n_out = step.size
        ks = (np.arange(n_out, dtype=np.int64) if outputs is None
              else np.asarray(outputs, dtype=np.int64).reshape(-1))
        id_ds = self._results["ID"]
        data_ds = self._results[field.name]["DATA"]
        ncomp = len(cols)
        if ks.size and bool(np.all(np.diff(ks) == 1)):
            a, b = int(offsets[ks[0]]), int(offsets[ks[-1] + 1])
            pids = np.asarray(id_ds[a:b], dtype=np.int64)
            vals = np.asarray(data_ds[a:b])[:, cols]
            local = offsets[ks[0]: ks[-1] + 2] - a
        else:
            id_parts, val_parts, counts = [], [], [0]
            for k in ks:
                a, b = int(offsets[k]), int(offsets[k + 1])
                id_parts.append(np.asarray(id_ds[a:b], dtype=np.int64))
                val_parts.append(np.asarray(data_ds[a:b])[:, cols])
                counts.append(b - a)
            pids = (np.concatenate(id_parts) if id_parts
                    else np.zeros(0, dtype=np.int64))
            vals = (np.concatenate(val_parts) if val_parts
                    else np.zeros((0, ncomp), dtype=data_ds.dtype))
            local = np.cumsum(counts, dtype=np.int64)
        registry_ids = self.registry().ids
        if ids is not None:
            keep = np.isin(pids, ids)
            cum = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))
            local = cum[local]
            pids, vals = pids[keep], vals[keep]
            registry_ids = registry_ids[np.isin(registry_ids, ids)]
        return ParticleSlab(
            component=component,
            values=vals.reshape(-1, ncomp),
            particle_ids=pids,
            offsets=np.asarray(local, dtype=np.int64),
            time=time[ks],
            step=step[ks],
            registry_ids=registry_ids,
            components=tuple(field.components[c] for c in cols),
        )
