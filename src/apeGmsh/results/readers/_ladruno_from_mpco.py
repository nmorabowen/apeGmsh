"""Convert an STKO ``.mpco`` into a ``.ladruno`` (ADR 0122 D5).

The conversion is explicit, one way and atomic. It works at the HDF5
level, bucket for bucket, and never goes through the slab API (which
drops the class, rule and section tags). The ``.mpco`` is opened
read-only and is never modified.

What is copied verbatim, and what is transformed (fork schema v1 at
``FORK_PIN``; the scoping note of #2120):

==============================  ==============================================
``.mpco``                       ``.ladruno``
==============================  ==============================================
``INFO`` datasets               ``INFO`` attributes, plus ``GENERATOR``,
                                ``FORMAT_VERSION=1``, ``STORED_PRECISION``,
                                ``PARTITIONED`` / ``PARTITION_ID`` /
                                ``NUM_PARTITIONS`` and an empty
                                ``PROVENANCE`` group. No
                                ``FORMAT_MINOR_VERSION``: minor 1 also
                                promises the fork's section 2a stage
                                boundaries (a static/transient flip opens a
                                stage), and the stages here are MPCO's,
                                which split on a domain change only.
``MODEL_STAGE[k]`` @STEP/TIME   copied, plus ``KIND`` from the program
``MODEL/NODES``                 copied
``MODEL/ELEMENTS/<key>``        a group: ``CONNECTIVITY``, the source
(flat dataset)                  attributes, the BASIS attributes and, where
                                known, ``QUADRATURE/GP_PARAM`` (below)
``MODEL/SECTION_ASSIGNMENTS``   copied, plus ``@KIND`` (``fiber`` when the
                                section has ``FIBER_DATA``, else
                                ``resultant``)
``MODEL/LOCAL_AXES``            dropped (``unsupported``: ``local_axes``)
``ON_NODES/<R>/DATA/STEP_k``    ``DATA[T x n x c]`` (chunked, deflate) plus
                                ``STEP[T]`` and ``TIME[T]``
``ON_NODES`` modal groups       copied (``DATA/STEP_k/MODE_i``)
``ON_ELEMENTS/<token>`` attrs   moved onto each bucket
bucket ``META``                 ``COLUMN_MAP`` + ``@COMP_NAMES``
bucket ``DATA/STEP_k``          ``DATA[T x n x NUM_COLUMNS]`` + ``STEP`` +
                                ``TIME``
bucket names, tokens, ``ID``    verbatim
``<stem>.part-N.mpco``          ``<stem>.part-N.ladruno``
==============================  ==============================================

``KIND`` is evidence, so it comes only from the program in ``model_h5``:
the ``MODEL_STAGE`` groups that hold step data pair, in order, with the
program's ``ops.stage`` blocks, whose ``analysis`` record names the kind
(:func:`apeGmsh.results._bind._stage_kind`). A ``MODEL_STAGE`` that holds
only modes is an ``eigen`` stage. Any other count or an unknown kind
refuses.

A program with no ``ops.stage`` block pairs with no stage. Its kind is not
archived as fact: the archive cannot tell an in-process run from a deck,
and it does not record a deck's ``analyze`` call (#1950, coordinator
ruling (iii)). ``read_deck_mpco`` passes the private
``_implicit_stage=True``, which refuses every unstaged program with that
reason (after the lineage check); #2210 records the deck's ``analyze``
call.

``COLUMN_MAP`` follows the fork recorder's own derivation from the same
header MPCO's ``META`` encodes: ``LEVELS`` is the deepest path code of a
block, ``GAUSS_ID``, ``NUM_COMP`` and ``MULTIPLICITY`` are the block's,
and ``SECTION_TAG`` and ``FIBER_ID`` are ``-1`` (the recorder does not
track them at the header level either).

BASIS and QUADRATURE come from the source and from ``RESPONSE_CATALOG``
(coordinator ruling 1 on #2120), never from a copied fork table:

* a custom-rule (``1000``) line group takes ``GP_PARAM`` from its own
  ``GP_X`` attribute;
* a standard-rule group takes the parent coordinates of the catalog rows
  for its class and rule, when they agree; ``TOPOLOGY`` and
  ``PARAM_DOMAIN`` follow from the catalog's coordinate system;
* any other group gets ``TOPOLOGY="custom"`` and no ``QUADRATURE``, and
  ``unsupported`` names it (``quadrature:<group>``).

``PROVENANCE`` is written by
:func:`~apeGmsh.results.readers._ladruno_provenance.stamp_provenance`
on the staged files, with their final names, before they move into
place: ``writer`` is ``apeGmsh mpco->ladruno <version>`` and
``unsupported`` lists ``local_axes``, ``quadrature:<group>`` and
``springs`` (zero-length buckets are copied verbatim, but
``Results.from_ladruno`` reads no spring level).
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import h5py

__all__ = ["MPCOConversionRefused", "StageKindRefused", "convert_mpco"]

_STAGE_PREFIX = "MODEL_STAGE["
_PART_RE = re.compile(r"^(?P<stem>.+?)\.part-(?P<idx>\d+)\.ladruno$")
_LOCAL_AXES = "local_axes"
_SPRINGS = "springs"
_QUADRATURE = "quadrature:"


class MPCOConversionRefused(RuntimeError):
    """:func:`convert_mpco` refused before any target file existed.

    Raised for an existing target (unless ``overwrite=True``), a missing
    or unreadable ``model_h5``, MPCO stages or kinds that do not match the
    program's, a partition set that is incomplete or inconsistent, and a
    source that is unreadable or holds something this converter does not
    know how to carry. The source is never modified.
    """


class StageKindRefused(MPCOConversionRefused):
    """The refusal is about stage kinds alone: the file's stages cannot be
    paired with the program's, or a kind is not a fact (an unstaged
    program).

    Every other refusal (the archive is another model's, a partition set
    is incomplete, a target or staging directory is in the way, the source
    holds what the converter cannot carry) is a plain
    :class:`MPCOConversionRefused`. ``read_deck_mpco`` reads the ``.mpco``
    on this one only and re-raises the rest.
    """


@dataclass
class _Plan:
    """What one conversion writes."""

    sources: list[Path]
    targets: list[Path]
    staging: Path
    kinds: list[str]
    unsupported: set[str] = field(default_factory=set)


def convert_mpco(
    src: "str | Path",
    dst: "str | Path",
    *,
    model_h5: "str | Path",
    overwrite: bool = False,
    _implicit_stage: bool = False,
) -> list[Path]:
    """Convert an ``.mpco`` (or a partitioned set) into ``.ladruno``.

    Parameters
    ----------
    src
        One ``.mpco``, or any ``<stem>.part-<N>.mpco`` of a partitioned
        run (every part is converted).
    dst
        The target ``.ladruno``. For a partitioned source, ``dst`` names
        the set: ``out.ladruno`` (or ``out.part-<k>.ladruno``) writes
        ``out.part-0.ladruno`` ... ``out.part-<N-1>.ladruno``.
    model_h5
        The bridge-written model archive (``ops.h5``) the run was emitted
        from. It is required: stage kinds come only from its program, and
        ``INFO/PROVENANCE`` pairs the converted file with it.
    overwrite
        Replace existing targets. Without it an existing target refuses.
    _implicit_stage
        Private; only ``read_deck_mpco`` passes it. It refuses every
        program with no ``ops.stage`` block, whatever the file holds, with
        :class:`StageKindRefused` naming why: such a program's kind is not
        archived as fact. Without it, an unstaged program pairs with no
        stage (a ``.mpco`` with step data refuses, as before).

    Returns
    -------
    The target paths, in partition order.

    The checks on the targets, the archive, the stages and the partition
    set run before anything is written. The parts are then written into
    ``<dst parent>/<dst stem>.partial/`` under their final names, stamped,
    and moved into place. With ``overwrite=True`` every old part is first
    moved aside into the staging directory, and the backups are removed
    only after every new part is in place. Any failure on the way, a
    malformed bucket found while writing included, removes the new parts
    already moved, puts every old part back under its name, and removes the
    staging directory: either the whole new set is in place, or the old
    set is as it was (no target, without ``overwrite``). The source is
    never opened for writing. :class:`MPCOConversionRefused` is raised for
    a refusal; another failure (the stamp, the disk) re-raises its own
    error after the rollback.
    """
    plan = _plan(Path(src), Path(dst), model_h5, overwrite, _implicit_stage)
    if plan.staging.exists():  # left by an interrupted run; overwrite=True
        shutil.rmtree(plan.staging)
    try:
        plan.staging.mkdir(parents=False, exist_ok=False)
    except OSError as e:
        raise MPCOConversionRefused(
            f"cannot create the staging directory {plan.staging}: {e}"
        ) from e
    moved: list[Path] = []
    backups: list[tuple[Path, Path]] = []
    try:
        staged = [plan.staging / t.name for t in plan.targets]
        for i, (source, out) in enumerate(zip(plan.sources, staged)):
            _convert_part(
                source, out, kinds=plan.kinds, part=i,
                n_parts=len(plan.sources), unsupported=plan.unsupported,
            )
        from ._ladruno_provenance import stamp_provenance

        stamp_provenance(
            staged[0], model_h5,
            unsupported=sorted(plan.unsupported),
            writer=_writer(),
        )
        from ..._atomic_io import replace_with_retry

        # Every old part moves aside first, so a failure can put the whole
        # old set back; the backups go only once every new part is in place.
        for target in plan.targets:
            if target.exists():
                if not overwrite:
                    raise MPCOConversionRefused(
                        f"{target} appeared while converting; nothing was "
                        "written.")
                backup = plan.staging / f"{target.name}.old"
                replace_with_retry(target, backup)
                backups.append((backup, target))
        for path, target in zip(staged, plan.targets):
            replace_with_retry(path, target)
            moved.append(target)
    except BaseException:
        _roll_back(plan.staging, moved, backups)
        raise
    shutil.rmtree(plan.staging, ignore_errors=True)
    return list(plan.targets)


def _roll_back(
    staging: Path, moved: list[Path], backups: "list[tuple[Path, Path]]",
) -> None:
    """Remove the new parts, put every old part back, drop the staging.

    A backup that cannot be restored keeps the staging directory, and the
    old file with it, and is named in a warning.
    """
    import warnings

    from ..._atomic_io import replace_with_retry

    for target in moved:
        try:
            target.unlink()
        except OSError:
            pass
    stranded = []
    for backup, target in backups:
        try:
            replace_with_retry(backup, target)
        except OSError:
            stranded.append(backup)
    if stranded:
        warnings.warn(
            f"convert_mpco could not restore {[str(b) for b in stranded]}; "
            f"the old files stay in {staging}.", RuntimeWarning, stacklevel=3)
        return
    shutil.rmtree(staging, ignore_errors=True)


def _writer() -> str:
    import apeGmsh

    return f"apeGmsh mpco->ladruno {apeGmsh.__version__}"


# ---------------------------------------------------------------------------
# Planning: every refusal happens here, before anything is written
# ---------------------------------------------------------------------------


def _plan(
    src: Path, dst: Path, model_h5: "Optional[str | Path]", overwrite: bool,
    implicit_stage: bool = False,
) -> _Plan:
    if model_h5 is None:
        raise MPCOConversionRefused(
            "convert_mpco needs model_h5=: a stage KIND is evidence, and "
            "only the program archived in model.h5 says whether a stage "
            "was static or transient."
        )
    model_path = Path(model_h5)
    if not model_path.is_file():
        raise MPCOConversionRefused(f"no such model archive: {model_path}")
    sources = _discover_sources(src)
    targets = _targets(dst, len(sources), partitioned=_is_part_name(src))
    staging = targets[0].parent / f"{_set_stem(dst)}.partial"
    _refuse_existing(targets, staging, overwrite)
    model, names, kinds, pending, ranks = _program(model_path)
    # A kind refusal is held until the archive is proved to be this run's
    # (the partition count here, the lineage after the scan): a wrong
    # archive always refuses as one, never as a question of kinds.
    if pending is None and implicit_stage and not names:
        pending = StageKindRefused(
            f"{model_path} declares no ops.stage blocks: an unstaged "
            "program's kind is not archived as fact (#2210 records it). "
            "Nothing was written."
        )
    if ranks and len(ranks) != len(sources):
        raise MPCOConversionRefused(
            f"{model_path} emits {len(ranks)} partitions (ranks "
            f"{sorted(ranks)}), but {len(sources)} part file(s) were found "
            f"({[p.name for p in sources]}). Every part of the run must be "
            "present."
        )
    stage_kinds: Optional[list[str]] = None
    unsupported: set[str] = set()
    for i, source in enumerate(sources if pending is None else ()):
        try:
            part_kinds = _scan_part(source, names, kinds, unsupported)
        except StageKindRefused as e:
            if i == 0:
                pending = e
                break
            # Part 0 paired: a later part that does not is a set that
            # disagrees with itself, not a question of kinds.
            raise MPCOConversionRefused(
                f"{e} (part 0, {sources[0].name}, paired; the parts are not "
                "from one run)") from e
        if stage_kinds is None:
            stage_kinds = part_kinds
        elif part_kinds != stage_kinds:
            raise MPCOConversionRefused(
                f"{source.name} pairs its stages as {part_kinds}, but "
                f"{sources[0].name} as {stage_kinds}; the parts are not "
                "from one run."
            )
    # Last, the readers' own checks: modal records, on a set the stage
    # signatures and time vectors of every part, and the lineage. A held
    # kind refusal is raised only once they pass.
    _open_with_reader(sources, model, model_path)
    if pending is not None:
        raise pending
    assert stage_kinds is not None
    return _Plan(sources, targets, staging, stage_kinds, unsupported)


def _is_part_name(src: Path) -> bool:
    return re.match(r"^.+?\.part-\d+\.mpco$", src.name) is not None


def _discover_sources(src: Path) -> list[Path]:
    from ._mpco_multi import discover_partition_files

    if src.suffix != ".mpco":
        raise MPCOConversionRefused(f"{src} is not an .mpco file.")
    try:
        parts = discover_partition_files(src)
    except ValueError as e:
        raise MPCOConversionRefused(f"{src}: {e}") from e
    missing = [p for p in parts if not p.is_file()]
    if missing:
        raise MPCOConversionRefused(f"no such results file: {missing[0]}")
    return [Path(p) for p in parts]


def _set_stem(dst: Path) -> str:
    m = _PART_RE.match(dst.name)
    if m is not None:
        return m.group("stem")
    return dst.name[: -len(".ladruno")] if dst.name.endswith(".ladruno") \
        else dst.name


def _targets(dst: Path, n_parts: int, *, partitioned: bool) -> list[Path]:
    if dst.suffix != ".ladruno":
        raise MPCOConversionRefused(
            f"{dst}: the target must end in .ladruno.")
    if not dst.parent.is_dir():
        raise MPCOConversionRefused(f"no such directory: {dst.parent}")
    if not partitioned:
        if _PART_RE.match(dst.name) is not None:
            raise MPCOConversionRefused(
                f"{dst} is named like a part of a partitioned set, but the "
                "source is one file. Name the target <stem>.ladruno."
            )
        return [dst]
    stem = _set_stem(dst)
    return [dst.parent / f"{stem}.part-{i}.ladruno" for i in range(n_parts)]


def _refuse_existing(
    targets: list[Path], staging: Path, overwrite: bool,
) -> None:
    existing = [t for t in targets if t.exists()]
    if existing and not overwrite:
        raise MPCOConversionRefused(
            f"{existing[0]} exists; pass overwrite=True to replace it. "
            "Nothing was written."
        )
    m = _PART_RE.match(targets[0].name)
    if m is not None:
        stem = m.group("stem")
        stray = sorted(
            p for p in targets[0].parent.glob(f"{stem}.part-*.ladruno")
            if p not in targets
        )
        if stray:
            raise MPCOConversionRefused(
                f"{stray[0]} is a part of an earlier {stem!r} set beyond the "
                f"{len(targets)} parts being written; a reader would mix it "
                "in. Remove it first. Nothing was written."
            )
    if staging.exists():
        held = sorted(p.name for p in staging.glob("*.old"))
        if held:
            # A rollback that could not restore them left the only copy of
            # an earlier set here: never removed, overwrite or not.
            raise MPCOConversionRefused(
                f"the staging directory {staging} holds backups of an "
                f"earlier set that were never restored ({held}). Move them "
                "back to their names (drop the .old suffix), then remove the "
                "directory. Nothing was written."
            )
        if not overwrite:
            raise MPCOConversionRefused(
                f"the staging directory {staging} exists (a conversion that "
                "was interrupted, or one running now). Remove it, or pass "
                "overwrite=True. Nothing was written."
            )


def _program(
    model_h5: Path,
) -> "tuple[Any, list[str], list[str], Optional[StageKindRefused], set[int]]":
    """The model, its stage names and kinds, and its partition ranks.

    A stage whose kind is unknown is returned as a :class:`StageKindRefused`
    (with no kinds), not raised: the caller raises it only after the
    lineage check, so a wrong archive always refuses as one.
    """
    from ...opensees.opensees_model import OpenSeesModel
    from .._bind import _stage_kind

    try:
        model = OpenSeesModel.from_h5(model_h5)
    except Exception as e:  # any load failure is a refusal, named
        raise MPCOConversionRefused(
            f"{model_h5} cannot be loaded as a bridge model archive "
            f"({type(e).__name__}: {e})."
        ) from e
    stages = list(model.stages())
    kind_error: Optional[StageKindRefused] = None
    try:
        kinds = [_stage_kind(s, model_path=model_h5) for s in stages]
    except ValueError as e:
        kinds, kind_error = [], StageKindRefused(str(e))
        kind_error.__cause__ = e
    ranks = {int(p.rank) for p in model.partitions()}
    return model, [str(s.name) for s in stages], kinds, kind_error, ranks


def _open_with_reader(sources: list[Path], model: Any, model_h5: Path) -> None:
    """Refuse a source ``from_mpco`` would refuse, or would bind elsewhere.

    The readers' own checks run first (stages, modes, and on a set the
    parts' stage signatures and time vectors). Then the archive must be the
    capture's model: :func:`apeGmsh.results._bind._resolve_fem_via_model`,
    the check behind ``from_mpco``'s ``ModelFemMismatchWarning``, is run
    and its warning is a refusal here. A converted file is stamped with the
    archive's hashes and kinds, so pairing it with another model is never a
    warning.
    """
    import warnings

    from .._bind import ModelFemMismatchWarning, _resolve_fem_via_model
    from ._mpco import MPCOReader
    from ._mpco_multi import MPCOMultiPartitionReader

    try:
        reader: Any = (MPCOReader(sources[0]) if len(sources) == 1
                       else MPCOMultiPartitionReader(list(sources)))
    except (OSError, ValueError, KeyError) as e:
        raise MPCOConversionRefused(
            f"{sources[0]} cannot be read as an MPCO capture "
            f"({type(e).__name__}: {e})."
        ) from e
    try:
        try:
            reader.stages()
        except (OSError, ValueError, KeyError) as e:
            raise MPCOConversionRefused(
                f"{sources[0]} cannot be read as an MPCO capture "
                f"({type(e).__name__}: {e})."
            ) from e
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ModelFemMismatchWarning)
            _resolve_fem_via_model(reader, None, model, model_path=model_h5)
    finally:
        reader.close()
    for w in caught:
        if issubclass(w.category, ModelFemMismatchWarning):
            raise MPCOConversionRefused(
                f"{sources[0]} is not a run of {model_h5}: {w.message} "
                "Nothing was written."
            )
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def _scan_part(
    source: Path, names: Sequence[str], kinds: Sequence[str],
    unsupported: set[str],
) -> list[str]:
    """Check one source part and return its stage kinds."""
    import h5py

    try:
        h5 = h5py.File(source, "r")
    except OSError as e:
        raise MPCOConversionRefused(
            f"{source} cannot be read as HDF5 ({e}).") from e
    with h5:
        info = h5["INFO"] if "INFO" in h5 else None
        if info is None or not isinstance(info, h5py.Group):
            raise MPCOConversionRefused(f"{source} has no INFO group.")
        if "GENERATOR" in info.attrs:
            raise MPCOConversionRefused(
                f"{source}: INFO/GENERATOR is set, so this is not an STKO "
                ".mpco (a .ladruno renamed?)."
            )
        _refuse_unknown(info, {"SOLVER_NAME", "SOLVER_VERSION", "SPATIAL_DIM"},
                        f"{source}:INFO")
        if "SPATIAL_DIM" not in info:
            raise MPCOConversionRefused(f"{source} has no INFO/SPATIAL_DIM.")
        stages = _stage_names(h5)
        _refuse_unknown(h5, {"INFO", *stages}, f"{source}:/")
        if not stages:
            raise MPCOConversionRefused(f"{source} holds no MODEL_STAGE.")
        mode_only = []
        for name in stages:
            stage = h5[name]
            _scan_stage(stage, f"{source}:{name}", unsupported)
            mode_only.append(_holds_only_modes(stage))
        hosts = [n for n, m in zip(stages, mode_only) if not m]
        if len(hosts) != len(names):
            raise StageKindRefused(
                f"{source} holds {len(hosts)} MODEL_STAGE group(s) with "
                f"step data ({hosts}) but the program declares "
                f"{len(names)} stage(s) ({list(names)}). Stage kinds cannot "
                "be paired, and a KIND is never guessed. Nothing was "
                "written."
            )
        paired = iter(kinds)
        return ["eigen" if m else next(paired) for m in mode_only]


def _stage_names(h5: "h5py.File | h5py.Group") -> list[str]:
    def key(name: str) -> "tuple[int, str]":
        inner = name[len(_STAGE_PREFIX):-1]
        return (int(inner), name) if inner.isdigit() else (1 << 62, name)

    return sorted((k for k in h5 if k.startswith(_STAGE_PREFIX)), key=key)


def _refuse_unknown(group: "h5py.Group", known: "set[str]", where: str) -> None:
    unknown = sorted(set(group) - set(known))
    if unknown:
        raise MPCOConversionRefused(
            f"{where} holds {unknown}, which this converter does not know "
            f"how to carry (known: {sorted(known)}). Nothing was written."
        )


def _scan_stage(stage: "h5py.Group", where: str, unsupported: set[str]) -> None:
    """Refuse what the writer would refuse; collect the lossy categories.

    ``springs`` is set by every ZeroLength-family bucket the stage holds,
    on any custom-rule index: each one is copied verbatim and none is read
    back. The scan is the spring reader's, without its warnings, which
    say a bucket is left unread; here every bucket is copied.
    """
    from ._mpco_spring_io import scan_spring_buckets

    _refuse_unknown(stage, {"MODEL", "RESULTS"}, where)
    if "MODEL" not in stage:
        raise MPCOConversionRefused(f"{where} has no MODEL group.")
    model = stage["MODEL"]
    _refuse_unknown(
        model, {"NODES", "ELEMENTS", "LOCAL_AXES", "SECTION_ASSIGNMENTS"},
        f"{where}/MODEL",
    )
    if "LOCAL_AXES" in model and len(model["LOCAL_AXES"]):
        unsupported.add(_LOCAL_AXES)
    if "ELEMENTS" in model:
        for gname, dset in model["ELEMENTS"].items():
            basis = _basis(gname, dset, f"{where}/MODEL/ELEMENTS/{gname}")
            if basis.gp_param is None:
                unsupported.add(_QUADRATURE + gname)
    if "RESULTS" not in stage:
        return
    results = stage["RESULTS"]
    _refuse_unknown(results, {"ON_NODES", "ON_ELEMENTS"}, f"{where}/RESULTS")
    if "ON_ELEMENTS" in results:
        on_el = results["ON_ELEMENTS"]
        for comp in ("spring_force", "spring_deformation"):
            scan = scan_spring_buckets(on_el, canonical_component=comp)
            if scan.buckets or scan.custom_rule:
                unsupported.add(_SPRINGS)


def _holds_only_modes(stage: "h5py.Group") -> bool:
    """True when the stage's only data are modal groups (an eigen stage)."""
    from ._mpco_translation import is_modal_result_name

    if "RESULTS" not in stage:
        return False
    results = stage["RESULTS"]
    has_modes = False
    if "ON_NODES" in results:
        for name, grp in results["ON_NODES"].items():
            if is_modal_result_name(name):
                has_modes = True
            elif _step_count(grp) > 0:
                return False
    if "ON_ELEMENTS" in results:
        for token in results["ON_ELEMENTS"].values():
            for bucket in token.values():
                if _step_count(bucket) > 0:
                    return False
    return has_modes


def _step_count(grp: "h5py.Group") -> int:
    if "DATA" not in grp:
        return 0
    return len(grp["DATA"])


# ---------------------------------------------------------------------------
# BASIS / QUADRATURE
# ---------------------------------------------------------------------------


@dataclass
class _Basis:
    topology: str
    family: str
    param_domain: str
    gp_param: "Optional[np.ndarray]"


_TOPOLOGY_BY_COORDS = {
    ("isoparametric", 1): ("line", "[-1,1]"),
    ("isoparametric_1d", 1): ("line", "[-1,1]"),
    ("isoparametric", 2): ("quad", "[-1,1]"),
    ("isoparametric", 3): ("hex", "[-1,1]"),
    ("barycentric_tri", 2): ("tri", "bary"),
    ("barycentric_tet", 3): ("tet", "bary"),
}


def _basis(gname: str, dset: Any, where: str) -> _Basis:
    """The BASIS of one ``MODEL/ELEMENTS`` group; ``gp_param`` None = unknown."""
    import h5py

    from ...opensees._response_catalog import (
        RESPONSE_CATALOG,
        IntRule,
        parse_mpco_element_key,
    )

    if not isinstance(dset, h5py.Dataset):
        raise MPCOConversionRefused(
            f"{where} is a group, not the flat connectivity dataset an "
            ".mpco writes.")
    if dset.ndim != 2 or dset.shape[1] < 2:
        raise MPCOConversionRefused(
            f"{where}: connectivity has shape {dset.shape}; expected "
            "[nElem x (1 + nNodes)].")
    try:
        key = parse_mpco_element_key(gname)
    except ValueError as e:
        raise MPCOConversionRefused(f"{where}: {e}") from e
    unknown = _Basis("custom", "custom", "[-1,1]", None)
    if key.int_rule == IntRule.Custom:
        dim = int(np.asarray(
            dset.attrs.get("CUSTOM_INTEGRATION_RULE_DIMENSION", 1)).ravel()[0])
        if "GP_X" not in dset.attrs or dim != 1:
            return unknown
        gp_x = np.asarray(dset.attrs["GP_X"], dtype=np.float64).ravel()
        if gp_x.size == 0:
            return unknown
        return _Basis("line", "lagrange", "[-1,1]", gp_x.reshape(-1, 1))
    if key.int_rule == IntRule.NoIntegrationRule:
        return unknown
    # Every catalog row of this class and rule, over all tokens, must give
    # the same parent coordinates; otherwise the group is not described.
    rows: "dict[tuple[str, tuple[int, ...], bytes], np.ndarray]" = {}
    for (cls, rule, _tok), lay in RESPONSE_CATALOG.items():
        if cls == key.class_name and rule == key.int_rule:
            coords = np.asarray(lay.natural_coords, dtype=np.float64)
            rows[(lay.coord_system, coords.shape, coords.tobytes())] = coords
    if len(rows) != 1:
        return unknown
    ((coord_system, shape, _), coords), = rows.items()
    if len(shape) != 2:
        return unknown
    topo = _TOPOLOGY_BY_COORDS.get((coord_system, int(shape[1])))
    if topo is None:
        return unknown
    return _Basis(topo[0], "lagrange", topo[1], coords.copy())


# ---------------------------------------------------------------------------
# Writing one part
# ---------------------------------------------------------------------------


def _s(text: str) -> np.ndarray:
    """A fixed-length string attribute, the way the fork recorder writes one."""
    return np.array([text.encode("utf-8")])


def _i(value: int) -> np.ndarray:
    return np.array([int(value)], dtype=np.int32)


def _convert_part(
    source: Path, out: Path, *, kinds: list[str], part: int, n_parts: int,
    unsupported: set[str],
) -> None:
    import h5py

    with h5py.File(source, "r") as src, h5py.File(out, "w-", track_order=True) as dst:
        # Groups keep their creation order, as the recorder writes them:
        # a reader that takes the first ON_NODES group naming a component
        # (REACTION_FORCE before RAYLEIGH_FORCE) must see the source order.
        info = dst.create_group("INFO", track_order=True)
        precisions: set[str] = set()
        for name, kind in zip(_stage_names(src), kinds):
            _write_stage(src[name], dst.create_group(name, track_order=True),
                         kind, precisions, where=f"{source.name}:{name}")
        if len(precisions) > 1:
            raise MPCOConversionRefused(
                f"{source} stores result data in mixed precisions "
                f"{sorted(precisions)}.")
        _write_info(src["INFO"], info,
                    precision=precisions.pop() if precisions else "f64",
                    part=part, n_parts=n_parts)


def _write_info(
    src: "h5py.Group", dst: "h5py.Group", *, precision: str, part: int,
    n_parts: int,
) -> None:
    dst.attrs["GENERATOR"] = _s("Ladruno")
    dst.attrs["FORMAT_VERSION"] = _i(1)
    for name in ("SOLVER_NAME", "SOLVER_VERSION", "SPATIAL_DIM"):
        if name in src:
            dst.attrs[name] = np.asarray(src[name][...])
    dst.attrs["STORED_PRECISION"] = _s(precision)
    dst.attrs["PARTITIONED"] = _i(1 if n_parts > 1 else 0)
    dst.attrs["PARTITION_ID"] = _i(part)
    dst.attrs["NUM_PARTITIONS"] = _i(n_parts)
    dst.create_group("PROVENANCE", track_order=True)


def _write_stage(
    src: "h5py.Group", dst: "h5py.Group", kind: str, precisions: set[str],
    *, where: str,
) -> None:
    for name, value in src.attrs.items():
        dst.attrs[name] = value
    dst.attrs["KIND"] = _s(kind)
    model = src["MODEL"]
    out_model = dst.create_group("MODEL", track_order=True)
    if "NODES" in model:
        src.file.copy(model["NODES"], out_model, "NODES")
    elements = out_model.create_group("ELEMENTS", track_order=True)
    if "ELEMENTS" in model:
        for gname, dset in model["ELEMENTS"].items():
            _write_element_group(gname, dset, elements.create_group(gname, track_order=True),
                                 where=f"{where}/MODEL/ELEMENTS/{gname}")
    sections = out_model.create_group("SECTION_ASSIGNMENTS", track_order=True)
    if "SECTION_ASSIGNMENTS" in model:
        for sname, sgrp in model["SECTION_ASSIGNMENTS"].items():
            src.file.copy(sgrp, sections, sname)
            sections[sname].attrs["KIND"] = _s(
                "fiber" if "FIBER_DATA" in sgrp else "resultant")
    if "RESULTS" not in src:
        return
    results = src["RESULTS"]
    out_results = dst.create_group("RESULTS", track_order=True)
    if "ON_NODES" in results:
        on_nodes = out_results.create_group("ON_NODES", track_order=True)
        for rname, rgrp in results["ON_NODES"].items():
            _write_node_result(rgrp, on_nodes, rname, precisions,
                               where=f"{where}/ON_NODES/{rname}")
    if "ON_ELEMENTS" in results:
        on_el = out_results.create_group("ON_ELEMENTS", track_order=True)
        for token, tgrp in results["ON_ELEMENTS"].items():
            _write_token(tgrp, on_el, token, precisions,
                         where=f"{where}/ON_ELEMENTS/{token}")


def _write_element_group(
    gname: str, dset: "h5py.Dataset", dst: "h5py.Group", *, where: str,
) -> None:
    basis = _basis(gname, dset, where)
    conn = np.asarray(dset[...])
    dst.create_dataset("CONNECTIVITY", data=conn)
    for name, value in dset.attrs.items():
        dst.attrs[name] = value
    dst.attrs["TOPOLOGY"] = _s(basis.topology)
    dst.attrs["FAMILY"] = _s(basis.family)
    dst.attrs["PARAM_DOMAIN"] = _s(basis.param_domain)
    dst.attrs["RATIONAL"] = _i(0)
    dst.attrs["NUM_CTRL"] = _i(conn.shape[1] - 1)
    if basis.gp_param is not None:
        dst.attrs["NUM_GP"] = _i(basis.gp_param.shape[0])
        dst.attrs["NDIR"] = _i(basis.gp_param.shape[1])
        quad = dst.create_group("QUADRATURE", track_order=True)
        quad.create_dataset("GP_PARAM", data=basis.gp_param)


def _write_node_result(
    src: "h5py.Group", parent: "h5py.Group", name: str, precisions: set[str],
    *, where: str,
) -> None:
    import h5py

    if not isinstance(src, h5py.Group):
        raise MPCOConversionRefused(f"{where} is not a result group.")
    _refuse_unknown(src, {"ID", "DATA"}, where)
    if "ID" not in src:
        raise MPCOConversionRefused(
            f"{where} has no ID dataset; the rows cannot be named.")
    data = src["DATA"] if "DATA" in src else None
    if data is not None and len(data) and all(
            isinstance(v, h5py.Group) for v in data.values()):
        # Modes of vibration: DATA/STEP_k/MODE_i, the same layout in both
        # formats, so the group is copied as it is.
        src.file.copy(src, parent, name)
        return
    dst = parent.create_group(name, track_order=True)
    for aname, value in src.attrs.items():
        dst.attrs[aname] = value
    src.file.copy(src["ID"], dst, "ID")
    _restack(data, dst, n_cols=None, precisions=precisions, where=where)


def _write_token(
    src: "h5py.Group", parent: "h5py.Group", token: str, precisions: set[str],
    *, where: str,
) -> None:
    import h5py

    if not isinstance(src, h5py.Group):
        raise MPCOConversionRefused(f"{where} is not a result group.")
    if not len(src):
        return  # an empty token group: the fork writes none
    dst = parent.create_group(token, track_order=True)
    for bname, bucket in src.items():
        bwhere = f"{where}/{bname}"
        if not isinstance(bucket, h5py.Group):
            raise MPCOConversionRefused(f"{bwhere} is not a bucket group.")
        _refuse_unknown(bucket, {"META", "ID", "DATA"}, bwhere)
        for required in ("META", "ID"):
            if required not in bucket:
                raise MPCOConversionRefused(f"{bwhere} has no {required}.")
        out = dst.create_group(bname, track_order=True)
        for aname, value in src.attrs.items():
            out.attrs[aname] = value
        for aname, value in bucket.attrs.items():
            if aname in src.attrs and not np.array_equal(
                    np.asarray(value), np.asarray(src.attrs[aname])):
                raise MPCOConversionRefused(
                    f"{bwhere}: attribute {aname} differs between the bucket "
                    "and its result group.")
            out.attrs[aname] = value
        if "NUM_COLUMNS" not in bucket.attrs:
            raise MPCOConversionRefused(f"{bwhere} has no NUM_COLUMNS.")
        n_cols = int(np.asarray(bucket.attrs["NUM_COLUMNS"]).ravel()[0])
        _write_column_map(bucket["META"], out, n_cols, where=bwhere)
        src.file.copy(bucket["ID"], out, "ID")
        _restack(bucket["DATA"] if "DATA" in bucket else None, out,
                 n_cols=n_cols, precisions=precisions, where=bwhere)


# ---------------------------------------------------------------------------
# META -> COLUMN_MAP
# ---------------------------------------------------------------------------


def _meta_column(meta: "h5py.Group", name: str, where: str) -> np.ndarray:
    if name not in meta:
        raise MPCOConversionRefused(f"{where}/META has no {name}.")
    return np.asarray(meta[name][...], dtype=np.int64).ravel()


def _split_block(block: str, n_comp: int, where: str) -> "tuple[list[int], list[str]]":
    """``"0.1.4.s11,s22"`` -> ``([0, 1, 4], ["s11", "s22"])``.

    The path is the leading run of integer fields; a component name may
    itself contain a dot or be a number, so the split is the one whose
    name count equals the block's ``NUM_COMPONENTS``, taking the longest
    integer path.
    """
    fields = block.split(".")
    for n_path in range(len(fields) - 1, 0, -1):
        head = fields[:n_path]
        if not all(f.lstrip("-").isdigit() for f in head):
            continue
        names = ".".join(fields[n_path:]).split(",")
        if len(names) == n_comp:
            return [int(f) for f in head], names
    raise MPCOConversionRefused(
        f"{where}: META COMPONENTS block {block!r} does not split into a "
        f"level path and {n_comp} component names.")


def _write_column_map(
    meta: "h5py.Group", bucket: "h5py.Group", n_cols: int, *, where: str,
) -> None:
    mult = _meta_column(meta, "MULTIPLICITY", where)
    gauss = _meta_column(meta, "GAUSS_IDS", where)
    ncomp = _meta_column(meta, "NUM_COMPONENTS", where)
    if "COMPONENTS" not in meta:
        raise MPCOConversionRefused(f"{where}/META has no COMPONENTS.")
    raw = np.asarray(meta["COMPONENTS"][...]).ravel()
    if raw.size != 1:
        raise MPCOConversionRefused(
            f"{where}/META/COMPONENTS holds {raw.size} strings, expected 1.")
    text = raw[0].decode("utf-8") if isinstance(raw[0], bytes) else str(raw[0])
    blocks = text.split(";")
    k = ncomp.size
    if not (mult.size == gauss.size == k == len(blocks)) or k == 0:
        raise MPCOConversionRefused(
            f"{where}/META: {mult.size} MULTIPLICITY, {gauss.size} "
            f"GAUSS_IDS, {k} NUM_COMPONENTS and {len(blocks)} COMPONENTS "
            "blocks; they must agree.")
    if int(np.sum(mult * ncomp)) != n_cols:
        raise MPCOConversionRefused(
            f"{where}: sum(MULTIPLICITY * NUM_COMPONENTS) = "
            f"{int(np.sum(mult * ncomp))} != NUM_COLUMNS {n_cols}.")
    levels = np.zeros(k, dtype=np.int32)
    lines = []
    for i, block in enumerate(blocks):
        path, names = _split_block(block, int(ncomp[i]), where)
        levels[i] = path[-1]
        lines.append(",".join(names))
    cmap = bucket.create_group("COLUMN_MAP", track_order=True)

    def col(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.int32).reshape(k, 1)

    cmap.create_dataset("LEVELS", data=col(levels))
    cmap.create_dataset("GAUSS_ID", data=col(gauss))
    cmap.create_dataset("SECTION_TAG", data=col(np.full(k, -1)))
    cmap.create_dataset("FIBER_ID", data=col(np.full(k, -1)))
    cmap.create_dataset("NUM_COMP", data=col(ncomp))
    cmap.create_dataset("MULTIPLICITY", data=col(mult))
    cmap.attrs["COMP_NAMES"] = _s("\n".join(lines))


# ---------------------------------------------------------------------------
# DATA/STEP_k -> DATA[T x n x c] + STEP[T] + TIME[T]
# ---------------------------------------------------------------------------


def _restack(
    data: "Optional[h5py.Group]", dst: "h5py.Group", *, n_cols: Optional[int],
    precisions: set[str], where: str,
) -> None:
    import h5py

    n_rows = int(np.asarray(dst["ID"]).shape[0])
    steps: list[tuple[int, float, h5py.Dataset]] = []
    for sname, ds in (data.items() if data is not None else ()):
        if not isinstance(ds, h5py.Dataset):
            raise MPCOConversionRefused(
                f"{where}/DATA/{sname} is a group among step datasets.")
        for attr in ("STEP", "TIME"):
            if attr not in ds.attrs:
                raise MPCOConversionRefused(
                    f"{where}/DATA/{sname} has no {attr} attribute.")
        steps.append((
            int(np.asarray(ds.attrs["STEP"]).ravel()[0]),
            float(np.asarray(ds.attrs["TIME"]).ravel()[0]),
            ds,
        ))
    steps.sort(key=lambda s: s[0])
    ids = [s[0] for s in steps]
    if len(set(ids)) != len(ids):
        raise MPCOConversionRefused(f"{where}: repeated STEP ids {ids}.")
    shapes = {s[2].shape for s in steps}
    if n_cols is None:
        n_cols = next(iter(shapes))[1] if shapes else 0
    want = (n_rows, n_cols)
    bad = sorted(str(s) for s in shapes if tuple(s) != want)
    if bad:
        raise MPCOConversionRefused(
            f"{where}: step datasets have shapes {bad}; expected {want} "
            "(ID rows x columns).")
    dtypes = {s[2].dtype for s in steps}
    if len(dtypes) > 1:
        raise MPCOConversionRefused(f"{where}: mixed step dtypes {dtypes}.")
    dtype = dtypes.pop() if dtypes else np.dtype(np.float64)
    if dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
        raise MPCOConversionRefused(f"{where}: step data are {dtype}.")
    precisions.add("f64" if dtype == np.float64 else "f32")
    n_t = len(steps)
    step_ids = np.asarray(ids, dtype=np.int32)
    times = np.asarray([s[1] for s in steps], dtype=np.float64)
    if not (n_rows and n_cols):
        dst.create_dataset("DATA", data=np.empty((n_t, n_rows, n_cols), dtype))
        dst.create_dataset("STEP", data=step_ids)
        dst.create_dataset("TIME", data=times)
        return
    ct, chunk_ids = _chunk_plan(n_rows, n_cols, dtype.itemsize)
    out = dst.create_dataset(
        "DATA", shape=(n_t, n_rows, n_cols), dtype=dtype,
        maxshape=(None, n_rows, n_cols), chunks=(ct, chunk_ids, n_cols),
        shuffle=True, compression="gzip", compression_opts=4,
    )
    # One block of ``ct`` steps at a time: every chunk is complete in one
    # write (no partial chunk re-deflated), and memory holds one block, not
    # the whole history.
    for t0 in range(0, n_t, ct):
        block = steps[t0:t0 + ct]
        buf = np.empty((len(block), n_rows, n_cols), dtype=dtype)
        for k, (_, _, ds) in enumerate(block):
            ds.read_direct(buf, dest_sel=np.s_[k])
        out[t0:t0 + len(block)] = buf
        del buf
    dst.create_dataset("STEP", data=step_ids, maxshape=(None,), chunks=(ct,))
    dst.create_dataset("TIME", data=times, maxshape=(None,), chunks=(ct,))


def _chunk_plan(n_ids: int, n_comp: int, elem_bytes: int) -> "tuple[int, int]":
    """``(steps, ids)`` per chunk: the recorder's WP-164 rule.

    Chunks target about 1 MiB (``Ladruno_Hdf5.h::planTimeSeries`` at
    ``FORK_PIN``). A slab of at most 1 MiB keeps every id in one chunk and
    stacks up to 1024 steps; a larger slab tiles the id axis and stacks
    at most 16 steps, so no chunk nears HDF5's 4 GiB limit.
    """
    target, row_budget = 1 << 20, 32 << 20
    row_bytes = n_comp * elem_bytes
    slab = n_ids * row_bytes
    if slab <= target:
        return max(1, min(1024, target // slab if slab else 1)), n_ids
    ct = max(1, min(16, row_budget // slab))
    return ct, max(1, min(n_ids, target // (ct * row_bytes)))

