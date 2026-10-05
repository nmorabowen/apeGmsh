"""The D1 overwrite policy: when an automatic artifact write may replace a file.

ADR 0112 D1 makes every session run leave ``model.h5`` (and its geometry
sibling) at a conventional path, and the bridge will leave the fuller
``model.h5`` (neutral + ``/opensees``) at the same path (V2d, #1307).
Two writers at one path need one rule for who may replace what.  The
rule lives here, as module functions both writers call; it is the
maintainer's ruling of 2026-10-04 on #1307 (rules P1 to P4), with the
rulings of 2026-10-05 on PR #1439:

* **P2.** The session's default name is the ``__main__`` script's stem
  (:func:`main_script`).  With no real script (a notebook, ``-c``,
  stdin) nothing is written automatically, and one warning says so.
* **P3.** A file from another run is replaced only when its
  ``/provenance`` names a script this run's provenance also names, or
  names no script (:func:`provenance_scripts`).  A notebook cell's path
  (``.../ipykernel_<pid>/...``) carries the kernel's process id and is
  no stable identity, so it names no script: a notebook's file is
  replaced, and the docs say to give each notebook its own
  ``model_name``.  Under MPI only rank 0 writes automatically
  (:func:`mpi_rank`); a run whose mesh is partitioned (an MPI deck's)
  gets no automatic write.  A composed model is not partitioned for D1:
  it writes.  An explicit ``save_to=`` bypasses both gates (P7).
* **P4.** A file from this run (the same ``session_id``) that holds a
  zone the write would drop is this run's fuller output: it is kept
  silently when the content it holds is what the write would produce
  (:func:`content_hash`: **everything** the neutral zone carries, mesh,
  groups, labels, loads, masses, constraints and ties, never
  ``snapshot_id`` alone), and with a "stale" warning when anything
  changed after it was written.
* A refused model write also skips the geometry sibling, with one
  warning naming both files, so the pair stays consistent.

A foreign file, an existing one under ``overwrite=False``, or an older
file holding a zone the write would drop keep V2b's warnings (#1305).
An explicit ``apeSees.h5(path)`` never consults this module (P7): it
keeps its ``'w'`` behaviour, because it is the user's intent.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np

if TYPE_CHECKING:
    from ._internal.provenance import ProvenanceTable
    from .mesh.FEMData import FEMData

__all__ = [
    "MPI_RANK_ENV",
    "Verdict",
    "artifact_content_hash",
    "artifact_identity",
    "artifact_target_is_ours",
    "artifact_verdict",
    "content_hash",
    "main_script",
    "mpi_rank",
    "neutral_content_hash",
    "provenance_scripts",
]

#: Environment variables an MPI launcher (or ``srun``) sets to the rank
#: of the process it started, in the order they are consulted.
MPI_RANK_ENV: tuple[str, ...] = (
    "OMPI_COMM_WORLD_RANK",   # Open MPI
    "PMI_RANK",               # MPICH, Intel MPI, MS-MPI
    "PMIX_RANK",              # PMIx launchers
    "MV2_COMM_WORLD_RANK",    # MVAPICH2
    "SLURM_PROCID",           # srun
)

#: What :func:`artifact_verdict` answers: write the target; keep it as it
#: is, silently (this run's fuller output); or refuse, having warned.
Verdict = Literal["write", "keep", "refuse"]

#: Root groups that are not neutral-zone content (h5-schema.md, "Zone
#: registry"): the other zones, and ``/meta`` with its timestamp and ids.
_NON_NEUTRAL_ROOTS = frozenset({"meta", "provenance", "opensees", "stages", "geometry"})

#: A notebook cell as ipykernel names it: a file under a per-kernel
#: temp directory (``ipykernel_<pid>``), or IPython's older pseudo-file.
#: Neither outlives the kernel, so neither is a script identity for P3.
_NOTEBOOK_CELL = re.compile(r"(^|[\\/])ipykernel_\d+[\\/]|^<ipython-input-")

#: ``warnings.warn`` depth from these functions to the user's line:
#: the function, the writer's private method, its public caller
#: (``end()``, ``tcl()``, ...), the user's call.
_STACKLEVEL = 4


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def main_script() -> Path | None:
    """The user's ``__main__`` script, resolved, or ``None`` (P2).

    ``None`` when Python runs no real file as ``__main__`` (a REPL, a
    notebook cell, ``-c``, stdin), and when the ``__main__`` module is a
    launcher from the standard library or site-packages (``python -m
    pytest``, ``ipykernel``): the provenance classifier decides, so the
    name a session takes and the script its ``/provenance`` names are
    the same file.  A script run in place under the apeGmsh tree is the
    user's, as it is for provenance.
    """
    from ._internal.provenance import _APEGMSH, _USER, _classify

    main = sys.modules.get("__main__")
    try:
        file = main.__file__ if main is not None else None
    except AttributeError:  # a REPL or notebook __main__ has no file
        file = None
    if not file or not os.path.isfile(file):
        return None
    if _classify(file) not in (_USER, _APEGMSH):
        return None
    return Path(file).resolve()


def mpi_rank() -> int | None:
    """This process's MPI rank from the launcher's environment, or
    ``None`` outside an MPI launch (P3).

    Raises ``RuntimeError`` on a rank variable that is not an integer:
    a writer that cannot tell whether it is rank 0 must not write.
    """
    for name in MPI_RANK_ENV:
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            raise RuntimeError(
                f"{name}={value!r} is not an integer MPI rank; cannot tell "
                f"whether this process is rank 0, so nothing is written"
            ) from None
    return None


# ---------------------------------------------------------------------------
# The content
# ---------------------------------------------------------------------------


def _feed_token(h: "hashlib.blake2b", tag: bytes, data: bytes) -> None:
    h.update(tag)
    h.update(len(data).to_bytes(8, "little"))
    h.update(data)


def _feed_object(h: "hashlib.blake2b", el: object) -> None:
    """Feed one element of an object array: a string, bytes, ``None``, a
    number or an array.  Anything else is refused: its ``repr`` could
    carry an address, and a silent fallback would make the digest drift."""
    if el is None:
        h.update(b"N|")
    elif isinstance(el, bytes):
        _feed_token(h, b"B|", el)
    elif isinstance(el, str):
        _feed_token(h, b"U|", el.encode("utf-8"))
    elif isinstance(el, (bool, int, float)):
        _feed_token(h, b"P|", repr(el).encode("ascii"))
    elif isinstance(el, np.ndarray):
        _feed_array(h, el)
    elif isinstance(el, np.generic):
        _feed_array(h, np.asarray(el))
    else:
        raise TypeError(
            f"content hash: an object element of type {type(el).__name__} "
            f"in the neutral zone has no stable encoding"
        )


def _feed_array(h: "hashlib.blake2b", arr: np.ndarray) -> None:
    """Feed an array: a structured one field by field in dtype order (a
    nested compound recurses), an object one element by element, any
    other as its dtype tag, shape and contiguous bytes."""
    dt = arr.dtype
    if dt.names:
        h.update(b"S|")
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        for name in dt.names:
            _feed_token(h, b"F|", name.encode("utf-8"))
            _feed_array(h, arr[name])
        h.update(b"E|")
    elif dt.kind == "O":
        h.update(b"O|")
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        elements = arr.ravel().tolist()
        if all(isinstance(el, bytes) for el in elements):
            # The common case (a variable-length string column read
            # back as bytes): one length-prefixed run, one update.
            h.update(b"".join(
                b"B|" + len(el).to_bytes(8, "little") + el for el in elements
            ))
        else:
            for el in elements:
                _feed_object(h, el)
        h.update(b"E|")
    else:
        _feed_token(h, b"A|", dt.str.encode("ascii"))
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(arr).tobytes())


def _feed_node(h: "hashlib.blake2b", node: "h5py.Group | h5py.Dataset") -> None:
    for key in sorted(node.attrs.keys()):
        _feed_token(h, b"@|", key.encode("utf-8"))
        _feed_object(h, node.attrs[key])
    if isinstance(node, h5py.Dataset):
        h.update(b"D|")
        _feed_array(h, np.asarray(node[()]))
        return
    h.update(b"G|")
    for name in sorted(node.keys()):
        _feed_token(h, b"M|", name.encode("utf-8"))
        _feed_node(h, node[name])
    h.update(b"E|")


def neutral_content_hash(root: "h5py.Group") -> str:
    """One digest of every neutral-zone group under ``root``.

    The groups are walked in name order; each dataset contributes its
    attributes (sorted by name), dtype tag, shape and contiguous bytes,
    a structured dataset field by field and an object (variable-length
    string) dataset element by element, so chunked and contiguous
    storage, and a write and its reload, give the same digest, and no
    element is ever encoded through its ``repr``.  ``/meta`` (timestamp,
    ids, name), ``/provenance`` and the other zones are not content and
    are skipped.
    """
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(k for k in root.keys() if k not in _NON_NEUTRAL_ROOTS):
        _feed_token(h, b"R|", name.encode("utf-8"))
        _feed_node(h, root[name])
    return h.hexdigest()


def content_hash(fem: "FEMData") -> str:
    """The digest of everything the neutral write of ``fem`` would put in
    the file (P4): the zone is written to an in-memory HDF5 file by the
    one neutral writer and hashed with :func:`neutral_content_hash`, so
    it equals :func:`artifact_content_hash` of the file that write
    leaves, whoever wrote it (the session or the bridge).
    """
    from .mesh._femdata_h5_io import write_neutral_zone

    name = f"apegmsh-content-{uuid.uuid4().hex}.h5"
    with h5py.File(name, "w", driver="core", backing_store=False) as f:
        write_neutral_zone(fem, f)
        return neutral_content_hash(f)


def artifact_content_hash(path: "str | Path") -> str:
    """:func:`neutral_content_hash` of the artifact at ``path``."""
    with h5py.File(str(path), "r") as f:
        return neutral_content_hash(f)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def _attr_str(attrs: "h5py.AttributeManager", key: str) -> str:
    if key not in attrs:
        return ""
    raw = attrs[key]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def artifact_identity(path: "str | Path") -> tuple[str, frozenset[str]]:
    """``(session_id, scripts)`` of an apeGmsh artifact.

    ``session_id`` is the ``/meta`` attribute, or ``""`` when absent.
    ``scripts`` holds every ``/provenance/files`` row of kind ``script``
    (the ``__main__`` file of the run that wrote the file), absolute and
    case-normalised for a real file, as written for a pseudo-file
    (``<string>``, ``<stdin>``); a notebook cell is left out, as it is
    no stable identity.  Empty when the file carries no ``/provenance``
    or names no script.
    """
    with h5py.File(str(path), "r") as f:
        session_id = _attr_str(f["meta"].attrs, "session_id") if "meta" in f else ""
        scripts: set[str] = set()
        if "provenance" in f and "files" in f["provenance"]:
            prov = f["provenance"]
            base_dir = _attr_str(prov.attrs, "base_dir")
            files = prov["files"]
            paths = files["path"].asstr()[()]
            kinds = files["kind"].asstr()[()]
            for p, k in zip(paths, kinds):
                if k == "script":
                    key = _script_key(str(p), base_dir)
                    if key is not None:
                        scripts.add(key)
    return session_id, frozenset(scripts)


def _script_key(path: str, base_dir: str) -> str | None:
    """One comparable key for a provenance script path: as written for a
    pseudo-file, else absolute (``base_dir`` resolves a relative
    ``/provenance/files`` path) and case-normalised; ``None`` for a
    notebook cell (:data:`_NOTEBOOK_CELL`), which names no script."""
    from ._internal.provenance import _absolute

    if _NOTEBOOK_CELL.search(path):
        return None
    if path.startswith("<"):
        return path
    return os.path.normcase(os.path.abspath(_absolute(path, base_dir)))


def provenance_scripts(table: "ProvenanceTable | None") -> frozenset[str]:
    """The scripts a snapshot's provenance names, as :func:`artifact_identity`
    keys them: what the write would put in ``/provenance/files`` with kind
    ``script``.  Empty for a snapshot without a table, one whose
    declarations were made with no user ``__main__`` frame around them (a
    test function), or one made in a notebook.  P3 compares these with
    the target's: a file from another run is replaced when the two sets
    meet, or the file's is empty.
    """
    if table is None:
        return frozenset()
    keys = (_script_key(f.path, "") for f in table.files if f.kind == "script")
    return frozenset(k for k in keys if k is not None)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def _refuse(message: str, skips: tuple[Path, ...]) -> Literal["refuse"]:
    """Warn once for a refused target and the files skipped with it."""
    if skips:
        names = ", ".join(str(p) for p in skips)
        message = (
            f"{message} {names} is not written either, so the pair stays "
            f"consistent."
        )
    warnings.warn(message, stacklevel=_STACKLEVEL + 1)
    return "refuse"


def artifact_verdict(
    target: Path,
    *,
    writes: frozenset[str],
    overwrite: bool,
    session_id: str,
    content: Callable[[], str],
    scripts: frozenset[str],
    explicit: bool,
    skips: tuple[Path, ...] = (),
) -> Verdict:
    """What an automatic D1 write may do at ``target``.

    ``writes`` names the zones the write produces; ``session_id`` is the
    ``/meta/session_id`` it would stamp; ``content`` gives, when asked,
    the :func:`content_hash` of what it would write (``lambda: ""`` for
    a file that carries no neutral zone, the geometry sibling);
    ``scripts`` is :func:`provenance_scripts` of the snapshot;
    ``explicit`` is True when the user named the target (``save_to=``),
    which exempts it from P3's same-script rule; ``skips`` are the files
    the caller leaves unwritten when the target is refused (the
    geometry sibling), named in that one warning.

    ``"write"`` when ``target`` does not exist, or when all of these hold:

    * ``overwrite`` is on, and the file is an apeGmsh artifact (a per-zone
      ``/meta`` version key; V2b);
    * **this run's file** (equal ``session_id``): it holds no zone the
      write would drop.  One that does is this run's fuller output (the
      bridge's neutral + ``/opensees``): ``"keep"``, silently, when its
      neutral content equals what the write would produce (P4), and
      ``"refuse"`` with one warning when it holds different content (a
      filtered ``get_fem_data``, or a change after it was written);
    * **another run's file**: it holds no zone the write would drop
      (V2b's warning otherwise), and, unless ``explicit``, its
      ``/provenance`` names a script in ``scripts`` or names no script
      (P3; a parameter sweep that keeps each run sets ``model_name``
      per run, and so does each notebook, whose cells name no script).

    Every ``"refuse"`` is one warning (``UserWarning``); the file is
    never written elsewhere.
    """
    from .mesh._geometry_h5_io import artifact_zones, is_apegmsh_artifact

    if not target.exists():
        return "write"
    if not overwrite:
        return _refuse(f"{target} exists and overwrite=False; not written.", skips)
    if not is_apegmsh_artifact(target):
        return _refuse(
            f"{target} exists and is not an apeGmsh artifact (no /meta "
            f"schema key); not overwritten. Pass save_to= to write the "
            f"model elsewhere.",
            skips,
        )
    held = artifact_zones(target)
    file_session, file_scripts = artifact_identity(target)
    dropped = sorted(held - writes)
    if file_session == session_id:
        if not dropped:
            return "write"
        if artifact_content_hash(target) == content():
            return "keep"
        return _refuse(
            f"{target} holds different content than this session would "
            f"write (a filtered get_fem_data, or a change after it was "
            f"written) with its {', '.join(dropped)} zone(s); not replaced. "
            f"Emit again from the session's snapshot, or pass save_to=.",
            skips,
        )
    if dropped:
        return _refuse(
            f"{target} holds the {', '.join(dropped)} zone(s) that the "
            f"automatic write would drop; not overwritten. Write that "
            f"file under another name, or pass save_to=.",
            skips,
        )
    if not explicit and file_scripts and not (file_scripts & scripts):
        return _refuse(
            f"{target} was written by another script "
            f"({', '.join(sorted(file_scripts))}); not overwritten. Set "
            f"model_name= per run to keep each run's output, or pass "
            f"save_to= to replace it.",
            skips,
        )
    return "write"


def artifact_target_is_ours(
    target: Path,
    *,
    writes: frozenset[str],
    overwrite: bool,
    session_id: str,
    content: Callable[[], str],
    scripts: frozenset[str],
    explicit: bool,
    skips: tuple[Path, ...] = (),
) -> bool:
    """May an automatic D1 write replace ``target``?  True for the
    ``"write"`` verdict of :func:`artifact_verdict`, whose parameters and
    rows this shares; ``"keep"`` and ``"refuse"`` are both False."""
    return artifact_verdict(
        target, writes=writes, overwrite=overwrite, session_id=session_id,
        content=content, scripts=scripts, explicit=explicit, skips=skips,
    ) == "write"
