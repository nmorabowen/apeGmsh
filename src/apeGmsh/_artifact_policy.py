"""The D1 overwrite policy: when an automatic artifact write may replace a file.

ADR 0112 D1 makes every session run leave ``model.h5`` (and its geometry
sibling) at a conventional path, and the bridge will leave the fuller
``model.h5`` (neutral + ``/opensees``) at the same path (V2d, #1307).
Two writers at one path need one rule for who may replace what.  The
rule lives here, as module functions both writers call; it is the
maintainer's ruling of 2026-10-04 on #1307 (rules P1 to P4):

* **P2.** The session's default name is the ``__main__`` script's stem
  (:func:`main_script`).  With no real script (a notebook, ``-c``,
  stdin) nothing is written automatically, and one warning says so.
* **P3.** A file from another run is replaced only when its
  ``/provenance`` names a script this run's provenance also names, or
  names no script (:func:`provenance_scripts`).  Under MPI only rank 0
  writes (:func:`mpi_rank`); a partitioned run gets no automatic write.
* **P4.** A file from this run (the same ``session_id``) that holds a
  zone the write would drop is this run's fuller output: it is kept
  silently when its ``snapshot_id`` equals the snapshot's, and with a
  "stale" warning when the model changed after it was written.

A foreign file, an existing one under ``overwrite=False``, or an older
file holding a zone the write would drop keep V2b's warnings (#1305).
An explicit ``apeSees.h5(path)`` never consults this module (P7): it
keeps its ``'w'`` behaviour, because it is the user's intent.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:
    from ._internal.provenance import ProvenanceTable

__all__ = [
    "MPI_RANK_ENV",
    "artifact_identity",
    "artifact_target_is_ours",
    "main_script",
    "mpi_rank",
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
# The file
# ---------------------------------------------------------------------------


def _attr_str(attrs: "h5py.AttributeManager", key: str) -> str:
    if key not in attrs:
        return ""
    raw = attrs[key]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def artifact_identity(path: "str | Path") -> tuple[str, str, frozenset[str]]:
    """``(session_id, snapshot_id, scripts)`` of an apeGmsh artifact.

    ``session_id`` and ``snapshot_id`` are the ``/meta`` attributes, or
    ``""`` when absent.  ``scripts`` holds every ``/provenance/files``
    row of kind ``script`` (the ``__main__`` file of the run that wrote
    the file), absolute and case-normalised for a real file, as written
    for a pseudo-file (``<string>``, a notebook cell); empty when the
    file carries no ``/provenance`` or names no script.
    """
    with h5py.File(str(path), "r") as f:
        attrs = f["meta"].attrs if "meta" in f else {}
        session_id = _attr_str(attrs, "session_id")
        snapshot_id = _attr_str(attrs, "snapshot_id")
        scripts: set[str] = set()
        if "provenance" in f and "files" in f["provenance"]:
            prov = f["provenance"]
            base_dir = _attr_str(prov.attrs, "base_dir")
            files = prov["files"]
            paths = files["path"].asstr()[()]
            kinds = files["kind"].asstr()[()]
            for p, k in zip(paths, kinds):
                if k == "script":
                    scripts.add(_script_key(str(p), base_dir))
    return session_id, snapshot_id, frozenset(scripts)


def _script_key(path: str, base_dir: str) -> str:
    """One comparable key for a provenance file path: as written for a
    pseudo-file, else absolute (``base_dir`` resolves a relative
    ``/provenance/files`` path) and case-normalised."""
    from ._internal.provenance import _absolute

    if path.startswith("<"):
        return path
    return os.path.normcase(os.path.abspath(_absolute(path, base_dir)))


def provenance_scripts(table: "ProvenanceTable | None") -> frozenset[str]:
    """The scripts a snapshot's provenance names, as :func:`artifact_identity`
    keys them: what the write would put in ``/provenance/files`` with kind
    ``script``.  Empty for a snapshot without a table, or one whose
    declarations were made with no user ``__main__`` frame around them (a
    test function).  P3 compares these with the target's: a file from
    another run is replaced when the two sets meet, or the file's is empty.
    """
    if table is None:
        return frozenset()
    return frozenset(
        _script_key(f.path, "") for f in table.files if f.kind == "script"
    )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def artifact_target_is_ours(
    target: Path,
    *,
    writes: frozenset[str],
    overwrite: bool,
    session_id: str,
    fem_hash: str,
    scripts: frozenset[str],
    explicit: bool,
) -> bool:
    """May an automatic D1 write replace ``target``?

    ``writes`` names the zones the write produces; ``session_id`` and
    ``fem_hash`` are the ``/meta/session_id`` and ``/meta/snapshot_id``
    the write would stamp (``""`` for a file that carries no snapshot,
    the geometry sibling); ``scripts`` is :func:`provenance_scripts` of
    the snapshot; ``explicit`` is True when the user named the target
    (``save_to=``), which exempts it from P3's same-script rule.

    Yes when ``target`` does not exist, or when all of these hold:

    * ``overwrite`` is on, and the file is an apeGmsh artifact (a per-zone
      ``/meta`` version key; V2b);
    * **this run's file** (equal ``session_id``): it holds no zone the
      write would drop.  One that does is this run's fuller output (the
      bridge's neutral + ``/opensees``): kept **silently** when its
      ``snapshot_id`` equals ``fem_hash`` (P4), kept with one "stale"
      warning when the model changed after it was written;
    * **another run's file**: it holds no zone the write would drop
      (V2b's warning otherwise), and, unless ``explicit``, its
      ``/provenance`` names a script in ``scripts`` or names no script
      (P3; a parameter sweep that keeps each run sets ``model_name``
      per run).  A notebook names its cells, so an edited notebook does
      not refresh the file it wrote earlier: ``save_to=`` does.

    Every refusal is one warning (``UserWarning``) and ``False``; the
    file is never written elsewhere.
    """
    from .mesh._geometry_h5_io import artifact_zones, is_apegmsh_artifact

    if not target.exists():
        return True
    if not overwrite:
        warnings.warn(
            f"{target} exists and overwrite=False; not written",
            stacklevel=_STACKLEVEL,
        )
        return False
    if not is_apegmsh_artifact(target):
        warnings.warn(
            f"{target} exists and is not an apeGmsh artifact (no /meta "
            f"schema key); not overwritten. Pass save_to= to write the "
            f"model elsewhere.",
            stacklevel=_STACKLEVEL,
        )
        return False
    held = artifact_zones(target)
    file_session, file_hash, file_scripts = artifact_identity(target)
    dropped = sorted(held - writes)
    if file_session == session_id:
        if not dropped:
            return True
        if file_hash == fem_hash:
            return False
        warnings.warn(
            f"{target} is stale: it holds this run's {', '.join(dropped)} "
            f"zone(s), written from an earlier snapshot of a model that "
            f"changed afterwards; not overwritten. Emit again after the "
            f"last change, or pass save_to=.",
            stacklevel=_STACKLEVEL,
        )
        return False
    if dropped:
        warnings.warn(
            f"{target} holds the {', '.join(dropped)} zone(s) that the "
            f"automatic write would drop; not overwritten. Write that "
            f"file under another name, or pass save_to=.",
            stacklevel=_STACKLEVEL,
        )
        return False
    if not explicit and file_scripts and not (file_scripts & scripts):
        warnings.warn(
            f"{target} was written by another script "
            f"({', '.join(sorted(file_scripts))}); not overwritten. Set "
            f"model_name= per run to keep each run's output, or pass "
            f"save_to= to replace it.",
            stacklevel=_STACKLEVEL,
        )
        return False
    return True
