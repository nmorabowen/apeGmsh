"""
Shared ``model.h5`` composition (ADR 0018 / modeldata-enrichment-scope C1).

The single composer both authoring front doors use:

* ``apeSees.h5`` — bridge typed primitives → ``BuiltModel`` → ``H5Emitter``.
* ``ModelData.write`` — declarative orientation inject → ``H5Emitter``.

This module owns the broker-zone / bridge-zone / cuts composition
order, the stub-FEM fallback, the schema-version stamp, and the
partial-write teardown — exactly once.  Neither front door
reimplements any of it (ADR 0018 INV-1/3, scope C1).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import h5py


__all__ = [
    "_compose_model_h5",
    "_try_write_broker_zone",
    "_override_schema_version",
    "_path_stem",
]


def _compose_model_h5(
    fem: object,
    emitter: Any,
    path: str,
    *,
    model_name: str,
    ndf: int,
    cuts: "Sequence[Any]" = (),
    sweeps: "Sequence[Any]" = (),
    snapshot_id: str | None = None,
) -> None:
    """Compose a ``model.h5`` from a broker ``fem`` + a populated ``emitter``.

    The one composition path.  Order: broker ``/meta`` + neutral zone
    (with a stub-FEM fallback to the bridge's own ``/meta`` +
    schema-version override), then the ``emitter``'s ``/opensees/...``
    enrichment, then apeGmsh.cuts v4 ``/opensees/cuts`` / ``/sweeps``.

    Parameters
    ----------
    fem
        The broker snapshot.  A hand-rolled stub lacking the FEMData
        surface triggers the bridge-only fallback (no neutral zone).
    emitter
        An already-populated :class:`H5Emitter`.
    path
        Destination HDF5 path (opened ``"w"``).
    model_name, ndf
        Written into ``/meta`` by the broker writer.
    cuts, sweeps
        apeGmsh.cuts v4 sequences; empty ⇒ no cuts/sweeps groups.
    snapshot_id
        When not ``None``, overwrite ``/meta/snapshot_id`` with this
        exact string after meta is written (ADR 0018 INV-8 — opaque
        carry-through for ``ModelData.from_h5``).  ``None`` ⇒ leave
        whatever the broker / bridge wrote: the pre-extraction
        behaviour, so ``apeSees.h5`` is byte-invariant under C1.
    """
    import h5py

    from ...cuts._h5_io import write_cuts_into
    from ...mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION
    from ..emitter.h5 import SCHEMA_VERSION

    with h5py.File(path, "w") as f:
        broker_used = _try_write_broker_zone(
            fem, f,
            schema_version=NEUTRAL_SCHEMA_VERSION,
            model_name=model_name,
            ndf=ndf,
        )
        if not broker_used:
            # Stub FEM or otherwise missing broker surface — fall back
            # to bridge-only /meta with the bridge's own SCHEMA_VERSION
            # (the file still validates; absent neutral zone is the
            # right "no broker" signal).
            emitter._write_meta(f)
            _override_schema_version(f, SCHEMA_VERSION)
        if snapshot_id is not None and "meta" in f:
            # INV-8: opaque carry-through. A ModelData has no FEMData
            # to legitimately recompute the hash from; preserve the
            # exact /meta/snapshot_id byte string read off the source.
            f["meta"].attrs["snapshot_id"] = snapshot_id
        emitter.write_opensees_into(f)
        # Empty sequences are a no-op inside write_cuts_into — neither
        # /opensees/cuts/ nor /opensees/sweeps/ is created when nothing
        # was supplied.
        write_cuts_into(f, cuts=cuts, sweeps=sweeps)


def _try_write_broker_zone(
    fem: object,
    f: "h5py.File",
    *,
    schema_version: str,
    model_name: str,
    ndf: int,
) -> bool:
    """Attempt to write the broker's ``/meta`` + neutral zone.

    Returns ``True`` if the broker writer ran end-to-end, ``False`` if
    the FEM lacks the surface the writer needs (typically a hand-rolled
    test stub).  On ``False`` the file is rewound to a fresh state — no
    half-populated groups linger.
    """
    from ...mesh._femdata_h5_io import write_meta, write_neutral_zone

    if not hasattr(fem, "snapshot_id"):
        return False
    try:
        write_meta(
            fem, f,  # type: ignore[arg-type]
            schema_version=schema_version,
            model_name=model_name,
            ndf=ndf,
        )
        write_neutral_zone(fem, f)  # type: ignore[arg-type]
    except (AttributeError, TypeError):
        # Stub FEM didn't expose enough surface.  Tear down any
        # partial groups so the bridge's fallback `/meta` write
        # doesn't collide.
        for key in list(f.keys()):
            del f[key]
        return False
    return True


def _override_schema_version(f: "h5py.File", schema_version: str) -> None:
    """Overwrite ``/meta/schema_version`` after the bridge wrote it.

    The bridge stamps :data:`SCHEMA_VERSION` so even bridge-only files
    declare the post-Phase-8.5 schema; this is a no-op when the bridge
    already wrote that exact version, but guards against future drift
    between the constants.
    """
    if "meta" in f:
        f["meta"].attrs["schema_version"] = schema_version


def _path_stem(path: str) -> str:
    """Return ``path``'s file-name stem (no extension). Used as the
    default H5 ``/meta/model_name``."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem or "model"
