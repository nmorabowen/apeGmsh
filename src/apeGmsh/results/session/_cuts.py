"""Persisted section cuts, as view clips (ADR 0098 S6b).

ADR 0098 retires the ``section_cut`` DIAGRAM kind but keeps the
contract behind it: *h5-persisted section cuts keep their auto-load
contract as clip-on-a-view — a new session boots with persisted cuts
as view clips*. This module is the LOAD half. The render half already
exists (:meth:`MeshView.add_clip`, ``viewers.session._realize
._apply_clips``) and is untouched.

Only the auto-load half ports. "kwarg-wins" (H14) is moot: the
``viewer(cuts=)`` kwarg was retired at S6a and :meth:`Results.session`
grew no replacement, so there is no kwarg left to win.

Precedence, mirrored from the retired ``ResultsDirector
.load_cuts_from_h5``
--------------------------------------------------------------------
A bound ``OpenSeesModel`` handle is the cuts source when there is one
— ``model.cuts()`` / ``model.sweeps()``, *even when both are empty*.
Only with NO handle bound does this fall back to reading the file with
:func:`apeGmsh.cuts.read_cuts_and_sweeps`. A port that always read the
file would silently diverge from the door it replaces.

NOTICE-AND-SKIP — the honesty rule
----------------------------------
A :class:`~apeGmsh.cuts.SectionCutDef` cut only the elements it NAMED,
optionally narrowed further by a convex ``bounding_polygon`` on the cut
plane. A :class:`~apeGmsh.results.session.ViewClip` has neither field:
it cuts the WHOLE view. So translating a cut that named a strict subset
of the model — or one carrying a bounding polygon — would silently
widen what disappears from the screen. Those are **skipped with a
notice** instead; only cuts that translate honestly are attached.

**Silence is the only forbidden outcome.** Every cut that is read
either becomes a clip or produces a line saying why it did not.

"Strict subset" is decided in OpenSees-tag space, against the model's
element ids: ``SectionCutDef.element_ids`` are OpenSees element tags
(not FEM eids — the two spaces differ, see
:class:`~apeGmsh.cuts.FemToOpsTagMap`), so the only set they may be
compared to is the model's own tag universe. That universe is taken
from the same place the cuts came from — ``model.elements()`` first,
then the ``/opensees/element_meta/*/ids`` table in the results file
(the very table the tag map is built from). A cut covering every one of
those tags cuts the whole model and translates honestly; anything less
is a strict subset and is skipped.

When the universe cannot be established at all (no handle, no
``element_meta`` zone) the cut is skipped too, with a notice saying the
check could not be made — the conservative direction. Widening the
picture on an unverified guess is exactly the failure this rule exists
to prevent.

Never fails the boot
--------------------
A malformed or unreadable cuts zone must not stop
:meth:`Results.session` returning a usable session — the same non-fatal
log-and-continue shape ``ResultsViewer._apply_pending_cuts`` used. Any
failure becomes one notice; the session boots with the clips that did
translate (possibly none).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apeGmsh.cuts import SectionCutDef
    from apeGmsh.results.Results import Results

    from ._views import MeshView


@dataclass(frozen=True)
class ClipRequest:
    """One :class:`~apeGmsh.cuts.SectionCutDef` that translates
    honestly, in :meth:`MeshView.add_clip`'s own terms."""

    name: str
    normal: tuple[float, float, float]
    offset: float
    flipped: bool


# ======================================================================
# Reading — the retired director's precedence, verbatim
# ======================================================================

def _named_cuts(results: "Results") -> "list[tuple[str, SectionCutDef]]":
    """``(display name, cut)`` for every persisted cut, sweeps flattened.

    Standalone cuts first in writer order, then each sweep's cuts in
    sweep order — the same order ``load_cuts_from_h5`` attached them
    in. Raises whatever the source raises; the caller owns the
    never-fail-the-boot guarantee.
    """
    model = getattr(results, "model", None)
    if model is not None:
        # Chain-forward path. Takes precedence even when it yields
        # nothing: a bound handle IS the answer about this model's
        # cuts, and re-walking the file behind its back is the silent
        # divergence this branch exists to avoid.
        cuts = tuple(model.cuts())
        sweeps = tuple(model.sweeps())
    else:
        path = getattr(results, "_path", None)
        if path is None or not _file_has_cuts_zone(path):
            return []
        from apeGmsh.cuts import read_cuts_and_sweeps
        cuts, sweeps = read_cuts_and_sweeps(path)

    named: "list[tuple[str, SectionCutDef]]" = [
        (cut.label or f"Section cut {i + 1}", cut)
        for i, cut in enumerate(cuts)
    ]
    for j, sweep in enumerate(sweeps):
        for k, cut in enumerate(sweep):
            named.append((cut.label or f"Sweep {j + 1} cut {k + 1}", cut))
    return named


def _file_has_cuts_zone(path: Any) -> bool:
    """Whether ``path`` carries a non-empty ``/opensees/cuts`` or
    ``/opensees/sweeps`` group.

    A cheap probe so a file with nothing persisted never pays for the
    validating reader — and, more importantly, never produces a notice
    about a schema it was never asked to satisfy. "No cuts" must boot
    silently.
    """
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            for zone in ("opensees/cuts", "opensees/sweeps"):
                group = f.get(zone)
                if group is not None and len(group) > 0:
                    return True
    except Exception:
        return False
    return False


# ======================================================================
# The model's element ids — the set "strict subset" is strict against
# ======================================================================

def _model_element_tags(results: "Results") -> Optional[frozenset[int]]:
    """Every OpenSees element tag in the model, or ``None`` if unknown.

    Same precedence as the cuts themselves: the bound handle's
    ``elements()`` first, the file's ``/opensees/element_meta/*/ids``
    table second. ``None`` means *not established* — never "empty
    model", which is why the caller must skip rather than compare
    against a set it does not have.
    """
    model = getattr(results, "model", None)
    if model is not None:
        try:
            tags = {int(record.tag) for record in model.elements()}
        except Exception:
            tags = set()
        if tags:
            return frozenset(tags)

    path = getattr(results, "_path", None)
    if path is None:
        return None
    try:
        import h5py
        tags = set()
        with h5py.File(str(path), "r") as f:
            meta = f.get("opensees/element_meta")
            if meta is None:
                return None
            for token in meta:
                ids = meta[token].get("ids")
                if ids is None:
                    continue
                tags.update(int(t) for t in ids[...].ravel())
    except Exception:
        return None
    return frozenset(tags) if tags else None


# ======================================================================
# Translation — honest, or a notice
# ======================================================================

def _translate(
    name: str,
    cut: "SectionCutDef",
    universe: Optional[frozenset[int]],
) -> "tuple[Optional[ClipRequest], Optional[str]]":
    """``(clip, None)`` for a cut that translates honestly, else
    ``(None, notice)``. Never both, never neither."""
    if cut.bounding_polygon is not None:
        return None, (
            f"section cut {name!r} was not loaded: it is bounded by a "
            f"polygon on the cut plane, and a view clip cuts the whole "
            f"view — attaching it would hide more than the cut did."
        )

    named = {int(e) for e in cut.element_ids}
    if universe is None:
        return None, (
            f"section cut {name!r} was not loaded: its {len(named)} "
            f"element id(s) could not be checked against the model's "
            f"element ids, so whether it cuts the whole model is unknown."
        )
    if not named >= universe:
        return None, (
            f"section cut {name!r} was not loaded: it names "
            f"{len(named & universe)} of the model's {len(universe)} "
            f"elements, and a view clip cuts the whole view — attaching "
            f"it would hide more than the cut did."
        )

    n = cut.plane_normal  # already unit (SectionCutDef.__post_init__)
    p = cut.plane_point
    return ClipRequest(
        name=name,
        normal=(float(n[0]), float(n[1]), float(n[2])),
        offset=float(p[0] * n[0] + p[1] * n[1] + p[2] * n[2]),
        flipped=(cut.side == "negative"),
    ), None


def persisted_clips(
    results: "Results",
) -> "tuple[tuple[ClipRequest, ...], tuple[str, ...]]":
    """``(clips, notices)`` for the cuts persisted against ``results``.

    The whole decision, with no view in hand — so the policy can be
    tested without booting one. Never raises: a source that cannot be
    read becomes a notice and an empty clip list.
    """
    try:
        named = _named_cuts(results)
    except Exception as exc:  # noqa: BLE001 - a bad file never blocks the boot
        return (), (
            f"persisted section cuts could not be read, so none were "
            f"loaded as view clips: {type(exc).__name__}: {exc}",
        )
    if not named:
        return (), ()

    try:
        universe = _model_element_tags(results)
    except Exception:  # noqa: BLE001 - unknown, not fatal; every cut skips
        universe = None

    clips: list[ClipRequest] = []
    notices: list[str] = []
    for name, cut in named:
        try:
            clip, notice = _translate(name, cut, universe)
        except Exception as exc:  # noqa: BLE001 - one bad cut, not all of them
            notices.append(
                f"section cut {name!r} was not loaded: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if clip is not None:
            clips.append(clip)
        if notice is not None:
            notices.append(notice)
    return tuple(clips), tuple(notices)


def attach_persisted_cuts(
    results: "Results", view: "MeshView",
) -> "tuple[str, ...]":
    """Attach every honestly-translating persisted cut to ``view``.

    Returns the notices the caller owes the human — one per cut that
    was read and NOT attached. Never raises.
    """
    clips, notices = persisted_clips(results)
    attached: list[str] = []
    for clip in clips:
        try:
            view.add_clip(
                clip.normal,
                offset=clip.offset,
                name=clip.name,
                flipped=clip.flipped,
            )
        except Exception as exc:  # noqa: BLE001 - the session still boots
            attached.append(
                f"section cut {clip.name!r} could not be attached as a "
                f"view clip: {type(exc).__name__}: {exc}"
            )
    return tuple(notices) + tuple(attached)


__all__ = [
    "ClipRequest",
    "attach_persisted_cuts",
    "persisted_clips",
]
