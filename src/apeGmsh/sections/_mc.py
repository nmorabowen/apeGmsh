"""Moment–curvature for fiber-lane section documents (ADR 0080 B7).

Productizes the in-process ``zeroLengthSection`` harness that gate G-D
(ADR 0078) and gate G-E (ADR 0080 B3) proved: two coincident nodes, the
section on a zero-length element, displacement control on one rotation
DOF, the moment read off the reference pattern's load factor.

The section is lowered through exactly the construction the bridge
handoff uses (:func:`~apeGmsh.sections._document.typed_fiber_items` →
each primitive's own ``_emit``), so an M–κ curve and a deck built from
the same document integrate the *same* fibers with the *same* material
tags. Nothing about the section is re-derived here.

Two properties of this module follow from openseespy being a **single
process-global domain**, and both are contracts, not accidents:

* :func:`moment_curvature` **wipes** that domain before building its
  harness. Do not call it while a live analysis is open — the live
  model is destroyed.
* It runs **synchronously on the calling thread** and must stay that
  way. openseespy is a process-global C++ runtime with no reentrancy;
  driving it from a worker while another thread drives it (or Gmsh)
  aborts the interpreter below Python's reach. The B6 properties
  worker is threaded because it builds *Gmsh* sections; this is
  deliberately not.

openseespy absent → ``ImportError`` with guidance (fail-soft: nothing
else in :mod:`apeGmsh.sections` needs it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from ._document import SectionDocument


__all__ = [
    "MomentCurvature",
    "MomentCurvatureError",
    "backend_available",
    "moment_curvature",
]


def backend_available() -> bool:
    """``True`` when an OpenSees backend module appears importable.

    A cheap :func:`importlib.util.find_spec` probe — it never imports
    the backend (which is slow and prints the fork's banner). Used only
    to enable/disable the builder's M–κ button; the real gate is
    :func:`moment_curvature` itself, which fails soft with guidance.
    """
    from importlib.util import find_spec

    for module in ("opensees", "openseespy"):
        try:
            if find_spec(module) is not None:
                return True
        except (ImportError, ValueError):    # pragma: no cover - odd paths
            continue
    return False


#: axis label → the OpenSees node DOF that bends about it. ``"z"`` is
#: the strong-axis default: its elastic slope is the analyzer's
#: ``EIxx_c`` (gate G-D pairs DOF 6 with ``EIxx_c``, DOF 5 with
#: ``EIyy_c``).
_DOF_OF_AXIS: "dict[str, int]" = {"z": 6, "y": 5}

#: torsional stiffness used when the document declares no ``GJ``. Inert
#: by construction — the harness fixes the torsional DOF at both nodes,
#: so this only satisfies OpenSees' "torsion not specified" refusal.
_PLACEHOLDER_GJ: float = 1.0


class MomentCurvatureError(RuntimeError):
    """The M–κ harness could not be built or its first step failed."""


@dataclass(frozen=True, slots=True)
class MomentCurvature:
    """One moment–curvature curve (ADR 0080 B7).

    ``curvature`` and ``moment`` are parallel, start at ``(0, 0)``, and
    carry the sign of ``kappa_max``. ``complete`` is ``False`` when a
    step failed to converge before ``n_steps`` — the curve up to that
    point is still valid, which is the normal end of an RC section that
    crushes.
    """

    axis: str
    curvature: tuple[float, ...]
    moment: tuple[float, ...]
    axial: float
    complete: bool

    @property
    def EI0(self) -> float:
        """Initial (first-step secant) flexural stiffness ``M/κ``.

        For elastic materials this is the fiber-sum ``Σ E·A·r²`` about
        the bending axis exactly — the identity the B7 gate checks.
        """
        if len(self.curvature) < 2 or self.curvature[1] == 0.0:
            raise MomentCurvatureError(
                "EI0 needs at least one converged step with non-zero "
                "curvature."
            )
        return self.moment[1] / self.curvature[1]

    @property
    def M_max(self) -> float:
        """Largest ``|M|`` reached, carrying its sign."""
        return max(self.moment, key=abs) if self.moment else 0.0


def _ops_module() -> Any:
    """The resolved OpenSees backend, or a guided ``ImportError``."""
    from apeGmsh.opensees.emitter.live import _get_ops

    try:
        return _get_ops()
    except ImportError as exc:
        raise ImportError(
            "moment_curvature() needs an OpenSees backend (openseespy or "
            "the Ladruno fork). Install one — pip install openseespy — or "
            "point APEGMSH_OPENSEES_BIN at the fork's dist\\bin. The rest "
            "of apeGmsh.sections works without it."
        ) from exc


def _material_from_spec(name: str, spec: "dict[str, Any] | None") -> Any:
    """Construct one typed ``UniaxialMaterial`` from a document spec.

    The spec's ``type`` is the same token ``ops.uniaxialMaterial.<Type>``
    takes, and its ``params`` go straight to the typed primitive's
    constructor. Specs that lean on bridge-level convenience the
    primitive does not accept (e.g. naming another registered material)
    fail loudly here rather than silently curving differently than the
    deck would.
    """
    from apeGmsh.opensees.material import uniaxial as _uniaxial

    if not spec:
        raise MomentCurvatureError(
            f"material {name!r} has no uniaxial spec — moment_curvature "
            f"needs set_material({name!r}, ..., uniaxial=('<Type>', "
            f"{{...}}))."
        )
    cls = getattr(_uniaxial, spec["type"], None)
    if cls is None:
        raise MomentCurvatureError(
            f"material {name!r}: no typed uniaxial primitive named "
            f"{spec['type']!r}."
        )
    try:
        return cls(**spec["params"])
    except (TypeError, ValueError) as exc:
        raise MomentCurvatureError(
            f"material {name!r} ({spec['type']}): {exc}"
        ) from exc


def _emit_section(ops: Any, doc: "SectionDocument") -> None:
    """Emit the document's materials + ``section Fiber 1`` into the live
    domain, through the primitives' own ``_emit``."""
    from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter
    from apeGmsh.opensees.section.fiber import Fiber

    from ._document import typed_fiber_items

    recipe = doc.build()
    table = doc.to_dict()["materials"]
    used = sorted({
        i["material"]
        for i in (*recipe.patches, *recipe.layers, *recipe.points)
    })
    mats = {
        name: _material_from_spec(name, table.get(name, {}).get("uniaxial"))
        for name in used
    }
    patches, layers, points = typed_fiber_items(recipe, mats)
    section = Fiber(
        patches=patches, layers=layers, fibers=points,
        # OpenSees refuses a 3D fiber section with no torsion at all, but
        # the harness fixes the torsional DOF at both nodes — so a
        # document that declares no GJ gets a placeholder that provably
        # cannot reach the answer.
        GJ=recipe.GJ if recipe.GJ is not None else _PLACEHOLDER_GJ,
    )

    tags = {id(m): i for i, m in enumerate(mats.values(), start=1)}
    emitter = LiveOpsEmitter(wipe=False)
    set_tag_resolver(emitter, lambda prim: tags[id(prim)])
    for material in mats.values():
        material._emit(emitter, tags[id(material)])
    section._emit(emitter, 1)


def moment_curvature(
    doc: "SectionDocument",
    *,
    axis: "Literal['z', 'y']" = "z",
    kappa_max: float,
    n_steps: int = 40,
    axial: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 25,
) -> MomentCurvature:
    """Push a fiber-lane document's section to ``kappa_max`` and record
    its moment–curvature response.

    Parameters
    ----------
    doc
        A **fiber-lane** :class:`~apeGmsh.sections.SectionDocument`.
        Every material it uses needs a ``uniaxial`` spec. (Continuum
        documents lower to fibers only through a bridge — hand
        ``doc.to_section(ops)`` to a frame model for those.)
    axis
        ``"z"`` (default) bends about the section's z-axis — DOF 6,
        whose elastic slope is the analyzer's ``EIxx_c``. ``"y"`` is
        DOF 5 / ``EIyy_c``.
    kappa_max
        Target curvature. **Signed**: a negative value walks the curve
        into the other quadrant, which is how an asymmetric section's
        two directions get compared.
    n_steps
        Displacement-control steps from 0 to ``kappa_max``.
    axial
        Constant axial force held during the push, applied first and
        frozen (``loadConst``). OpenSees convention — **compression is
        negative**.
    tol, max_iter
        Newton convergence test (``NormDispIncr``) parameters.

    Returns
    -------
    MomentCurvature

    Raises
    ------
    ImportError
        No OpenSees backend is installed.
    MomentCurvatureError
        The document is not fiber-lane, a material has no usable
        uniaxial spec, or the axial pre-load did not converge.

    Notes
    -----
    Wipes the process-global OpenSees domain — see the module
    docstring.
    """
    if doc.kind != "fiber":
        raise MomentCurvatureError(
            f"moment_curvature is a fiber-lane operation; this document "
            f"is kind={doc.kind!r}. Lower a continuum document on a "
            f"bridge instead: doc.to_section(ops)."
        )
    if axis not in _DOF_OF_AXIS:
        raise MomentCurvatureError(
            f"axis must be 'z' or 'y', got {axis!r}."
        )
    if kappa_max == 0.0:
        raise MomentCurvatureError("kappa_max must be non-zero.")
    if n_steps < 1:
        raise MomentCurvatureError(
            f"n_steps must be >= 1, got {n_steps}."
        )

    ops = _ops_module()
    dof = _DOF_OF_AXIS[axis]

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0.0, 0.0, 0.0)
    ops.node(2, 0.0, 0.0, 0.0)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    # a fiber section carries no shear stiffness (and no torsional
    # stiffness without -GJ): free axial + both rotations, restrain the
    # rest — the same restraint set gate G-D used.
    ops.fix(2, 0, 1, 1, 1, 0, 0)

    _emit_section(ops, doc)
    ops.element("zeroLengthSection", 1, 1, 2, 1)

    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")

    if axial != 0.0:
        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, float(axial), 0.0, 0.0, 0.0, 0.0, 0.0)
        ops.integrator("LoadControl", 1.0)
        ops.test("NormDispIncr", tol, max_iter)
        ops.algorithm("Newton")
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            raise MomentCurvatureError(
                f"the axial pre-load ({axial}) did not converge; the "
                f"section cannot carry it."
            )
        ops.loadConst("-time", 0.0)

    # unit reference moment → the load factor IS the moment
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    reference = [0.0] * 6
    reference[dof - 1] = 1.0
    ops.load(2, *reference)

    dkappa = float(kappa_max) / n_steps
    ops.integrator("DisplacementControl", 2, dof, dkappa)
    ops.test("NormDispIncr", tol, max_iter)
    ops.algorithm("Newton")
    ops.analysis("Static")

    curvature = [0.0]
    moment = [0.0]
    complete = True
    for _step in range(n_steps):
        if ops.analyze(1) != 0:
            complete = False
            break
        curvature.append(float(ops.nodeDisp(2, dof)))
        moment.append(float(ops.getLoadFactor(2)))

    return MomentCurvature(
        axis=axis,
        curvature=tuple(curvature),
        moment=tuple(moment),
        axial=float(axial),
        complete=complete,
    )
