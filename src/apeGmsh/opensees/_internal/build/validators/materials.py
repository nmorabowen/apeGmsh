"""materials: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

import warnings
from typing import Iterable, Sequence

import numpy as np

from ...types import Element
from ..errors import BridgeError
from ..pg_expand import PGElementFanout
from ..transforms import _node_coord
from .solver import (
    _GENERAL_SOLVER_MSG,
    _UNSYMMETRIC_SAFE_SYSTEMS,
)


if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


class ManzariTangentSolverWarning(UserWarning):
    """A Manzari-family consistent tangent met a symmetric-storage solver.

    ``TanType != 0`` selects the continuum elasto-plastic tangent, and the
    tangent of a **non-associated** model is genuinely unsymmetric.  A
    half-storage solver reads only the ``col >= row`` half of each element
    matrix — no averaging, no detection — so the run solves a *different*
    system and converges to a plausible-looking wrong answer.

    Fail-soft, unlike the u-p gate: the deck is still runnable, and there
    are legitimate reasons to take the symmetrized tangent knowingly (a
    calibration deck with no global solve, an associated-flow parameter
    set).  It is the *answer* that is not trustworthy, so this is a
    warning the author must weigh, not a build-stopper.
    """


def validate_manzari_tangent_solver(
    primitives: "Iterable[object]",
    *,
    enforce: bool,
    staged: bool,
    partitioned: bool,
    flat_systems: "Sequence[object]",
    stage_systems: "Sequence[tuple[str, object | None]]",
) -> None:
    """Warn when a consistent Manzari tangent will be solved symmetrically.

    ``LadrunoSANISAND`` defaults to ``tan_type=2`` (the fork measured 800
    vs 283 Newton iterations against the elastic tangent), and the fork
    parser's own default moved ``0 -> 2`` in PR #792, so a deck can also
    reach the consistent tangent through a tail apeGmsh did not write.
    That tangent is unsymmetric; nothing else in the bridge checks the
    solver against it.

    Scope mirrors :func:`validate_ladruno_up_solver` exactly — a DECLARED
    symmetric system is wrong whether or not this emit solves, while the
    MISSING-system branch (OpenSees' no-``system`` default is ProfileSPD)
    is gated on ``enforce`` and skipped for a partitioned deck, which
    rides the ADR-0027 INV-5 general auto-emit.
    """
    # Lazily imported, like every other material import in this module.
    # The isinstance tuple is spelled out rather than built by a helper so
    # the narrowing survives to ``m.tan_type`` -- and it is an explicit
    # tuple rather than a ``hasattr(m, "tan_type")`` duck test, because the
    # field name is not reserved and a future unrelated material carrying
    # one must not silently inherit this gate.
    from ....material.nd import LadrunoSANISAND, ManzariDafalias, SAniSandMS

    offenders = sorted({
        f"{type(m).__name__}(tan_type={m.tan_type})"
        for m in primitives
        if isinstance(m, (ManzariDafalias, SAniSandMS, LadrunoSANISAND))
        and m.tan_type != 0
    })
    if not offenders:
        return
    who = ", ".join(offenders)

    def _why(token: str, where: str, detail: str) -> str:
        return (
            f"{who} with system {token} ({where}): {detail} The consistent "
            f"tangent (TanType != 0) of a non-associated model is "
            f"UNSYMMETRIC, so the solve would use only half of it and "
            f"converge to a plausible but WRONG answer. Declare a general "
            f"solver: {_GENERAL_SOLVER_MSG} — or set tan_type=0 (the "
            f"elastic tangent, symmetric but ~2.8x the Newton iterations)."
        )

    def _check(system: object, where: str) -> None:
        token = type(system).__name__
        if token not in _UNSYMMETRIC_SAFE_SYSTEMS:
            warnings.warn(
                _why(
                    repr(token), where,
                    f"{token} does not store the full matrix.",
                ),
                ManzariTangentSolverWarning,
                stacklevel=2,
            )
            return
        # Pardiso / Mumps are safe only in their unsymmetric mode — the
        # half-storage modes are exactly the silent-drop this gate exists
        # to surface (fork ADR-75 P1d).
        mtype = getattr(system, "matrix_type", "unsymmetric")
        if mtype != "unsymmetric":
            warnings.warn(
                _why(
                    f"{token}(matrix_type={mtype!r})", where,
                    "this mode stores only the upper triangle.",
                ),
                ManzariTangentSolverWarning,
                stacklevel=2,
            )

    if staged:
        for name, system in stage_systems:
            if system is not None:
                _check(system, f"stage {name}")
        if enforce:
            for name, system in stage_systems:
                if system is None:
                    warnings.warn(
                        _why(
                            "ProfileSPD", f"stage {name}, undeclared",
                            "the stage analyzes with no linear system, so it "
                            "runs on the OpenSees ProfileSPD default after "
                            "the prior stage's wipeAnalysis.",
                        ),
                        ManzariTangentSolverWarning,
                        stacklevel=2,
                    )
        return

    if flat_systems:
        _check(flat_systems[-1], "global")
        return
    if not enforce or partitioned:
        # Never solves (archival / eigen-only / model-only), or rides the
        # ADR-0027 INV-5 general auto-emit.
        return
    warnings.warn(
        _why(
            "ProfileSPD", "undeclared",
            "no linear system is declared, so the deck runs on the "
            "OpenSees no-`system` default, ProfileSPD.",
        ),
        ManzariTangentSolverWarning,
        stacklevel=2,
    )


class ManzariConvergenceTestWarning(UserWarning):
    """A SANISAND deck asks for a convergence test it cannot reach.

    ``NormDispIncr`` is unreachable on this material: the
    displacement-increment residual stalls and never meets a tight
    tolerance.  Measured in apeGmsh's own live suite, a single-hex
    triaxial driver floored at **4.2587e-08** against the ``1e-8`` it
    asked for, and the deviatoric leg never converged.  It is not
    mesh-neutral either, so the same number means different things at
    different ``h``.

    Fail-soft: a looser tolerance may still be reachable, and only the
    model author knows what their deck needs.
    """


#: Convergence tests whose residual IS the displacement increment, and so
#: inherit the stall.  ``EnergyIncr`` mixes the increment with the
#: unbalance and was measured to converge on the same deck, so it is not
#: here; the force-residual tests are the recommendation, not the problem.
_DISP_INCREMENT_TESTS: frozenset[str] = frozenset(
    {"NormDispIncr", "RelativeNormDispIncr"}
)


def validate_manzari_convergence_test(
    primitives: "Iterable[object]",
    *,
    staged: bool,
    flat_tests: "Sequence[object]",
    stage_tests: "Sequence[tuple[str, object | None]]",
) -> None:
    """Warn when a SANISAND deck drives on the displacement increment.

    Unlike :func:`validate_manzari_tangent_solver` this does not look at
    ``tan_type``: the stall is a property of the material's integrator,
    not of the tangent, and it was measured on a ``ManzariDafalias`` leg
    running the elastic one.  A missing test is not this gate's business
    (the analysis-chain validation owns that), so only a DECLARED test is
    checked, staged decks per stage.
    """
    from ....material.nd import LadrunoSANISAND, ManzariDafalias, SAniSandMS

    who = sorted({
        type(m).__name__
        for m in primitives
        if isinstance(m, (ManzariDafalias, SAniSandMS, LadrunoSANISAND))
    })
    if not who:
        return
    names = ", ".join(who)

    def _check(test: object, where: str) -> None:
        token = type(test).__name__
        if token not in _DISP_INCREMENT_TESTS:
            return
        warnings.warn(
            f"{names} with test {token} ({where}): the "
            f"displacement-increment residual is UNREACHABLE on this "
            f"material — it stalls (measured: a 4.2587e-08 floor against a "
            f"1e-8 tolerance, deviatoric leg never converged) and it is not "
            f"mesh-neutral, so the same tolerance means different things at "
            f"different mesh sizes. Drive on the force residual instead: "
            f"ops.test.NormUnbalance(tol=<fraction of the model's own "
            f"load or weight>, max_iter=...).",
            ManzariConvergenceTestWarning,
            stacklevel=2,
        )

    if staged:
        for name, test in stage_tests:
            if test is not None:
                _check(test, f"stage {name}")
        return
    if flat_tests:
        _check(flat_tests[-1], "global")


def _material_graph(prim: object) -> "list[object]":
    """``prim`` and every material it wraps, transitively.

    ``PlaneStrain(base=...)`` / ``LogStrain(inner=...)`` and friends put
    the real constitutive model one or more levels down, so a capped
    SANISAND can reach an element without being its ``material``.
    """
    seen: list[object] = []
    stack = [prim]
    while stack:
        cur = stack.pop()
        if cur is None or any(cur is s for s in seen):
            continue
        seen.append(cur)
        deps = getattr(cur, "dependencies", None)
        if callable(deps):
            try:
                stack.extend(deps())
            except Exception:                  # pragma: no cover - defensive
                pass
    return seen


def validate_sanisand_substep_cap(elements: "Iterable[Element]") -> None:
    """Refuse a ``max_substeps`` cap under an element MEASURED to swallow it.

    The cap makes the material REFUSE, at commit, an increment its
    ``-maxSubsteps`` companion cannot integrate — a COMMIT-time refusal.
    Since fork PR #838, ``Domain::commit()`` aborts any commit-time refusal
    element-independently (the material declares it out of band via
    ``ladrunoNoteCommitRefusal()``; a discarding element can no longer
    swallow it — see :func:`element_propagates_material_refusal`), so this
    build-time gate is now a belt on top of that runtime abort rather than
    the only thing standing between a capped material and a silently wrong
    answer. It still earns its keep: it fails at build time instead of at
    analyze time, and it still matters on any build predating #838. Keyed
    on the measured per-element flag (fork PR #838's refusal-propagation
    audit, ``Ladruno_implementation/LEDGER_quirks.md`` "Element refusal
    roster"), not on a name allow-list: only a host MEASURED to discard the
    return code (``False``) raises; an unmeasured host (``None``) stays
    silent, the same policy :func:`validate_asdplastic_host` uses.
    """
    from ...._element_capabilities import element_propagates_material_refusal
    from ....material.nd import LadrunoSANISAND

    for spec in elements:
        cls = type(spec).__name__
        if element_propagates_material_refusal(cls) is not False:
            continue
        for mat in _material_graph(getattr(spec, "material", None)):
            if not isinstance(mat, LadrunoSANISAND) or not mat.max_substeps:
                continue
            raise BridgeError(
                f"LadrunoSANISAND(max_substeps={mat.max_substeps}) reaches "
                f"{cls!r} (pg {getattr(spec, 'pg', '?')!r}), which is "
                f"MEASURED to discard a material's refusal. The cap makes "
                f"the material REFUSE an increment it cannot integrate; an "
                f"element that discards that return code hands the analysis "
                f"a PARTIALLY integrated stress with a partial tangent and "
                f"it converges on it — worse than the uncapped force-accept, "
                f"which at least integrates the whole increment. Use an "
                f"element MEASURED to propagate a material refusal (e.g. "
                f"LadrunoBrick, LadrunoQuad, or TenNodeTetrahedron as the "
                f"fork builds it — stock's is 6x too soft), or leave "
                f"max_substeps=0 (uncapped)."
            )


class ASDPlasticHostWarning(UserWarning):
    """An ``ASDPlasticMaterial3D`` sits on an element that swallows refusals.

    After fork ADR-94 the material returns ``LADRUNO_MATERIAL_REFUSED``
    from every failure site — but only a host that ACTS on the return
    code turns that into a failed step.  ``stdBrick`` (``Brick::update()``
    returns 0 unconditionally) was measured to report 20/20 successes on
    a deck ``LadrunoBrick`` refuses 0/20 (fork ADR-94 B2, deliberately
    left on the fork).  ``strict_convergence`` — on by default since
    ADR 0105 — is therefore inert on such a host.  This is the TRIAL-time
    path (``setTrialStrain``): it still needs a forwarding element after
    fork PR #838, unlike a COMMIT-time refusal, which ``Domain::commit()``
    now aborts element-independently regardless of host.

    Fail-soft: a vanilla-host deck is legal and was the SSI-1 default;
    it is the fail-loud contract that does not reach it.
    """


def validate_asdplastic_host(elements: "Iterable[Element]") -> None:
    """ADR 0105 D4 — warn once per deck when an ASDP material is swallowed.

    Keyed on :func:`element_propagates_material_refusal`, the measured
    per-element flag, not on element names: only a host MEASURED to
    discard the return code (``False``) warns; an unmeasured one
    (``None``) stays silent.  Materials are found through the wrapper
    graph (``PlaneStrain(base=...)`` and friends), the way
    :func:`validate_sanisand_substep_cap` finds a capped SANISAND.
    """
    from ...._element_capabilities import element_propagates_material_refusal
    from ....material.nd import ASDPlasticMaterial3D

    hosts: dict[str, set[str]] = {}
    for spec in elements:
        cls = type(spec).__name__
        if element_propagates_material_refusal(cls) is not False:
            continue
        if any(
            isinstance(m, ASDPlasticMaterial3D)
            for m in _material_graph(getattr(spec, "material", None))
        ):
            hosts.setdefault(cls, set()).add(str(getattr(spec, "pg", "?")))
    if not hosts:
        return
    where = "; ".join(
        f"{cls} (pg {', '.join(sorted(pgs))})" for cls, pgs in sorted(hosts.items())
    )
    warnings.warn(
        f"ASDPlasticMaterial3D on {where}: TRIAL-time material refusals are "
        f"swallowed by this element (fork ADR-94 B2) — strict_convergence "
        f"and every other fail-loud material contract never reach the "
        f"analysis, so a non-converged or inadmissible state is committed "
        f"as if it had converged. Use LadrunoBrick or TenNodeTetrahedron "
        f"for a fail-loud deck, on a fork build: the refusal contract is "
        f"fork-only, and stock's TenNodeTetrahedron is 6x too soft (the live "
        f"run refuses it there).",
        ASDPlasticHostWarning,
        stacklevel=2,
    )


# Corner signs of the 8-node brick in OpenSees order (bottom face, then top).
_HEX_CORNERS = np.array([
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
], dtype=float)
_GP = 1.0 / np.sqrt(3.0)


def _hex_volume(x: np.ndarray) -> float:
    """Reference volume of a trilinear hex: 2x2x2 Gauss of ``detJ`` (exact).

    The same rule as the fork's ``LadrunoBrick::computeVolume``; signed, so a
    mis-ordered or inverted element reads ``<= 0`` there and here alike.
    """
    vol = 0.0
    for g in _HEX_CORNERS * _GP:
        terms = 1.0 + _HEX_CORNERS * g  # (8, 3): (1 + xi_i xi), ...
        dN = np.empty((8, 3))
        dN[:, 0] = _HEX_CORNERS[:, 0] * terms[:, 1] * terms[:, 2]
        dN[:, 1] = _HEX_CORNERS[:, 1] * terms[:, 0] * terms[:, 2]
        dN[:, 2] = _HEX_CORNERS[:, 2] * terms[:, 0] * terms[:, 1]
        vol += float(np.linalg.det((dN / 8.0).T @ x))
    return vol


def _min_node_distance(x: np.ndarray) -> float:
    """``Element::getCharacteristicLength``: the minimum inter-node distance."""
    n = len(x)
    return min(
        float(np.linalg.norm(x[i] - x[j])) for i in range(n) for j in range(i + 1, n)
    )


def _measure_lch(cls: str, x: np.ndarray) -> float | None:
    """The fork's override of ``getCharacteristicLength`` for ``cls``, or ``None``.

    Measure-based, from the corner nodes (exact for straight-edged quadratic
    elements, the mesher's default): ``cbrt(V)`` for the bricks, ``cbrt(6V)``
    for ``BezierTet10``, ``sqrt(A)`` for ``LadrunoQuad`` and ``sqrt(2A)`` for
    the triangles (fork 23d937bc4). ``None`` means the class keeps the base
    rule. A degenerate measure falls back to the base, as the fork does.
    """
    if cls in ("LadrunoBrick", "LadrunoBrick20") and len(x) >= 8:
        v = _hex_volume(x[:8])
        return float(np.cbrt(v)) if v > 0.0 else _min_node_distance(x)
    if cls == "BezierTet10" and len(x) >= 4:
        v6 = abs(float(np.linalg.det(np.stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]]))))
        return float(np.cbrt(v6)) if v6 > 0.0 else _min_node_distance(x)
    if cls == "LadrunoQuad" and len(x) >= 4:
        a = 0.5 * float(np.linalg.norm(np.cross(x[2] - x[0], x[3] - x[1])))
        return float(np.sqrt(a)) if a > 0.0 else _min_node_distance(x)
    if cls in ("LadrunoCST", "LadrunoLST", "BezierTri6") and len(x) >= 3:
        a = 0.5 * float(np.linalg.norm(np.cross(x[1] - x[0], x[2] - x[0])))
        return float(np.sqrt(2.0 * a)) if a > 0.0 else _min_node_distance(x)
    return None


def _element_lch(cls: str, x: np.ndarray) -> float:
    """The characteristic length the fork hands ``cls``'s materials."""
    measured = _measure_lch(cls, x)
    return measured if measured is not None else _min_node_distance(x)


def _ladruno_concrete_ceiling(mat: object) -> tuple[str, float, float | None] | None:
    """``(name, ceiling, fixed_lch)`` for a Ladruno concrete the sweep checks.

    ``fixed_lch`` is the one length every element uses when the material does
    not regularize per element (``None`` when it does). An ``_LadrunoRC`` under
    ``-autoRegularization`` with an explicit ``-lch`` regularizes with that
    length (``P.lch > 0`` wins, ``LadrunoRCConcrete.cpp:336``). ``None`` overall
    means there is nothing to check: an ``_LadrunoRC`` without
    ``-autoRegularization`` never rescales its tables by element size.
    """
    from ....material.nd import LadrunoConcrete3D, _LadrunoRC

    if isinstance(mat, LadrunoConcrete3D):
        return ("LadrunoConcrete3D", min(mat.l_max()),
                None if mat.auto_regularize else float(mat.lch))
    if isinstance(mat, _LadrunoRC):
        if mat.auto_regularization is None:
            return None
        return (type(mat).__name__, min(mat.l_max()),
                float(mat.lch) if mat.lch > 0.0 else None)
    return None


def _crack_band_ceiling(mat: object) -> tuple[str, float, float | None, str] | None:
    """``(name, l_max, fixed_lch, ceiling text)`` for a crack-band concrete, or ``None``.

    ``None`` when ``mat`` is not one, or has no ceiling to check: an ASD
    concrete without ``auto_regularize`` (the fork never rescales its curves,
    ``ASDConcrete3DMaterial.cpp:1689``, ``ASDConcrete1DMaterial.cpp:1061``), one
    raw-constructed without ``ft``/``Gf`` provenance or with a zero ``ft`` or
    ``Gf``, or an ``_LadrunoRC`` without ``-autoRegularization``.
    """
    from ....material.nd import ASDConcrete3D
    from ....material.uniaxial import ASDConcrete1D

    if isinstance(mat, (ASDConcrete3D, ASDConcrete1D)):
        if not mat.auto_regularize or mat.ft is None or mat.ft <= 0.0:
            return None
        asd_lmax = mat.l_max()
        if asd_lmax is None or asd_lmax <= 0.0:
            return None
        return ("ASDConcrete", asd_lmax, None,
                f"crack-band snapback ceiling l_max=2*E*Gf/ft^2={asd_lmax:g}")
    found = _ladruno_concrete_ceiling(mat)
    if found is None:
        return None
    name, lmax, fixed_lch = found
    return (name, lmax, fixed_lch,
            f"crack-band snap-back ceiling min(l_max_t, l_max_c)={lmax:g}")


# Shells with no update() at the fork pin: their sections are first driven from
# the assembly loops, which do not set the active element, so an auto-regularized
# material latches the length of whichever element was left active.
_LATCHING_SHELLS = frozenset({"ShellMITC4", "ShellDKGQ", "ShellNLDKGQ"})


def _latches_host_length(mat: object) -> bool:
    """Whether ``mat`` reads its length from the active element at the fork pin."""
    from ....material.nd import ASDConcrete3D, LadrunoConcrete3D, _LadrunoRC
    from ....material.uniaxial import ASDConcrete1D

    if isinstance(mat, (ASDConcrete3D, ASDConcrete1D, LadrunoConcrete3D)):
        return mat.auto_regularize
    if isinstance(mat, _LadrunoRC):
        return mat.auto_regularization is not None and not mat.lch > 0.0
    return False


def _section_host_lch(spec: object, x: np.ndarray) -> float | None:
    """The length a section-nested material reads from its host element, or ``None``.

    The materials latch ``ops_TheActiveElement->getCharacteristicLength()``
    (``ASDConcrete3DMaterial.cpp:1687``, ``ASDConcrete1DMaterial.cpp:1059``,
    ``LadrunoConcrete3D.cpp:382``, ``LadrunoRCConcrete.cpp:337``), and the
    domain's update loop sets the active element to the shell before its
    sections are driven (``Domain.cpp:2766``/``2785``), so the length is the
    shell's own on a shell whose ``update()`` drives them. Fork 23d937bc4.
    ``None`` means the host is not mirrored and the sweep skips it.
    """
    cls = type(spec).__name__
    if cls == "ASDShellQ4":
        # ASDShellQ4.cpp:1876: the base length, halved while the EAS membrane
        # is on (it localizes in one row of Gauss points); -noeas drops it.
        lch = _min_node_distance(x)
        return lch if getattr(spec, "no_eas", False) else lch / 2.0
    if cls == "ASDShellT3":
        # ASDShellT3.cpp:893: update() drives the sections; no length override.
        return _min_node_distance(x)
    return None


def _sweep_section_concrete(
    spec: "Element",
    elements: "PGElementFanout | list[tuple[int, tuple[int, ...]]]",
    fem: "FEMData",
) -> None:
    """The ADR 0044 sweep for crack-band concrete inside ``spec.section``.

    Every concrete reached through the section's wrapper graph
    (``LayeredShell`` layers, ``PlateFiber``, ``LadrunoShellModifier``, ...)
    is checked against the host's length (:func:`_section_host_lch`), or
    against its fixed ``lch`` when it does not regularize per element. One
    warning per PG lists each concrete over its ceiling. On a
    :data:`_LATCHING_SHELLS` host the length is unknown, so an auto-regularized
    concrete gets its own warning instead of a ceiling check.
    """
    section = getattr(spec, "section", None)
    if section is None:
        return
    from ....material.nd import ASDRegularizationWarning

    graph = _material_graph(section)
    found = [c for c in map(_crack_band_ceiling, graph) if c is not None]
    cls = type(spec).__name__
    pg = getattr(spec, "pg", None)
    where = f" in PG {pg!r}" if pg is not None else ""
    if cls in _LATCHING_SHELLS:
        found = [c for c in found if c[2] is not None]
        latching = sorted({type(m).__name__ for m in graph if _latches_host_length(m)})
        if latching and len(elements) > 0:
            warnings.warn(
                f"Auto-regularized concrete ({', '.join(latching)}) in the section "
                f"of {cls}{where}: the fork latches the characteristic length from "
                f"another element for this host (apeGmsh #2680), so the softening "
                f"energy can change with what else is in the model. The length is "
                f"unknown, so the snap-back ceiling is not checked. ASDShellQ4 and "
                f"ASDShellT3 read their own length.",
                ASDRegularizationWarning,
                stacklevel=3,
            )
    if not found:
        return
    per_element = [c for c in found if c[2] is None]
    lchs: list[float] = []
    if per_element:
        for _eid, node_tags in elements:
            coords = np.array([_node_coord(fem, int(t)) for t in node_tags])
            if len(coords) < 2:
                continue
            lch = _section_host_lch(spec, coords)
            if lch is None:
                per_element = []  # host not mirrored: nothing to compare with
                break
            lchs.append(lch)

    lines: list[str] = []
    for name, lmax, fixed_lch, ceiling in found:
        if fixed_lch is not None:
            if fixed_lch > lmax:
                lines.append(
                    f"{name}: the fixed lch={fixed_lch:g} every element uses "
                    f"exceeds the {ceiling} (ratio {fixed_lch / lmax:.2f})")
            continue
        if not per_element:
            continue
        over = [h for h in lchs if h > lmax]
        if over:
            worst = max(over)
            lines.append(
                f"{name}: {len(over)}/{len(elements)} elements exceed the "
                f"{ceiling} (worst lch={worst:g}, ratio {worst / lmax:.2f})")
    if not lines:
        return
    eas = (" ASDShellQ4 halves the node distance while EAS is on (no_eas=False)."
           if cls == "ASDShellQ4" else "")
    warnings.warn(
        f"Concrete in the section of {cls}{where}: " + "; ".join(lines)
        + f". The length is the host's getCharacteristicLength.{eas} Over the "
        f"ceiling the softening fracture energy is floored, so the response is "
        f"over-brittle and not mesh-objective; refine the mesh or increase Gf.",
        ASDRegularizationWarning,
        stacklevel=3,
    )


def sweep_asdconcrete_element_size(
    spec: "Element",
    elements: "PGElementFanout | list[tuple[int, tuple[int, ...]]]",
    fem: "FEMData",
) -> None:
    """Warn (once, aggregated) when concrete elements exceed their snap-back ceiling (ADR 0044).

    For elements whose ``spec.material`` is *directly* (solids and 2-node
    members) one of the crack-band concretes, compares the length the binary
    regularizes with against the material's snap-back ceiling. Elements over
    it have their softening energy floored, so the response is over-brittle
    and no longer mesh-objective.

    * ``ASDConcrete3D``/``ASDConcrete1D`` with ``auto_regularize``:
      ``l_max = 2 E Gf / ft^2`` against the minimum inter-node distance
      (``Element::getCharacteristicLength``). Without it the input curves are
      used as given, whatever the mesh.
    * ``LadrunoConcrete3D``: ``min(l_max_t, l_max_c)``
      (:meth:`~apeGmsh.opensees.material.nd.LadrunoConcrete3D.l_max`) against
      each element's own length under ``auto_regularize``, else against the
      fixed ``lch`` every element uses.
    * ``LadrunoRCConcrete``/``LadrunoRCFiniteStrain`` with
      ``auto_regularization``: the ceiling read off the emitted tables, against
      each element's own length, or an explicit ``lch``. Without it nothing
      depends on the mesh.

    A Ladruno concrete's element length follows the fork's per-class override
    where one is mirrored (:func:`_measure_lch`); any other class uses the
    base rule.

    Concrete nested in ``spec.section`` is swept by
    :func:`_sweep_section_concrete` against the host's length
    (:func:`_section_host_lch`): the shells so far. Fiber beams are a
    documented follow-up, and an unmirrored host is skipped. A material with
    a zero ``ft`` or ``Gf`` is skipped. The pass is read-only and never blocks emit; the warning is CI-promotable via
    ``-W error::...ASDRegularizationWarning``.
    """
    mat = getattr(spec, "material", None)
    if mat is None:
        _sweep_section_concrete(spec, elements, fem)
        return
    # Deferred import: avoids an _internal -> material import cycle at load.
    from ....material.nd import ASDRegularizationWarning

    pg = getattr(spec, "pg", None)
    where = f" in PG {pg!r}" if pg is not None else ""
    advice = (
        "Their softening fracture energy is floored, so the response is "
        "over-brittle and not mesh-objective; refine the mesh or increase Gf."
    )
    found = _crack_band_ceiling(mat)
    if found is None:
        return
    name, lmax, fixed_lch, ceiling = found

    if fixed_lch is not None:
        if fixed_lch > lmax:
            warnings.warn(
                f"{name}: the fixed lch={fixed_lch:g} every element{where} uses "
                f"exceeds the {ceiling} (ratio {fixed_lch / lmax:.2f}). {advice}",
                ASDRegularizationWarning,
                stacklevel=2,
            )
        return

    cls = type(spec).__name__
    worst = 0.0
    over = 0
    for _eid, node_tags in elements:
        coords = np.array([_node_coord(fem, int(t)) for t in node_tags])
        if len(coords) < 2:
            continue
        lch = (_min_node_distance(coords) if name == "ASDConcrete"
               else _element_lch(cls, coords))
        if lch > lmax:
            over += 1
            worst = max(worst, lch)
    if over:
        warnings.warn(
            f"{name}: {over}/{len(elements)} elements{where} exceed the "
            f"{ceiling} (worst lch={worst:g}, ratio {worst / lmax:.2f}). {advice}",
            ASDRegularizationWarning,
            stacklevel=2,
        )
