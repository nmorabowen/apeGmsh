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


def sweep_asdconcrete_element_size(
    spec: "Element",
    elements: "PGElementFanout | list[tuple[int, tuple[int, ...]]]",
    fem: "FEMData",
) -> None:
    """Warn (once, aggregated) when ASDConcrete elements exceed ``l_max`` (ADR 0044).

    For elements whose ``spec.material`` is an ``ASDConcrete3D``/``ASDConcrete1D``
    *directly* (solids and 2-node members), compares the element's
    characteristic length — the minimum inter-node distance, matching
    OpenSees' ``Element::getCharacteristicLength`` — against the material's
    crack-band snapback ceiling ``l_max = 2 E Gf / ft^2``. Elements over the
    ceiling have their softening fracture energy floored by the binary, so the
    response is over-brittle and no longer mesh-objective.

    Minimal cut: section-nested ASDConcrete (fiber beams / shells, where the
    material lives inside a section rather than ``spec.material``) is skipped —
    its per-class characteristic-length mapping is a documented follow-up. The
    pass is read-only and never blocks emit; the warning is CI-promotable via
    ``-W error::...ASDRegularizationWarning``.
    """
    mat = getattr(spec, "material", None)
    if mat is None:
        return
    # Deferred import: avoids an _internal -> material import cycle at load.
    from ....material.nd import ASDConcrete3D, ASDRegularizationWarning
    from ....material.uniaxial import ASDConcrete1D

    if not isinstance(mat, (ASDConcrete3D, ASDConcrete1D)):
        return
    lmax = mat.l_max()
    if lmax is None:  # raw-constructed without ft/Gf provenance — no ceiling
        return

    worst = 0.0
    over = 0
    for _eid, node_tags in elements:
        coords = [_node_coord(fem, int(t)) for t in node_tags]
        n = len(coords)
        if n < 2:
            continue
        lch = min(
            float(np.linalg.norm(coords[i] - coords[j]))
            for i in range(n)
            for j in range(i + 1, n)
        )
        if lch > lmax:
            over += 1
            worst = max(worst, lch)
    if over:
        pg = getattr(spec, "pg", None)
        where = f" in PG {pg!r}" if pg is not None else ""
        warnings.warn(
            f"ASDConcrete: {over}/{len(elements)} elements{where} exceed the "
            f"crack-band snapback ceiling l_max=2*E*Gf/ft^2={lmax:g} "
            f"(worst lch={worst:g}, ratio {worst / lmax:.2f}). Their softening "
            f"fracture energy is floored, so the response is over-brittle and "
            f"not mesh-objective; refine the mesh or increase Gf.",
            ASDRegularizationWarning,
            stacklevel=2,
        )
