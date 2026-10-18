"""
``element LadrunoUP`` — unified Biot u-p saturated-porous continuum.

Split out of :mod:`apeGmsh.opensees.element.solid` (program slice S4) as a
pure move; the public path ``apeGmsh.opensees.element.LadrunoUP`` is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._element_capabilities import (
    LADRUNO_UP_SHAPES_BY_NDM,
    LADRUNO_UP_TH_NODE_COUNTS,
)
from .._internal.tag_resolution import (
    current_element_nodes,
    resolve_tag,
)
from .._internal.types import Element, NDMaterial, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = ["LadrunoUP"]


# LadrunoUP ``-formulation`` selector: std (full integration) or bbar
# (mean-dilatation).  ssp/eas/uri are NOT u-p axes (fork ADR 71 §3.4).
_UP_FORMULATIONS: tuple[str, ...] = ("std", "bbar")

# LadrunoUP shape family (OPS_LadrunoUP.cpp:149-157): (2,3) T3 · (2,4) Q4 ·
# (2,6) Bézier T6 · (3,8) H8 · (3,10) Bézier Tet10.  The shape tables are
# owned by _element_capabilities (the single source of truth); these are the
# node-count views ``_emit`` needs to pick a dim and append ``-pOrder
# linear``.  The authoritative etype-vs-ndm legality check runs at build time
# (validate_ladruno_up_specs), where the mesh's per-cell etype is available —
# ``_emit``'s node-count dispatch is a last-line fallback that cannot see the
# etype (it holds only node tags).
_UP_NODE_COUNTS_2D: frozenset[int] = LADRUNO_UP_SHAPES_BY_NDM[2]
_UP_NODE_COUNTS_3D: frozenset[int] = LADRUNO_UP_SHAPES_BY_NDM[3]
_UP_TH_NODE_COUNTS: frozenset[int] = LADRUNO_UP_TH_NODE_COUNTS


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoUP(Element):
    """``element LadrunoUP`` — unified Biot u-p saturated-porous continuum.

    The Ladruno-fork coupled solid-displacement + pore-pressure element
    (class tag ``33017``, ADR 0074 / fork ADR 71): one class spanning five
    shapes selected by the fanned-out mesh cell — tri3 → T3, quad4 → Q4,
    hexa8 → H8 (equal-order linear p, stabilized), tri6 → Bézier T6 and
    tet10 → Bézier Tet10 (Taylor–Hood: quadratic Bernstein u, vertex-linear
    p).  Gmsh connectivity is used verbatim for every shape.  **Fork-only:**
    emitting the command works on any build, but ``ops.run()`` needs the
    fork (gated at run, not emit).

    Tcl signature::

        element LadrunoUP $tag $n1 … $nk $matTag \\
            <-thick $t> -Kf $Kf -poro $n -rhoF $rhof \\
            -perm $k1 $k2 <$k3> | -permH $k1 $k2 <$k3> -gammaW $gw \\
            <-alpha $biot> <-Ks $Ks> <-body …> <-fluidBody …> \\
            <-formulation std|bbar> <-pOrder equal|linear> <-lumped> \\
            <-stab auto <$a0> | off | $alpha> <-dynSeepage on|off> \\
            <-geom linear>

    **The honest-p contract.**  Nodal DOF ``ndm+1`` IS the pore pressure
    (a displacement-channel DOF — not upstream's ∫p·dt velocity trick):

    * drained boundary = fix that slot, e.g. ``ops.fix(pg="Top",
      dofs=(0, 0, 1))`` on a 2D equal-order region.  Fix masks are
      validated against each node's RESOLVED ndf (a longer mask fails
      loud; a shorter one fixes leading DOFs) — on a Taylor–Hood mesh a
      boundary pg also holds 2-dof mid-edge nodes, so target the p-mask
      at a vertex-only node set (mid-edge nodes carry no p to drain);
    * prescribed head = an ``sp`` on slot ``ndm+1`` inside a pattern.  For
      *staged* heads use ``constraints('Penalty', 1e12, 1e12)`` — a nonzero
      pressure ``sp`` added mid-analysis under ``Transformation`` converges
      cleanly to a WRONG steady state (fork quirks row) — and pair Penalty
      with ``test('NormDispIncr', ...)`` (``NormUnbalance`` stalls at the
      penalty scale);
    * read p back with ``recorder Node -dof <ndm+1> disp`` /
      ``nodeDisp(node, ndm+1)`` — and the fork's ``.ladruno`` recorder
      PRESSURE channel is contract-aware (reads the disp slot for
      LadrunoUP nodes), so ``ops.recorder`` pressure requests work
      unchanged.

    **ndf inference (ADR 0074 D2).**  Equal-order shapes put ``ndf =
    ndm+1`` on every node.  The Taylor–Hood shapes are heterogeneous
    *inside one element*: vertex nodes ``ndm+1``, mid-edge nodes ``ndm`` —
    apeGmsh resolves this per node-slot automatically and emits the
    ``-ndf`` tokens, so the hand-written two-step ``model(ndf=…)`` dance
    disappears.  For saturated-only models declare the envelope
    ``ops.model(ndm=…, ndf=ndm+1)`` (vertex-heavy elision); TH meshes then
    emit ``-ndf ndm`` tokens on mid-edge nodes only.  Mixed dry/saturated
    regions need SEPARATE coincident interface nodes tied on the u-DOFs
    with explicit-DOF-list ``equal_dof`` (ADR 0069; fork guide §6.3).

    **The solver requirement (ADR 0074 D4 — enforced at build).**  The
    honest-p tangent is UNSYMMETRIC; symmetric-storage solvers silently
    drop one coupling block and return plausible garbage (measured p ~1e88
    with rc=0 on ProfileSPD — the no-``system``-command default).  apeGmsh
    refuses to build a LadrunoUP deck whose ``system`` is missing or
    symmetric: declare ``ops.system.UmfPack()`` (serial first choice; also
    Pardiso on a fork+MKL build, SparseGeneral/FullGeneral/BandGeneral;
    Mumps for MPI).

    Parameters
    ----------
    pg
        Physical-group label whose cells are realized as LadrunoUP
        elements at fan-out time.  2D pgs must carry tri3/quad4/tri6
        cells, 3D pgs hexa8/tet10 — validated at build.
    material
        Any 2D/3D :class:`NDMaterial`; the element copies it per GP as
        PlaneStrain (2D) / ThreeDimensional (3D).  The material sees only
        the **effective stress** σ′ (p never enters the constitutive
        update); its ``rho`` is the saturated mixture density.
    Kf
        Fluid bulk modulus, entered **raw** — the element forms the
        storage ``1/Q̄ = n/K_f + (α−n)/K_s`` itself.  (quadUP's ``bulk``
        is the pre-combined ``Q̄ ≈ K_f/n``; do not pass that here.)
    poro
        Porosity ``n``, ``0 < n <= 1`` (and ``n <= alpha``).
    rhoF
        Fluid density — used only in the seepage body-force term.
    perm
        Per-axis permeability ``k̄ = k_hydraulic / γ_w`` (units L³·T/M),
        2 or 3 components matching the pg dimension.  Give EITHER this OR
        the ``permH``/``gammaW`` pair.
    thick
        Out-of-plane thickness — 2D only (plane strain implied), fork
        default 1.0 when omitted.
    permH, gammaW
        Sugar pair (both or neither, excludes ``perm``): hydraulic
        conductivity per axis (e.g. m/s) and unit weight of water; the
        parser divides internally.  Kills the classic quadUP unit error.
    alpha
        Biot–Willis α, ``0 < α <= 1``; fork default 1.0 when omitted.
    Ks
        Grain bulk modulus; fork default ``<= 0`` ⇒ incompressible grains.
    body, fluidBody
        Per-axis **accelerations** (NOT force densities — unlike the other
        continuum elements' ``body_force``), always-on; a ``SelfWeight``
        load pattern replaces them with loadFactor-scaled values (the
        staging knob of the gravity init recipe).  ``fluidBody`` drives
        the seepage source and defaults to ``body`` in the fork when
        omitted.
    formulation
        ``"std"`` (default) or ``"bbar"`` (mean-dilatation).
    lumped
        Emit ``-lumped`` for a row-sum diagonal mass matrix.
    stab
        Equal-order pressure-Laplacian stabilization: ``"auto"``,
        ``("auto", alpha0)``, ``"off"``, or a manual float α.  ``None``
        (default) emits nothing — the fork defaults equal-order pairs to
        auto (α₀=0.25, with a one-line notice).  Wave-propagation runs
        should opt out explicitly with ``stab="off"`` (the auto Laplacian
        injects ~10% spurious deep-station ringing on fast-wave columns).
        **Fatal on Taylor–Hood pgs** (TH is inf-sup stable; the fork
        parser refuses ``-stab`` there) — validated at build.
    dynSeepage
        ``"on"``/``"off"`` (or bool).  ``None`` (default) emits nothing —
        the fork default is OFF (the −ü seepage drive amplifies trial-
        acceleration noise; keep it off for consolidation AND dynamics
        unless you are doing SWANDYNE-parity research).
    geom
        Only ``"linear"`` is accepted (the axis is reserved in the fork);
        ``None``/``"linear"`` emit nothing.

    Notes
    -----
    The user never types ``-pOrder``: tri6/tet10 fan-outs append
    ``-pOrder linear`` automatically (the only legal value per shape at
    v1); equal-order shapes omit it.  Every ``None`` kwarg is simply not
    emitted — the fork parser's defaults stay the single source of truth
    (pass-through, no re-defaulting).  Transient integration wants
    Newmark γ ≥ 0.6 (γ=0.5 has an undamped parasitic root on the p-row);
    the ZS84 production set is γ=0.6, β=0.3025.
    """

    pg: str
    material: NDMaterial
    Kf: float
    poro: float
    rhoF: float
    perm: tuple[float, ...] | None = None
    thick: float | None = None
    permH: tuple[float, ...] | None = None
    gammaW: float | None = None
    alpha: float | None = None
    Ks: float | None = None
    body: tuple[float, ...] | None = None
    fluidBody: tuple[float, ...] | None = None
    formulation: str = "std"
    lumped: bool = False
    stab: str | float | tuple[str, float] | None = None
    dynSeepage: str | bool | None = None
    geom: str | None = None

    def __post_init__(self) -> None:
        if self.Kf <= 0:
            raise ValueError(
                f"LadrunoUP: Kf must be > 0 (raw fluid bulk modulus — NOT "
                f"quadUP's pre-combined bulk), got {self.Kf!r}."
            )
        if not (0.0 < self.poro <= 1.0):
            raise ValueError(
                f"LadrunoUP: poro must satisfy 0 < n <= 1, got {self.poro!r}."
            )
        if self.rhoF < 0:
            raise ValueError(
                f"LadrunoUP: rhoF must be >= 0, got {self.rhoF!r}."
            )
        # -perm XOR (-permH + -gammaW): the parser refuses both and refuses
        # an unpaired permH/gammaW (OPS_LadrunoUP.cpp:254,485).
        if (self.permH is None) != (self.gammaW is None):
            raise ValueError(
                "LadrunoUP: permH and gammaW come together (conductivity / "
                "gammaW = the perm value) — give both or neither."
            )
        if self.perm is not None and self.permH is not None:
            raise ValueError(
                "LadrunoUP: give perm OR the permH/gammaW pair, not both."
            )
        if self.perm is None and self.permH is None:
            raise ValueError(
                "LadrunoUP: permeability is REQUIRED — give perm=(k1, k2"
                "[, k3]) (k_hydraulic/gammaW per axis) or permH=(...) with "
                "gammaW=."
            )
        if self.gammaW is not None and self.gammaW <= 0:
            raise ValueError(
                f"LadrunoUP: gammaW must be > 0, got {self.gammaW!r}."
            )
        kvec = self.perm if self.perm is not None else self.permH
        assert kvec is not None
        if len(kvec) not in (2, 3):
            raise ValueError(
                f"LadrunoUP: permeability needs 2 (2D) or 3 (3D) per-axis "
                f"components, got {len(kvec)}."
            )
        if any(k < 0 for k in kvec):
            raise ValueError(
                f"LadrunoUP: permeability components must be >= 0, got "
                f"{tuple(kvec)!r}."
            )
        if self.alpha is not None and not (0.0 < self.alpha <= 1.0):
            raise ValueError(
                f"LadrunoUP: alpha must satisfy 0 < alpha <= 1, got "
                f"{self.alpha!r}."
            )
        # Storage-domain police (kernel 1/Q̄ contract): 0 < n <= alpha <= 1.
        eff_alpha = 1.0 if self.alpha is None else self.alpha
        if self.poro > eff_alpha:
            raise ValueError(
                f"LadrunoUP: porosity n={self.poro!r} exceeds Biot "
                f"alpha={eff_alpha!r} (storage 1/Qbar needs 0 < n <= "
                f"alpha <= 1)."
            )
        if self.thick is not None:
            if self.thick <= 0:
                raise ValueError(
                    f"LadrunoUP: thick must be > 0, got {self.thick!r}."
                )
            if len(kvec) == 3:
                raise ValueError(
                    "LadrunoUP: thick is 2D only (plane strain implied) — "
                    "a 3-component permeability declares a 3D element."
                )
        for name, vec in (("body", self.body), ("fluidBody", self.fluidBody)):
            if vec is not None and len(vec) != len(kvec):
                raise ValueError(
                    f"LadrunoUP: {name} needs {len(kvec)} components to "
                    f"match the permeability dimension, got {len(vec)}."
                )
        if self.formulation not in _UP_FORMULATIONS:
            raise ValueError(
                f"LadrunoUP: formulation must be one of {_UP_FORMULATIONS} "
                f"(ssp/eas/uri are not u-p axes), got {self.formulation!r}."
            )
        self._validate_stab()
        if self.dynSeepage not in (None, True, False, "on", "off"):
            raise ValueError(
                f"LadrunoUP: dynSeepage must be 'on'/'off' (or bool), got "
                f"{self.dynSeepage!r}."
            )
        if self.geom is not None and self.geom != "linear":
            raise ValueError(
                f"LadrunoUP: geom accepts only 'linear' (the axis is "
                f"reserved in the fork; finite-strain u-p is a research "
                f"phase), got {self.geom!r}."
            )

    def _validate_stab(self) -> None:
        """Mirror the parser's ``-stab`` grammar (OPS_LadrunoUP.cpp:381-440).

        ``"auto"`` | ``("auto", alpha0 > 0)`` | ``"off"`` | manual float
        ``alpha >= 0``.  The TH-fatality check needs the mesh and lives in
        the ADR-0074 build pass (and defensively in ``_emit``).
        """
        s = self.stab
        if s is None or s in ("auto", "off"):
            return
        if isinstance(s, tuple):
            if (
                len(s) == 2 and s[0] == "auto"
                and isinstance(s[1], (int, float)) and not isinstance(s[1], bool)
            ):
                if s[1] <= 0:
                    raise ValueError(
                        f"LadrunoUP: stab=('auto', alpha0) needs alpha0 > 0 "
                        f"(a non-positive alpha0 DEstabilizes), got {s[1]!r}."
                    )
                return
            raise ValueError(
                f"LadrunoUP: tuple stab must be ('auto', alpha0), got {s!r}."
            )
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            raise ValueError(
                f"LadrunoUP: stab must be 'auto', ('auto', alpha0), 'off', "
                f"or a manual numeric alpha, got {s!r}."
            )
        if s < 0:
            raise ValueError(
                f"LadrunoUP: manual stab alpha must be >= 0 (a negative "
                f"alpha anti-stabilizes the pressure block), got {s!r}."
            )

    @property
    def perm_dim(self) -> int:
        """2 or 3 — the spatial dimension the permeability declares."""
        kvec = self.perm if self.perm is not None else self.permH
        assert kvec is not None
        return len(kvec)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        k = len(nodes)
        if k in _UP_NODE_COUNTS_2D:
            dim = 2
        elif k in _UP_NODE_COUNTS_3D:
            dim = 3
        else:
            raise ValueError(
                f"LadrunoUP: no shape provider for {k} nodes (want 3/4/6 "
                f"in 2D or 8/10 in 3D)."
            )
        if dim != self.perm_dim:
            raise ValueError(
                f"LadrunoUP: a {k}-node ({dim}D) cell fanned out under a "
                f"{self.perm_dim}-component permeability — perm/permH (and "
                f"body/fluidBody) must carry {dim} components for this pg."
            )
        is_th = k in _UP_TH_NODE_COUNTS
        if is_th and self.stab is not None:
            raise ValueError(
                "LadrunoUP: -stab is parser-fatal on the Taylor–Hood "
                "shapes (tri6/tet10 are inf-sup stable without it); drop "
                "stab= for this pg."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, mat_tag]
        # Flag-prefixed, order-free in the fork's while-loop parser — emitted
        # in the guide's grammar order for readable decks.  None kwargs are
        # NOT emitted (pass-through: the parser's defaults are the single
        # source of truth); the std/linear defaults are elided.
        if self.thick is not None:
            args += ["-thick", self.thick]
        args += ["-Kf", self.Kf, "-poro", self.poro, "-rhoF", self.rhoF]
        if self.perm is not None:
            args += ["-perm", *self.perm]
        else:
            assert self.permH is not None and self.gammaW is not None
            args += ["-permH", *self.permH, "-gammaW", self.gammaW]
        if self.alpha is not None:
            args += ["-alpha", self.alpha]
        if self.Ks is not None:
            args += ["-Ks", self.Ks]
        if self.body is not None:
            args += ["-body", *self.body]
        if self.fluidBody is not None:
            args += ["-fluidBody", *self.fluidBody]
        if self.formulation != "std":
            args += ["-formulation", self.formulation]
        # The emitter owns -pOrder: exactly one legal value per shape at v1
        # (TH shapes REQUIRE linear; equal-order elides the synonym).
        if is_th:
            args += ["-pOrder", "linear"]
        if self.lumped:
            args.append("-lumped")
        if self.stab is not None:
            if isinstance(self.stab, tuple):
                args += ["-stab", "auto", self.stab[1]]
            elif self.stab in ("auto", "off"):
                args += ["-stab", self.stab]
            else:
                args += ["-stab", float(self.stab)]
        if self.dynSeepage is not None:
            on = self.dynSeepage in (True, "on")
            args += ["-dynSeepage", "on" if on else "off"]
        # geom: 'linear' is both the only legal value and the parser
        # default — elided like formulation='std'.
        emitter.element("LadrunoUP", tag, *args)

