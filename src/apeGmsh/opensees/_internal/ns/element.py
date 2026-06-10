"""
``_ElementNS`` — backs ``ops.element.<Type>(pg=..., ...)``.

Phase 2 populates the namespace with one typed method per OpenSees
element. The bridge fans the spec across its physical group at build
time; the typed class never carries node tags.

Each method is fully kw-only with explicit types — no ``**kwargs``
(P12). Methods are grouped by family in alphabetical-ish order.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ...element.absorbing import ASDAbsorbingBoundary2D, ASDAbsorbingBoundary3D
from ...element.beam_column import (
    ElasticTimoshenkoBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from ...element.shell import (
    ASDShellQ4,
    ASDShellT3,
    ShellDKGQ,
    ShellMITC3,
    ShellMITC4,
)
from ...element.solid import (
    BezierTet10,
    BezierTri6,
    FourNodeQuad,
    FourNodeTetrahedron,
    LadrunoBrick,
    SixNodeTri,
    TenNodeTetrahedron,
    Tri31,
    stdBrick,
)
from ...element.truss import CorotTruss, InertiaTruss, Truss
from ...element.two_node_link import TwoNodeLink
from ...element.zero_length import (
    CoupledZeroLength,
    NodeRef,
    ZeroLength,
    ZeroLengthMatDir,
    ZeroLengthSection,
)
from ...section.plate import (
    ElasticMembranePlateSection,
    LayeredShell,
    LayeredShellFiberSection,
)
from ...material.nd import ElasticIsotropic
from ...transform import Corotational, Linear, PDelta
from ..types import (
    BeamIntegration,
    Damping,
    GeomTransf,
    NDMaterial,
    Section,
    TimeSeries,
    UniaxialMaterial,
)
from ._base import _BridgeNamespace

if TYPE_CHECKING:
    from ...parts.plane_wave_box import AbsorbingSkinResult


__all__ = ["_ElementNS"]


# Beam-column transforms — concrete subclasses of GeomTransf.
_AnyTransf = Linear | PDelta | Corotational

# Annotation-safe aliases: inside ``_ElementNS`` the facade METHODS shadow the
# imported absorbing classes, so class-scope annotations would resolve to the
# methods ("Function is not valid as a type").
_Absorbing2D = ASDAbsorbingBoundary2D
_Absorbing3D = ASDAbsorbingBoundary3D

# Shell sections — only plate-flavored sections accepted at construction
# (catches "wired a Fiber section into a shell" at the type checker).
_ShellSection = (
    ElasticMembranePlateSection
    | LayeredShell
    | LayeredShellFiberSection
)


class _ElementNS(_BridgeNamespace):
    """``ops.element.<Type>(pg=..., ...)`` — Phase 2 element namespace."""

    # -- Beam-column family (Phase 2α) ----------------------------------

    def elasticBeamColumn(
        self,
        *,
        pg: str,
        transf: _AnyTransf | str,
        A: float,
        E: float,
        Iz: float,
        Iy: float | None = None,
        G: float | None = None,
        J: float | None = None,
        mass: float | None = None,
        c_mass: bool = False,
        damp: Damping | None = None,
    ) -> elasticBeamColumn:
        transf = cast(_AnyTransf, self._bridge._resolve(transf, base=GeomTransf))
        return self._bridge._register(
            elasticBeamColumn(
                pg=pg, transf=transf,
                A=A, E=E, Iz=Iz,
                Iy=Iy, G=G, J=J,
                mass=mass, c_mass=c_mass, damp=damp,
            )
        )

    def forceBeamColumn(
        self,
        *,
        pg: str,
        transf: _AnyTransf | str,
        integration: BeamIntegration | str,
        mass: float | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        damp: Damping | None = None,
    ) -> forceBeamColumn:
        """``element forceBeamColumn`` — force-based distributed-plasticity.

        Compose the integration rule first (e.g. ``ops.beamIntegration.Lobatto(
        section=sec, n_ip=5)``) and pass it as ``integration=``.  Both
        ``transf`` and ``integration`` accept object handles or
        registered names.  ``damp`` attaches a ``damping`` object directly to
        this element (ADR 0053 D3b) instead of via a region.
        """
        transf = cast(_AnyTransf, self._bridge._resolve(transf, base=GeomTransf))
        integration = self._bridge._resolve(integration, base=BeamIntegration)
        return self._bridge._register(
            forceBeamColumn(
                pg=pg, transf=transf, integration=integration,
                mass=mass, max_iter=max_iter, tol=tol, damp=damp,
            )
        )

    def dispBeamColumn(
        self,
        *,
        pg: str,
        transf: _AnyTransf | str,
        integration: BeamIntegration | str,
        mass: float | None = None,
        c_mass: bool = False,
        damp: Damping | None = None,
    ) -> dispBeamColumn:
        """``element dispBeamColumn`` — displacement-based distributed-plasticity."""
        transf = cast(_AnyTransf, self._bridge._resolve(transf, base=GeomTransf))
        integration = self._bridge._resolve(integration, base=BeamIntegration)
        return self._bridge._register(
            dispBeamColumn(
                pg=pg, transf=transf, integration=integration,
                mass=mass, c_mass=c_mass, damp=damp,
            )
        )

    def ElasticTimoshenkoBeam(
        self,
        *,
        pg: str,
        transf: _AnyTransf | str,
        E: float,
        G: float,
        A: float,
        Iz: float,
        Avy: float,
        Iy: float | None = None,
        J: float | None = None,
        Avz: float | None = None,
        mass: float | None = None,
        c_mass: bool = False,
    ) -> ElasticTimoshenkoBeam:
        transf = cast(_AnyTransf, self._bridge._resolve(transf, base=GeomTransf))
        return self._bridge._register(
            ElasticTimoshenkoBeam(
                pg=pg, transf=transf,
                E=E, G=G, A=A, Iz=Iz, Avy=Avy,
                Iy=Iy, J=J, Avz=Avz,
                mass=mass, c_mass=c_mass,
            )
        )

    # -- Truss family (Phase 2β) ----------------------------------------

    def Truss(
        self,
        *,
        pg: str,
        A: float,
        material: UniaxialMaterial | str,
        rho: float | None = None,
        c_mass: bool = False,
        do_rayleigh: bool = False,
    ) -> Truss:
        material = self._bridge._resolve(material, base=UniaxialMaterial)
        return self._bridge._register(
            Truss(
                pg=pg, A=A, material=material,
                rho=rho, c_mass=c_mass, do_rayleigh=do_rayleigh,
            )
        )

    def CorotTruss(
        self,
        *,
        pg: str,
        A: float,
        material: UniaxialMaterial | str,
        rho: float | None = None,
        c_mass: bool = False,
        do_rayleigh: bool = False,
    ) -> CorotTruss:
        material = self._bridge._resolve(material, base=UniaxialMaterial)
        return self._bridge._register(
            CorotTruss(
                pg=pg, A=A, material=material,
                rho=rho, c_mass=c_mass, do_rayleigh=do_rayleigh,
            )
        )

    def InertiaTruss(
        self,
        *,
        pg: str,
        mass: float,
    ) -> InertiaTruss:
        return self._bridge._register(InertiaTruss(pg=pg, mass=mass))

    # -- ZeroLength family (Phase 2β) -----------------------------------

    def ZeroLength(
        self,
        *,
        pg: str | None = None,
        nodes: tuple[NodeRef, NodeRef] | None = None,
        mat_dirs: tuple[ZeroLengthMatDir, ...],
        orient: tuple[float, float, float, float, float, float]
        | None = None,
        do_rayleigh: bool = False,
        damp: Damping | None = None,
    ) -> ZeroLength:
        """``element zeroLength`` — coupled (material, dof) springs.

        Pass **exactly one** of ``pg=`` (fan across a 2-node "line"
        physical group) or ``nodes=(node_i, node_j)`` (ADR 0049 node-pair
        form — wire one spring to explicit endpoints, e.g. a boundary node
        to a ``g.decouple_node`` ground, without a meshed line).
        """
        return self._bridge._register(
            ZeroLength(
                pg=pg,
                nodes=nodes,
                mat_dirs=mat_dirs,
                orient=orient,
                do_rayleigh=do_rayleigh,
                damp=damp,
            )
        )

    def ZeroLengthSection(
        self,
        *,
        pg: str,
        section: Section | str,
        orient: tuple[float, float, float, float, float, float]
        | None = None,
        do_rayleigh: bool = True,
    ) -> ZeroLengthSection:
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            ZeroLengthSection(
                pg=pg,
                section=section,
                orient=orient,
                do_rayleigh=do_rayleigh,
            )
        )

    def CoupledZeroLength(
        self,
        *,
        pg: str | None = None,
        nodes: tuple[NodeRef, NodeRef] | None = None,
        material: UniaxialMaterial | str,
        dir1: int,
        dir2: int,
        use_rayleigh: bool = False,
    ) -> CoupledZeroLength:
        """``element CoupledZeroLength`` — one material on the resultant of 2 dirs.

        Pass **exactly one** of ``pg=`` or ``nodes=(node_i, node_j)`` (ADR
        0049 node-pair form).
        """
        material = self._bridge._resolve(material, base=UniaxialMaterial)
        return self._bridge._register(
            CoupledZeroLength(
                pg=pg,
                nodes=nodes,
                material=material,
                dir1=dir1,
                dir2=dir2,
                use_rayleigh=use_rayleigh,
            )
        )

    def TwoNodeLink(
        self,
        *,
        pg: str | None = None,
        nodes: tuple[NodeRef, NodeRef] | None = None,
        mat_dirs: tuple[ZeroLengthMatDir, ...],
        orient: tuple[float, ...] | None = None,
        p_delta: tuple[float, ...] | None = None,
        shear_dist: tuple[float, ...] | None = None,
        do_rayleigh: bool = False,
        mass: float | None = None,
    ) -> TwoNodeLink:
        """``element twoNodeLink`` — finite-length coupled springs.

        Pass **exactly one** of ``pg=`` or ``nodes=(node_i, node_j)`` (ADR
        0049 node-pair form).
        """
        return self._bridge._register(
            TwoNodeLink(
                pg=pg,
                nodes=nodes,
                mat_dirs=mat_dirs,
                orient=orient,
                p_delta=p_delta,
                shear_dist=shear_dist,
                do_rayleigh=do_rayleigh,
                mass=mass,
            )
        )

    # -- Shell family (Phase 2γ) ----------------------------------------

    def ShellMITC3(
        self, *, pg: str, section: _ShellSection | str
    ) -> ShellMITC3:
        section_r = cast(Section, self._bridge._resolve(section, base=Section))
        return self._bridge._register(
            ShellMITC3(pg=pg, section=section_r)
        )

    def ShellMITC4(
        self, *, pg: str, section: _ShellSection | str,
        damp: Damping | None = None,
    ) -> ShellMITC4:
        section_r = cast(Section, self._bridge._resolve(section, base=Section))
        return self._bridge._register(
            ShellMITC4(pg=pg, section=section_r, damp=damp)
        )

    def ShellDKGQ(
        self, *, pg: str, section: _ShellSection | str,
        damp: Damping | None = None,
    ) -> ShellDKGQ:
        section_r = cast(Section, self._bridge._resolve(section, base=Section))
        return self._bridge._register(
            ShellDKGQ(pg=pg, section=section_r, damp=damp)
        )

    def ASDShellQ4(
        self,
        *,
        pg: str,
        section: _ShellSection | str,
        corotational: bool = False,
        drilling_nt_alpha: float | None = None,
        local_cs: tuple[float, ...] | None = None,
        damp: Damping | None = None,
    ) -> ASDShellQ4:
        section_r = cast(Section, self._bridge._resolve(section, base=Section))
        return self._bridge._register(
            ASDShellQ4(
                pg=pg,
                section=section_r,
                corotational=corotational,
                drilling_nt_alpha=drilling_nt_alpha,
                local_cs=local_cs,
                damp=damp,
            )
        )

    def ASDShellT3(
        self,
        *,
        pg: str,
        section: _ShellSection | str,
        corotational: bool = False,
        drilling_dof: int | None = None,
        local_cs: tuple[float, ...] | None = None,
        damp: Damping | None = None,
    ) -> ASDShellT3:
        section_r = cast(Section, self._bridge._resolve(section, base=Section))
        return self._bridge._register(
            ASDShellT3(
                pg=pg,
                section=section_r,
                corotational=corotational,
                drilling_dof=drilling_dof,
                local_cs=local_cs,
                damp=damp,
            )
        )

    # -- Solid family (Phase 2δ) ----------------------------------------

    def FourNodeTetrahedron(
        self,
        *,
        pg: str,
        material: NDMaterial | str,
        body_force: tuple[float, float, float] | None = None,
    ) -> FourNodeTetrahedron:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            FourNodeTetrahedron(
                pg=pg, material=material, body_force=body_force,
            )
        )

    def TenNodeTetrahedron(
        self,
        *,
        pg: str,
        material: NDMaterial | str,
        body_force: tuple[float, float, float] | None = None,
    ) -> TenNodeTetrahedron:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            TenNodeTetrahedron(
                pg=pg, material=material, body_force=body_force,
            )
        )

    def stdBrick(  # noqa: N802 — mirrors the OpenSees Tcl token
        self,
        *,
        pg: str,
        material: NDMaterial | str,
        body_force: tuple[float, float, float] | None = None,
        damp: Damping | None = None,
    ) -> stdBrick:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            stdBrick(
                pg=pg, material=material, body_force=body_force, damp=damp,
            )
        )

    def LadrunoBrick(
        self,
        *,
        pg: str,
        material: NDMaterial | str,
        formulation: str = "std",
        geom: str = "linear",
        hourglass: str | None = None,
        hourglass_coeff: float | None = None,
        lumped: bool = False,
        body_force: tuple[float, float, float] | None = None,
        damp: Damping | None = None,
    ) -> LadrunoBrick:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            LadrunoBrick(
                pg=pg,
                material=material,
                formulation=formulation,
                geom=geom,
                hourglass=hourglass,
                hourglass_coeff=hourglass_coeff,
                lumped=lumped,
                body_force=body_force,
                damp=damp,
            )
        )

    # -- Absorbing boundary (ADR 0054) ----------------------------------

    def _absorbing_props(
        self,
        material: ElasticIsotropic | str | None,
        G: float | None,
        v: float | None,
        rho: float | None,
        *,
        who: str = "ASDAbsorbingBoundary3D",
    ) -> tuple[float, float, float]:
        """Resolve the ``(G, v, rho)`` triple from either a material or raw nums."""
        if material is not None:
            if G is not None or v is not None or rho is not None:
                raise ValueError(
                    f"{who}: pass either material= or "
                    "G/v/rho, not both."
                )
            mat = self._bridge._resolve(material, base=NDMaterial)
            if not isinstance(mat, ElasticIsotropic):
                raise TypeError(
                    f"{who}: material= must be an "
                    f"ElasticIsotropic (got {type(mat).__name__}); or pass "
                    "raw G/v/rho."
                )
            return mat.E / (2.0 * (1.0 + mat.nu)), mat.nu, mat.rho
        if G is None or v is None or rho is None:
            raise ValueError(
                f"{who}: pass either material= or all of "
                "G, v, rho."
            )
        return float(G), float(v), float(rho)

    def ASDAbsorbingBoundary3D(  # noqa: N802 — mirrors the OpenSees token
        self,
        *,
        pg: str,
        btype: str,
        material: ElasticIsotropic | str | None = None,
        G: float | None = None,
        v: float | None = None,
        rho: float | None = None,
        fx: TimeSeries | str | None = None,
        fy: TimeSeries | str | None = None,
        fz: TimeSeries | str | None = None,
    ) -> _Absorbing3D:
        """One absorbing-boundary declaration over a single-btype skin PG.

        Supply the soil properties either as ``material=ElasticIsotropic(...)``
        (``G = E/(2(1+v))`` is derived) or as raw ``G=/v=/rho=``.  The
        ``-fx/-fy/-fz`` base-input series are only valid on a bottom boundary.
        Usually you call :meth:`absorbing_boundary` instead, which fans this
        across all skin PGs of an :class:`AbsorbingSkinResult`.
        """
        Gval, vval, rhoval = self._absorbing_props(material, G, v, rho)
        fxs = self._bridge._resolve(fx, base=TimeSeries) if fx is not None else None
        fys = self._bridge._resolve(fy, base=TimeSeries) if fy is not None else None
        fzs = self._bridge._resolve(fz, base=TimeSeries) if fz is not None else None
        return self._bridge._register(
            ASDAbsorbingBoundary3D(
                pg=pg, G=Gval, v=vval, rho=rhoval, btype=btype,
                fx=fxs, fy=fys, fz=fzs,
            )
        )

    def ASDAbsorbingBoundary2D(  # noqa: N802 — mirrors the OpenSees token
        self,
        *,
        pg: str,
        btype: str,
        thickness: float,
        material: ElasticIsotropic | str | None = None,
        G: float | None = None,
        v: float | None = None,
        rho: float | None = None,
        fx: TimeSeries | str | None = None,
        fy: TimeSeries | str | None = None,
    ) -> _Absorbing2D:
        """One 2D absorbing-boundary declaration over a single-btype skin PG.

        The plane-strain sibling of :meth:`ASDAbsorbingBoundary3D` (ADR 0054,
        AB-5).  ``thickness`` is the **out-of-plane slab thickness** (it
        scales the element's mass/stiffness/dashpot terms — match your 2D
        continuum elements' thickness).  ``btype`` draws from ``B``/``L``/``R``;
        ``-fx/-fy`` base-input series are only valid on a bottom boundary.
        Usually you call :meth:`absorbing_boundary` instead, which fans this
        across all skin PGs of a 2D :class:`AbsorbingSkinResult`.
        """
        Gval, vval, rhoval = self._absorbing_props(
            material, G, v, rho, who="ASDAbsorbingBoundary2D",
        )
        fxs = (self._bridge._resolve(fx, base=TimeSeries)
               if fx is not None else None)
        fys = (self._bridge._resolve(fy, base=TimeSeries)
               if fy is not None else None)
        return self._bridge._register(
            ASDAbsorbingBoundary2D(
                pg=pg, G=Gval, v=vval, rho=rhoval,
                thickness=float(thickness), btype=btype,
                fx=fxs, fy=fys,
            )
        )

    def absorbing_boundary(
        self,
        *,
        skin: "AbsorbingSkinResult",
        material: ElasticIsotropic | str | None = None,
        G: float | None = None,
        v: float | None = None,
        rho: float | None = None,
        materials: "list[ElasticIsotropic | str] | None" = None,
        base_series: TimeSeries | str | None = None,
        base_dirs: tuple[str, ...] = ("x",),
        thickness: float | None = None,
    ) -> "list[_Absorbing3D | _Absorbing2D]":
        """Emit ``ASDAbsorbingBoundary3D``/``2D`` over every btype of a skin.

        Dispatches on ``skin.ndm``: a 3D skin fans ``ASDAbsorbingBoundary3D``
        bricks; a 2D skin (built with ``add_plane_wave_box_2d`` /
        ``add_absorbing_shell_2d``) fans ``ASDAbsorbingBoundary2D`` quads and
        **requires** ``thickness=`` (the out-of-plane plane-strain slab
        thickness — match your soil quads').

        For a **homogeneous** skin, pass a single ``material=`` (or raw
        ``G/v/rho``): one declaration per ``skin.skin_pgs`` entry (each with its
        fixed btype).  For a **stratified** skin (built with ``z=[...]`` /
        ``layers=``), pass ``materials=[m0, m1, …]`` — one per layer, top → bottom,
        ``len(materials) == skin.n_layers`` — and each layer's skin cells get that
        layer's derived ``G/v/rho`` (the base skin takes the bottom layer's
        material).  ``base_series`` is attached as ``-fx/-fy/-fz`` (per
        ``base_dirs``; ``"z"`` is invalid for a 2D skin) to every bottom
        (``B``-containing) skin PG only.  Returns the registered specs.
        """
        ndm = int(getattr(skin, "ndm", 3))
        who = ("ASDAbsorbingBoundary2D" if ndm == 2
               else "ASDAbsorbingBoundary3D")
        if ndm == 2 and thickness is None:
            raise ValueError(
                "absorbing_boundary: a 2D skin requires thickness= (the "
                "out-of-plane plane-strain slab thickness; match your soil "
                "quads')."
            )
        if ndm == 3 and thickness is not None:
            raise ValueError(
                "absorbing_boundary: thickness= is 2D-only (the 3D element "
                "derives everything from its hex geometry); drop it for a "
                "3D skin."
            )
        if materials is not None and (
            material is not None or G is not None or v is not None or rho is not None
        ):
            raise ValueError(
                "absorbing_boundary: pass either materials= (per-layer) or a "
                "single material=/G/v/rho, not both."
            )
        series = (
            self._bridge._resolve(base_series, base=TimeSeries)
            if base_series is not None else None
        )
        dirs = tuple(base_dirs)
        ok_dirs = ("x", "y") if ndm == 2 else ("x", "y", "z")
        for d in dirs:
            if d not in ok_dirs:
                raise ValueError(
                    f"absorbing_boundary: base_dirs entries must be "
                    f"{'/'.join(repr(o) for o in ok_dirs)} for a {ndm}D "
                    f"skin, got {d!r}."
                )

        def _emit(
            pg: str, btype: str, gv: float, vv: float, rv: float,
        ) -> "_Absorbing3D | _Absorbing2D":
            bottom = "B" in btype and series is not None
            if ndm == 2:
                return self._bridge._register(
                    ASDAbsorbingBoundary2D(
                        pg=pg, G=gv, v=vv, rho=rv,
                        thickness=float(thickness),  # type: ignore[arg-type]
                        btype=btype,
                        fx=series if (bottom and "x" in dirs) else None,
                        fy=series if (bottom and "y" in dirs) else None,
                    )
                )
            return self._bridge._register(
                ASDAbsorbingBoundary3D(
                    pg=pg, G=gv, v=vv, rho=rv, btype=btype,
                    fx=series if (bottom and "x" in dirs) else None,
                    fy=series if (bottom and "y" in dirs) else None,
                    fz=series if (bottom and "z" in dirs) else None,
                )
            )

        out: "list[_Absorbing3D | _Absorbing2D]" = []
        if materials is not None:
            if len(materials) != skin.n_layers:
                raise ValueError(
                    f"absorbing_boundary: materials has {len(materials)} entries "
                    f"but the skin has {skin.n_layers} layer(s); pass one material "
                    "per layer, top → bottom."
                )
            props = [
                self._absorbing_props(m, None, None, None, who=who)
                for m in materials
            ]
            for layer, by_btype in skin.skin_pgs_by_layer.items():
                gv, vv, rv = props[layer]
                for btype, pg in by_btype.items():
                    out.append(_emit(pg, btype, gv, vv, rv))
        else:
            Gval, vval, rhoval = self._absorbing_props(
                material, G, v, rho, who=who,
            )
            for btype, pg in skin.skin_pgs.items():
                out.append(_emit(pg, btype, Gval, vval, rhoval))
        return out

    def FourNodeQuad(
        self,
        *,
        pg: str,
        thickness: float,
        material: NDMaterial | str,
        plane_type: str = "PlaneStrain",
        pressure: float | None = None,
        rho: float | None = None,
        body_force: tuple[float, float] | None = None,
        damp: Damping | None = None,
    ) -> FourNodeQuad:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            FourNodeQuad(
                pg=pg,
                thickness=thickness,
                material=material,
                plane_type=plane_type,
                pressure=pressure,
                rho=rho,
                body_force=body_force,
                damp=damp,
            )
        )

    def Tri31(
        self,
        *,
        pg: str,
        thickness: float,
        material: NDMaterial | str,
        plane_type: str = "PlaneStrain",
        pressure: float | None = None,
        rho: float | None = None,
        body_force: tuple[float, float] | None = None,
    ) -> Tri31:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            Tri31(
                pg=pg,
                thickness=thickness,
                material=material,
                plane_type=plane_type,
                pressure=pressure,
                rho=rho,
                body_force=body_force,
            )
        )

    def SixNodeTri(
        self,
        *,
        pg: str,
        thickness: float,
        material: NDMaterial | str,
        plane_type: str = "PlaneStrain",
        pressure: float | None = None,
        rho: float | None = None,
        body_force: tuple[float, float] | None = None,
    ) -> SixNodeTri:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            SixNodeTri(
                pg=pg,
                thickness=thickness,
                material=material,
                plane_type=plane_type,
                pressure=pressure,
                rho=rho,
                body_force=body_force,
            )
        )

    def BezierTri6(
        self,
        *,
        pg: str,
        thickness: float,
        material: NDMaterial | str,
        plane_type: str = "PlaneStrain",
        bbar: bool = False,
        consistent_mass: bool = False,
        pressure: float | None = None,
        rho: float | None = None,
        body_force: tuple[float, float] | None = None,
    ) -> BezierTri6:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            BezierTri6(
                pg=pg,
                thickness=thickness,
                material=material,
                plane_type=plane_type,
                bbar=bbar,
                consistent_mass=consistent_mass,
                pressure=pressure,
                rho=rho,
                body_force=body_force,
            )
        )

    def BezierTet10(
        self,
        *,
        pg: str,
        material: NDMaterial | str,
        bbar: bool = False,
        consistent_mass: bool = False,
        rho: float | None = None,
        body_force: tuple[float, float, float] | None = None,
        pressure: float | None = None,
        geom: str = "linear",
        fbar: str = "centroid",
    ) -> BezierTet10:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            BezierTet10(
                pg=pg,
                material=material,
                bbar=bbar,
                consistent_mass=consistent_mass,
                rho=rho,
                body_force=body_force,
                pressure=pressure,
                geom=geom,
                fbar=fbar,
            )
        )
