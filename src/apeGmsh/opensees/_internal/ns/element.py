"""
``_ElementNS`` — backs ``ops.element.<Type>(pg=..., ...)``.

Phase 2 populates the namespace with one typed method per OpenSees
element. The bridge fans the spec across its physical group at build
time; the typed class never carries node tags.

Each method is fully kw-only with explicit types — no ``**kwargs``
(P12). Methods are grouped by family in alphabetical-ish order.
"""
from __future__ import annotations

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
    SixNodeTri,
    TenNodeTetrahedron,
    Tri31,
    stdBrick,
)
from ...element.truss import CorotTruss, InertiaTruss, Truss
from ...element.zero_length import (
    ZeroLength,
    ZeroLengthMatDir,
    ZeroLengthSection,
)
from ...section.plate import (
    ElasticMembranePlateSection,
    LayeredShell,
    LayeredShellFiberSection,
)
from ...transform import Corotational, Linear, PDelta
from ..types import (
    BeamIntegration,
    GeomTransf,
    NDMaterial,
    Section,
    UniaxialMaterial,
)
from ._base import _BridgeNamespace


__all__ = ["_ElementNS"]


# Beam-column transforms — concrete subclasses of GeomTransf.
_AnyTransf = Linear | PDelta | Corotational

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
    ) -> elasticBeamColumn:
        transf = self._bridge._resolve(transf, base=GeomTransf)
        return self._bridge._register(
            elasticBeamColumn(
                pg=pg, transf=transf,
                A=A, E=E, Iz=Iz,
                Iy=Iy, G=G, J=J,
                mass=mass, c_mass=c_mass,
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
    ) -> forceBeamColumn:
        """``element forceBeamColumn`` — force-based distributed-plasticity.

        Compose the integration rule first (e.g. ``ops.beamIntegration.Lobatto(
        section=sec, n_ip=5)``) and pass it as ``integration=``.  Both
        ``transf`` and ``integration`` accept object handles or
        registered names.
        """
        transf = self._bridge._resolve(transf, base=GeomTransf)
        integration = self._bridge._resolve(integration, base=BeamIntegration)
        return self._bridge._register(
            forceBeamColumn(
                pg=pg, transf=transf, integration=integration,
                mass=mass, max_iter=max_iter, tol=tol,
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
    ) -> dispBeamColumn:
        """``element dispBeamColumn`` — displacement-based distributed-plasticity."""
        transf = self._bridge._resolve(transf, base=GeomTransf)
        integration = self._bridge._resolve(integration, base=BeamIntegration)
        return self._bridge._register(
            dispBeamColumn(
                pg=pg, transf=transf, integration=integration,
                mass=mass, c_mass=c_mass,
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
        transf = self._bridge._resolve(transf, base=GeomTransf)
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
        pg: str,
        mat_dirs: tuple[ZeroLengthMatDir, ...],
        orient: tuple[float, float, float, float, float, float]
        | None = None,
        do_rayleigh: bool = False,
    ) -> ZeroLength:
        return self._bridge._register(
            ZeroLength(
                pg=pg,
                mat_dirs=mat_dirs,
                orient=orient,
                do_rayleigh=do_rayleigh,
            )
        )

    def ZeroLengthSection(
        self,
        *,
        pg: str,
        section: Section | str,
        orient: tuple[float, float, float, float, float, float]
        | None = None,
        do_rayleigh: bool = False,
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

    # -- Shell family (Phase 2γ) ----------------------------------------

    def ShellMITC3(
        self, *, pg: str, section: _ShellSection | str
    ) -> ShellMITC3:
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            ShellMITC3(pg=pg, section=section)
        )

    def ShellMITC4(
        self, *, pg: str, section: _ShellSection | str
    ) -> ShellMITC4:
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            ShellMITC4(pg=pg, section=section)
        )

    def ShellDKGQ(
        self, *, pg: str, section: _ShellSection | str
    ) -> ShellDKGQ:
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            ShellDKGQ(pg=pg, section=section)
        )

    def ASDShellQ4(
        self,
        *,
        pg: str,
        section: _ShellSection | str,
        corotational: bool = False,
        drilling_nt_alpha: float | None = None,
        local_cs: tuple[float, ...] | None = None,
    ) -> ASDShellQ4:
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            ASDShellQ4(
                pg=pg,
                section=section,
                corotational=corotational,
                drilling_nt_alpha=drilling_nt_alpha,
                local_cs=local_cs,
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
    ) -> ASDShellT3:
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            ASDShellT3(
                pg=pg,
                section=section,
                corotational=corotational,
                drilling_dof=drilling_dof,
                local_cs=local_cs,
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
    ) -> stdBrick:
        material = self._bridge._resolve(material, base=NDMaterial)
        return self._bridge._register(
            stdBrick(
                pg=pg, material=material, body_force=body_force,
            )
        )

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
            )
        )
