"""
Plate / shell sections — typed primitives for OpenSees plate-bending
section commands.

Four section types live here:

* :class:`ElasticMembranePlateSection` — single-layer linear-elastic
  plate (membrane + bending) used by ``ShellMITC4``, ``ShellDKGQ``,
  ``ASDShellQ4``.
* :class:`LayeredShell` — stacked nDMaterial layers (a thin-shell
  composite section) for ``ShellMITC4`` and friends.
* :class:`LayeredShellFiberSection` — the same section under the C++
  class name. OpenSees registers ``LayeredShellFiberSection`` only
  under the ``LayeredShell`` keyword, so both primitives emit
  ``section LayeredShell``.
* :class:`LadrunoShellModifier` — a *decorator* that applies
  ETABS-style stiffness modifiers to any of the above (fork-only,
  Ladruno ADR 91).

Plus one builder, :func:`RCLayeredShell`, which composes a
reinforced-concrete :class:`LayeredShell` from a concrete law and
:class:`RebarMesh` bar meshes (each mesh its own :class:`PlateRebar`
layer, the concrete reduced around it). It adds no emit surface.

The ``LayeredShell*`` sections compose a tuple of nDMaterials and
declare them via :meth:`dependencies`; their ``_emit`` references
resolved nDMaterial tags via the same closure-based resolver Fiber
uses (see :mod:`section.fiber`).

The OpenSees commands per the manual:

* ``section ElasticMembranePlateSection $tag $E $nu $h <$rho>
  <$Ep_mod>``
* ``section LayeredShell $tag $nLayers $matTag1 $h1 ... $matTagN $hN``
* (``LayeredShellFiberSection`` emits the ``LayeredShell`` line above.)
* ``section LadrunoShellModifier $tag $innerSecTag <-f11 v> ...
  <-mass v>``
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .._internal.types import NDMaterial, Primitive, Section, UniaxialMaterial
from ..material.nd import LogStrain2D, PlaneStrain, PlaneStressRebar, PlateRebar
from ._tag_resolver import resolve_mat_tag

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "SHELL_MODIFIER_FLAGS",
    "ShellModifierNonlinearInnerWarning",
    "ElasticMembranePlateSection",
    "LadrunoShellModifier",
    "MIN_SHELL_LAYERS",
    "LayeredShell",
    "LayeredShellFiberSection",
    "ShellLayer",
    "RCLayeredShell",
    "RebarMesh",
    "CoarseShellLayeringWarning",
    "MIN_RC_CONCRETE_LAYERS",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class ElasticMembranePlateSection(Section):
    """``section ElasticMembranePlateSection`` — single-layer plate.

    Linear-elastic plate-bending section with isotropic membrane
    stiffness. Used by ``ShellMITC4``, ``ShellDKGQ``, ``ASDShellQ4``.

    Parameters
    ----------
    E
        Young's modulus.
    nu
        Poisson's ratio. Must satisfy ``0 <= nu < 0.5``.
    h
        Section thickness.
    rho
        Mass density per unit volume. Defaults to 0.0 (no mass).
    Ep_mod
        Out-of-plane (bending) stiffness modifier — the section's
        bending modulus becomes ``E * Ep_mod`` while the membrane
        modulus stays ``E``. Defaults to 1.0, and is emitted only when
        it differs, so existing decks are unchanged.

        Present for round-trip fidelity with decks written elsewhere.
        For new work prefer :class:`LadrunoShellModifier`, which is
        strictly more expressive; ``Ep_mod=r`` is exactly equivalent to
        ``m11=m22=m12=v13=v23=r`` there.
    """

    E:      float
    nu:     float
    h:      float
    rho:    float = 0.0
    Ep_mod: float = 1.0

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(
                f"ElasticMembranePlateSection: E must be > 0, "
                f"got {self.E}."
            )
        if not (0.0 <= self.nu < 0.5):
            raise ValueError(
                f"ElasticMembranePlateSection: nu must be in [0, 0.5), "
                f"got {self.nu}."
            )
        if self.h <= 0:
            raise ValueError(
                f"ElasticMembranePlateSection: h must be > 0, "
                f"got {self.h}."
            )
        if self.rho < 0:
            raise ValueError(
                f"ElasticMembranePlateSection: rho must be >= 0, "
                f"got {self.rho}."
            )
        if self.Ep_mod < 0:
            raise ValueError(
                f"ElasticMembranePlateSection: Ep_mod must be >= 0, "
                f"got {self.Ep_mod}."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        # Ep_mod is the optional 5th double and is positional, so it can
        # only be emitted after rho. Omit it at its default to leave
        # existing decks byte-identical.
        params: list[float] = [self.E, self.nu, self.h, self.rho]
        if self.Ep_mod != 1.0:
            params.append(self.Ep_mod)
        emitter.section("ElasticMembranePlateSection", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LayeredShell / LayeredShellFiberSection — composite layered sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ShellLayer:
    """One layer of a ``LayeredShell`` / ``LayeredShellFiberSection``.

    A value object — not a :class:`Primitive`, no tag, not registered
    standalone. Held in a tuple by the parent section.

    Parameters
    ----------
    material
        The nDMaterial constitutive law for this layer.
    thickness
        Layer thickness.
    """

    material: NDMaterial
    thickness: float

    def __post_init__(self) -> None:
        if isinstance(self.material, UniaxialMaterial):
            # The section parser looks each layer tag up among the
            # nDMaterials (OPS_getNDMaterial), a tag space separate from the
            # uniaxials: the tag names a missing or an unrelated nD material.
            raise TypeError(
                f"ShellLayer: {type(self.material).__name__} is a "
                "UniaxialMaterial, and a layered shell takes nDMaterial "
                "layers only. Wrap the bar steel as a smeared rebar layer: "
                "PlateRebar(material=<uniaxial>, angle=<deg>) with the layer "
                "thickness = bar area / spacing (or build the whole section "
                "with RCLayeredShell)."
            )
        if not isinstance(self.material, NDMaterial):
            raise TypeError(
                "ShellLayer: material must be an NDMaterial primitive, got "
                f"{type(self.material).__name__!r}."
            )
        if isinstance(self.material, PlaneStressRebar):
            # LayeredShellFiberSection asks each layer for
            # getCopy("PlateFiber"); PlaneStressRebar answers null and the
            # C++ side calls exit(-1), killing the interpreter.
            raise TypeError(
                "ShellLayer: PlaneStressRebar is a plane-stress material "
                "with no PlateFiber view, and OpenSees exits the process "
                "on such a layer. Use PlateRebar for a smeared rebar layer."
            )
        if isinstance(self.material, LogStrain2D):
            # getCopy("PlateFiber") answers null -> exit(-1), as above.
            raise TypeError(
                "ShellLayer: LogStrain2D has only PlaneStrain / PlaneStress "
                "views, and OpenSees exits the process on a layer with no "
                "PlateFiber view. Use a 3-D law instead."
            )
        if isinstance(self.material, PlaneStrain):
            # PlaneStrainMaterial::getCopy(type) returns an order-3
            # plane-strain copy for ANY type, so the section would drive an
            # order-3 material with order-5 plate strains.
            raise TypeError(
                "ShellLayer: PlaneStrain answers getCopy('PlateFiber') with "
                "an order-3 plane-strain copy, not a PlateFiber material. "
                "Use the 3-D law it wraps as the layer material directly."
            )
        if self.thickness <= 0:
            raise ValueError(
                f"ShellLayer: thickness must be > 0, got {self.thickness}."
            )


#: The C++ parser (``OPS_LayeredShellFiberSection``) rejects fewer layers
#: with "number of layers must be larger than 2".
MIN_SHELL_LAYERS = 3


def _validate_layers(
    cls_name: str, layers: tuple[ShellLayer, ...]
) -> None:
    for i, layer in enumerate(layers):
        if not isinstance(layer, ShellLayer):
            raise TypeError(
                f"{cls_name}: layers[{i}] must be a ShellLayer(material=..., "
                f"thickness=...), got {type(layer).__name__!r}."
            )
    if len(layers) < MIN_SHELL_LAYERS:
        raise ValueError(
            f"{cls_name}: OpenSees needs at least {MIN_SHELL_LAYERS} "
            f"ShellLayers, got {len(layers)}. Split a thick layer in two "
            f"(e.g. top and bottom cover) to reach the minimum."
        )


def _layer_dependencies(
    layers: tuple[ShellLayer, ...]
) -> tuple[Primitive, ...]:
    """Deduplicate materials referenced by layers, preserving order."""
    seen: dict[int, NDMaterial] = {}
    for layer in layers:
        seen.setdefault(id(layer.material), layer.material)
    return tuple(seen.values())


@dataclass(frozen=True, kw_only=True, slots=True)
class LayeredShell(Section):
    """``section LayeredShell`` — stacked nDMaterial layers.

    Parameters
    ----------
    layers
        Tuple of :class:`ShellLayer` describing each through-thickness
        layer (bottom to top). At least three layers are required; the
        OpenSees parser refuses fewer.

    Notes
    -----
    Material-tag resolution at emit time is delegated to a
    closure-captured resolver attached to the emitter; see
    :mod:`section._tag_resolver` for the contract and the open
    coordinator question flagged for the Phase 4 emitters.
    """

    layers: tuple[ShellLayer, ...]

    def __post_init__(self) -> None:
        _validate_layers("LayeredShell", self.layers)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        params: list[float | str] = [len(self.layers)]
        for layer in self.layers:
            mat_tag = resolve_mat_tag(emitter, layer.material)
            params.append(mat_tag)
            params.append(layer.thickness)
        emitter.section("LayeredShell", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return _layer_dependencies(self.layers)


@dataclass(frozen=True, kw_only=True, slots=True)
class LayeredShellFiberSection(Section):
    """``LayeredShellFiberSection`` — :class:`LayeredShell` under its C++ name.

    Emits ``section LayeredShell $tag $nLayers ...``, the same line as
    :class:`LayeredShell`. OpenSees has one class for layered shells,
    ``LayeredShellFiberSection``, and registers it only under the
    ``LayeredShell`` keyword in the Tcl interpreter, the Python
    interpreter, and the newer runtime. Before this was fixed the
    primitive emitted ``section LayeredShellFiberSection``, and no
    interpreter accepted that line. The class name is kept so existing
    scripts still work.

    Parameters
    ----------
    layers
        Tuple of :class:`ShellLayer` describing each through-thickness
        layer (bottom to top). At least three layers are required.
    """

    layers: tuple[ShellLayer, ...]

    def __post_init__(self) -> None:
        _validate_layers("LayeredShellFiberSection", self.layers)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        params: list[float | str] = [len(self.layers)]
        for layer in self.layers:
            mat_tag = resolve_mat_tag(emitter, layer.material)
            params.append(mat_tag)
            params.append(layer.thickness)
        # The only keyword OpenSees registers for this C++ class.
        emitter.section("LayeredShell", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return _layer_dependencies(self.layers)


# ---------------------------------------------------------------------------
# RCLayeredShell — reinforced-concrete layered-shell builder
# ---------------------------------------------------------------------------

class CoarseShellLayeringWarning(UserWarning):
    """An :func:`RCLayeredShell` with fewer than six concrete layers.

    Subclass of :class:`UserWarning`, so it can be silenced per call or
    promoted to an error in CI (the warn-as-contract idiom of
    :class:`ShellModifierNonlinearInnerWarning`).

    ``LayeredShellFiberSection`` integrates each layer at its mid-plane
    (one point per layer), so every layer loses its own ``t**3 / 12``
    bending inertia. For ``n`` equal layers of a homogeneous plate that is
    exactly ``1 / n**2`` of the bending stiffness (4 % at 5 layers, 1 % at
    10), and a coarse stack also smears the cracked depth of a nonlinear
    concrete law over few points.
    """


#: Below this many concrete layers :func:`RCLayeredShell` warns
#: (:class:`CoarseShellLayeringWarning`). 8-12 is the recommended range.
MIN_RC_CONCRETE_LAYERS = 6


@dataclass(frozen=True, kw_only=True, slots=True)
class RebarMesh:
    """One smeared reinforcement mesh (one bar direction) of an RC shell.

    A value object for :func:`RCLayeredShell`: not a :class:`Primitive`, no
    tag. The builder turns it into a thin :class:`PlateRebar` layer of
    thickness ``area_per_width`` centred on the bar centroid.

    Parameters
    ----------
    material
        The bar steel, a uniaxial law (e.g. ``Steel02``, or a
        ``LadrunoRebarBuckling``-wrapped steel for discrete boundary bars).
    angle
        Bar direction in **degrees** from the x axis of the section frame
        (see :class:`PlateRebar` for which axis that is, and pin it with
        ``ASDShellQ4(local_cs=...)``).
    area_per_width
        Steel area per unit width, ``A_s / s`` (bar area over spacing).
        This is the smeared steel thickness. Finite and ``> 0``.
    z
        Bar centroid measured from the shell mid-surface, positive towards
        the top face. Give either ``z``, or ``cover`` and ``face``.
    cover
        Distance from ``face`` to the bar centroid (clear cover plus half
        a bar diameter, plus the diameter of any bar layer outside this
        one). Finite and ``>= 0``.
    face
        ``"bottom"`` (the ``-h/2`` face, where the first layer starts) or
        ``"top"``.
    """

    material: UniaxialMaterial
    angle: float
    area_per_width: float
    z: float | None = None
    cover: float | None = None
    face: Literal["bottom", "top"] | None = None

    @classmethod
    def from_bars(
        cls, *,
        material: UniaxialMaterial,
        angle: float,
        bar_area: float,
        spacing: float,
        z: float | None = None,
        cover: float | None = None,
        face: Literal["bottom", "top"] | None = None,
    ) -> "RebarMesh":
        """Build from one bar's area and the bar spacing (``A_s / s``)."""
        for label, value in (("bar_area", bar_area), ("spacing", spacing)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"RebarMesh.from_bars: {label} must be finite and > 0, "
                    f"got {value!r}."
                )
        return cls(
            material=material, angle=angle,
            area_per_width=bar_area / spacing,
            z=z, cover=cover, face=face,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.material, UniaxialMaterial):
            raise TypeError(
                "RebarMesh: material must be a UniaxialMaterial primitive "
                f"(the bar steel), got {type(self.material).__name__!r}."
            )
        if not math.isfinite(self.angle):
            raise ValueError(
                f"RebarMesh: angle must be finite, got {self.angle!r}."
            )
        if not (math.isfinite(self.area_per_width) and self.area_per_width > 0):
            raise ValueError(
                "RebarMesh: area_per_width must be finite and > 0, got "
                f"{self.area_per_width!r}."
            )
        if self.z is not None:
            if self.cover is not None or self.face is not None:
                raise ValueError(
                    "RebarMesh: give either z, or cover and face, not both."
                )
            if not math.isfinite(self.z):
                raise ValueError(f"RebarMesh: z must be finite, got {self.z!r}.")
            return
        if self.cover is None or self.face is None:
            raise ValueError(
                "RebarMesh: give the bar centroid as z, or as cover and "
                "face ('bottom' or 'top')."
            )
        if self.face not in ("bottom", "top"):
            raise ValueError(
                "RebarMesh: face must be 'bottom' or 'top', got "
                f"{self.face!r}."
            )
        if not (math.isfinite(self.cover) and self.cover >= 0):
            raise ValueError(
                f"RebarMesh: cover must be finite and >= 0, got {self.cover!r}."
            )

    def centroid(self, h: float) -> float:
        """Bar centroid from the mid-surface of a section of thickness ``h``."""
        if self.z is not None:
            return self.z
        assert self.cover is not None  # __post_init__ guarantees it
        return -h / 2 + self.cover if self.face == "bottom" else h / 2 - self.cover


def _apportion(lengths: list[float], n: int) -> list[int]:
    """Split ``n`` layers over regions in proportion to their lengths.

    Every region gets at least one layer (``n >= len(lengths)`` is the
    caller's job); the rest go by largest remainder, ties to the lower
    region.
    """
    total = math.fsum(lengths)
    ideal = [n * g / total for g in lengths]
    counts = [max(1, math.floor(x)) for x in ideal]
    while sum(counts) > n:
        i = max(
            (k for k in range(len(counts)) if counts[k] > 1),
            key=lambda k: counts[k] - ideal[k],
        )
        counts[i] -= 1
    while sum(counts) < n:
        i = max(range(len(counts)), key=lambda k: ideal[k] - counts[k])
        counts[i] += 1
    return counts


def RCLayeredShell(
    *,
    h: float,
    concrete: NDMaterial,
    meshes: Sequence[RebarMesh] = (),
    n_concrete: int = 10,
) -> LayeredShell:
    """Build a reinforced-concrete :class:`LayeredShell` from bar meshes.

    Composes existing primitives only; the result is a plain
    ``LayeredShell`` (``section LayeredShell``), listed bottom (``-h/2``)
    to top (``+h/2``). Each :class:`RebarMesh` becomes its **own** thin
    :class:`PlateRebar` layer, of thickness ``A_s / s``, centred on the
    bar centroid. The concrete is split into sub-layers that fill the
    gaps between the bar layers, so the concrete thickness is reduced by
    exactly the steel it gives way to (``h - sum(A_s / s)``). A smeared
    bar is never a ``rho``-weighted overlay inside a concrete layer: that
    would count ``c + rho*s`` of material where there is
    ``(1 - rho)*c + rho*s``.

    ``LayeredShellFiberSection`` places layer ``i`` at the midpoint of
    its thickness interval, so the bar sits at the given depth. The
    thicknesses sum to ``h`` (to rounding: the topmost concrete layer
    takes the remainder).

    Parameters
    ----------
    h
        Total shell thickness. Finite and ``> 0``.
    concrete
        The concrete layer law: an nD material with a PlateFiber view,
        e.g. :class:`~apeGmsh.opensees.material.nd.LadrunoRCConcrete`,
        :class:`~apeGmsh.opensees.material.nd.ASDConcrete3D`,
        :class:`~apeGmsh.opensees.material.nd.LadrunoConcrete3D`,
        ``ElasticIsotropic``, or a plane-stress law wrapped in
        :class:`~apeGmsh.opensees.material.nd.PlateFromPlaneStress`.
        One instance is shared by every concrete layer.
    meshes
        The reinforcement meshes, in any order. One mesh per bar
        direction and curtain: a two-way, two-curtain wall is four
        meshes at four depths. Each bar layer must lie inside
        ``[-h/2, h/2]``, and bar layers must not overlap (they may
        touch). Meshes with the same ``material`` instance and ``angle``
        share one :class:`PlateRebar`, so the deck carries one
        ``nDMaterial`` line per distinct bar law and direction.
    n_concrete
        Target number of concrete layers, split over the concrete regions
        (below, between and above the bar layers) in proportion to their
        thickness. Every region gets at least one layer, so the total is
        ``max(n_concrete, number of regions)``.

    Returns
    -------
    LayeredShell
        Standalone (P11): register it and its new :class:`PlateRebar`
        layers with a bridge before ``build()``. ``ops.section.RCLayeredShell``
        does that for you; the concrete and steel must already be
        registered.

    Raises
    ------
    ValueError
        ``h`` or ``n_concrete`` invalid, a bar layer outside the section,
        two bar layers overlapping, or no concrete left between them.
    TypeError
        ``concrete`` is not a valid shell-layer material (see
        :class:`ShellLayer`), or a mesh is not a :class:`RebarMesh`.

    Warns
    -----
    CoarseShellLayeringWarning
        Fewer than :data:`MIN_RC_CONCRETE_LAYERS` concrete layers.

    Notes
    -----
    * **Layer count.** Use 8-12 concrete layers for nonlinear RC bending.
      Each layer is integrated at its mid-plane, which drops ``1/n**2`` of
      the bending stiffness of ``n`` equal layers (see
      :class:`CoarseShellLayeringWarning`); below 6 the builder warns.
    * **Boundary bars.** Discrete bars at a wall boundary can use a
      ``LadrunoRebarBuckling``-wrapped steel as the mesh ``material``
      (fork only); :class:`PlateRebar` accepts any uniaxial law.
    * **Tension stiffening, once.** Represent it in the concrete law
      (e.g. the tension-stiffening options of ``LadrunoRCConcrete``) *or*
      in a stiffened steel law, not both: two contributions double-count
      the concrete between cracks.
    * **Bar direction.** ``angle`` is measured in the section frame the
      shell hands its layers. Pass ``ASDShellQ4(local_cs=(...))`` so that
      frame is the same on every build (:class:`PlateRebar`, Notes).

    Examples
    --------
    >>> steel = Steel01(fy=420e6, E=200e9, b=0.01)
    >>> conc = ElasticIsotropic(E=30e9, nu=0.2)
    >>> d12 = 113.1e-6                        # one 12 mm bar, m^2
    >>> sec = RCLayeredShell(h=0.20, concrete=conc, meshes=[
    ...     RebarMesh.from_bars(material=steel, angle=0.0, bar_area=d12,
    ...                         spacing=0.15, cover=0.031, face="bottom"),
    ...     RebarMesh.from_bars(material=steel, angle=90.0, bar_area=d12,
    ...                         spacing=0.15, cover=0.043, face="bottom"),
    ... ])
    """
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"RCLayeredShell: h must be finite and > 0, got {h!r}.")
    if isinstance(n_concrete, bool) or not isinstance(n_concrete, int) or n_concrete < 1:
        raise ValueError(
            f"RCLayeredShell: n_concrete must be an int >= 1, got {n_concrete!r}."
        )
    # Validate the concrete once, with ShellLayer's messages.
    ShellLayer(material=concrete, thickness=h)

    tol = 1e-9 * h
    half = h / 2
    bars: list[tuple[float, float, float, int, RebarMesh]] = []
    for i, mesh in enumerate(meshes):
        if not isinstance(mesh, RebarMesh):
            raise TypeError(
                f"RCLayeredShell: meshes[{i}] must be a RebarMesh, got "
                f"{type(mesh).__name__!r}."
            )
        z = mesh.centroid(h)
        t = mesh.area_per_width
        lo, hi = z - t / 2, z + t / 2
        if lo < -half - tol or hi > half + tol:
            raise ValueError(
                f"RCLayeredShell: meshes[{i}] (angle={mesh.angle}) spans "
                f"z in [{lo:.6g}, {hi:.6g}], outside the section "
                f"[{-half:.6g}, {half:.6g}]. Check its z / cover and "
                "area_per_width."
            )
        bars.append((lo, hi, z, i, mesh))
    bars.sort(key=lambda b: (b[2], b[3]))

    # Concrete gaps: below the first bar, between bars, above the last.
    edges = [-half] + [e for b in bars for e in (b[0], b[1])] + [half]
    gaps = [edges[2 * k + 1] - edges[2 * k] for k in range(len(bars) + 1)]
    for k in range(1, len(bars)):
        if gaps[k] < -tol:
            (_, _, _, ia, a), (_, _, _, ib, b) = bars[k - 1], bars[k]
            raise ValueError(
                f"RCLayeredShell: meshes[{ia}] (angle={a.angle}) and "
                f"meshes[{ib}] (angle={b.angle}) overlap by {-gaps[k]:.6g}. "
                "Bar layers must not overlap: put each mesh at its own "
                "depth (e.g. the second direction one bar diameter inside "
                "the first)."
            )
    regions = [k for k, g in enumerate(gaps) if g > tol]
    if not regions:
        raise ValueError(
            "RCLayeredShell: the bar layers fill the whole thickness; no "
            "concrete is left. Check area_per_width (A_s / s) against h."
        )
    counts = dict(zip(
        regions,
        _apportion([gaps[k] for k in regions], max(n_concrete, len(regions))),
    ))

    rebar: dict[tuple[int, float], PlateRebar] = {}
    layers: list[ShellLayer] = []
    last_concrete = -1
    for k in range(len(gaps)):
        for _ in range(counts.get(k, 0)):
            last_concrete = len(layers)
            layers.append(ShellLayer(material=concrete, thickness=gaps[k] / counts[k]))
        if k < len(bars):
            mesh = bars[k][4]
            key = (id(mesh.material), float(mesh.angle))
            bar = rebar.get(key)
            if bar is None:
                bar = rebar[key] = PlateRebar(material=mesh.material, angle=mesh.angle)
            layers.append(ShellLayer(material=bar, thickness=mesh.area_per_width))

    # The topmost concrete layer absorbs the rounding (and any gap below
    # tol that was treated as contact), so the stack sums to h.
    rest = math.fsum(
        layer.thickness for j, layer in enumerate(layers) if j != last_concrete
    )
    layers[last_concrete] = ShellLayer(material=concrete, thickness=h - rest)

    n_layers = sum(counts.values())
    if n_layers < MIN_RC_CONCRETE_LAYERS:
        warnings.warn(
            f"RCLayeredShell: {n_layers} concrete layers. Each layer is "
            f"integrated at its mid-plane, which drops about 1/n**2 of the "
            f"bending stiffness; use 8-12 concrete layers for nonlinear RC "
            f"bending (n_concrete=).",
            CoarseShellLayeringWarning,
            stacklevel=2,
        )
    return LayeredShell(layers=tuple(layers))


# ---------------------------------------------------------------------------
# LadrunoShellModifier — ETABS-style stiffness modifiers (fork, ADR 91)
# ---------------------------------------------------------------------------

class ShellModifierNonlinearInnerWarning(UserWarning):
    """``LadrunoShellModifier`` wrapping something other than an elastic plate.

    Subclass of :class:`UserWarning` so it can be silenced per-call or
    promoted to an error in CI (the warn-as-contract idiom used by
    :class:`~apeGmsh.opensees.material.nd.SanisandIntegrationWarning`).

    The supported use is a **linear-elastic** plate section: the
    modifiers are a stiffness fiction (cracked-section factors), and the
    fork implements them as a congruence, driving the wrapped section at
    a *scaled* deformation ``S·e`` and scaling its resultants back by
    ``S`` (``scale[i] = sqrt(mod[i])``). For an elastic inner that is
    exactly equivalent to scaling the section stiffness. For a
    path-dependent inner it is not: the constitutive law integrates at a
    fictitious strain, so yield and damage arrive at the wrong place, and
    per-layer stresses recovered from the inner materials are the
    response at ``S·e`` rather than the physical layer response — no
    post-hoc scaling recovers the true values.

    Emit is never blocked: the fork accepts any order-8 plate section
    (ADR 91), so this stays a warning rather than a refusal.
    """


#: The nine modifier flags of ``section LadrunoShellModifier``, in the
#: order the ETABS OAPI area-modifier array reports them (its tenth
#: entry, ``weight``, has no fork-side counterpart — see the class
#: docstring).
SHELL_MODIFIER_FLAGS: tuple[str, ...] = (
    "f11", "f22", "f12", "m11", "m22", "m12", "v13", "v23", "mass",
)


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoShellModifier(Section):
    """``section LadrunoShellModifier`` — ETABS-style stiffness modifiers.

    A *decorator* section: it wraps any order-8 plate section and scales
    its tangent, resultants, and density. This is the cracked-section
    idiom of practice — ``f11 = f22 = f12 = 0.35`` on a shear wall per
    ACI 318-25 §6.6.3.1.1 — expressed without disturbing the wrapped
    section's own constitutive law.

    Fork primitive (Ladruno ADR 91); not available in stock OpenSees.

    Parameters
    ----------
    inner
        The wrapped plate section. **Use
        :class:`ElasticMembranePlateSection`** — that is the supported
        case. The fork accepts any order-8 section with the standard
        plate response codes (so :class:`LayeredShell` /
        :class:`LayeredShellFiberSection` are legal), but wrapping a
        path-dependent section raises
        :class:`ShellModifierNonlinearInnerWarning`; see that class for
        why the result is not physically meaningful.
    f11, f22, f12
        Membrane stiffness modifiers.
    m11, m22, m12
        Bending stiffness modifiers.
    v13, v23
        Transverse shear stiffness modifiers.
    mass
        Density modifier; scales the wrapped section's ``getRho()``.

    Notes
    -----
    Every flag defaults to ``1.0`` and only non-default flags are
    emitted, so an all-defaults wrap is a no-op — which lets a generator
    wrap unconditionally.

    Modifiers are applied as a congruence, ``D' = S·D·S`` with
    ``S = diag(√f11 … √v23)``, so the Poisson coupling term moves as
    ``√(f11·f22)``. When ``f11 == f22`` — every standard cracked-wall
    recipe — this is indistinguishable from scaling the whole membrane
    block. A diagonal-only rescale was rejected by ADR 91 §4 because it
    destroys positive definiteness at exactly these modifier values.

    ``Ep_mod=r`` on :class:`ElasticMembranePlateSection` is exactly
    equivalent to ``m11=m22=m12=v13=v23=r`` here; this section is
    strictly more expressive and is the preferred spelling.

    A modifier of exactly ``0.0`` is accepted (it is ETABS-legal) but
    leaves the section singular in that response mode; the fork warns
    once per ``section`` command. Negative modifiers are refused here
    and by the fork.

    There is deliberately **no weight modifier**: OpenSees derives the
    shell self-weight body force from the same ``getRho()`` that builds
    the mass matrix, so a weight flag could only alias ``mass``
    (ADR 91 §5). Scale self-weight at the load level instead.
    """

    inner: Section
    f11:  float = 1.0
    f22:  float = 1.0
    f12:  float = 1.0
    m11:  float = 1.0
    m22:  float = 1.0
    m12:  float = 1.0
    v13:  float = 1.0
    v23:  float = 1.0
    mass: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.inner, Section):
            raise TypeError(
                "LadrunoShellModifier: inner must be a Section primitive, "
                f"got {type(self.inner).__name__!r}."
            )
        for name in SHELL_MODIFIER_FLAGS:
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(
                    f"LadrunoShellModifier: {name} must be >= 0.0, "
                    f"got {value}."
                )
        if not isinstance(self.inner, ElasticMembranePlateSection):
            warnings.warn(
                f"LadrunoShellModifier wraps "
                f"{type(self.inner).__name__!r}, which is not a "
                f"linear-elastic plate section. Stiffness modifiers are "
                f"only meaningful there: the fork drives the wrapped "
                f"section at a scaled deformation S*e "
                f"(S = diag(sqrt(f11)..sqrt(v23))), so a path-dependent "
                f"section yields and damages at a fictitious strain, and "
                f"per-layer results read from its materials are the "
                f"response at S*e, not the physical layer response. Use "
                f"ElasticMembranePlateSection here, or model the "
                f"stiffness reduction constitutively instead.",
                ShellModifierNonlinearInnerWarning,
                stacklevel=3,
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        inner_tag = resolve_mat_tag(emitter, self.inner)
        params: list[float | str] = [inner_tag]
        for name in SHELL_MODIFIER_FLAGS:
            value = getattr(self, name)
            if value != 1.0:
                params += [f"-{name}", value]
        emitter.section("LadrunoShellModifier", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.inner,)
