"""
Plate / shell sections — typed primitives for OpenSees plate-bending
section commands.

Four section types live here:

* :class:`ElasticMembranePlateSection` — single-layer linear-elastic
  plate (membrane + bending) used by ``ShellMITC4``, ``ShellDKGQ``,
  ``ASDShellQ4``.
* :class:`LayeredShell` — stacked nDMaterial layers (a thin-shell
  composite section) for ``ShellMITC4`` and friends.
* :class:`LayeredShellFiberSection` — same shape as ``LayeredShell``
  but routed through OpenSees's ``LayeredShellFiberSection`` C++
  class (the in-tree fiber-based stack).
* :class:`LadrunoShellModifier` — a *decorator* that applies
  ETABS-style stiffness modifiers to any of the above (fork-only,
  Ladruno ADR 91).

The ``LayeredShell*`` sections compose a tuple of nDMaterials and
declare them via :meth:`dependencies`; their ``_emit`` references
resolved nDMaterial tags via the same closure-based resolver Fiber
uses (see :mod:`section.fiber`).

The OpenSees commands per the manual:

* ``section ElasticMembranePlateSection $tag $E $nu $h <$rho>
  <$Ep_mod>``
* ``section LayeredShell $tag $nLayers $matTag1 $h1 ... $matTagN $hN``
* ``section LayeredShellFiberSection $tag $nLayers $matTag1 $h1 ...
  $matTagN $hN``
* ``section LadrunoShellModifier $tag $innerSecTag <-f11 v> ...
  <-mass v>``
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import NDMaterial, Primitive, Section
from ._tag_resolver import resolve_mat_tag

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "SHELL_MODIFIER_FLAGS",
    "ShellModifierNonlinearInnerWarning",
    "ElasticMembranePlateSection",
    "LadrunoShellModifier",
    "LayeredShell",
    "LayeredShellFiberSection",
    "ShellLayer",
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
        if self.thickness <= 0:
            raise ValueError(
                f"ShellLayer: thickness must be > 0, got {self.thickness}."
            )


def _validate_layers(
    cls_name: str, layers: tuple[ShellLayer, ...]
) -> None:
    if not layers:
        raise ValueError(
            f"{cls_name}: at least one ShellLayer is required."
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
        layer (bottom to top). At least one layer is required.

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
    """``section LayeredShellFiberSection`` — stacked nDMaterial layers
    (fiber-based variant).

    Same input shape as :class:`LayeredShell`; emits the OpenSees
    ``LayeredShellFiberSection`` type token instead. The C++ class
    behind it integrates the layers as fibers through the thickness.

    Parameters
    ----------
    layers
        Tuple of :class:`ShellLayer` describing each through-thickness
        layer.
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
        emitter.section("LayeredShellFiberSection", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return _layer_dependencies(self.layers)


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
