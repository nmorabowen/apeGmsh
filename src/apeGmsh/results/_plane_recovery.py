"""Per-element out-of-plane stress/strain recovery from the model broker.

For 2-D continuum results only the three in-plane tensor components are
stored. The physically-correct out-of-plane component depends on the
element's plane idealization, which OpenSees records:

* **Plane stress** element → σ_zz = 0 exactly (any material); the
  conjugate ε_zz = -ν/(1-ν)(ε_xx+ε_yy).
* **Plane strain** element → ε_zz = 0 exactly; the conjugate
  σ_zz = ν(σ_xx+σ_yy) (exact for linear elasticity — for a nonlinear
  material the true σ_zz is the material's internal σ33, which stock
  OpenSees does not record, so this is an elastic estimate).

This module reads each element's plane type + Poisson's ratio from the
``OpenSeesModel`` broker (``model.elements()`` / ``model.materials()``)
and synthesizes the missing ``*_zz`` column *per element*, so a model
that mixes plane-stress and plane-strain elements is handled correctly.
The result is injected into the columns dict the derived-scalar layer
already consumes — so if a file ever *does* store ``stress_zz`` (e.g. a
fork that records the material's σ33), that real value is used and this
reconstruction is skipped.

Everything here is best-effort and defensive: any element/material it
cannot parse contributes no recovery (that element falls back to a zero
out-of-plane component, i.e. plane stress for σ / plane strain for ε).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Materials whose Poisson's ratio is a direct positional param.
_NU_AT_PARAM1 = frozenset({
    "ElasticIsotropic",
    "ElasticIsotropicPlaneStrain2D",
    "ElasticIsotropicPlaneStress2D",
    "ASDConcrete3D",
})
# Materials parameterised by (K, G) at params[0], params[1].
_KG_MATERIALS = frozenset({
    "J2Plasticity", "DruckerPrager", "LadrunoJ2", "LadrunoJ2Finite",
})
# 2-D elements whose args are [thickness, plane_type, mat_tag, ...] after
# the connectivity prefix is stripped on a from_h5 load.
_POSITIONAL_PLANE_ELEMENTS = frozenset({
    "quad", "tri31", "tri6n", "BezierTri6",
})
# 2-D Ladruno elements: args are [mat_tag, ...flags]; plane_type is the
# value after a "-type" flag (absent → the "PlaneStrain" default).
_FLAG_PLANE_ELEMENTS = frozenset({"LadrunoQuad", "LadrunoCST"})

_CACHE: dict[int, dict[int, tuple[Optional[str], Optional[float]]]] = {}


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_plane(text: Any) -> Optional[str]:
    s = str(text).lower()
    if "strain" in s:
        return "PlaneStrain"
    if "stress" in s:
        return "PlaneStress"
    return None


def _material_nu(rec: Any) -> Optional[float]:
    """Poisson's ratio for an nD material record, or ``None``."""
    token = getattr(rec, "type_token", "")
    params = tuple(getattr(rec, "params", ()) or ())
    try:
        if token in _NU_AT_PARAM1 and len(params) >= 2:
            return _as_float(params[1])
        if token in _KG_MATERIALS and len(params) >= 2:
            k, g = _as_float(params[0]), _as_float(params[1])
            if k is not None and g is not None and (3.0 * k + g) != 0.0:
                return (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))
        # ASDPlastic-family (MohrCoulomb, generic): scan for the token pair.
        for i, p in enumerate(params[:-1]):
            if isinstance(p, str) and p == "PoissonsRatio":
                return _as_float(params[i + 1])
    except Exception:
        return None
    return None


def _element_plane_and_mat(rec: Any) -> tuple[Optional[str], Optional[int]]:
    """``(plane_type, mat_tag)`` for a 2-D element record, else ``(None, None)``."""
    token = getattr(rec, "type_token", "")
    args = tuple(getattr(rec, "args", ()) or ())
    try:
        if token in _POSITIONAL_PLANE_ELEMENTS and len(args) >= 3:
            return _normalize_plane(args[1]), int(args[2])
        if token in _FLAG_PLANE_ELEMENTS and len(args) >= 1:
            mat_tag = int(args[0])
            plane = "PlaneStrain"     # Ladruno default when no -type flag
            for i, a in enumerate(args[:-1]):
                if a == "-type":
                    plane = _normalize_plane(args[i + 1]) or plane
                    break
            return plane, mat_tag
    except Exception:
        return None, None
    return None, None


def plane_recovery_map(
    model: Any,
) -> dict[int, tuple[Optional[str], Optional[float]]]:
    """``{fem_eid: (plane_type, nu)}`` for the model's 2-D elements.

    Cached per model object. Empty when the model carries no parseable
    element/material records (e.g. a synthetic results file with no
    ``/opensees`` zone) — callers then fall back to a zero out-of-plane
    component.
    """
    key = id(model)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    out: dict[int, tuple[Optional[str], Optional[float]]] = {}
    try:
        nd_materials = model.materials(family="nd")
    except Exception:
        nd_materials = ()
    nu_by_tag: dict[int, float] = {}
    for rec in nd_materials or ():
        nu = _material_nu(rec)
        tag = getattr(rec, "tag", None)
        if nu is not None and tag is not None:
            nu_by_tag[int(tag)] = nu

    try:
        elements = model.elements()
    except Exception:
        elements = ()
    for rec in elements or ():
        plane, mat_tag = _element_plane_and_mat(rec)
        if plane is None:
            continue
        fem_eid = getattr(rec, "fem_eid", None)
        if fem_eid is None or int(fem_eid) < 0:
            continue
        nu = nu_by_tag.get(int(mat_tag)) if mat_tag is not None else None
        out[int(fem_eid)] = (plane, nu)

    _CACHE[key] = out
    return out


def inject_out_of_plane(
    columns: dict[str, np.ndarray], element_index: np.ndarray, *,
    prefix: str, model: Any,
) -> bool:
    """Synthesize the per-element ``{prefix}_zz`` column from the model.

    Mutates ``columns`` in place, adding ``stress_zz`` / ``strain_zz``
    computed per element from its plane type + ν. No-op (returns
    ``False``) when the ``_zz`` column already exists (3-D data or a
    fork that recorded it), the in-plane columns are missing, or the
    model resolves no 2-D element — the caller then leaves the
    out-of-plane component at zero. Returns ``True`` if a column was
    injected.
    """
    zz_key = f"{prefix}_zz"
    stored_zz = columns.get(zz_key)
    if stored_zz is not None:
        stored_zz = np.asarray(stored_zz, dtype=np.float64)
        if np.isfinite(stored_zz).all():
            return False      # fully recorded real σ_zz → use verbatim
        # else: partially recorded (NaN sentinel where the material could
        # not supply it) — fall through and fill only the NaN entries.
    xx = columns.get(f"{prefix}_xx")
    yy = columns.get(f"{prefix}_yy")
    if xx is None or yy is None:
        return False
    pmap = plane_recovery_map(model)
    if not pmap:
        return False

    eidx = np.asarray(element_index, dtype=np.int64)
    uniq, inv = np.unique(eidx, return_inverse=True)
    # Per-unique-element: code (1=plane strain, 2=plane stress, 0=other) + ν.
    code_u = np.zeros(uniq.size, dtype=np.int8)
    nu_u = np.full(uniq.size, np.nan, dtype=np.float64)
    for i, e in enumerate(uniq):
        plane, nu = pmap.get(int(e), (None, None))
        if plane == "PlaneStrain":
            code_u[i] = 1
        elif plane == "PlaneStress":
            code_u[i] = 2
        if nu is not None:
            nu_u[i] = nu
    code_gp = code_u[inv]          # (N,)
    nu_gp = nu_u[inv]              # (N,)
    has_nu = np.isfinite(nu_gp)

    in_plane_sum = np.asarray(xx, dtype=np.float64) + np.asarray(yy, dtype=np.float64)
    if prefix == "stress":
        # Plane strain: σ_zz = ν(σ_xx+σ_yy). Plane stress / other: 0.
        factor = np.where((code_gp == 1) & has_nu, nu_gp, 0.0)
        zz = factor * in_plane_sum
    else:  # strain
        # Plane stress: ε_zz = -ν/(1-ν)(ε_xx+ε_yy). Plane strain / other: 0.
        safe = has_nu & (nu_gp != 1.0)
        coef = np.where((code_gp == 2) & safe, -nu_gp / (1.0 - nu_gp), 0.0)
        zz = coef * in_plane_sum

    if stored_zz is not None:
        # Keep the recorded (finite) σ_zz; fill only the NaN sentinels.
        columns[zz_key] = np.where(np.isfinite(stored_zz), stored_zz, zz)
    else:
        columns[zz_key] = zz
    return True
