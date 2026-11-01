"""``element LadrunoEmbeddedRebar`` — emit-grammar builder (Ladruno fork).

``LadrunoEmbeddedRebar`` (``ELE_TAG`` **33005**) ties **one** discrete
rebar node to a solid host element's nodes through precomputed
shape-function weights, so a reinforcement mesh embeds in a
**non-matching** concrete mesh. It is a pure **coupling** element (the
bar's own axial stiffness lives on a separate ``corotTruss``); it only
enforces the bar↔host attachment, with either perfect bond or a 1D
:class:`~apeGmsh.opensees.material.uniaxial.LadrunoBondSlip` τ–s law in
the axial direction. It is **Mode P (penalty)** — stiffness only, so it
works in both implicit and explicit.

Unlike the mesh-fanned typed elements (``Truss``, ``BezierTri6``, …),
``LadrunoEmbeddedRebar`` is **per-instance**: one element per rebar
node, with explicit nodes + weights produced at build time by the
``g.reinforce`` generator's guarded inverse map (it is the sibling of
``ASDEmbeddedNodeElement``, which is likewise resolver-produced, not a
``ops.element.*`` primitive). This module is therefore the **single
source of truth for the emit grammar** — a pure builder that turns
typed inputs into the positional argument list the OpenSees parser
expects — shared by the generator (and, later, the H5 round-trip).

The grammar mirrors ``OPS_LadrunoEmbeddedRebar.cpp`` exactly::

    element LadrunoEmbeddedRebar tag rebarNode
        {nHost h1..hN | -host eleTag}
        {-shape N1..NN | -xi x1..x_ndm}
        -dir dx dy [dz]
        ( -bond matTag [-bondScale bs] | -perfect kAxial )
        [-kt {kt | auto}] [-ktAlpha a]
        [-corot {-xiB b1..b_ndm | -shapeB N1..NN}]
        [-enforce {penalty | al}]
        [-bipenalty {-dtcr dt | -wcap beta}]

Fork-only: emission produces deck text on any build; the element is
unavailable on stock ``openseespy`` and bites only at ``ops.run()``.
"""
from __future__ import annotations

from collections.abc import Sequence


__all__ = ["embedded_rebar_args"]

# OpenSees command token + live class tag (SRC/classTags.h, ladruno
# private >=33000 band — the dead pre-300 value is gone, do not hardcode it).
ELEMENT_TYPE = "LadrunoEmbeddedRebar"
ELE_TAG = 33005


def embedded_rebar_args(
    *,
    rebar_node: int,
    direction: Sequence[float],
    host_ele: int | None = None,
    host_nodes: Sequence[int] | None = None,
    xi: Sequence[float] | None = None,
    shape: Sequence[float] | None = None,
    perfect: float | None = None,
    bond: int | None = None,
    bond_scale: float | None = None,
    kt: float | str | None = None,
    kt_alpha: float | None = None,
    corot: bool = False,
    xi_b: Sequence[float] | None = None,
    shape_b: Sequence[float] | None = None,
    enforce: str = "penalty",
    bipenalty: bool = False,
    dtcr: float | None = None,
    wcap: float | None = None,
) -> list[int | float | str]:
    """Build the positional argument list **after** ``tag`` for one
    ``element LadrunoEmbeddedRebar`` call.

    The returned list is ``[rebar_node, <host>, <weights>, -dir…,
    <axial>, …optional…]`` — pass it as
    ``emitter.element("LadrunoEmbeddedRebar", tag, *args)``.

    Exactly one host spec (``host_ele`` xor ``host_nodes``), exactly one
    weight spec (``xi`` xor ``shape``), and exactly one axial law
    (``perfect`` xor ``bond``) must be supplied. ``xi`` and ``kt="auto"``
    and ``wcap`` require the ``host_ele`` form (the host element is
    queried). See the module docstring for the grammar.
    """
    ndm = len(direction)
    if ndm not in (2, 3):
        raise ValueError(
            f"LadrunoEmbeddedRebar: direction must have 2 or 3 components, "
            f"got {ndm} ({direction!r})"
        )

    # -- host spec: exactly one of host_ele / host_nodes ---------------------
    if (host_ele is None) == (host_nodes is None):
        raise ValueError(
            "LadrunoEmbeddedRebar: supply exactly one of host_ele (-host) or "
            "host_nodes (nHost h1..hN)"
        )

    # -- weights: exactly one of xi / shape ---------------------------------
    if (xi is None) == (shape is None):
        raise ValueError(
            "LadrunoEmbeddedRebar: supply exactly one of xi (-xi, host-queried)"
            " or shape (-shape, explicit weights)"
        )
    if xi is not None:
        if host_ele is None:
            raise ValueError(
                "LadrunoEmbeddedRebar: xi (-xi) requires the host_ele (-host) "
                "form (no host element to query); use shape (-shape) instead"
            )
        if len(xi) != ndm:
            raise ValueError(
                f"LadrunoEmbeddedRebar: xi must have ndm={ndm} components "
                f"(matching direction), got {len(xi)}"
            )
    if shape is not None and host_nodes is not None and len(shape) != len(host_nodes):
        raise ValueError(
            f"LadrunoEmbeddedRebar: shape has {len(shape)} weights but "
            f"host_nodes has {len(host_nodes)} nodes"
        )

    # -- axial law: exactly one of perfect / bond ---------------------------
    if (perfect is None) == (bond is None):
        raise ValueError(
            "LadrunoEmbeddedRebar: supply exactly one axial law — perfect "
            "(-perfect kAxial) or bond (-bond matTag)"
        )
    if bond_scale is not None and bond is None:
        raise ValueError(
            "LadrunoEmbeddedRebar: bond_scale (-bondScale) is only valid with "
            "the bond (-bond) axial law"
        )

    # -- kt auto needs the host form ----------------------------------------
    if kt == "auto" and host_ele is None:
        raise ValueError(
            "LadrunoEmbeddedRebar: kt='auto' (-kt auto) reads the host "
            "stiffness and requires the host_ele (-host) form"
        )
    if kt_alpha is not None and kt != "auto":
        raise ValueError(
            "LadrunoEmbeddedRebar: kt_alpha (-ktAlpha) only applies to "
            "kt='auto'"
        )

    # -- enforce ------------------------------------------------------------
    if enforce not in ("penalty", "al"):
        raise ValueError(
            f"LadrunoEmbeddedRebar: enforce must be 'penalty' or 'al', got "
            f"{enforce!r}"
        )

    # -- corot needs exactly one point-B spec -------------------------------
    if corot:
        if (xi_b is None) == (shape_b is None):
            raise ValueError(
                "LadrunoEmbeddedRebar: corot requires exactly one point-B "
                "spec — xi_b (-xiB) or shape_b (-shapeB)"
            )
        if xi_b is not None and host_ele is None:
            raise ValueError(
                "LadrunoEmbeddedRebar: xi_b (-xiB) requires the host_ele "
                "(-host) form; use shape_b (-shapeB) otherwise"
            )
    elif xi_b is not None or shape_b is not None:
        raise ValueError(
            "LadrunoEmbeddedRebar: xi_b/shape_b are only valid with corot=True"
        )

    # -- bipenalty: explicit-only, exactly one budget, penalty-gated --------
    if bipenalty:
        if (dtcr is None) == (wcap is None):
            raise ValueError(
                "LadrunoEmbeddedRebar: bipenalty requires exactly one budget "
                "— dtcr (-dtcr) or wcap (-wcap)"
            )
        if enforce != "penalty":
            raise ValueError(
                "LadrunoEmbeddedRebar: bipenalty is gated on enforce='penalty'"
                " (it is auto-disabled under augmented Lagrangian)"
            )
        if wcap is not None and host_ele is None:
            raise ValueError(
                "LadrunoEmbeddedRebar: wcap (-wcap) reads the host frequency "
                "and requires the host_ele (-host) form"
            )
    elif dtcr is not None or wcap is not None:
        raise ValueError(
            "LadrunoEmbeddedRebar: dtcr/wcap are only valid with bipenalty=True"
        )

    # -- assemble in the exact parser order ---------------------------------
    args: list[int | float | str] = [rebar_node]

    if host_ele is not None:
        args += ["-host", host_ele]
    else:
        assert host_nodes is not None
        args += [len(host_nodes), *host_nodes]

    if xi is not None:
        args += ["-xi", *xi]
    else:
        assert shape is not None
        args += ["-shape", *shape]

    args += ["-dir", *direction]

    if perfect is not None:
        args += ["-perfect", perfect]
    else:
        assert bond is not None
        args += ["-bond", bond]
        if bond_scale is not None:
            args += ["-bondScale", bond_scale]

    if kt is not None:
        args += ["-kt", kt]
    if kt_alpha is not None:
        args += ["-ktAlpha", kt_alpha]

    if corot:
        args.append("-corot")
        if xi_b is not None:
            args += ["-xiB", *xi_b]
        else:
            assert shape_b is not None
            args += ["-shapeB", *shape_b]

    if enforce != "penalty":
        args += ["-enforce", enforce]

    if bipenalty:
        args.append("-bipenalty")
        if dtcr is not None:
            args += ["-dtcr", dtcr]
        else:
            assert wcap is not None
            args += ["-wcap", wcap]

    return args
