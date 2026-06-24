"""``element LadrunoEmbeddedNode`` — emit-grammar builder (Ladruno fork).

``LadrunoEmbeddedNode`` (``ELE_TAG`` **33006**) ties **one** constrained
node to a solid host element's nodes through precomputed shape-function
weights, so a node embeds in a **non-matching** host mesh. It is the
isotropic sibling of :mod:`~apeGmsh.opensees.element.embedded_rebar` over
the shared ``LadrunoEmbeddedKernel`` — a pure **U-translational** coupling
(rotations left free) with optional augmented-Lagrangian enforcement,
explicit-safe bipenalty, and **g0 stress-free birth** (a node added onto an
already-deformed host activates at zero force).

Like ``LadrunoEmbeddedRebar`` / ``ASDEmbeddedNodeElement`` it is
**per-instance** (one element per constrained node, with explicit nodes +
weights produced at build time by the ``g.embed`` generator's guarded
inverse map), so this module is the **single source of truth for the emit
grammar**.

The grammar mirrors ``OPS_LadrunoEmbeddedNode.cpp``::

    element LadrunoEmbeddedNode tag cNode
        {nHost h1..hN | -host eleTag}
        {-shape N1..NN | -xi x1..x_ndm}
        [-k {Ku | auto}] [-kAlpha a]
        [-enforce {penalty | al}]
        [-bipenalty {-dtcr dt | -wcap beta}]
        [-absolute]

The experimental ``-rot`` / ``-pressure`` / ``-normal`` / ``-corot`` modes
are **not** emitted by ``g.embed`` (U-only core only). Fork-only: emission
produces deck text on any build; the element bites only at ``ops.run()``.
"""
from __future__ import annotations

from collections.abc import Sequence


__all__ = ["embedded_node_args"]

# OpenSees command token + live class tag (SRC/classTags.h, ladruno
# private >=33000 band — do not hardcode the dead pre-300 value).
ELEMENT_TYPE = "LadrunoEmbeddedNode"
ELE_TAG = 33006


def embedded_node_args(
    *,
    cnode: int,
    host_ele: int | None = None,
    host_nodes: Sequence[int] | None = None,
    xi: Sequence[float] | None = None,
    shape: Sequence[float] | None = None,
    k: float | str | None = None,
    k_alpha: float | None = None,
    enforce: str = "penalty",
    bipenalty: bool = False,
    dtcr: float | None = None,
    wcap: float | None = None,
    staged: bool = True,
) -> list[int | float | str]:
    """Build the positional argument list **after** ``tag`` for one
    ``element LadrunoEmbeddedNode`` call.

    The returned list is ``[cnode, <host>, <weights>, …optional…]`` — pass
    it as ``emitter.element("LadrunoEmbeddedNode", tag, *args)``.

    Exactly one host spec (``host_ele`` xor ``host_nodes``) and exactly one
    weight spec (``xi`` xor ``shape``) must be supplied. ``xi`` and
    ``k="auto"`` and ``wcap`` require the ``host_ele`` form (the host
    element is queried). ``staged=True`` (default) emits g0 stress-free
    birth (no ``-absolute``); ``staged=False`` emits ``-absolute``.
    """
    # -- host spec: exactly one of host_ele / host_nodes ---------------------
    if (host_ele is None) == (host_nodes is None):
        raise ValueError(
            "LadrunoEmbeddedNode: supply exactly one of host_ele (-host) or "
            "host_nodes (nHost h1..hN)"
        )

    # -- weights: exactly one of xi / shape ---------------------------------
    if (xi is None) == (shape is None):
        raise ValueError(
            "LadrunoEmbeddedNode: supply exactly one of xi (-xi, host-queried)"
            " or shape (-shape, explicit weights)"
        )
    if xi is not None and host_ele is None:
        raise ValueError(
            "LadrunoEmbeddedNode: xi (-xi) requires the host_ele (-host) form "
            "(no host element to query); use shape (-shape) instead"
        )
    if shape is not None and host_nodes is not None and len(shape) != len(host_nodes):
        raise ValueError(
            f"LadrunoEmbeddedNode: shape has {len(shape)} weights but "
            f"host_nodes has {len(host_nodes)} nodes"
        )

    # -- k auto needs the host form -----------------------------------------
    if k == "auto" and host_ele is None:
        raise ValueError(
            "LadrunoEmbeddedNode: k='auto' (-k auto) reads the host stiffness "
            "and requires the host_ele (-host) form"
        )
    if k_alpha is not None and k != "auto":
        raise ValueError(
            "LadrunoEmbeddedNode: k_alpha (-kAlpha) only applies to k='auto'"
        )

    # -- enforce ------------------------------------------------------------
    if enforce not in ("penalty", "al"):
        raise ValueError(
            f"LadrunoEmbeddedNode: enforce must be 'penalty' or 'al', got "
            f"{enforce!r}"
        )

    # -- bipenalty: explicit-only, exactly one budget, penalty-gated --------
    if bipenalty:
        if (dtcr is None) == (wcap is None):
            raise ValueError(
                "LadrunoEmbeddedNode: bipenalty requires exactly one budget "
                "— dtcr (-dtcr) or wcap (-wcap)"
            )
        if enforce != "penalty":
            raise ValueError(
                "LadrunoEmbeddedNode: bipenalty is gated on enforce='penalty' "
                "(it is auto-disabled under augmented Lagrangian)"
            )
        if wcap is not None and host_ele is None:
            raise ValueError(
                "LadrunoEmbeddedNode: wcap (-wcap) reads the host frequency "
                "and requires the host_ele (-host) form"
            )
    elif dtcr is not None or wcap is not None:
        raise ValueError(
            "LadrunoEmbeddedNode: dtcr/wcap are only valid with bipenalty=True"
        )

    # -- assemble in the exact parser order ---------------------------------
    args: list[int | float | str] = [cnode]

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

    if k is not None:
        args += ["-k", k]
    if k_alpha is not None:
        args += ["-kAlpha", k_alpha]

    if enforce != "penalty":
        args += ["-enforce", enforce]

    if bipenalty:
        args.append("-bipenalty")
        if dtcr is not None:
            args += ["-dtcr", dtcr]
        else:
            assert wcap is not None
            args += ["-wcap", wcap]

    # g0 stress-free birth is ON by default in the fork; -absolute opts out.
    if not staged:
        args.append("-absolute")

    return args
