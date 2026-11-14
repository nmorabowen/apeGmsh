"""Gmsh-free FEMData builders and small structured meshes for the MPM seeding tests."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite

#: family -> (gmsh code, gmsh name, dim, order, npe)
FAMILIES: dict[str, tuple[int, str, int, int, int]] = {
    "line2": (1, "Line 2", 1, 1, 2),
    "tri3": (2, "Triangle 3", 2, 1, 3),
    "quad4": (3, "Quadrilateral 4", 2, 1, 4),
    "tet4": (4, "Tetrahedron 4", 3, 1, 4),
    "hex8": (5, "Hexahedron 8", 3, 1, 8),
    "prism6": (6, "Prism 6", 3, 1, 6),
    "tri6": (9, "Triangle 6", 2, 2, 6),
}

Element = tuple[str, Sequence[int]]  # (family, connectivity)


def make_fem(nodes: Mapping[int, Sequence[float]],
             bodies: Mapping[str, Iterable[Element]]) -> FEMData:
    """A FEMData whose physical group ``name`` holds ``bodies[name]``'s elements.

    Element ids are numbered from 1 in the order given; coordinates are padded to 3-D.
    """
    node_ids = np.array(sorted(nodes), dtype=np.int64)
    xyz = np.zeros((node_ids.size, 3))
    for i, nid in enumerate(node_ids):
        c = list(nodes[int(nid)])
        xyz[i, :len(c)] = c
    row = {int(n): i for i, n in enumerate(node_ids)}

    by_family: dict[str, tuple[list[int], list[Sequence[int]]]] = {}
    pg: dict[tuple[int, int], dict[str, object]] = {}
    eid = 0
    for tag, (name, elems_in) in enumerate(bodies.items(), start=1):
        elems = list(elems_in)
        ids: list[int] = []
        dims = set()
        for family, conn in elems:
            eid += 1
            ids.append(eid)
            by_family.setdefault(family, ([], []))
            by_family[family][0].append(eid)
            by_family[family][1].append(list(conn))
            dims.add(FAMILIES[family][2])
        pnodes = sorted({int(n) for _, c in elems for n in c})
        pg[(max(dims), tag)] = {
            "name": name,
            "node_ids": np.array(pnodes, dtype=np.int64),
            "node_coords": xyz[[row[n] for n in pnodes]],
            "element_ids": np.array(ids, dtype=np.int64),
        }

    groups = {}
    infos = []
    for family, (ids, conns) in by_family.items():
        code, gname, dim, order, npe = FAMILIES[family]
        info = make_type_info(code=code, gmsh_name=gname, dim=dim, order=order, npe=npe,
                              count=len(ids))
        infos.append(info)
        groups[code] = ElementGroup(element_type=info, ids=np.array(ids, dtype=np.int64),
                                    connectivity=np.array(conns, dtype=np.int64))
    physical = PhysicalGroupSet(pg)
    labels = LabelSet({})
    node_comp = NodeComposite(node_ids=node_ids, node_coords=xyz, physical=physical,
                              labels=labels)
    elem_comp = ElementComposite(groups=groups, physical=physical, labels=labels,
                                 constraints=None, loads=[])
    info = MeshInfo(n_nodes=int(node_ids.size), n_elems=eid, bandwidth=0, types=infos)
    return FEMData(nodes=node_comp, elements=elem_comp, info=info)


def grid_nodes(shape: Sequence[int], lo: Sequence[float], hi: Sequence[float],
               first: int = 1) -> tuple[dict[int, list[float]], np.ndarray]:
    """Structured node grid; returns ``(nodes, ids)`` with ``ids[i, j(, k)]``."""
    axes = [np.linspace(a, b, m + 1) for a, b, m in zip(lo, hi, shape)]
    ids = np.arange(first, first + int(np.prod([m + 1 for m in shape]))).reshape(
        [m + 1 for m in shape], order="F")
    nodes = {}
    for idx in itertools.product(*[range(m + 1) for m in shape]):
        nodes[int(ids[idx])] = [float(axes[d][idx[d]]) for d in range(len(shape))]
    return nodes, ids


def quads(ids: np.ndarray) -> list[Element]:
    out: list[Element] = []
    for i in range(ids.shape[0] - 1):
        for j in range(ids.shape[1] - 1):
            out.append(("quad4", [ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]]))
    return out


def tris(ids: np.ndarray) -> list[Element]:
    """Each quad split by its (i,j)-(i+1,j+1) diagonal, counter-clockwise."""
    out: list[Element] = []
    for i in range(ids.shape[0] - 1):
        for j in range(ids.shape[1] - 1):
            a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            out += [("tri3", [a, b, c]), ("tri3", [a, c, d])]
    return out


def hexes(ids: np.ndarray) -> list[Element]:
    out: list[Element] = []
    for i, j, k in itertools.product(*[range(m - 1) for m in ids.shape]):
        out.append(("hex8", [ids[i, j, k], ids[i + 1, j, k], ids[i + 1, j + 1, k],
                             ids[i, j + 1, k], ids[i, j, k + 1], ids[i + 1, j, k + 1],
                             ids[i + 1, j + 1, k + 1], ids[i, j + 1, k + 1]]))
    return out


def tets(ids: np.ndarray, nodes: Mapping[int, Sequence[float]]) -> list[Element]:
    """Kuhn split of every hex into 6 tets along its (0,0,0)-(1,1,1) diagonal (conforming),
    each ordered to a positive volume."""
    out: list[Element] = []
    for i, j, k in itertools.product(*[range(m - 1) for m in ids.shape]):
        for perm in itertools.permutations(range(3)):
            cur = [0, 0, 0]
            path = [ids[i, j, k]]
            for axis in perm:
                cur[axis] += 1
                path.append(ids[i + cur[0], j + cur[1], k + cur[2]])
            out.append(("tet4", positive_tet(path, nodes)))
    return out


def positive_tet(conn: Sequence[int], nodes: Mapping[int, Sequence[float]]) -> list[int]:
    p = [np.asarray(nodes[int(n)], dtype=float) for n in conn]
    det = np.linalg.det(np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], axis=1))
    c = [int(n) for n in conn]
    return c if det > 0 else [c[0], c[2], c[1], c[3]]
