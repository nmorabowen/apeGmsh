# Gmsh under the hood

This page tells you what apeGmsh delegates to [Gmsh](https://gmsh.info) and
where to learn the Gmsh-level details when you need them — so you know which
manual to open when a question goes deeper than this site.

apeGmsh is a wrapper, not a re-implementation. Geometry lives in Gmsh's
OpenCASCADE kernel; meshing is Gmsh's meshers behind
[`g.mesh`](meshing.md); physical groups are Gmsh physical groups; the
`.msh` format is Gmsh's own. What apeGmsh adds is the structural-FEM
workflow on top — names instead of tags, declare-then-resolve physics, the
frozen snapshot, the solver bridge. The wrapper is deliberately thin and
non-magical: no mesh option is changed behind your back, and the full
`gmsh.*` Python API remains available inside any session for anything
without a first-class apeGmsh name.

That thinness means Gmsh's own documentation stays relevant to you:

- the [Gmsh reference manual](https://gmsh.info/doc/texinfo/gmsh.html) —
  every mesh option, algorithm, and file-format detail;
- the [Gmsh Python API](https://gitlab.onelab.info/gmsh/gmsh/blob/master/api/gmsh.py)
  — the surface you reach through when apeGmsh doesn't wrap a call;
- the [Gmsh tutorials](https://gmsh.info/doc/texinfo/gmsh.html#Tutorial) —
  meshing concepts (transfinite, fields, partitioning) in their original
  habitat.

A practical rule for mixed code: do through apeGmsh what apeGmsh names
(geometry, sizing, groups, selection — they feed the metadata the rest of
the workflow depends on), and reach through to raw `gmsh.*` only for
option tweaks and queries that change no topology. Raw calls that create
or delete entities bypass the label registry, and names are what the whole
downstream workflow runs on.

---

*Next: [Selection & queries](selection.md).*
