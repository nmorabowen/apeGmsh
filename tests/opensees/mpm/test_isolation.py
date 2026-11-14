"""Charter P3/P9 for ``apeGmsh.opensees._mpm``: no gmsh, nothing outside ``apeGmsh.mesh``.

``import apeGmsh`` itself imports gmsh (its ``__init__`` builds the session API), so
"import the package, then check ``sys.modules``" cannot isolate the package: any import of
``apeGmsh.opensees._mpm`` runs the parent ``__init__`` first. The proof is therefore made
of two parts:

* a fresh interpreter in which ``gmsh`` and ``apeGmsh`` are both made unimportable loads
  the ``_mpm`` directory as a stand-alone package, seeds a duck-typed mesh and writes a
  particle set: so the package needs neither at runtime;
* every import statement in the package is stdlib, numpy, h5py, relative, or
  ``apeGmsh.mesh`` under ``TYPE_CHECKING``.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import apeGmsh.opensees._mpm as mpm_pkg

PKG = Path(mpm_pkg.__file__).resolve().parent
ALLOWED_TOP = {"__future__", "collections", "dataclasses", "datetime", "math", "os",
               "pathlib", "re", "typing", "uuid", "h5py", "numpy"}


def test_package_runs_with_gmsh_and_apegmsh_unimportable(tmp_path):
    script = textwrap.dedent(f"""
        import importlib.util, sys
        from types import SimpleNamespace
        import numpy as np
        sys.modules["gmsh"] = None      # any "import gmsh" now raises ImportError
        sys.modules["apeGmsh"] = None   # and so does any apeGmsh import
        spec = importlib.util.spec_from_file_location(
            "mpm_isolated", {str(PKG / "__init__.py")!r},
            submodule_search_locations=[{str(PKG)!r}])
        mpm = importlib.util.module_from_spec(spec)
        sys.modules["mpm_isolated"] = mpm
        spec.loader.exec_module(mpm)

        group = SimpleNamespace(element_type=SimpleNamespace(name="quad4"),
                                ids=np.array([1]), connectivity=np.array([[1, 2, 3, 4]]))
        fem = SimpleNamespace(
            nodes=SimpleNamespace(ids=np.array([1, 2, 3, 4]),
                                  coords=np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0],
                                                   [0, 1, 0]])),
            elements=SimpleNamespace(select=lambda pg: SimpleNamespace(
                groups=lambda: [group])))
        s = mpm.seed_lattice(fem, mpm.MPMGrid((0., 0.), (0.5, 0.5), (2, 2)),
                             [mpm.MPMBody("b", "b", 1.0)], n=2, thickness=1.0)
        assert s.num_particles == 16, s.num_particles
        out = {str(tmp_path / "iso.particles.h5")!r}
        mpm.write_particle_set(mpm.preflight_particle_set(
            out, s, material_tags={{"b": 1}}, units=mpm.Units("m", "kN", "s"),
            provenance=mpm.Provenance("9b2e7c41-5d3a-4f6e-8c1b-2a3d4e5f6a7b", "x",
                                      "unknown")))
        assert sys.modules["gmsh"] is None and sys.modules["apeGmsh"] is None
        leaked = sorted(m for m in sys.modules
                        if m.startswith("apeGmsh.") or m.startswith("gmsh."))
        assert not leaked, leaked
        print("ISOLATED-OK")
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "ISOLATED-OK" in proc.stdout
    assert (tmp_path / "iso.particles.h5").is_file()


def _imports(tree: ast.AST):
    """(module, under_type_checking) for every import in the module."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            for sub in ast.walk(node):
                guarded.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, id(node) in guarded
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield "." * node.level + (node.module or ""), id(node) in guarded
            else:
                yield node.module or "", id(node) in guarded


def test_every_import_is_stdlib_numpy_h5py_or_typing_only_mesh():
    files = sorted(PKG.glob("*.py"))
    assert {f.name for f in files} >= {"__init__.py", "_seeding.py", "_particle_set.py"}
    for f in files:
        for mod, typing_only in _imports(ast.parse(f.read_text(encoding="utf-8"))):
            if mod.startswith("."):
                continue
            top = mod.split(".")[0]
            if mod == "apeGmsh.mesh" or mod.startswith("apeGmsh.mesh."):
                assert typing_only, f"{f.name}: {mod} imported at runtime"
                continue
            assert top in ALLOWED_TOP, f"{f.name} imports {mod}"
