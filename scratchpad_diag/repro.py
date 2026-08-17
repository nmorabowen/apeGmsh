"""Minimal repro probe for the ADR 0080 B6 Linux segfault.

Knobs (env, "1" = on):
  GL=1      create a live offscreen VTK/Mesa render context first
  VIEWERS=1 import apeGmsh.viewers (pulls pyvista/VTK + session code)
  FORK=1    run subprocess.run from a WORKER thread
  MAINFORK=1 run the same subprocess.run from the MAIN thread instead
  GMSH=1    run an in-process gmsh session on the main thread afterwards
  N=<int>   repetitions (default 3)
"""
import faulthandler
import os
import subprocess
import sys
import threading

faulthandler.enable()

def on(name):
    return os.environ.get(name, "0") == "1"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

print("== import qt", flush=True)
from qtpy import QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
print("   platform:", app.platformName(), flush=True)

print("== import gmsh", flush=True)
import gmsh

if on("VIEWERS"):
    print("== import apeGmsh.viewers", flush=True)
    import apeGmsh.viewers  # noqa: F401

plotter = None
if on("GL"):
    print("== create live offscreen VTK context", flush=True)
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere())
    plotter.show(auto_close=False)
    print("   render window:", type(plotter.ren_win).__name__, flush=True)

def child_call(tag):
    r = subprocess.run(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        capture_output=True, text=True, timeout=120,
    )
    print(f"   [{tag}] child rc={r.returncode}", flush=True)

def gmsh_session(tag):
    gmsh.initialize(interruptible=True)
    gmsh.model.add(f"m{tag}")
    gmsh.model.occ.addRectangle(0, 0, 0, 4, 4)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
    gmsh.model.mesh.generate(2)
    n = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    print(f"   [{tag}] gmsh nodes={n}", flush=True)

N = int(os.environ.get("N", "3"))
for i in range(N):
    print(f"== round {i}", flush=True)
    if on("FORK"):
        t = threading.Thread(target=child_call, args=(f"worker{i}",), daemon=True)
        t.start()
        t.join(120)
    if on("MAINFORK"):
        child_call(f"main{i}")
    if on("GMSH"):
        gmsh_session(i)

if plotter is not None:
    plotter.close()
print("== OK, survived", flush=True)
