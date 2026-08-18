import faulthandler, gc, os, sys, threading
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
faulthandler.enable()
from qtpy import QtCore, QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
MODE = sys.argv[1]

def spin():
    for _ in range(300000):
        pass
    app.processEvents()
    for _ in range(300000):
        pass

def rect_doc():
    from apeGmsh.sections import SectionDocument
    d = SectionDocument.new(name="t", kind="continuum")
    d.set_material("s", E=200e3, nu=0.3)
    d.add_shape("rect_face", id="r", b=4.0, h=4.0, material="s")
    d.set_mesh(lc=1.0)
    return d

if MODE == "A":                        # live C++ object dropped on a worker
    box = [QtCore.QTimer()]
    t = threading.Thread(target=box.clear); t.start(); t.join()

elif MODE == "B":                      # STALE wrapper (C++ already destroyed)
    parent = QtWidgets.QWidget()
    child = QtWidgets.QLabel(parent)
    box = [child]
    del child, parent
    t = threading.Thread(target=box.clear); t.start(); t.join()

elif MODE == "C":                      # gc reaps a Qt-holding cycle on a worker
    class Node: pass
    def make():
        p = QtWidgets.QWidget()
        n = Node(); n.me = n; n.w = p; n.c = QtWidgets.QLabel(p)
    gc.disable()
    for _ in range(200): make()
    t = threading.Thread(target=gc.collect); t.start(); t.join()
    gc.enable()

elif MODE == "D":                      # real builder window finalized off-thread
    from apeGmsh.sections._builder_gui import SectionBuilderWindow
    box = []
    for _ in range(3):
        w = SectionBuilderWindow(rect_doc())
        w.refresh_properties()
        w._controller.join(60.0); w._controller.drain()
        w.close()
        box.append(w)
    gc.disable()
    def worker():
        box.clear()
        gc.collect()
    t = threading.Thread(target=worker); t.start(); t.join()
    gc.enable()

elif MODE == "E":                      # gc forced ON the properties worker itself
    from apeGmsh.sections._builder_gui import SectionBuilderWindow
    import apeGmsh.sections._properties as props
    real = props.build_document
    def gc_during_build(doc_dict):
        gc.collect()                   # cyclic gc runs ON the worker thread
        r = real(doc_dict)
        gc.collect()
        return r
    props.build_document = gc_during_build
    for _ in range(5):
        w = SectionBuilderWindow(rect_doc())
        w._controller = props.PropertiesController(
            builder=gc_during_build, on_result=w._on_properties)
        w.refresh_properties()
        w._controller.join(60.0); w._controller.drain()
        w.close()
    spin()

spin()
print("SURVIVED", MODE, flush=True)
