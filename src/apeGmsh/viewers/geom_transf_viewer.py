"""
GeomTransfViewer
================

Interactive 3D viewer for OpenSees geometric transformation (geomTransf).

Opens a Three.js-based HTML page in the default browser showing:

* The **beam element** as a cylinder between node I and node J.
* **Global axes** (X, Y, Z) at the origin.
* **Local axes** (x, y, z) computed with the OpenSees convention
  from ``vecxz``.
* A readout panel with the local frame vectors, element length,
  and orthogonality checks.

Single-beam mode exposes interactive controls (input fields) to
adjust nodes and ``vecxz`` in real time.  Multi-beam mode draws
all frames simultaneously with the controls hidden.

The page does no frame math.  Python computes every frame with
:func:`~apeGmsh.viewers.diagrams._beam_geometry.compute_local_axes`,
the rule the results diagrams use, and an edit in the controls asks
the local server for the new frame.

Usage
-----
::

    from apeGmsh.viewers import GeomTransfViewer

    viewer = GeomTransfViewer()

    # Single beam — interactive controls in the browser
    viewer.show(node_i=[0, 0, 0], node_j=[0.3, 0.5, 3], vecxz=[1, 0, 0])

    # vecxz defaults to what the bridge emits: -X for this +Z column
    viewer.show(node_i=[0, 0, 0], node_j=[0, 0, 3])

    # Multiple beams
    viewer.show(beams=[
        {"node_i": [0,0,0], "node_j": [0,0,3],  "vecxz": [1,0,0]},
        {"node_i": [0,0,3], "node_j": [3,0,3],  "vecxz": [0,0,1]},
        {"node_i": [3,0,0], "node_j": [3,0,3],  "vecxz": [1,0,0]},
    ])

Dependencies
------------
stdlib and numpy.
Three.js r128 loaded from cdnjs (requires internet on first open).
"""
from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Sequence

import numpy as np


class GeomTransfViewer:
    """
    Browser-based 3D viewer for OpenSees geomTransf visualisation.

    Spins up a lightweight local HTTP server, opens the viewer in the
    default browser, and **blocks until the browser tab is closed**.
    The server shuts itself down automatically — no ``input()`` prompt,
    no leftover temp files.

    Parameters
    ----------
    title : str
        Window / browser-tab title (default: ``"OpenSees geomTransf viewer"``).
    """

    def __init__(self, title: str = "OpenSees geomTransf viewer") -> None:
        self._title = title

    # ── public API ────────────────────────────────────────────────────

    def show(
        self,
        node_i: Sequence[float] | None = None,
        node_j: Sequence[float] | None = None,
        vecxz: Sequence[float] | None = None,
        beams: list[dict] | None = None,
    ) -> None:
        """
        Open the interactive geomTransf viewer in the default browser.

        Blocks until the browser tab is closed.

        Parameters
        ----------
        node_i : [x, y, z]
            Start node (single-beam convenience).
        node_j : [x, y, z]
            End node (single-beam convenience).
        vecxz : [vx, vy, vz]
            Vector in local x-z plane.  Defaults to what the bridge
            emits for a member with no orientation
            (``apeSees(default_orientation=Cartesian())``): global Z
            projected off the member, or ``x × Y`` for a vertical one
            (``-X`` for a +Z column).
        beams : list[dict]
            List of dicts with keys ``'node_i'``, ``'node_j'``,
            ``'vecxz'`` (optional).  Pass this for multi-beam mode.
        """
        beam_list = self._build_beam_list(node_i, node_j, vecxz, beams)
        html = _build_html(beam_list, self._title)

        # ── local HTTP server ─────────────────────────────────────────
        shutdown_event = threading.Event()

        handler_cls = partial(_ViewerHandler, html=html,
                              shutdown_event=shutdown_event)
        # Threaded: browsers pre-open sockets they may never use, and a
        # one-connection-at-a-time server would hold every /frame edit
        # behind such a socket.
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        port = server.server_address[1]

        server_thread = threading.Thread(target=server.serve_forever,
                                         daemon=True)
        server_thread.start()

        webbrowser.open(f"http://127.0.0.1:{port}")

        # Block until the browser tab fires the /shutdown beacon
        shutdown_event.wait()
        server.shutdown()
        server_thread.join(timeout=2)

    # ── internals ─────────────────────────────────────────────────────

    @staticmethod
    def _build_beam_list(
        node_i: Sequence[float] | None,
        node_j: Sequence[float] | None,
        vecxz: Sequence[float] | None,
        beams: list[dict] | None,
    ) -> list[dict]:
        """Normalise the caller's arguments into a flat beam list."""
        if beams is not None:
            return [
                _beam_payload(b["node_i"], b["node_j"], b.get("vecxz"))
                for b in beams
            ]
        if node_i is not None and node_j is not None:
            return [_beam_payload(node_i, node_j, vecxz)]
        raise ValueError("Provide either (node_i, node_j) or beams=[...]")


_GLOBAL_Y = np.array([0.0, 1.0, 0.0])
_GLOBAL_Z = np.array([0.0, 0.0, 1.0])
_VERTICAL_TOL = 1e-9    # opensees/_orientation.py ``_TOL``


def _bridge_default_vecxz(x_local: np.ndarray) -> np.ndarray:
    """The ``vecxz`` the bridge emits for a member with no orientation.

    ``resolve_vecxz`` with the default ``Cartesian()`` triad
    (``e2 = +Y``, ``e3 = +Z``): global Z projected off the member, or,
    for a vertical member, ``x × Y`` (``-X`` for a +Z column).  Viewers
    may not import ``apeGmsh.opensees`` (ADR 0014), so this mirrors the
    rule and ``tests/viewers/test_geom_transf_viewer.py`` pins it to
    ``resolve_vecxz``.
    """
    if abs(float(np.dot(x_local, _GLOBAL_Z))) < 1.0 - _VERTICAL_TOL:
        local_y = np.cross(_GLOBAL_Z, x_local)
        local_y /= float(np.linalg.norm(local_y))
    else:
        local_y = _GLOBAL_Y
    vecxz = np.cross(x_local, local_y)
    return vecxz / float(np.linalg.norm(vecxz))


def _beam_payload(
    node_i: Sequence[float],
    node_j: Sequence[float],
    vecxz: Sequence[float] | None = None,
) -> dict:
    """One beam as the page draws it: nodes, ``vecxz`` and local frame.

    The frame is :func:`compute_local_axes`, the rule the results
    diagrams use, so the page cannot drift from them.  An omitted
    ``vecxz`` is what the bridge would emit for the member
    (:func:`_bridge_default_vecxz`).  OpenSees itself has no default.
    ``frame`` is ``None``, with a ``degenerate`` reason, where OpenSees
    refuses the transformation: the nodes coincide, or ``vecxz`` is
    zero or parallel to the beam axis.  ``compute_local_axes`` would
    quietly substitute its own default there and draw a frame OpenSees
    never builds.
    """
    # Deferred: the diagrams package costs ~0.2 s to import, which
    # ``import apeGmsh.viewers`` should not pay for this viewer.
    from .diagrams._beam_geometry import (
        _DEGENERATE_EPS,
        compute_local_axes,
    )

    ci = np.asarray(node_i, dtype=np.float64)
    cj = np.asarray(node_j, dtype=np.float64)
    for name, node in (("node_i", ci), ("node_j", cj)):
        if node.shape != (3,):
            raise ValueError(
                f"GeomTransfViewer is 3-D: {name} needs 3 coordinates, "
                f"got {node.tolist()}. Give a 2-D model's nodes z=0."
            )
    chord = cj - ci
    length = float(np.linalg.norm(chord))
    beam: dict = {"node_i": ci.tolist(), "node_j": cj.tolist(), "frame": None}
    if length <= _DEGENERATE_EPS:
        # No axis, so no default either; show the Cartesian reference.
        v = _GLOBAL_Z if vecxz is None else np.asarray(vecxz, dtype=np.float64)
        beam["vecxz"] = v.tolist()
        beam["degenerate"] = "node I and node J coincide"
        return beam
    x_local = chord / length
    v = (
        _bridge_default_vecxz(x_local) if vecxz is None
        else np.asarray(vecxz, dtype=np.float64)
    )
    beam["vecxz"] = v.tolist()
    # The same test compute_local_axes applies before its fallback.
    if np.linalg.norm(v - np.dot(v, x_local) * x_local) < _DEGENERATE_EPS:
        beam["degenerate"] = "vecxz is zero or parallel to the beam axis"
        return beam
    ex, ey, ez, length = compute_local_axes(ci, cj, v)
    beam["frame"] = {
        "ex": ex.tolist(), "ey": ey.tolist(), "ez": ez.tolist(),
        "L": length,
    }
    return beam


# ======================================================================
# Local HTTP handler  (module-private)
# ======================================================================

class _ViewerHandler(BaseHTTPRequestHandler):
    """Serves the viewer HTML, recomputes edited frames, and listens
    for a /shutdown beacon."""

    def __init__(self, *args, html: str, shutdown_event: threading.Event,
                 **kwargs):
        self._html = html
        self._shutdown_event = shutdown_event
        super().__init__(*args, **kwargs)

    # GET /  -> serve the viewer page
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self._html.encode("utf-8"))

    # POST /frame     -> the frames of the edited beams, as JSON
    # POST /shutdown  -> signal Python to stop blocking
    def do_POST(self):  # noqa: N802
        if self.path == "/frame":
            size = int(self.headers.get("Content-Length", 0))
            beams = json.loads(self.rfile.read(size))
            body = json.dumps([
                _beam_payload(b["node_i"], b["node_j"], b["vecxz"])
                for b in beams
            ]).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.end_headers()
        self._shutdown_event.set()

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence request logs


# ======================================================================
# HTML / JS generation  (module-private)
# ======================================================================

def _build_html(beams: list[dict], title: str) -> str:
    beams_json = json.dumps(beams)

    b0 = beams[0]
    ix, iy, iz = b0["node_i"]
    jx, jy, jz = b0["node_j"]
    vx, vy, vz = b0["vecxz"]
    multi = len(beams) > 1
    controls_display = "none" if multi else "block"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; color: #ccc; font-family: system-ui, sans-serif;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  #viewport {{ flex: 1; position: relative; }}
  canvas {{ width: 100% !important; height: 100% !important; display: block; }}
  /* HUD */
  #hud {{ position: absolute; top: 12px; left: 12px; font-size: 12px;
          background: rgba(0,0,0,0.55); border-radius: 8px;
          padding: 10px 14px; line-height: 1.9; pointer-events: none; }}
  #hud b {{ font-weight: 600; }}
  #nodeinfo {{ position: absolute; top: 12px; right: 12px; font-size: 11px;
               background: rgba(0,0,0,0.55); border-radius: 8px;
               padding: 8px 12px; min-width: 160px; }}
  #hint {{ position: absolute; bottom: 10px; left: 12px; font-size: 11px;
           color: #666; pointer-events: none; }}
  /* Controls panel */
  #controls {{ display: {controls_display}; background: #222; padding: 10px 16px;
               border-top: 1px solid #333; font-size: 12px; }}
  .row {{ display: flex; gap: 10px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }}
  .row label {{ color: #888; min-width: 90px; }}
  .xyz {{ display: flex; gap: 6px; }}
  .xyz span {{ color: #666; font-size: 11px; margin-right: 2px; }}
  input[type=number] {{
    width: 72px; background: #2a2a2a; border: 1px solid #444;
    color: #ddd; border-radius: 4px; padding: 3px 6px; font-size: 12px;
  }}
  input[type=number]:focus {{ outline: none; border-color: #666; }}
  /* Axis readout */
  #readout {{ background: #222; padding: 8px 16px; font-size: 11px;
              font-family: 'Courier New', monospace; border-top: 1px solid #333;
              line-height: 1.8; color: #aaa; }}
</style>
</head>
<body>
<div id="viewport">
  <canvas id="c"></canvas>
  <div id="hud">
    <div>Global &nbsp;<b style="color:#cc2222">X</b>
                    <b style="color:#229944"> Y</b>
                    <b style="color:#2255cc"> Z</b></div>
    <div>Local &nbsp;&nbsp;<b style="color:#ff7777">x</b>
                    <b style="color:#66cc88"> y</b>
                    <b style="color:#6699ee"> z</b></div>
    <div id="hud_beams" style="color:#555;margin-top:4px;"></div>
  </div>
  <div id="nodeinfo">
    <div style="font-weight:600;margin-bottom:4px;color:#eee;">Nodes</div>
    <div id="ni_i"></div>
    <div id="ni_j"></div>
  </div>
  <div id="hint">Drag to rotate · Scroll to zoom · Right-drag to pan</div>
</div>
<div id="controls">
  <div class="row">
    <label>Node I</label>
    <div class="xyz">
      <span>x</span><input type="number" id="ix" value="{ix}" step="0.5">
      <span>y</span><input type="number" id="iy" value="{iy}" step="0.5">
      <span>z</span><input type="number" id="iz" value="{iz}" step="0.5">
    </div>
    <label style="margin-left:16px;">Node J</label>
    <div class="xyz">
      <span>x</span><input type="number" id="jx" value="{jx}" step="0.5">
      <span>y</span><input type="number" id="jy" value="{jy}" step="0.5">
      <span>z</span><input type="number" id="jz" value="{jz}" step="0.5">
    </div>
    <label style="margin-left:16px;">vecxz</label>
    <div class="xyz">
      <span>x</span><input type="number" id="vx" value="{vx}" step="0.1">
      <span>y</span><input type="number" id="vy" value="{vy}" step="0.1">
      <span>z</span><input type="number" id="vz" value="{vz}" step="0.1">
    </div>
  </div>
</div>
<div id="readout">—</div>
<script>
// ── shutdown beacon: tell the local server when the tab closes ────────
window.addEventListener('pagehide', () => {{
  navigator.sendBeacon('/shutdown');
}});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const BEAMS_INIT = {beams_json};
const MULTI = {str(multi).lower()};
// ── renderer / scene / camera ─────────────────────────────────────────────
const canvas   = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x1a1a1a);
const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45,1,0.01,500);
let theta=0.5, phi=1.1, radius=16;
let target = new THREE.Vector3(0,0,1.5);
function updateCamera(){{
  camera.position.set(
    target.x + radius*Math.sin(phi)*Math.sin(theta),
    target.y + radius*Math.sin(phi)*Math.cos(theta),
    target.z + radius*Math.cos(phi)
  );
  camera.up.set(0,0,1);
  camera.lookAt(target);
}}
updateCamera();
// ── orbit controls ────────────────────────────────────────────────────────
let drag=null;
canvas.addEventListener('mousedown', e=>{{ drag={{x:e.clientX,y:e.clientY,btn:e.button}}; }});
window.addEventListener('mouseup', ()=>drag=null);
window.addEventListener('mousemove', e=>{{
  if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  drag.x=e.clientX; drag.y=e.clientY;
  if(drag.btn===0){{               // rotate
    theta -= dx*0.012;
    phi = Math.max(0.05,Math.min(Math.PI-0.05, phi-dy*0.012));
  }} else if(drag.btn===2){{       // pan
    const right = new THREE.Vector3();
    const up    = new THREE.Vector3();
    camera.getWorldDirection(new THREE.Vector3());
    right.crossVectors(camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
    up.copy(camera.up).normalize();
    const f = radius*0.001;
    target.addScaledVector(right,-dx*f).addScaledVector(up,dy*f);
  }}
  updateCamera();
}});
canvas.addEventListener('wheel', e=>{{
  radius = Math.max(1,Math.min(80, radius+e.deltaY*0.02));
  updateCamera(); e.preventDefault();
}},{{passive:false}});
canvas.addEventListener('contextmenu', e=>e.preventDefault());
// ── math helpers ──────────────────────────────────────────────────────────
const V3      = (a,b,c) => new THREE.Vector3(a,b,c);
const norm    = v => Math.hypot(...v);
const normalz = v => {{ const l=norm(v); return l<1e-10?[0,0,0]:v.map(x=>x/l); }};
const cross   = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const dot     = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const add     = (a,b) => [a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const scl     = (v,s) => v.map(x=>x*s);
const sub     = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const toV3    = a => V3(a[0],a[1],a[2]);
const fmt     = v => `[${{v.map(x=>x.toFixed(3)).join(', ')}}]`;
function perpTo(v){{
  const a=Math.abs(v[0]),b=Math.abs(v[1]),c=Math.abs(v[2]);
  const ref=(a<=b&&a<=c)?[1,0,0]:(b<=a&&b<=c)?[0,1,0]:[0,0,1];
  return normalz(cross(v,ref));
}}
// ── colors ────────────────────────────────────────────────────────────────
const C = {{
  gx:{{h:0xcc2222,c:'#cc2222'}}, gy:{{h:0x229944,c:'#229944'}}, gz:{{h:0x2255cc,c:'#2255cc'}},
  lx:{{h:0xff7777,c:'#ff7777'}}, ly:{{h:0x66cc88,c:'#66cc88'}}, lz:{{h:0x6699ee,c:'#6699ee'}},
  vxz:{{h:0x888888,c:'#888888'}},
}};
// ── sprite label ──────────────────────────────────────────────────────────
function makeLabel(text, cssColor, s=0.5){{
  const cv=document.createElement('canvas'); cv.width=cv.height=128;
  const ctx=cv.getContext('2d');
  ctx.font='bold 40px sans-serif'; ctx.fillStyle=cssColor;
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(text,64,64);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:new THREE.CanvasTexture(cv),depthTest:false,transparent:true}}));
  sp.scale.set(s,s,1); return sp;
}}
// ── geometry helpers ──────────────────────────────────────────────────────
function arrow(from,to,ch,group){{
  const dir=sub(to,from); const l=norm(dir); if(l<1e-6)return;
  group.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([toV3(from),toV3(to)]),
    new THREE.LineBasicMaterial({{color:ch}})
  ));
  const cH=Math.min(0.2,l*0.22), dn=normalz(dir);
  const cone=new THREE.Mesh(new THREE.ConeGeometry(0.05,cH,10),new THREE.MeshBasicMaterial({{color:ch}}));
  cone.position.copy(toV3(to)).sub(V3(...dn).multiplyScalar(cH*0.5));
  cone.quaternion.setFromUnitVectors(V3(0,1,0),V3(...dn));
  group.add(cone);
}}
function dashed(from,to,ch,group){{
  for(let i=0;i<24;i+=2){{
    const a=toV3(from).lerp(toV3(to),i/24), b=toV3(from).lerp(toV3(to),(i+1)/24);
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a,b]),
      new THREE.LineBasicMaterial({{color:ch,opacity:0.55,transparent:true}})));
  }}
}}
function tube(from,to,ch,group){{
  const dir=sub(to,from); const l=norm(dir); if(l<1e-6)return;
  const m=new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.05,l,8),new THREE.MeshBasicMaterial({{color:ch}}));
  m.position.copy(toV3(from).lerp(toV3(to),0.5));
  m.quaternion.setFromUnitVectors(V3(0,1,0),V3(...normalz(dir)));
  group.add(m);
}}
function sphere(pos,ch,r,group){{
  const s=new THREE.Mesh(new THREE.SphereGeometry(r,16,16),new THREE.MeshBasicMaterial({{color:ch}}));
  s.position.copy(toV3(pos)); group.add(s);
}}
function grid(group){{
  const mat=new THREE.LineBasicMaterial({{color:0x444444,opacity:0.3,transparent:true}});
  for(let i=-8;i<=8;i++){{
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([V3(i,-8,0),V3(i,8,0)]),mat));
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([V3(-8,i,0),V3(8,i,0)]),mat));
  }}
}}
// ── build scene ───────────────────────────────────────────────────────────
// Each beam arrives with its local frame computed in Python
// (compute_local_axes, the OpenSees convention); the page only draws it.
let grp = null;
function buildScene(beams){{
  if(grp){{ scene.remove(grp); }}
  grp = new THREE.Group();
  grid(grp);
  // Global axes at origin
  const axG=1.6, O=[0,0,0];
  arrow(O,[axG,0,0],C.gx.h,grp); arrow(O,[0,axG,0],C.gy.h,grp); arrow(O,[0,0,axG],C.gz.h,grp);
  const lp=0.35;
  const lGX=makeLabel('X',C.gx.c); lGX.position.set(axG+lp,0,0);  grp.add(lGX);
  const lGY=makeLabel('Y',C.gy.c); lGY.position.set(0,axG+lp,0);  grp.add(lGY);
  const lGZ=makeLabel('Z',C.gz.c); lGZ.position.set(0,0,axG+lp);  grp.add(lGZ);
  let readoutLines = [];
  let allNodes = [];
  let cx=0,cy=0,cz=0;
  beams.forEach((b,idx)=>{{
    const ni=b.node_i, nj=b.node_j, vxz=b.vecxz;
    allNodes.push(ni,nj);
    cx+=ni[0]+nj[0]; cy+=ni[1]+nj[1]; cz+=ni[2]+nj[2];
    tube(ni,nj,0x666666,grp);
    sphere(ni,C.gy.h,0.1,grp);
    sphere(nj,C.gx.h,0.1,grp);
    const f=b.frame;
    if(!f){{
      readoutLines.push(`Beam ${{idx+1}}: degenerate (${{b.degenerate}})`);
      return;
    }}
    const {{ex,ey,ez,L}}=f;
    const mid=scl(add(ni,nj),0.5);
    const axL=Math.max(0.8,L*0.42);
    const off=0.38;
    arrow(mid,add(mid,scl(ex,axL)),C.lx.h,grp);
    arrow(mid,add(mid,scl(ey,axL)),C.ly.h,grp);
    arrow(mid,add(mid,scl(ez,axL)),C.lz.h,grp);
    const suffix = beams.length>1?` (${{idx}})` : '';
    const lLX=makeLabel('x'+suffix,C.lx.c,0.45); lLX.position.copy(toV3(add(mid,scl(ex,axL+off)))); grp.add(lLX);
    const lLY=makeLabel('y'+suffix,C.ly.c,0.45); lLY.position.copy(toV3(add(mid,scl(ey,axL+off)))); grp.add(lLY);
    const lLZ=makeLabel('z'+suffix,C.lz.c,0.45); lLZ.position.copy(toV3(add(mid,scl(ez,axL+off)))); grp.add(lLZ);
    const vn=normalz(vxz), vLen=axL*0.85;
    dashed(ni,add(ni,scl(vn,vLen)),C.vxz.h,grp);
    const vLbl=makeLabel('vecxz',C.vxz.c,0.55);
    const vPerp=perpTo(vn);
    vLbl.position.copy(toV3(add(add(ni,scl(vn,vLen)),scl(vPerp,0.4)))); grp.add(vLbl);
    readoutLines.push(
      `<b>Beam ${{idx+1}}</b>  L=${{L.toFixed(4)}}`,
      `  <span style="color:${{C.lx.c}}">ex</span>=${{fmt(ex)}}`,
      `  <span style="color:${{C.ly.c}}">ey</span>=${{fmt(ey)}}`,
      `  <span style="color:${{C.lz.c}}">ez</span>=${{fmt(ez)}}`,
      `  ex*ey=${{Math.abs(dot(ex,ey)).toFixed(6)}}  ey*ez=${{Math.abs(dot(ey,ez)).toFixed(6)}}  ex*ez=${{Math.abs(dot(ex,ez)).toFixed(6)}}`
    );
  }});
  scene.add(grp);
  // Center target
  const n=beams.length*2;
  target.set(cx/n,cy/n,cz/n);
  // Auto-fit radius to bounding box
  let maxD=0;
  allNodes.forEach(p=>{{maxD=Math.max(maxD,target.distanceTo(toV3(p)));}} );
  radius = Math.max(5, maxD*3.2);
  updateCamera();
  document.getElementById('readout').innerHTML = readoutLines.join('<br>');
  if(!MULTI){{
    const ni=beams[0].node_i, nj=beams[0].node_j;
    document.getElementById('ni_i').textContent=`I: (${{ni.map(v=>v.toFixed(2)).join(', ')}})`;
    document.getElementById('ni_j').textContent=`J: (${{nj.map(v=>v.toFixed(2)).join(', ')}})`;
  }}
  document.getElementById('hud_beams').textContent = beams.length>1?`${{beams.length}} beams`:'';
}}
// ── interactive controls (single-beam only) ───────────────────────────────
// An edit asks the local server for the new frame, so frames are only
// ever computed in Python.  A stale reply (an older edit) is dropped.
function readInputBeam(){{
  const g=id=>parseFloat(document.getElementById(id).value)||0;
  return [{{
    node_i:[g('ix'),g('iy'),g('iz')],
    node_j:[g('jx'),g('jy'),g('jz')],
    vecxz: [g('vx'),g('vy'),g('vz')],
  }}];
}}
let editSeq=0;
function refreshFromInputs(){{
  const seq=++editSeq;
  fetch('/frame',{{method:'POST',body:JSON.stringify(readInputBeam())}})
    .then(r=>r.json())
    .then(beams=>{{ if(seq===editSeq) buildScene(beams); }})
    .catch(()=>{{
      document.getElementById('readout').textContent=
        'Frame not updated: the viewer server did not answer. Call show() again.';
    }});
}}
if(!MULTI){{
  ['ix','iy','iz','jx','jy','jz','vx','vy','vz'].forEach(id=>
    document.getElementById(id).addEventListener('input',refreshFromInputs)
  );
}}
buildScene(BEAMS_INIT);
// ── resize ────────────────────────────────────────────────────────────────
function resize(){{
  const vp=document.getElementById('viewport');
  renderer.setSize(vp.clientWidth,vp.clientHeight,false);
  camera.aspect=vp.clientWidth/vp.clientHeight;
  camera.updateProjectionMatrix();
}}
resize();
window.addEventListener('resize',resize);
// ── render loop ───────────────────────────────────────────────────────────
(function loop(){{ requestAnimationFrame(loop); renderer.render(scene,camera); }})();
</script>
</body>
</html>
"""
