"""``GeomTransfViewer`` draws the local frame OpenSees builds.

The page used to derive the frame in JavaScript as ``y = x × vecxz``,
``z = x × y``, which negates both axes against OpenSees
(``LinearCrdTransf3d::getLocalAxes``: ``y = vecxz × x``, ``z = x × y``).
It also defaulted ``vecxz`` to ``[1, 0, 0]``, which is parallel to a beam
along X. The frame (``compute_local_axes``) and the default (the vecxz
the bridge emits, pinned to ``resolve_vecxz``) now come from Python and
reach the page as a payload, so these tests read that payload. The last three drive the real
``show()`` over HTTP with ``webbrowser.open`` replaced: no browser, no
window.
"""
from __future__ import annotations

import http.client
import json
import queue
import re
import socket
import threading
import webbrowser

import numpy as np
import pytest

from apeGmsh.opensees._orientation import Cartesian, resolve_vecxz
from apeGmsh.viewers.diagrams._beam_geometry import compute_local_axes
from apeGmsh.viewers.geom_transf_viewer import (
    GeomTransfViewer,
    _beam_payload,
    _build_html,
)

SKEW_I, SKEW_J = [1.0, -2.0, 0.5], [3.3, 0.7, 4.2]

# name -> (node_i, node_j, vecxz): along X, Y and Z and skew, each with
# the default vecxz (None) and with an explicit one.
BEAMS = {
    "x": ([0, 0, 0], [3, 0, 0], None),
    "x-vecxz": ([0, 0, 0], [3, 0, 0], [0, 1, 1]),
    "y": ([0, 0, 0], [0, 4, 0], None),
    "y-vecxz": ([0, 0, 0], [0, 4, 0], [1, 0, 0]),
    "z": ([0, 0, 0], [0, 0, 3], None),
    "z-vecxz": ([0, 0, 0], [0, 0, 3], [0, -1, 0]),
    "skew": (SKEW_I, SKEW_J, None),
    "skew-vecxz": (SKEW_I, SKEW_J, [0.3, -0.4, 0.8]),
}


def _frame(beam: dict) -> tuple:
    f = beam["frame"]
    return np.array(f["ex"]), np.array(f["ey"]), np.array(f["ez"]), f["L"]


def _opensees_axes(node_i, node_j, vecxz) -> tuple:
    """``LinearCrdTransf3d::getLocalAxes``, transcribed."""
    x = np.subtract(node_j, node_i, dtype=np.float64)
    x /= np.linalg.norm(x)
    y = np.cross(vecxz, x)
    y /= np.linalg.norm(y)
    return x, y, np.cross(x, y)


@pytest.mark.parametrize("name", BEAMS)
def test_frame_is_compute_local_axes(name):
    node_i, node_j, vecxz = BEAMS[name]
    beam = _beam_payload(node_i, node_j, vecxz)

    ex, ey, ez, length = _frame(beam)
    x, y, z, expected_length = compute_local_axes(
        node_i, node_j, beam["vecxz"],
    )

    np.testing.assert_allclose(ex, x, atol=1e-12)
    np.testing.assert_allclose(ey, y, atol=1e-12)
    np.testing.assert_allclose(ez, z, atol=1e-12)
    assert length == pytest.approx(expected_length)


@pytest.mark.parametrize("name", BEAMS)
def test_frame_is_the_opensees_frame(name):
    """An oracle that does not route through ``compute_local_axes``, so a
    sign slip there fails here, not only in the page."""
    node_i, node_j, _ = BEAMS[name]
    beam = _beam_payload(*BEAMS[name])

    ex, ey, ez, _ = _frame(beam)
    x, y, z = _opensees_axes(node_i, node_j, beam["vecxz"])

    np.testing.assert_allclose(ex, x, atol=1e-12)
    np.testing.assert_allclose(ey, y, atol=1e-12)
    np.testing.assert_allclose(ez, z, atol=1e-12)


def test_beam_along_x_with_vecxz_z_has_local_y_on_y_and_z_on_z():
    """The reported case. The old page drew y = -Y and z = -Z."""
    _, ey, ez, _ = _frame(_beam_payload([0, 0, 0], [1, 0, 0], [0, 0, 1]))

    np.testing.assert_allclose(ey, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(ez, [0.0, 0.0, 1.0], atol=1e-12)


_TILT = np.radians(5.7)
DEFAULT_CASES = {
    **{k: BEAMS[k][:2] for k in ("x", "y", "z", "skew")},
    "-z": ([0, 0, 3], [0, 0, 0]),
    "5.7deg-off-z": ([0, 0, 0], [np.sin(_TILT), 0, np.cos(_TILT)]),
}


@pytest.mark.parametrize("name", DEFAULT_CASES)
def test_omitted_vecxz_is_what_the_bridge_emits(name):
    """``apeSees(default_orientation=Cartesian())`` resolves an unoriented
    member's vecxz with ``resolve_vecxz``. The viewer may not import the
    bridge (ADR 0014), so this pins its mirror of the rule. A +Z column
    gets -X there; the diagrams' ``default_vecxz`` (+X) would draw it
    rolled 180 degrees. The beam along X is the one the old ``[1, 0, 0]``
    made degenerate."""
    node_i, node_j = DEFAULT_CASES[name]
    x = np.subtract(node_j, node_i, dtype=np.float64)
    x /= np.linalg.norm(x)

    beam = _beam_payload(node_i, node_j)

    expected = resolve_vecxz(x, *Cartesian().triad_at(node_i))
    np.testing.assert_allclose(beam["vecxz"], expected, atol=1e-12)
    assert beam["frame"] is not None


def test_a_defaulted_z_column_gets_minus_x():
    beam = _beam_payload([0, 0, 0], [0, 0, 3])

    np.testing.assert_allclose(beam["vecxz"], [-1.0, 0.0, 0.0], atol=1e-12)


def test_two_coordinate_nodes_are_refused_with_the_2d_hint():
    with pytest.raises(ValueError, match="z=0"):
        _beam_payload([0, 0], [3, 0])


def test_the_page_computes_no_frames():
    """The frame math left the JavaScript; the page draws ``b.frame``."""
    html = _build_html([_beam_payload([0, 0, 0], [3, 0, 0])], "t")

    assert "localFrame" not in html
    assert "b.frame" in html


@pytest.mark.parametrize("vecxz", [[2, 0, 0], [-1, 0, 0], [0, 0, 0]])
def test_vecxz_along_the_beam_is_degenerate(vecxz):
    """OpenSees refuses this vecxz. ``compute_local_axes`` would swap in
    the default, and the page must not draw that frame."""
    beam = _beam_payload([0, 0, 0], [3, 0, 0], vecxz)

    assert beam["frame"] is None
    assert "parallel" in beam["degenerate"]


def test_coincident_nodes_are_degenerate():
    beam = _beam_payload([1, 2, 3], [1, 2, 3])

    assert beam["frame"] is None
    assert "coincide" in beam["degenerate"]


def test_page_embeds_the_payload():
    beams = GeomTransfViewer._build_beam_list(
        None, None, None,
        beams=[
            {"node_i": [0, 0, 0], "node_j": [0, 0, 3]},
            {"node_i": [0, 0, 3], "node_j": [3, 0, 3], "vecxz": [0, 1, 0]},
        ],
    )

    html = _build_html(beams, "t")
    embedded = re.search(r"const BEAMS_INIT = (.*);\n", html)

    assert embedded is not None
    assert json.loads(embedded.group(1)) == beams


def test_single_beam_controls_start_from_the_default_vecxz():
    beams = GeomTransfViewer._build_beam_list(
        [0, 0, 0], [3, 0, 0], None, None,
    )

    html = _build_html(beams, "t")

    assert 'id="vx" value="0.0"' in html
    assert 'id="vz" value="1.0"' in html


# =====================================================================
# The live door: show() serves the page and recomputes edits in Python
# =====================================================================

def _post(port: int, path: str, body: str = "") -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", path, body=body)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


@pytest.fixture
def shown(monkeypatch):
    """The real ``show()``, with the browser launch replaced by a queue.

    ``show()`` blocks until the page's shutdown beacon, so it runs on a
    thread; the test plays the page. Yields ``(port, thread)``.
    """
    opened: queue.Queue = queue.Queue()
    monkeypatch.setattr(webbrowser, "open", opened.put)
    thread = threading.Thread(
        target=GeomTransfViewer().show,
        kwargs={"node_i": [0, 0, 0], "node_j": [3, 0, 0]},
        daemon=True,
    )
    thread.start()
    port = int(opened.get(timeout=30).rsplit(":", 1)[1])
    yield port, thread
    if thread.is_alive():
        _post(port, "/shutdown")
        thread.join(timeout=10)


def test_an_edit_is_recomputed_by_python(shown):
    port, thread = shown
    edit = {"node_i": [0, 0, 0], "node_j": [0, 5, 0], "vecxz": [0, 0, 1]}

    status, body = _post(port, "/frame", json.dumps([edit]))

    assert status == 200
    assert json.loads(body) == [_beam_payload(**edit)]
    assert thread.is_alive(), "a frame request ended the viewer"


def test_a_pre_opened_socket_does_not_stall_an_edit(shown):
    """Browsers pre-open sockets they may never send on. A server that
    handles one connection at a time waits on that socket forever, and
    every edit behind it stalls."""
    port, _ = shown
    edit = {"node_i": [0, 0, 0], "node_j": [0, 5, 0], "vecxz": [0, 0, 1]}

    with socket.create_connection(("127.0.0.1", port)):
        status, _ = _post(port, "/frame", json.dumps([edit]))

    assert status == 200


def test_the_shutdown_beacon_ends_show(shown):
    port, thread = shown

    status, _ = _post(port, "/shutdown")
    thread.join(timeout=10)

    assert status == 200
    assert not thread.is_alive()
