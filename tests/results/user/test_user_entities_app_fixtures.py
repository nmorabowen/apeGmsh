"""The app's ``/entities`` fixtures stay what Python reads and what the cut kernel computes (VU8).

``apeGmshViewer/test/reader/userEntities.test.ts`` reads the committed
``.user.h5`` files and compares the app's reading with ``goldens.json``. This
file holds that JSON to its two sources, so a change on either side fails here
or there and never passes silently:

* the Python reader (``UserFile.entities``, ``UserFile.results``) on the
  committed files gives the committed rows, refusals and index;
* the cut kernel (``STKO_to_python.cuts``) on the committed MPCO frame, fed the
  original :class:`~apeGmsh.cuts.SectionCutDef`, gives the committed ``F`` and
  ``M`` bit for bit, and so does the resultant the file stores for the cut.

Regenerate with ``python tests/results/user/make_app_entities_fixtures.py``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from apeGmsh.results.user import UserFile
from tests.results.user import make_app_entities_fixtures as fx

COMMITTED = json.loads((fx.OUT / "goldens.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("stem", ["frame_cut", "rows"])
def test_the_python_reader_reads_the_committed_goldens(stem: str) -> None:
    uf = UserFile(fx.OUT / f"{stem}.user.h5")
    index = uf.results()
    assert fx._entities(uf.entities()) == COMMITTED[stem]["entities"]
    assert [e.name for e in index.entries] == COMMITTED[stem]["entries"]
    assert [[r.name, r.reason] for r in index.refused] == COMMITTED[stem]["refused"]


def test_the_rows_fixture_refuses_what_it_says() -> None:
    """The edits of ``write_rows`` land: three bad rows, two results refused alone."""
    rows = COMMITTED["rows"]
    assert rows["entities"]["refused"] == [
        ["cut", 0, "bad-param"], ["cut", 1, "empty-name"], ["point_set", 0, "bad-param"]]
    assert rows["entries"] == ["on the storey", "on the second point set"]
    assert [r[1] for r in rows["refused"]] == ["unresolved", "unresolved"]


def test_the_kernel_gives_the_committed_resultant() -> None:
    pytest.importorskip("STKO_to_python.cuts")
    cut, force, moment = fx.kernel_resultant()
    golden = COMMITTED["frame_cut"]["kernel"]
    assert force.tobytes() == np.array(golden["F"], dtype="f8").tobytes()
    assert moment.tobytes() == np.array(golden["M"], dtype="f8").tobytes()
    assert list(cut.plane_normal) == golden["def"]["normal"]
    stored = UserFile(fx.OUT / "frame_cut.user.h5").read_result("frame cut resultant").data[:, 0, :]
    assert stored.tobytes() == np.concatenate([force, moment], axis=1).tobytes()
