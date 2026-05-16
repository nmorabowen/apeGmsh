"""Parser + sniffer tests for :mod:`apeGmsh.ground_motions`.

Tests use synthetic fixtures (tmp_path) for format coverage and a
sample of the homogenised corpus under ``examples/Records/03_Selected/``
for end-to-end smoke checks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.ground_motions import GroundMotion, from_file, sniff_format


# Locate the example corpus shipped with apeGmsh. Walking up from the
# test file lets these tests run both from the main repo and from a
# git worktree (where ``examples/`` isn't always copied).
def _find_corpus() -> Path | None:
    start = Path(__file__).resolve()
    for parent in start.parents:
        candidate = parent / "examples" / "Records" / "03_Selected"
        if candidate.is_dir():
            return candidate
    return None


_CORPUS = _find_corpus()


# ---------------------------------------------------------------------------
# Two-column parser
# ---------------------------------------------------------------------------

class TestTwoColumn:
    def _write(self, tmp_path: Path, lines: list[str]) -> Path:
        p = tmp_path / "rec.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def test_roundtrip_synthetic(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [
            "0.000000  1.0e-3",
            "0.005000  2.0e-3",
            "0.010000  3.0e-3",
            "0.015000  4.0e-3",
        ])
        gm = GroundMotion.from_two_column(path)
        assert gm.npts == 4
        assert gm.dt == pytest.approx(0.005)
        np.testing.assert_allclose(gm.accel, [1e-3, 2e-3, 3e-3, 4e-3])

    def test_scale_factor_multiplies_values(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [
            "0.000000  1.0",
            "0.005000  2.0",
            "0.010000  3.0",
        ])
        gm = GroundMotion.from_two_column(path, scale_factor=9.81)
        np.testing.assert_allclose(gm.accel, [9.81, 19.62, 29.43])
        assert gm.metadata["scale_factor"] == pytest.approx(9.81)

    def test_skips_comments_and_blank_lines(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [
            "# PEER-like header comment",
            "% matlab-style comment",
            "",
            "0.00  0.1",
            "0.01  0.2",
            "0.02  0.3",
        ])
        gm = GroundMotion.from_two_column(path)
        assert gm.npts == 3
        assert gm.dt == pytest.approx(0.01)

    def test_non_uniform_dt_accepted(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [
            "0.00  0.1",
            "0.01  0.2",
            "0.03  0.3",
        ])
        gm = GroundMotion.from_two_column(path)

        assert gm.is_uniform is False
        np.testing.assert_allclose(gm.time, [0.0, 0.01, 0.03])
        assert gm.dt == pytest.approx(0.015)
        assert gm.metadata["uniform"] is False
        assert "dt_rel_deviation" in gm.metadata

    def test_non_monotonic_time_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [
            "0.00  0.1",
            "0.02  0.2",
            "0.01  0.3",
        ])
        with pytest.raises(ValueError, match="strictly increasing"):
            GroundMotion.from_two_column(path)

    def test_uniform_within_tolerance_stays_uniform(
        self, tmp_path: Path
    ) -> None:
        path = self._write(tmp_path, [
            "0.000000000  0.1",
            "0.005000000  0.2",
            "0.010000000  0.3",
            "0.014999999  0.4",
            "0.020000000  0.5",
        ])
        gm = GroundMotion.from_two_column(path)
        assert gm.is_uniform is True
        assert gm.time is None

    def test_three_column_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [
            "0.00  0.1  0.5",
            "0.01  0.2  0.6",
        ])
        with pytest.raises(ValueError, match="2 columns"):
            GroundMotion.from_two_column(path)

    def test_too_few_rows_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ["0.00  0.1"])
        with pytest.raises(ValueError, match="at least 2"):
            GroundMotion.from_two_column(path)


# ---------------------------------------------------------------------------
# One-column parser
# ---------------------------------------------------------------------------

class TestOneColumn:
    def test_one_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.txt"
        path.write_text("0.1\n0.2\n0.3\n0.4\n", encoding="utf-8")
        gm = GroundMotion.from_one_column(path, dt=0.01)
        np.testing.assert_allclose(gm.accel, [0.1, 0.2, 0.3, 0.4])
        assert gm.dt == 0.01

    def test_many_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.txt"
        path.write_text("0.1 0.2 0.3\n0.4 0.5 0.6\n", encoding="utf-8")
        gm = GroundMotion.from_one_column(path, dt=0.01)
        np.testing.assert_allclose(gm.accel, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_scale_factor(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.txt"
        path.write_text("1.0\n2.0\n3.0\n", encoding="utf-8")
        gm = GroundMotion.from_one_column(path, dt=0.01, scale_factor=0.5)
        np.testing.assert_allclose(gm.accel, [0.5, 1.0, 1.5])

    def test_skiprows(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.txt"
        path.write_text(
            "HEADER LINE 1\nHEADER LINE 2\n0.1 0.2\n0.3 0.4\n",
            encoding="utf-8",
        )
        gm = GroundMotion.from_one_column(path, dt=0.01, skiprows=2)
        np.testing.assert_allclose(gm.accel, [0.1, 0.2, 0.3, 0.4])


# ---------------------------------------------------------------------------
# PEER NGA AT2 parser
# ---------------------------------------------------------------------------

_PEER_FIXTURE = """\
PEER NGA STRONG MOTION DATABASE RECORD
IMPERIAL VALLEY 10/15/79, BONDS CORNER, 140
ACCELERATION TIME HISTORY IN UNITS OF G
NPTS=  8, DT=   .0050 SEC
  .1000E-03  .2000E-03  .3000E-03  .4000E-03  .5000E-03
  .6000E-03  .7000E-03  .8000E-03
"""


class TestPeerAt2:
    def test_parses_synthetic(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.AT2"
        path.write_text(_PEER_FIXTURE, encoding="utf-8")

        gm = GroundMotion.from_peer_at2(path)
        assert gm.npts == 8
        assert gm.dt == pytest.approx(0.005)
        np.testing.assert_allclose(
            gm.accel,
            [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4],
            rtol=1e-6,
        )
        assert gm.metadata["format"] == "peer_at2"
        assert "BONDS CORNER" in gm.metadata["header2"]
        assert "UNITS OF G" in gm.metadata["units_declared"]

    def test_scale_factor(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.AT2"
        path.write_text(_PEER_FIXTURE, encoding="utf-8")

        # PEER values are in g — multiply to push into SI.
        gm = GroundMotion.from_peer_at2(path, scale_factor=9.81)
        np.testing.assert_allclose(
            gm.accel[0:2], [9.81e-4, 9.81 * 2e-4], rtol=1e-6
        )

    def test_truncated_data_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.AT2"
        path.write_text(
            "PEER NGA STRONG MOTION DATABASE RECORD\n"
            "Truncated test\n"
            "ACCELERATION TIME HISTORY IN UNITS OF G\n"
            "NPTS=8, DT=0.005 SEC\n"
            "0.1 0.2 0.3 0.4\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="NPTS=8"):
            GroundMotion.from_peer_at2(path)

    def test_malformed_header4_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.AT2"
        path.write_text(
            "PEER NGA STRONG MOTION DATABASE RECORD\n"
            "Bad header\n"
            "ACCELERATION TIME HISTORY IN UNITS OF G\n"
            "garbage line 4 without keywords\n"
            "0.1 0.2\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="PEER header pattern"):
            GroundMotion.from_peer_at2(path)


# ---------------------------------------------------------------------------
# ITACA parser
# ---------------------------------------------------------------------------

_ITACA_FIXTURE = """\
EVENT_NAME: CENTRAL_ITALY
EVENT_DATE_YYYYMMDD: 20161030
STATION_CODE: AMT
NETWORK: IT
SAMPLING_INTERVAL_S: 0.005
NDATA: 5
UNITS: cm/s^2
DATA_TYPE: ACCELERATION
0.001
0.002
0.003
0.004
0.005
"""


class TestItaca:
    def test_parses_synthetic(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.asc"
        path.write_text(_ITACA_FIXTURE, encoding="utf-8")

        gm = GroundMotion.from_itaca(path)
        assert gm.npts == 5
        assert gm.dt == pytest.approx(0.005)
        np.testing.assert_allclose(
            gm.accel, [0.001, 0.002, 0.003, 0.004, 0.005]
        )
        assert gm.metadata["header"]["STATION_CODE"] == "AMT"
        assert gm.metadata["units_declared"] == "cm/s^2"

    def test_scale_factor(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.asc"
        path.write_text(_ITACA_FIXTURE, encoding="utf-8")

        # cm/s² → m/s²
        gm = GroundMotion.from_itaca(path, scale_factor=0.01)
        np.testing.assert_allclose(
            gm.accel, [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        )

    def test_truncates_to_declared_npts(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.asc"
        path.write_text(_ITACA_FIXTURE + "0.006\n", encoding="utf-8")
        gm = GroundMotion.from_itaca(path)
        assert gm.npts == 5
        assert gm.accel[-1] == pytest.approx(0.005)

    def test_missing_dt_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.asc"
        path.write_text(
            "EVENT_NAME: TEST\nNETWORK: IT\n0.1\n0.2\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="sampling interval"):
            GroundMotion.from_itaca(path)


# ---------------------------------------------------------------------------
# Sniffer
# ---------------------------------------------------------------------------

class TestSniffer:
    def test_peer_at2_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.AT2"
        path.write_text(_PEER_FIXTURE, encoding="utf-8")
        assert sniff_format(path) == "peer_at2"

    def test_itaca_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.asc"
        path.write_text(_ITACA_FIXTURE, encoding="utf-8")
        assert sniff_format(path) == "itaca"

    def test_two_column_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.txt"
        path.write_text("0.00  0.1\n0.01  0.2\n", encoding="utf-8")
        assert sniff_format(path) == "two_column"

    def test_one_column_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.txt"
        path.write_text("0.1\n0.2\n0.3\n", encoding="utf-8")
        assert sniff_format(path) == "one_column"

    def test_unknown_when_first_line_not_numeric(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.bin"
        path.write_text("\x00\x01\x02not numeric\n", encoding="utf-8")
        assert sniff_format(path) == "unknown"


# ---------------------------------------------------------------------------
# from_file dispatch
# ---------------------------------------------------------------------------

class TestFromFile:
    def test_dispatches_peer(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.AT2"
        path.write_text(_PEER_FIXTURE, encoding="utf-8")
        gm = from_file(path)
        assert gm.metadata["format"] == "peer_at2"

    def test_dispatches_itaca(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.asc"
        path.write_text(_ITACA_FIXTURE, encoding="utf-8")
        gm = from_file(path)
        assert gm.metadata["format"] == "itaca"

    def test_two_column_dispatch_no_units_required(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "rec.txt"
        path.write_text("0.00  0.1\n0.01  0.2\n", encoding="utf-8")
        gm = from_file(path)
        assert gm.metadata["format"] == "two_column"

    def test_scale_factor_propagates(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.txt"
        path.write_text("0.00  1.0\n0.01  2.0\n", encoding="utf-8")
        gm = from_file(path, scale_factor=2.5)
        np.testing.assert_allclose(gm.accel, [2.5, 5.0])

    def test_one_column_requires_dt(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.txt"
        path.write_text("0.1\n0.2\n0.3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="do not encode dt"):
            from_file(path)


# ---------------------------------------------------------------------------
# Real corpus smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _CORPUS is None,
    reason="examples/Records/03_Selected/ not reachable from this checkout",
)
class TestRealCorpus:
    """End-to-end smoke checks against the bundled homogenised records."""

    @pytest.mark.parametrize("name", [
        "RSN160_IMPVALL.H_H-BCR140_N.txt",
        "Llolleo_Maule2010_X_N.txt",
        "AMNT_201604162359_E_100_E.txt",
        "VinadelMarAlgarrobo1985_X_N.txt",
    ])
    def test_sniffer_picks_two_column(self, name: str) -> None:
        assert sniff_format(_CORPUS / name) == "two_column"

    @pytest.mark.parametrize("name, expected_dt", [
        ("RSN160_IMPVALL.H_H-BCR140_N.txt", 0.005),
        ("Llolleo_Maule2010_X_N.txt", 0.005),
        ("AMNT_201604162359_E_100_E.txt", 0.010),
        ("VinadelMarAlgarrobo1985_X_N.txt", 0.024860),
    ])
    def test_parses_with_expected_dt(
        self, name: str, expected_dt: float
    ) -> None:
        gm = GroundMotion.from_two_column(_CORPUS / name)
        assert gm.dt == pytest.approx(expected_dt, rel=1e-4)
        assert gm.npts > 100
        assert gm.source == name
        assert gm.pga > 0


# ---------------------------------------------------------------------------
# ObsPy bridge — error path + mock trace
# ---------------------------------------------------------------------------

class TestObspyBridge:
    def test_read_obspy_raises_without_obspy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "obspy":
                raise ImportError("simulated missing obspy")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        path = tmp_path / "rec.bin"
        path.write_text("not a real format\n", encoding="utf-8")
        with pytest.raises(ImportError, match="pip install obspy"):
            from_file(path)

    def test_from_obspy_trace_with_mock(self) -> None:
        class FakeStats:
            delta = 0.005
            network = "IT"
            station = "AMT"
            channel = "HNE"
            location = ""
            starttime = "2016-10-30T06:40:18.000Z"

        class FakeTrace:
            data = np.array([0.1, 0.2, 0.3, 0.4])
            stats = FakeStats()

        gm = GroundMotion.from_obspy_trace(FakeTrace())
        assert gm.npts == 4
        assert gm.dt == pytest.approx(0.005)
        assert gm.metadata["station"] == "AMT"
        assert gm.metadata["channel"] == "HNE"

    def test_from_obspy_trace_scale_factor(self) -> None:
        class FakeStats:
            delta = 0.005
            network = ""
            station = ""
            channel = ""
            location = ""
            starttime = ""

        class FakeTrace:
            data = np.array([1.0, 2.0])
            stats = FakeStats()

        gm = GroundMotion.from_obspy_trace(FakeTrace(), scale_factor=0.01)
        np.testing.assert_allclose(gm.accel, [0.01, 0.02])
