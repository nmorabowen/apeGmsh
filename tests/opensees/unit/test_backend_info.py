"""BackendInfo: one fork signal (``ladrunoBuild()`` returning a sha).

Fork-free: every case classifies a fake module, never the real (maybe
stale) build. The oracle for each is the slice card's (#1498):

* ``ladrunoBuild`` returning a sha → fork, with that sha;
* ``criticalTimeStep`` without ``ladrunoBuild`` (a fork build predating
  fork PR #718) → stock;
* a ``ladrunoBuild`` that raises → stock with ``build=None``, no exception;
* an empty or non-sha build string → ``None``;
* the resolver names the module actually imported, even when a ``.pth``
  aliases ``openseespy.opensees`` to the fork;
* the resolver's cache follows the bound module;
* a native ``results.h5`` carries the stamp as root attrs, absent when
  unknown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from apeGmsh.opensees._target import (
    BackendInfo,
    LadrunoOption,
    LadrunoSchema,
    _schema_from_obj,
    backend_info_of,
)
from apeGmsh.opensees.emitter import live

_SHA = "a240b9183c0ffee0123456789abcdef012345678"


@pytest.fixture(autouse=True)
def _restore_resolver_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test leaves the resolver's caches as it found them."""
    monkeypatch.setattr(live, "_OPS_CACHE", live._OPS_CACHE)
    monkeypatch.setattr(live, "_BACKEND_INFO", live._BACKEND_INFO)


def _fake(
    name: str = "opensees", *, build: Any = None, raises: bool = False,
    critical: bool = False, file: str | None = None,
) -> ModuleType:
    """An OpenSees-shaped module (wipe / model / element)."""
    mod = ModuleType(name)
    for cmd in ("wipe", "model", "element"):
        setattr(mod, cmd, lambda *a: None)
    mod.version = lambda: "3.7.1"  # type: ignore[attr-defined]
    if critical:
        mod.criticalTimeStep = lambda: 1.0  # type: ignore[attr-defined]
    if raises:
        def _boom() -> str:
            raise RuntimeError("See stderr output")
        mod.ladrunoBuild = _boom  # type: ignore[attr-defined]
    elif build is not None:
        mod.ladrunoBuild = lambda: build  # type: ignore[attr-defined]
    if file is not None:
        mod.__file__ = file
    return mod


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------
def test_ladruno_build_sha_is_fork_with_that_sha() -> None:
    info = backend_info_of(_fake(build=_SHA, critical=True))
    assert info == BackendInfo(
        kind="fork", build=_SHA, version="3.7.1", source="opensees",
    )


def test_critical_time_step_alone_is_stock() -> None:
    # A fork build predating ladrunoBuild (fork PR #718) reads as stock.
    info = backend_info_of(_fake(critical=True))
    assert (info.kind, info.build) == ("stock", None)


def test_raising_ladruno_build_is_stock_not_an_exception() -> None:
    info = backend_info_of(_fake(raises=True, critical=True))
    assert (info.kind, info.build) == ("stock", None)


@pytest.mark.parametrize("raw", ["", "   ", "unknown", _SHA[:9], 7, None])
def test_degenerate_build_is_none(raw: Any) -> None:
    mod = _fake()
    mod.ladrunoBuild = lambda: raw  # type: ignore[attr-defined]
    info = backend_info_of(mod)
    assert (info.kind, info.build) == ("stock", None)


def test_padded_sha_is_stripped() -> None:
    assert backend_info_of(_fake(build=f" {_SHA}\n")).build == _SHA


def test_version_that_raises_is_none() -> None:
    mod = _fake(build=_SHA)

    def _boom() -> str:
        raise RuntimeError("no version")
    mod.version = _boom  # type: ignore[attr-defined]
    assert backend_info_of(mod).version is None


def test_source_is_the_file_actually_imported() -> None:
    mod = _fake(build=_SHA, file="C:/Ladruno/bin/opensees.pyd")
    assert backend_info_of(mod).source == "C:/Ladruno/bin/opensees.pyd"


# --------------------------------------------------------------------------
# The resolver owns it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("build, name, stamp", [
    (_SHA, "ladruno-fork", _SHA),
    (None, "stock-openseespy", None),
])
def test_name_build_and_info_agree(
    monkeypatch: pytest.MonkeyPatch, build: str | None, name: str,
    stamp: str | None,
) -> None:
    mod = _fake(build=build, critical=True)
    monkeypatch.setattr(live, "_get_ops", lambda: mod)
    assert live.get_backend_name() == name
    assert live.get_backend_build() == stamp
    assert live.get_backend_info().build == stamp


def test_info_follows_the_bound_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fork, stock = _fake(build=_SHA), _fake()
    bound = [fork]
    monkeypatch.setattr(live, "_get_ops", lambda: bound[0])
    monkeypatch.setattr(live, "_BACKEND_INFO", None)
    assert live.get_backend_info().kind == "fork"
    bound[0] = stock            # a re-resolve binds another module
    assert live.get_backend_info().kind == "stock"


def test_cache_reset_with_the_ops_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resetting _OPS_CACHE (the reload seam) drops the cached verdict too.
    old, new = _fake(build=_SHA), _fake()
    monkeypatch.setattr(live, "_OPS_CACHE", None)
    monkeypatch.setattr(live, "_BACKEND_INFO", (old, backend_info_of(old)))
    monkeypatch.setattr(live, "_resolve_ops", lambda: new)
    assert live._get_ops() is new
    assert live._BACKEND_INFO is None
    assert live.get_backend_info().kind == "stock"


def test_pth_alias_reports_the_module_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fork's .pth may bind ``openseespy.opensees`` to the fork module:
    # the stock loader then returns the fork, and the verdict says so.
    fork = _fake(
        "openseespy.opensees", build=_SHA,
        file="C:/Program Files/Ladruno/OpenSees/bin/opensees.pyd",
    )
    pkg = ModuleType("openseespy")
    pkg.opensees = fork  # type: ignore[attr-defined]
    monkeypatch.delenv("APEGMSH_OPENSEES_BIN", raising=False)
    monkeypatch.delenv("LADRUNO_OPENSEES_QUIET", raising=False)
    monkeypatch.setitem(sys.modules, "opensees", None)  # bare fork absent
    monkeypatch.setitem(sys.modules, "openseespy", pkg)
    monkeypatch.setitem(sys.modules, "openseespy.opensees", fork)
    monkeypatch.setattr(live, "_OPS_CACHE", None)
    monkeypatch.setattr(live, "_BACKEND_INFO", None)

    info = live.get_backend_info()
    assert live._get_ops() is fork
    assert (info.kind, info.build) == ("fork", _SHA)
    assert info.source == fork.__file__
    assert live.get_backend_name() == "ladruno-fork"


# --------------------------------------------------------------------------
# results.h5 root attrs
# --------------------------------------------------------------------------
def _open_attrs(path: Path, **kw: Any) -> dict[str, Any]:
    import h5py

    from apeGmsh.results.writers._native import NativeWriter

    with NativeWriter(path) as w:
        w.open(**kw)
        # stamped with the header: present before any stage data lands
        assert w._h5 is not None
        before = dict(w._h5.attrs)
    with h5py.File(path, "r") as f:
        assert dict(f.attrs).keys() >= before.keys()
        return {k: v for k, v in f.attrs.items()}


def test_results_h5_carries_the_fork_stamp(tmp_path: Path) -> None:
    attrs = _open_attrs(
        tmp_path / "r.h5", opensees_backend="fork", opensees_build=_SHA,
    )
    assert attrs["opensees_backend"] == "fork"
    assert attrs["opensees_build"] == _SHA


def test_results_h5_stock_has_no_build(tmp_path: Path) -> None:
    attrs = _open_attrs(tmp_path / "r.h5", opensees_backend="stock")
    assert attrs["opensees_backend"] == "stock"
    assert "opensees_build" not in attrs


@pytest.mark.parametrize("build", [None, ""])
def test_results_h5_unknown_writes_nothing(tmp_path: Path, build: Any) -> None:
    attrs = _open_attrs(tmp_path / "r.h5", opensees_build=build)
    assert "opensees_backend" not in attrs
    assert "opensees_build" not in attrs


def test_results_h5_refuses_unknown_kind_and_stock_build(tmp_path: Path) -> None:
    from apeGmsh.results.writers._native import NativeWriter

    with pytest.raises(ValueError, match="expected one of"):
        NativeWriter(tmp_path / "a.h5").open(opensees_backend="ladruno")
    with pytest.raises(ValueError, match="only a fork build"):
        NativeWriter(tmp_path / "b.h5").open(
            opensees_backend="stock", opensees_build=_SHA,
        )
    assert not (tmp_path / "a.h5").exists()
    assert not (tmp_path / "b.h5").exists()


# --------------------------------------------------------------------------
# The writer's callers and the other backend tags read the same signal
# --------------------------------------------------------------------------
def _capture_attrs(tmp_path: Path, ops: Any) -> dict[str, Any]:
    import h5py

    from apeGmsh.results.capture._domain import DomainCapture
    from tests.test_results_domain_capture import _make_spec, _MockFem

    fem = _MockFem([1, 2])
    path = tmp_path / "cap.h5"
    with DomainCapture(
        _make_spec(snapshot_id=fem.snapshot_id), path, fem, ops=ops,
    ):
        pass
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


def test_domain_capture_stamps_the_fork_it_samples(tmp_path: Path) -> None:
    attrs = _capture_attrs(tmp_path, _fake(build=_SHA))
    assert attrs["opensees_backend"] == "fork"
    assert attrs["opensees_build"] == _SHA


def test_domain_capture_unstamped_fork_is_stock(tmp_path: Path) -> None:
    attrs = _capture_attrs(tmp_path, _fake(critical=True))
    assert attrs["opensees_backend"] == "stock"
    assert "opensees_build" not in attrs


def test_strut_tie_backend_tag_uses_the_one_signal() -> None:
    from apeGmsh.interop.strut_tie import _backend_tag

    assert _backend_tag(_fake(build=_SHA)) == "ladruno-fork"
    assert _backend_tag(_fake(critical=True)) == "stock-openseespy"


def _fork_info() -> BackendInfo:
    return BackendInfo(kind="fork", build=_SHA, version="3.8.0", source="x")


def test_footfall_results_carry_the_live_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py
    import numpy as np

    from apeGmsh.opensees.analysis.footfall_result import FootfallResult
    from tests.test_results_domain_capture import _MockFem

    monkeypatch.setattr(live, "get_backend_info", _fork_info)
    res = object.__new__(FootfallResult)
    for name, value in {
        "nodes": (1,), "a_p": np.array([1.0]), "ratio": np.array([0.5]),
        "f_dom": np.array([4.0]),
    }.items():
        object.__setattr__(res, name, value)
    out = res.to_results(_MockFem([1, 2]), tmp_path / "ff.h5")
    with h5py.File(out, "r") as f:
        assert f.attrs["opensees_backend"] == "fork"
        assert f.attrs["opensees_build"] == _SHA


def test_parallel_modal_to_native_does_not_stamp_the_in_process_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The modes came from a separate binary: even with a fork bound in this
    # process, the file must not claim it.
    import h5py

    from apeGmsh.opensees.analysis.modal import ParallelModalResult
    from tests.test_parallel_modal_to_native import _write_job
    from tests.test_results_domain_capture import _MockFem

    monkeypatch.setattr(live, "get_backend_info", _fork_info)
    job = _write_job(
        tmp_path, eigenvalues="100.0", nodes=[1], ndf=3, ndm=3,
        rows=["0.1 0.2 0.3\n"],
    )
    out = tmp_path / "modes.h5"
    ParallelModalResult.from_job(str(job)).to_native(out, _MockFem([1]))
    with h5py.File(out, "r") as f:
        assert "opensees_backend" not in f.attrs
        assert "opensees_build" not in f.attrs


# --------------------------------------------------------------------------
# ladrunoSchema v0 (fork PR #972; slice #1931): the five #972 contract rules
# --------------------------------------------------------------------------
def _schema_obj(**over: Any) -> dict[str, Any]:
    """A v0 schema as the fork emits it (keys sorted), with overrides."""
    obj: dict[str, Any] = {
        "build": _SHA,
        "commands": ["LadrunoTie", "ladrunoBuild", "ladrunoSchema"],
        "engine": "dl",
        "options": {
            "LadrunoRCConcrete": {
                "-Cd": {"nargs": "list", "since": "PR-155"},
                "-implexControl": {"nargs": 2, "since": "PR-263"},
            },
            "LadrunoRCFiniteStrain": {
                "-implexControl": {"nargs": 2, "since": "PR-263"},
            },
        },
        "schema_version": 0,
    }
    obj.update(over)
    return obj


def _with_schema(
    raw: Any = None, *, build: Any = _SHA, raises: bool = False, **over: Any,
) -> ModuleType:
    """A fork fake whose ``ladrunoSchema()`` returns ``raw`` (or a v0 JSON)."""
    mod = _fake(build=build)
    if raises:
        def _boom() -> str:
            raise RuntimeError("See stderr output")
        mod.ladrunoSchema = _boom  # type: ignore[attr-defined]
    else:
        text = raw if raw is not None else json.dumps(
            _schema_obj(**over), sort_keys=True, separators=(",", ":"),
        )
        mod.ladrunoSchema = lambda: text  # type: ignore[attr-defined]
    return mod


def test_v0_schema_parses_into_a_hashable_value() -> None:
    info = backend_info_of(_with_schema())
    s = info.schema
    assert s is not None
    assert (s.schema_version, s.build, s.engine) == (0, _SHA, "dl")
    assert s.commands == frozenset(
        {"LadrunoTie", "ladrunoBuild", "ladrunoSchema"}
    )
    assert s.option("LadrunoRCConcrete", "-Cd") == LadrunoOption("list", "PR-155")
    assert s.option("LadrunoRCConcrete", "-implexControl") == LadrunoOption(
        2, "PR-263"
    )
    # kind/build still come from ladrunoBuild alone
    assert (info.kind, info.build) == ("fork", _SHA)
    # no bare dict field: the value hashes, and equal probes compare equal
    again = backend_info_of(_with_schema())
    assert info == again and hash(info) == hash(again)
    assert {info, again} == {info}


def test_existing_construction_sites_default_schema_none() -> None:
    info = BackendInfo(kind="fork", build=_SHA, version="3.7.1", source="x")
    assert info.schema is None
    assert BackendInfo("stock", None, None, "y").schema is None


# Rule 1: schema_version first; anything doubtful reads as unknown.
@pytest.mark.parametrize("version", [1, -1, "0", 0.0, None, False, True])
def test_other_schema_version_is_unknown(version: Any) -> None:
    info = backend_info_of(_with_schema(schema_version=version))
    assert info.schema is None
    assert (info.kind, info.build) == ("fork", _SHA)


def test_v1_with_a_v0_shape_is_not_read_as_v0() -> None:
    # A v1 may change the meaning of any other key: even a v0-shaped body
    # is not trusted under another version.
    assert _schema_from_obj(_schema_obj(schema_version=1), _SHA) is None
    assert _schema_from_obj(_schema_obj(), _SHA) is not None


@pytest.mark.parametrize("raw", [
    "{not json", "", "[]", '"a string"', "null", "0",
    json.dumps({"schema_version": 0}),
])
def test_malformed_schema_text_is_unknown(raw: str) -> None:
    info = backend_info_of(_with_schema(raw))
    assert info.schema is None
    assert (info.kind, info.build) == ("fork", _SHA)


@pytest.mark.parametrize("raw", [7, b"{}", ["x"], {"schema_version": 0}])
def test_non_string_schema_is_unknown(raw: Any) -> None:
    mod = _fake(build=_SHA)
    mod.ladrunoSchema = lambda: raw  # type: ignore[attr-defined]
    assert backend_info_of(mod).schema is None


def test_raising_schema_is_unknown_not_an_exception() -> None:
    info = backend_info_of(_with_schema(raises=True))
    assert info.schema is None
    assert (info.kind, info.build) == ("fork", _SHA)


@pytest.mark.parametrize("over", [
    {"engine": ""}, {"engine": 3},
    {"commands": "LadrunoTie"}, {"commands": ["LadrunoTie", 3]},
    {"commands": ["LadrunoTie", "LadrunoTie"]},
    {"options": []},
    {"options": {"LadrunoRCConcrete": []}},
    {"options": {"LadrunoRCConcrete": {"-x": {"nargs": 1}}}},
    {"options": {"LadrunoRCConcrete": {"-x": {"nargs": 1, "since": 5}}}},
    {"options": {"LadrunoRCConcrete": {"-x": {"nargs": True, "since": "P"}}}},
    {"options": {"LadrunoRCConcrete": {"-x": {"nargs": 1.5, "since": "P"}}}},
    {"options": {"LadrunoRCConcrete": {
        "-x": {"nargs": 1, "since": "P", "extra": 0}}}},
    {"extra": 1},
])
def test_wrong_shape_is_unknown(over: dict[str, Any]) -> None:
    assert backend_info_of(_with_schema(**over)).schema is None


_V0_BODY = (
    f'"build":"{_SHA}","commands":["LadrunoTie"],"engine":"dl",'
    '"options":OPTS'
)


@pytest.mark.parametrize("text", [
    # a doubled schema_version: json.loads alone keeps the last one (v0)
    '{"schema_version":1,' + _V0_BODY.replace(
        "OPTS", '{"O":{"-t":{"nargs":1,"since":"P"}}}') + ',"schema_version":0}',
    # a doubled owner
    '{' + _V0_BODY.replace("OPTS", '{"O":{"-t":{"nargs":1,"since":"P"}},'
                           '"O":{"-u":{"nargs":1,"since":"P"}}}')
    + ',"schema_version":0}',
    # a doubled token
    '{' + _V0_BODY.replace("OPTS", '{"O":{"-t":{"nargs":1,"since":"P"},'
                           '"-t":{"nargs":2,"since":"P"}}}')
    + ',"schema_version":0}',
    # a doubled field inside one token spec
    '{' + _V0_BODY.replace("OPTS", '{"O":{"-t":{"nargs":1,"nargs":2,'
                           '"since":"P"}}}')
    + ',"schema_version":0}',
])
def test_duplicate_keys_at_any_level_are_unknown(text: str) -> None:
    # Without the duplicate the same body parses, so the refusal is the
    # duplicate's alone.
    assert json.loads(text)["schema_version"] == 0
    info = backend_info_of(_with_schema(text))
    assert info.schema is None
    assert (info.kind, info.build) == ("fork", _SHA)


def test_the_duplicate_free_body_parses() -> None:
    text = '{' + _V0_BODY.replace(
        "OPTS", '{"O":{"-t":{"nargs":1,"since":"P"}}}') + ',"schema_version":0}'
    assert backend_info_of(_with_schema(text)).supports_option("O", "-t") is True


def test_missing_key_is_unknown() -> None:
    obj = _schema_obj()
    del obj["engine"]
    assert backend_info_of(_with_schema(json.dumps(obj))).schema is None


# The schema's build must agree with ladrunoBuild() of the same module.
def test_build_mismatch_is_unknown() -> None:
    text = json.dumps(_schema_obj(build="b" * 40))
    info = backend_info_of(_with_schema(text))
    assert info.schema is None
    assert (info.kind, info.build) == ("fork", _SHA)


# Rule 5: no schema leaves kind/build exactly as before.
def test_build_without_schema_is_unknown_with_kind_and_build_unchanged() -> None:
    info = backend_info_of(_fake(build=_SHA, critical=True))
    assert info == BackendInfo(
        kind="fork", build=_SHA, version="3.7.1", source="opensees",
    )
    assert info.schema is None
    assert info.supports_option("LadrunoRCConcrete", "-Cd") is None
    assert info.has_command("LadrunoTie") is None


@pytest.mark.parametrize("build", [None, "unknown"])
def test_schema_never_makes_a_build_fork(build: Any) -> None:
    # Without a usable ladrunoBuild() the schema cannot be checked: unknown.
    info = backend_info_of(_with_schema(build=build))
    assert (info.kind, info.build, info.schema) == ("stock", None, None)


# Rule 2: a token is supported iff present in options[owner]; unknown is None.
def test_token_queries_answer_true_false_or_unknown() -> None:
    known = backend_info_of(_with_schema())
    assert known.supports_option("LadrunoRCConcrete", "-implexControl") is True
    assert known.supports_option("LadrunoRCConcrete", "-noSuchToken") is False
    assert known.supports_option("NoSuchOwner", "-implexControl") is False
    # a token listed under another owner does not leak across owners
    assert known.supports_option("LadrunoRCFiniteStrain", "-Cd") is False
    unknown = backend_info_of(_with_schema(schema_version=1))
    assert unknown.supports_option("LadrunoRCConcrete", "-implexControl") is None
    assert unknown.has_command("LadrunoTie") is None


# Rule 3: since is opaque, never compared or sorted.
class _NoOrderStr(str):
    def _refuse(self, other: object) -> bool:
        raise AssertionError("since was compared")

    __lt__ = __le__ = __gt__ = __ge__ = _refuse  # type: ignore[assignment]


def test_since_is_never_ordered() -> None:
    obj = _schema_obj(options={"LadrunoRCConcrete": {
        "-b": {"nargs": 1, "since": _NoOrderStr("PR-99")},
        "-a": {"nargs": 1, "since": _NoOrderStr("PR-100")},
    }})
    s = _schema_from_obj(obj, _SHA)
    assert s is not None
    assert s.option("LadrunoRCConcrete", "-a") == LadrunoOption(1, "PR-100")
    assert s.option("LadrunoRCConcrete", "-b") == LadrunoOption(1, "PR-99")
    with pytest.raises(TypeError):
        LadrunoOption(1, "PR-1") < LadrunoOption(1, "PR-2")  # type: ignore[operator]
    with pytest.raises(TypeError):
        s < s  # type: ignore[operator]


@pytest.mark.parametrize("since", ["", "PR-263", "v9.9.9", "zzz"])
def test_since_is_kept_verbatim(since: str) -> None:
    obj = _schema_obj(options={"O": {"-t": {"nargs": 0, "since": since}}})
    s = _schema_from_obj(obj, _SHA)
    assert s is not None and s.option("O", "-t") == LadrunoOption(0, since)


# Rule 4: commands are per-engine; a query speaks only for that engine.
def test_command_queries_speak_only_for_the_engine_that_answered() -> None:
    info = backend_info_of(_with_schema())
    assert isinstance(info.schema, LadrunoSchema)
    assert info.has_command("LadrunoTie") is True
    assert info.has_command("LadrunoStaggeredAnalyze") is False
    assert info.has_command("LadrunoTie", engine="dl") is True
    # the classic Tcl binary a subprocess runs is another engine: unknown
    assert info.has_command("LadrunoTie", engine="classic") is None
