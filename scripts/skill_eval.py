#!/usr/bin/env python
"""Run the apeGmsh skill evals locally and write a record (SK1-S0, #2003).

The maintainer runs this under their own account; CI never does (no workflow
holds a key). CI runs only ``tests/test_skill_eval_graders.py``.

What one run does:

* Builds the skill tree from git at ``--skill-ref``: the canonical
  ``skills/apegmsh/`` tree (not the ``.claude/skills/apegmsh-helper`` mirror,
  whose frontmatter differs), installed under the name ``apegmsh``. Its content
  hash is the git tree id ``git rev-parse <ref>:skills/apegmsh``, which a
  reviewer recomputes.
* Runs each task of ``evals/apegmsh/tasks.toml`` ``--reps`` times in each arm:
  ``skill`` (the tree at the ref), ``no_skill`` (no apegmsh skill), and
  ``previous`` (the skill hash of the newest record under ``runs/``; absent for
  the first record, and skipped when it equals this hash).
* Runs each prompt of ``triggers.toml`` ``--reps`` times in the skill arm and
  records which skills fired.
* Each run is ``claude -p`` with an isolated config (``CLAUDE_CONFIG_DIR``): only
  the arm's skill, no user CLAUDE.md, no plugins. The config holds the login and
  settings, so it lives in a private per-user directory (``--config``; default
  ``%LOCALAPPDATA%/apegmsh-skill-eval/config`` or
  ``$XDG_STATE_HOME/apegmsh-skill-eval/config``, made 0700 on POSIX) and persists
  between invocations. Log it in once::

      CLAUDE_CONFIG_DIR=<config> claude          # bash; then /login, then exit
      $env:CLAUDE_CONFIG_DIR="<config>"; claude  # PowerShell

  (``--dry-run`` prints both with the real path), or export ``ANTHROPIC_API_KEY``.
* Each run starts in its own empty directory under the work root (``--scratch``;
  default a fixed directory outside the home directory). The CLI reads every
  ``CLAUDE.md`` above its working directory, so the runner refuses a work root with
  a ``CLAUDE.md``, ``CLAUDE.local.md`` or ``.claude/`` in it or above it.
* The agent never sees this repository, which holds the answer key (the fixtures,
  the graders, and the skill the no-skill arm must not read). apeGmsh's ``src`` is
  extracted at the skill ref (``git archive``) into the work root; the agent and the
  grader both import that copy (``PYTHONPATH``), with ``--python`` first on
  ``PATH``, and every environment variable naming the repository is dropped from
  the agent's environment.
* The CLI's init event must list the loaded skills: the skill and previous arms
  must load exactly ``apegmsh``, the no-skill arm must not. Otherwise the run
  stops, since the arm is not what the record would claim. Account-provisioned
  skills can still appear; the record names every skill that fired.
* Grades each ``solution.py`` with ``evals/apegmsh/graders.py`` (re-run alone,
  lint, oracle) and appends one JSON line per run to the record, so a crash
  keeps the runs already made.

The record ``evals/apegmsh/runs/<date>-<skill ref>.jsonl`` holds a header (git
SHA, skill ref and hash, previous record and hash, requested model, CLI version,
lint environment), one line per task or trigger run (arm, pass, grader output,
tokens, the model each run reported), and a summary line.

Usage:
    python scripts/skill_eval.py --dry-run                  # plan + skill hash, no claude
    python scripts/skill_eval.py --skill-ref 6308b303 --model <id>
    python scripts/skill_eval.py --grade TASK SCRIPT        # grade one script by hand

Never run in CI. Budget: 6 tasks x 3 reps x 3 arms + 20 triggers x 3 = 114 runs.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import getpass
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "evals" / "apegmsh"
RUNS_DIR = EVAL_DIR / "runs"
TRIGGERS_FILE = EVAL_DIR / "triggers.toml"
SKILL_PATH = "skills/apegmsh"
SKILL_NAME = "apegmsh"
ARMS = ("skill", "no_skill", "previous")
# Files whose content decides a grade; a dirty copy would make the record's git SHA lie.
# (apeGmsh itself comes from git at the skill ref, so a dirty src/ cannot reach a run.)
HARNESS_PATHS = ("evals/apegmsh", "scripts/skill_eval.py", "skills/apegmsh/scripts")
TASK_TOOLS = "Read,Write,Edit,Glob,Grep,Bash,Skill"
TASK_MAX_TURNS = 60
TRIGGER_MAX_TURNS = 3       # the skill fires on the first turn or not at all
RUN_TIMEOUT = 1800          # s, one claude -p run

sys.path.insert(0, str(EVAL_DIR))
import graders  # noqa: E402  (needs the two paths above)


# --------------------------------------------------------------------------- git

def _git(*args: str) -> str:
    run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {run.stderr.strip()}")
    return run.stdout.strip()


def resolve_skill(ref: str) -> tuple[str, str]:
    """``(commit, tree)`` for ``ref``: the commit SHA and the skill tree's hash."""
    commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    tree = _git("rev-parse", "--verify", f"{commit}:{SKILL_PATH}")
    return commit, tree


def install_skill_tree(tree: str, dest: Path) -> None:
    """Extract the git tree ``tree`` into ``dest`` and check it names itself ``apegmsh``."""
    if _git("cat-file", "-t", tree) != "tree":
        raise SystemExit(f"{tree} is not a git tree")
    archive = subprocess.run(["git", "archive", "--format=tar", tree], cwd=REPO,
                             capture_output=True, check=True).stdout
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    skill_md = dest / "SKILL.md"
    if not skill_md.is_file():
        raise SystemExit(f"tree {tree} has no SKILL.md")
    head = skill_md.read_text(encoding="utf-8").split("---")[1]
    names = [ln.split(":", 1)[1].strip() for ln in head.splitlines() if ln.startswith("name:")]
    if names != [SKILL_NAME]:
        raise SystemExit(f"tree {tree} declares name {names}, expected [{SKILL_NAME!r}]")


def harness_dirty() -> list[str]:
    out = _git("status", "--porcelain", "--", *HARNESS_PATHS, f":(exclude){RUNS_DIR.relative_to(REPO).as_posix()}")
    return [ln for ln in out.splitlines() if ln.strip()]


def previous_record() -> tuple[Path | None, dict[str, Any] | None]:
    """The newest committed record and its header, or ``(None, None)``."""
    records = sorted(RUNS_DIR.glob("*.jsonl")) if RUNS_DIR.is_dir() else []
    if not records:
        return None, None
    with open(records[-1], encoding="utf-8") as fh:
        header = json.loads(fh.readline())
    if header.get("kind") != "header" or "skill_hash" not in header:
        raise SystemExit(f"{records[-1]} does not start with a record header")
    return records[-1], header


def dir_hash(path: Path) -> str:
    """sha256 over the relative paths and bytes of every file under ``path``."""
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(file.read_bytes() + b"\0")
    return digest.hexdigest()


def default_scratch() -> Path:
    """A fixed work root outside the home directory.

    Outside home, because the CLI reads every ``CLAUDE.md`` above its working
    directory, and the home directory holds the user's ``.claude/``. It holds only
    work directories and the extracted ``src``; the config is private (see
    :func:`default_config`).
    """
    base = Path(tempfile.gettempdir()).resolve()
    home = Path.home().resolve()
    if base.is_relative_to(home):
        base = Path(home.anchor)
    return base / "apegmsh-skill-eval"


def default_config() -> Path:
    """The isolated CLI config: a private per-user directory, since it holds the login."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "apegmsh-skill-eval" / "config"


SYSTEM_SID = "*S-1-5-18"      # NT AUTHORITY\SYSTEM, by SID so a localised name cannot miss


def _windows_owner() -> str:
    domain = os.environ.get("USERDOMAIN")
    user = os.environ.get("USERNAME") or getpass.getuser()
    return f"{domain}\\{user}" if domain else user


def acl_principals(path: Path) -> list[tuple[str, bool]]:
    """``(principal, inherited)`` for each ACE ``icacls`` lists on ``path``."""
    run = subprocess.run(["icacls", str(path)], capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f"icacls {path} failed: {run.stderr.strip() or run.stdout.strip()}")
    aces = []
    for number, line in enumerate(run.stdout.splitlines()):
        line = line[len(str(path)):] if number == 0 else line
        line = line.strip()
        if not line or ":" not in line or line.lower().startswith("successfully"):
            continue
        principal, rights = line.rsplit(":", 1)
        aces.append((principal.strip(), "(I)" in rights))
    return aces


def make_private_dir(path: Path) -> None:
    """Create ``path`` readable by its owner only, and fail closed if it is not.

    POSIX: mode 0700. Windows: ``icacls /inheritance:r`` then full control for the
    owner and SYSTEM only, so a directory under ``C:\\`` does not keep the inherited
    ``Authenticated Users:(M)``; the resulting ACL is read back and checked.
    """
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(path, 0o700)
        return
    owner = _windows_owner()
    run = subprocess.run(["icacls", str(path), "/inheritance:r",
                          "/grant:r", f"{owner}:(OI)(CI)F", "/grant:r", f"{SYSTEM_SID}:(OI)(CI)F"],
                         capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f"could not lock {path} to its owner: {run.stderr.strip() or run.stdout.strip()}")
    aces = acl_principals(path)
    allowed = {owner.lower(), owner.split("\\")[-1].lower()}
    strangers = [p for p, _ in aces if p.lower() not in allowed and not p.upper().endswith("\\SYSTEM")]
    inherited = [p for p, inh in aces if inh]
    if strangers or inherited or not aces:
        raise SystemExit(f"{path} is not private after icacls: other principals {strangers}, "
                         f"inherited {inherited}")


def extract_src(commit: str, dest: Path) -> str:
    """Extract apeGmsh's ``src`` at ``commit`` into ``dest``; return its git tree id."""
    tree = _git("rev-parse", "--verify", f"{commit}:src")
    archive = subprocess.run(["git", "archive", "--format=tar", tree], cwd=REPO,
                             capture_output=True, check=True).stdout
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    if not (dest / "apeGmsh" / "__init__.py").is_file():
        raise SystemExit(f"{commit}:src holds no apeGmsh package")
    return tree


def _names_repo(value: str) -> bool:
    repo = str(REPO)
    if os.name == "nt":
        return repo.lower() in value.lower() or REPO.as_posix().lower() in value.lower()
    return repo in value


# What the CLI and the interpreter need from the caller's environment; nothing else
# (OLDPWD, VIRTUAL_ENV, editor and shell state) reaches the agent.
AGENT_ENV_ALLOW = (
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
    "USERPROFILE", "HOME", "HOMEDRIVE", "HOMEPATH", "USERNAME", "USER", "LOGNAME",
    "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL", "TERM",
    "CLAUDE_CODE_GIT_BASH_PATH", "ANTHROPIC_API_KEY",
)


def agent_env(python: str, src: Path, config: Path) -> dict[str, str]:
    """The agent's environment, from an allow-list.

    ``PATH`` with ``python`` first, ``PYTHONPATH`` set to the extracted ``src``,
    ``CLAUDE_CONFIG_DIR``, the grader's two quieting variables, and the
    :data:`AGENT_ENV_ALLOW` names the caller has set. A value naming this repository
    is dropped (``PATH`` loses those entries only).
    """
    if _names_repo(os.path.abspath(python)) or _names_repo(str(src)):
        raise SystemExit("the interpreter and the extracted src must live outside the repository")
    grader = graders.child_env(python, src=src)
    upper = {k.upper(): v for k, v in os.environ.items()}
    env = {}
    for name in AGENT_ENV_ALLOW:
        if name in upper and not _names_repo(upper[name]):
            env[name] = upper[name]
    path = grader.get("PATH") or grader.get("Path") or ""
    env["PATH"] = os.pathsep.join(e for e in path.split(os.pathsep) if e and not _names_repo(e))
    env["PYTHONPATH"] = str(src)
    env["MPLBACKEND"] = grader["MPLBACKEND"]
    env["LADRUNO_OPENSEES_QUIET"] = grader["LADRUNO_OPENSEES_QUIET"]
    env["CLAUDE_CONFIG_DIR"] = str(config)
    return env


def editable_root(python: str) -> Path | None:
    """Where ``python``'s apeGmsh install points when it is an editable checkout."""
    code = ("import importlib.metadata as m\n"
            "try:\n    print(m.distribution('apeGmsh').read_text('direct_url.json') or '')\n"
            "except m.PackageNotFoundError:\n    print('')\n")
    run = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120)
    try:
        info = json.loads(run.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return None
    url = info.get("url", "")
    if not (info.get("dir_info") or {}).get("editable") or not url.startswith("file:"):
        return None
    from urllib.parse import unquote, urlparse
    path = unquote(urlparse(url).path)
    if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return Path(path)


def _path_forms(root: Path) -> list[str]:
    """``root`` as a transcript may spell it: native, forward slashes, git-bash ``/c/...``."""
    forward = str(root).replace("\\", "/").rstrip("/").lower()
    forms = [forward]
    drive = re.match(r"^([a-z]):(/.*)$", forward)
    if drive:
        forms.append(f"/{drive.group(1)}{drive.group(2)}")
    return forms


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def contamination(tool_calls: list[dict[str, Any]], roots: list[Path]) -> list[str]:
    """Each tool call (Read, Bash, Glob, Grep, ...) whose input names a path under a root."""
    patterns = [re.compile(re.escape(form) + r"(?=$|[/\s'\"`;)])")
                for root in roots for form in _path_forms(root)]
    hits = []
    for call in tool_calls:
        for text in _strings(call.get("input")):
            haystack = text.replace("\\", "/").lower()
            if any(pattern.search(haystack) for pattern in patterns):
                hits.append(f"{call.get('name')}: {text[:300]}")
                break
    return hits


def instruction_files(root: Path) -> list[str]:
    """Every CLAUDE.md, CLAUDE.local.md or .claude/ in ``root`` or above it."""
    found = []
    for folder in [root, *root.parents]:
        for name in ("CLAUDE.md", "CLAUDE.local.md", ".claude"):
            if (folder / name).exists():
                found.append(str(folder / name))
    return found


# --------------------------------------------------------------------------- claude

def claude_version(claude: str) -> str | None:
    exe = shutil.which(claude)
    if exe is None:
        return None
    run = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    return run.stdout.strip() or None


def claude_command(claude: str, *, model: str | None, max_turns: int) -> list[str]:
    """The ``claude -p`` command line; the prompt goes on stdin.

    On stdin, not as an argument: an npm install on Windows is ``claude.cmd``, and
    cmd.exe cuts a multi-line argument at its first newline.
    """
    cmd = [claude, "-p", "--output-format", "stream-json", "--verbose",
           "--max-turns", str(max_turns), "--allowedTools", TASK_TOOLS]
    if model:
        cmd += ["--model", model]
    return cmd


def parse_stream(lines: list[str]) -> dict[str, Any]:
    """Pull the model, the skills that fired, and the usage out of stream-json events."""
    out: dict[str, Any] = {"model": None, "fired_skills": [], "available_skills": None,
                           "usage": None, "cost_usd": None, "num_turns": None,
                           "is_error": None, "final_text": None, "tool_calls": []}
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            out["model"] = event.get("model")
            if "skills" in event:
                out["available_skills"] = event["skills"]
        elif kind == "assistant":
            for item in (event.get("message") or {}).get("content") or []:
                if item.get("type") != "tool_use":
                    continue
                out["tool_calls"].append({"name": item.get("name"), "input": item.get("input")})
                if item.get("name") == "Skill":
                    out["fired_skills"].append((item.get("input") or {}).get("skill"))
        elif kind == "result":
            out["usage"] = event.get("usage")
            out["cost_usd"] = event.get("total_cost_usd")
            out["num_turns"] = event.get("num_turns")
            out["is_error"] = event.get("is_error")
            out["final_text"] = (event.get("result") or "")[-500:]
    return out


def skill_names(available: Any) -> list[str]:
    """Skill names from the init event's ``skills`` list (names, or objects with a name)."""
    names = []
    for item in available:
        names.append(item if isinstance(item, str) else (item or {}).get("name"))
    return names


def check_arm_skills(arm: str, available: Any) -> list[str]:
    """Refuse a run whose loaded skills contradict its arm; return the loaded names."""
    if not isinstance(available, list):
        raise SystemExit("the CLI's init event lists no skills, so the run cannot show which "
                         "skills it loaded; refusing to record it")
    names = skill_names(available)
    loaded = SKILL_NAME in names                     # exact name: apegmsh-helper is not apegmsh
    if arm in ("skill", "previous") and not loaded:
        raise SystemExit(f"the {arm} arm did not load {SKILL_NAME!r}; it loaded {names}")
    if arm == "no_skill" and loaded:
        raise SystemExit(f"the no_skill arm loaded {SKILL_NAME!r}; it loaded {names}")
    return names


def grade_safely(task: str, script: Path, *, python: str, tasks: dict[str, Any],
                 src: Path = graders.SRC) -> dict[str, Any]:
    """Grade one run; a grader exception is a failed run with its traceback, not a crash."""
    try:
        return graders.grade(task, script, python=python, tasks=tasks, src=src).to_dict()
    except Exception:  # noqa: BLE001 - recorded in full, and the plan goes on
        return {"task": task, "passed": False, "failed_at": "grader_error",
                "error": traceback.format_exc()[-graders.OUTPUT_TAIL:]}


def run_claude(cmd: list[str], prompt: str, *, cwd: Path, env: dict[str, str]) -> tuple[dict[str, Any], float]:
    start = time.time()
    try:
        run = subprocess.run(cmd, input=prompt, cwd=cwd, env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=RUN_TIMEOUT)
        stream = parse_stream(run.stdout.splitlines())
        stream["returncode"] = run.returncode
        if stream["usage"] is None:
            stream["stderr_tail"] = run.stderr[-1500:]
    except subprocess.TimeoutExpired:
        stream = parse_stream([])
        stream["returncode"] = None
        stream["stderr_tail"] = f"timed out after {RUN_TIMEOUT} s"
    return stream, time.time() - start


# --------------------------------------------------------------------------- plan

def build_plan(tasks: dict[str, Any], triggers: list[dict[str, Any]], arms: list[str],
               reps: int, with_triggers: bool) -> list[dict[str, Any]]:
    plan = []
    for task_id in tasks:
        for arm in arms:
            for rep in range(1, reps + 1):
                plan.append({"kind": "task", "task": task_id, "arm": arm, "rep": rep})
    if with_triggers:
        for trig in triggers:
            for rep in range(1, reps + 1):
                plan.append({"kind": "trigger", "trigger": trig["id"], "arm": "skill", "rep": rep})
    return plan


def load_triggers() -> list[dict[str, Any]]:
    with open(TRIGGERS_FILE, "rb") as fh:
        triggers = tomllib.load(fh)["trigger"]
    for trig in triggers:
        allowed = {"id", "expect", "prompt", "instead"}
        if set(trig) - allowed or trig.get("expect") not in ("apegmsh", "other"):
            raise SystemExit(f"malformed trigger {trig.get('id')!r}")
    return triggers


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tally: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        key = row["task"] if row["kind"] == "task" else "triggers"
        cell = tally.setdefault(key, {}).setdefault(row["arm"], [0, 0])
        cell[0] += int(bool(row["pass"]))
        cell[1] += 1
    return {k: {arm: f"{p}/{n}" for arm, (p, n) in v.items()} for k, v in tally.items()}


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skill-ref", default="HEAD", help="git ref of the skill to measure (default HEAD)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the skill hash; run nothing")
    parser.add_argument("--model", default=None, help="model id passed to claude --model")
    parser.add_argument("--reps", type=int, default=3, help="runs per task (and trigger) per arm")
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of " + ",".join(ARMS))
    parser.add_argument("--tasks", default=None, help="comma-separated subset of task ids")
    parser.add_argument("--no-triggers", action="store_true", help="skip the trigger set")
    parser.add_argument("--claude", default="claude", help="the claude CLI executable")
    parser.add_argument("--python", default=sys.executable, help="interpreter the agent and the grader use")
    parser.add_argument("--scratch", type=Path, default=None,
                        help=f"work root, outside home (default {default_scratch()})")
    parser.add_argument("--config", type=Path, default=None,
                        help=f"private isolated CLI config, holding the login (default {default_config()})")
    parser.add_argument("--out", type=Path, default=None, help="record path (default runs/<date>-<ref>.jsonl)")
    parser.add_argument("--competitor", type=Path, action="append", default=[],
                        help="an extra skill directory installed in every arm (repeatable), e.g. a "
                             "near-miss owner such as gmsh-structural; its content hash is recorded")
    parser.add_argument("--allow-dirty", action="store_true", help="run with uncommitted harness changes")
    parser.add_argument("--grade", nargs=2, metavar=("TASK", "SCRIPT"), help="grade one script and exit")
    args = parser.parse_args(argv)

    tasks = graders.load_tasks()
    if args.grade:
        sys.path.insert(0, str(graders.SRC))     # by hand: this checkout's apeGmsh
        result = graders.grade(args.grade[0], Path(args.grade[1]), python=args.python, tasks=tasks)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.passed else 1

    if args.tasks:
        wanted = args.tasks.split(",")
        unknown = sorted(set(wanted) - set(tasks))
        if unknown:
            raise SystemExit(f"unknown tasks {unknown}; known: {sorted(tasks)}")
        tasks = {t: tasks[t] for t in wanted}
    arms = args.arms.split(",")
    if set(arms) - set(ARMS):
        raise SystemExit(f"unknown arms {sorted(set(arms) - set(ARMS))}; known: {list(ARMS)}")
    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")
    triggers = load_triggers()
    common = graders.load_common()

    git_sha = _git("rev-parse", "HEAD")
    skill_commit, skill_hash = resolve_skill(args.skill_ref)
    prev_path, prev_header = previous_record()
    prev_hash = prev_header["skill_hash"] if prev_header else None
    notes = []
    if "previous" in arms and prev_hash is None:
        arms.remove("previous")
        notes.append("no previous record: this is the first, so it has no previous-hash arm")
    elif "previous" in arms and prev_hash == skill_hash:
        arms.remove("previous")
        notes.append(f"the previous record measured this same skill hash {skill_hash}: arm skipped")
    competitors = {}
    for comp in args.competitor:
        if not (comp / "SKILL.md").is_file():
            raise SystemExit(f"--competitor {comp} has no SKILL.md")
        competitors[comp.name] = dir_hash(comp)
    plan = build_plan(tasks, triggers, arms, args.reps, not args.no_triggers)
    version = claude_version(args.claude)
    stamp = _dt.date.today().isoformat()
    out = args.out or RUNS_DIR / f"{stamp}-{skill_commit[:8]}.jsonl"
    dirty = harness_dirty()
    scratch = (args.scratch or default_scratch()).resolve()
    config = (args.config or default_config()).resolve()
    leaks = instruction_files(scratch)
    src_tree = _git("rev-parse", "--verify", f"{skill_commit}:src")
    # A run that reads the repository or the editable checkout saw the answer key.
    roots = [REPO]
    checkout = editable_root(args.python)
    if checkout is not None and checkout.resolve() != REPO:
        roots.append(checkout)

    header = {
        "kind": "header",
        "date": stamp,
        "git_sha": git_sha,
        "harness_dirty": bool(dirty),
        "skill_ref": args.skill_ref,
        "skill_commit": skill_commit,
        "skill_hash": skill_hash,
        "skill_copy": f"{SKILL_PATH} (canonical tree, installed as '{SKILL_NAME}'; not the "
                      ".claude/skills/apegmsh-helper mirror)",
        "previous_record": prev_path.name if prev_path else None,
        "previous_skill_hash": prev_hash,
        "arms": arms,
        "reps": args.reps,
        "model_requested": args.model,
        "cli_version": version,
        "python": args.python,
        "lint_pyflakes": _has_pyflakes(args.python),
        "competitors": competitors,
        "apegmsh_src": f"src/ at {skill_commit} (tree {src_tree}), extracted; agent and grader",
        "scratch": str(scratch),
        "config": str(config),
        "instruction_files_above_scratch": leaks,
        "contamination_roots": [str(r) for r in roots],
        "notes": notes,
    }

    print(f"git SHA        {git_sha}{'  (harness dirty)' if dirty else ''}")
    print(f"skill ref      {args.skill_ref} -> {skill_commit}")
    print(f"skill hash     {skill_hash}   (git rev-parse {skill_commit[:12]}:{SKILL_PATH})")
    print(f"previous       {prev_path.name if prev_path else 'none'}  hash {prev_hash or '-'}")
    print(f"arms           {', '.join(arms)}   reps {args.reps}")
    print(f"claude CLI     {version or 'not found'}   model {args.model or '(CLI default)'}")
    for note in notes:
        print(f"note           {note}")
    print(f"record         {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    print(f"apeGmsh src    {skill_commit[:12]}:src (tree {src_tree}), extracted under the work root")
    print(f"work root      {scratch}")
    print(f"config         {config}   (private; holds the login)")
    print(f"log in once    bash:       CLAUDE_CONFIG_DIR=\"{config}\" {args.claude}   then /login")
    print(f"               PowerShell: $env:CLAUDE_CONFIG_DIR=\"{config}\"; {args.claude}   then /login")
    print(f"CLAUDE.md etc. {leaks or 'none in or above the scratch root'}")
    print(f"contamination  a run touching {[str(r) for r in roots]} is recorded contaminated")
    print(f"planned runs   {len(plan)}")
    for item in plan:
        name = item.get("task") or item.get("trigger")
        print(f"  {item['kind']:7s} {name:30s} {item['arm']:9s} rep {item['rep']}")
    if args.dry_run:
        return 0

    if dirty and not args.allow_dirty:
        raise SystemExit("uncommitted harness changes (the record's git SHA would lie):\n  "
                         + "\n  ".join(dirty) + "\ncommit them, or pass --allow-dirty")
    claude_exe = shutil.which(args.claude)
    claude_exe = os.path.abspath(claude_exe) if claude_exe else None   # runs start in their own cwd
    if version is None or claude_exe is None:
        raise SystemExit(f"claude CLI {args.claude!r} not found")
    if out.exists():
        raise SystemExit(f"{out} exists; pass --out to write a new record")
    if leaks:
        raise SystemExit("instruction files in or above the scratch root would reach every run:\n  "
                         + "\n  ".join(leaks) + "\npass a --scratch outside them")
    work_root = scratch / "work" / out.stem
    if work_root.exists():
        raise SystemExit(f"{work_root} exists from an earlier run; remove it or pass another --out")
    if _names_repo(str(scratch)):
        raise SystemExit("the work root must lie outside the repository (it holds the answer key)")
    make_private_dir(config)
    make_private_dir(work_root)
    src_copy = work_root / "apegmsh-src"
    extract_src(skill_commit, src_copy)
    # The grader's in-process reads (FEMData.from_h5) use the same copy.
    sys.path.insert(0, str(src_copy))
    import apeGmsh
    if not Path(apeGmsh.__file__).resolve().is_relative_to(src_copy.resolve()):
        raise SystemExit(f"apeGmsh imports from {apeGmsh.__file__}, not the extracted {src_copy}")
    trees = {"skill": skill_hash, "previous": prev_hash}
    trig_by_id = {t["id"]: t for t in triggers}
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with open(out, "w", encoding="utf-8") as record:
        record.write(json.dumps(header) + "\n")
        for n, item in enumerate(plan, start=1):
            arm = item["arm"]
            skills_dir = config / "skills"
            shutil.rmtree(skills_dir, ignore_errors=True)
            skills_dir.mkdir()
            if trees.get(arm):
                install_skill_tree(trees[arm], skills_dir / SKILL_NAME)
            for comp in args.competitor:
                shutil.copytree(comp, skills_dir / comp.name)
            name = item.get("task") or item.get("trigger")
            workdir = work_root / item["kind"] / name / arm / f"rep{item['rep']}"
            workdir.mkdir(parents=True)
            if item["kind"] == "task":
                prompt = common["preamble"].format(python=args.python) + tasks[name]["prompt"]
                cmd = claude_command(claude_exe, model=args.model, max_turns=TASK_MAX_TURNS)
            else:
                prompt = trig_by_id[name]["prompt"]
                cmd = claude_command(claude_exe, model=args.model, max_turns=TRIGGER_MAX_TURNS)
            stream, seconds = run_claude(cmd, prompt, cwd=workdir,
                                         env=agent_env(args.python, src_copy, config))
            if stream["usage"] is None and n == 1:
                raise SystemExit("the first claude run produced no result event (not logged in?):\n"
                                 + stream.get("stderr_tail", ""))
            check_arm_skills(arm, stream["available_skills"])
            tool_calls = stream.pop("tool_calls")
            row = {**item, "skill_hash": trees.get(arm), "seconds": round(seconds, 1), **stream,
                   "tool_call_count": len(tool_calls), "contaminated": contamination(tool_calls, roots)}
            if item["kind"] == "task":
                row["grade"] = grade_safely(name, workdir / graders.SCRIPT_NAME, python=args.python,
                                            tasks=tasks, src=src_copy)
                row["pass"] = row["grade"]["passed"]
            else:
                fired = SKILL_NAME in stream["fired_skills"]
                row["expect"] = trig_by_id[name]["expect"]
                row["pass"] = fired if row["expect"] == SKILL_NAME else not fired
            if row["contaminated"]:
                row["pass"] = False      # it saw the answer key: never a pass
            rows.append(row)
            record.write(json.dumps(row) + "\n")
            record.flush()
            print(f"[{n}/{len(plan)}] {item['kind']} {name} {arm} rep {item['rep']}: "
                  f"{'PASS' if row['pass'] else 'fail'}  fired {stream['fired_skills']}"
                  f"{'  CONTAMINATED' if row['contaminated'] else ''}")
        summary = {"kind": "summary", "pass_counts": summarise(rows)}
        record.write(json.dumps(summary) + "\n")
    print(json.dumps(summary["pass_counts"], indent=2))
    return 0


def _has_pyflakes(python: str) -> bool:
    run = subprocess.run([python, "-c", "import pyflakes"], capture_output=True)
    return run.returncode == 0


if __name__ == "__main__":
    sys.exit(main())
