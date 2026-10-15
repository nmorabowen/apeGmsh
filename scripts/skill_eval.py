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
* Each run is ``claude -p`` from its own empty scratch directory with an
  isolated config (``CLAUDE_CONFIG_DIR``): only the arm's skill, no user
  CLAUDE.md, no plugins. Log that config in once before the first run
  (``CLAUDE_CONFIG_DIR=<scratch>/config claude`` then ``/login``), or export
  ``ANTHROPIC_API_KEY``. Account-provisioned skills can still appear; the
  record names every skill that fired.
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
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
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
HARNESS_PATHS = ("evals/apegmsh", "scripts/skill_eval.py", "skills/apegmsh/scripts")
TASK_TOOLS = "Read,Write,Edit,Glob,Grep,Bash,Skill"
TASK_MAX_TURNS = 60
TRIGGER_MAX_TURNS = 3       # the skill fires on the first turn or not at all
RUN_TIMEOUT = 1800          # s, one claude -p run

sys.path.insert(0, str(REPO / "src"))
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
                           "is_error": None, "final_text": None}
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
                if item.get("type") == "tool_use" and item.get("name") == "Skill":
                    out["fired_skills"].append((item.get("input") or {}).get("skill"))
        elif kind == "result":
            out["usage"] = event.get("usage")
            out["cost_usd"] = event.get("total_cost_usd")
            out["num_turns"] = event.get("num_turns")
            out["is_error"] = event.get("is_error")
            out["final_text"] = (event.get("result") or "")[-500:]
    return out


def run_claude(cmd: list[str], prompt: str, *, cwd: Path, config: Path) -> tuple[dict[str, Any], float]:
    env = dict(os.environ)
    env["CLAUDE_CONFIG_DIR"] = str(config)
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
    parser.add_argument("--scratch", type=Path, default=None, help="scratch root (default: a new temp dir)")
    parser.add_argument("--out", type=Path, default=None, help="record path (default runs/<date>-<ref>.jsonl)")
    parser.add_argument("--competitor", type=Path, action="append", default=[],
                        help="an extra skill directory installed in every arm (repeatable), e.g. a "
                             "near-miss owner such as gmsh-structural; its content hash is recorded")
    parser.add_argument("--allow-dirty", action="store_true", help="run with uncommitted harness changes")
    parser.add_argument("--grade", nargs=2, metavar=("TASK", "SCRIPT"), help="grade one script and exit")
    args = parser.parse_args(argv)

    tasks = graders.load_tasks()
    if args.grade:
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

    scratch = (args.scratch or Path(tempfile.mkdtemp(prefix="apegmsh-skill-eval-"))).resolve()
    config = scratch / "config"
    config.mkdir(parents=True, exist_ok=True)
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
            workdir = scratch / "work" / item["kind"] / name / arm / f"rep{item['rep']}"
            workdir.mkdir(parents=True)
            if item["kind"] == "task":
                prompt = common["preamble"].format(python=args.python) + tasks[name]["prompt"]
                cmd = claude_command(claude_exe, model=args.model, max_turns=TASK_MAX_TURNS)
            else:
                prompt = trig_by_id[name]["prompt"]
                cmd = claude_command(claude_exe, model=args.model, max_turns=TRIGGER_MAX_TURNS)
            stream, seconds = run_claude(cmd, prompt, cwd=workdir, config=config)
            if stream["usage"] is None and n == 1:
                raise SystemExit("the first claude run produced no result event (not logged in?):\n"
                                 + stream.get("stderr_tail", ""))
            available = stream["available_skills"]
            if arm == "skill" and available is not None and SKILL_NAME not in json.dumps(available):
                raise SystemExit(f"the skill arm's init event does not list {SKILL_NAME!r}: {available}")
            row = {**item, "skill_hash": trees.get(arm), "seconds": round(seconds, 1), **stream}
            if item["kind"] == "task":
                grade = graders.grade(name, workdir / graders.SCRIPT_NAME, python=args.python, tasks=tasks)
                row["grade"] = grade.to_dict()
                row["pass"] = grade.passed
            else:
                fired = SKILL_NAME in stream["fired_skills"]
                row["expect"] = trig_by_id[name]["expect"]
                row["pass"] = fired if row["expect"] == SKILL_NAME else not fired
            rows.append(row)
            record.write(json.dumps(row) + "\n")
            record.flush()
            print(f"[{n}/{len(plan)}] {item['kind']} {name} {arm} rep {item['rep']}: "
                  f"{'PASS' if row['pass'] else 'fail'}  fired {stream['fired_skills']}")
        summary = {"kind": "summary", "pass_counts": summarise(rows)}
        record.write(json.dumps(summary) + "\n")
    print(json.dumps(summary["pass_counts"], indent=2))
    return 0


def _has_pyflakes(python: str) -> bool:
    run = subprocess.run([python, "-c", "import pyflakes"], capture_output=True)
    return run.returncode == 0


if __name__ == "__main__":
    sys.exit(main())
