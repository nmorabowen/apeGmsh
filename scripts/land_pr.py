"""Land one PR on main: the "How work lands" checklist (AGENTS.md) as code.

    python scripts/land_pr.py <number>            # check, squash-merge, prove it landed
    python scripts/land_pr.py <number> --dry-run  # run the checks and stop before the merge
    python scripts/land_pr.py <number> --verify-only  # only the post-merge proof, for a merged PR
    python scripts/land_pr.py --check-card <issue>  # print a slice card's parsed Owned set and skipped amendment lines
    python scripts/land_pr.py <number> --owned-only   # only check 9 (Owned files), on any PR

The checks run in order and the first failure refuses, naming the check and
the lesson. Checks 7 and 8 are ADR 0121 D4 (invariant 8): a diff that sets an
ADR's Status to Accepted is refused outright, and an edit to a baseline file
(the freeze census, the retirement ledger) needs the ``human-gate`` label. The maintainer's web merge is the
exception path for both, by design. Check 9 (lesson #1788) refuses a ``program``
PR whose changed files fall outside its slice card's Owned list; the ``scope-ok``
label, applied by the maintainer or coordinator, overrides it, the way
``human-gate`` admits a baseline edit at check 8. ``--owned-only`` runs check 9
alone on any PR, merged ones included.

Checks 10 and 11 are lesson #1877 (3af31061 merged #1698 into a red ``main``).
Check 10 refuses while ``main``'s latest completed Tests run is red, when main's
head H has no run, and, while H's run is in progress, unless the PR's checks ran
on a base that contains H; a read that fails refuses. The ``fixes-red-main``
label, for the fix or the revert, overrides it. Check 11 refuses a PR that touches ``tests/`` or a BASELINE file
when its required checks ran on a merge base more than STALE_COMMITS commits, or
STALE_HOURS hours, behind ``main``. The base is read from the job log's
``Merge <head> into <base>`` checkout line, the only place the API exposes it.

Check 12 (#2138 bumped ``FORK_PIN`` with only the stock lane run; the fork side went red that night, #2143) refuses
a PR that changes ``FORK_PIN`` unless a ``live-fork`` ``workflow_dispatch`` run on the PR head SHA completed
green. ``live-fork`` is not a required check, so nothing else proves a new pin. No label overrides it.

Whatever copy is invoked, the rules that run are origin/main's: ``main``
fetches origin/main and re-executes ``git show origin/main:scripts/land_pr.py``
from a temp file (maintainer ruling 2 on #1686), so a PR that edits this file
is judged by the rules it would replace. Then ``gh pr merge --squash --match-head-commit <checked head>``
(never ``--auto``, never ``--delete-branch``), so a push between the checks
and the merge is refused by GitHub; then ``compare/main...<squash sha>`` must
read ``behind`` or ``identical``; then the branch tip must still be the
merged head, so a push that raced the merge is flagged rather than left on
an orphaned branch. A push after the landing is the nightly orphan
detector's job (.github/workflows/orphans.yml, #1230).

Every read after the merge is retried with bounded backoff (the merge call
never is); if one keeps failing the script exits non-zero with ``MERGED BUT
UNVERIFIED: #N`` and the ``--verify-only`` command to run (#1682).

GitHub is reached only through ``run_gh`` and the checkout only through
``run_git``, so tests/test_land_pr.py replays recorded responses through
those seams with neither ``gh`` nor a network. Stdlib only.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

Runner = Callable[[list[str]], str]
#: `_real_files`: whether a name is a file on origin/main.
Real = Callable[[str], bool]

#: The checks `main` requires (AGENTS.md "How work lands"). One that has not
#: reported is a refusal: zero checks visible means a conflict or a stalled
#: Actions, not green (#630).
REQUIRED_CHECKS = ("lock-tests", "emit-cost-gate", "static-gates", "suite", "live-stock")
PR_FIELDS = (
    "state,isDraft,baseRefName,headRefName,headRefOid,mergeable,statusCheckRollup,"
    "labels,body,closingIssuesReferences"
)
#: The one PR files listing, read by checks 5, 7 and 8: one compact JSON row per
#: file, [filename, status, previous_filename, changes, patch]. `previous_filename`
#: makes a rename out of a frozen or baseline path visible (#1775).
FILES_JQ = ".[] | [.filename, .status, .previous_filename, .changes, (.patch // null)] | @json"
FileRow = list[Any]
#: ADR 0121 D4 (b): the census baseline. `freeze_census.py` holds FROZEN and
#: ADMITTED; `test_freeze_lock.py` holds FREEZE_POINT_DIGEST and the lock that
#: compares them, so editing either one edits the baseline. X4-a (#1806) adds
#: the retirement ledger: adding, re-dating or removing a row is a maintainer gate.
#: `test_loc_ratchet.py` holds the X4-b shrink-only LOC baselines (#1807).
#: `test_loc_ratchet_escapes.py` holds the escape guards' pins (PACKAGES_BASELINE,
#: FROM_RECORDERS_CLOSURE_BASELINE), gated like the ratchet's own (maintainer G4 on #1884, #1886).
#: C4 (#1967): `test_lazy_import_ratchet.py` pins the lazy cross-package import share and
#: `lazy_import_snapshot.txt` is the site list its failure diff names.
BASELINE = (
    "tests/freeze_census.py",
    "tests/test_freeze_lock.py",
    "internal_docs/program/retirement_ledger.toml",
    "tests/test_loc_ratchet.py",
    "tests/test_loc_ratchet_escapes.py",
    "tests/test_lazy_import_ratchet.py",
    "tests/lazy_import_snapshot.txt",
)
HUMAN_GATE = "human-gate"
ADR_FILE = re.compile(r"architecture/decisions/\d{4}-[^/]*\.md")
#: `scripts/adr_index.py::parse_adr` reads the Status line from the first 30 lines.
ADR_HEADER_LINES = 30
#: A slice PR names its slice issue as `Slice: #<n>` (PROGRAM.md section 7).
SLICE_REF = re.compile(r"^Slice:\s*#(\d+)", re.MULTILINE)
#: A body written from a Windows tool can begin with a UTF-8 BOM and end lines in CRLF
#: (#1887: "<BOM>Slice: #1857" missed the line-start match, so check 9 refused). An
#: invisible BOM or zero-width space before a `#` heading or a bold label would hide it and
#: keep an Owned section open, so a line-start run is deleted. Elsewhere a mark becomes a
#: space: deleting it would glue "Do<ZWSP>not" into "Donot" and hide the negation (review of #1917).
_LEADING_INVISIBLE = re.compile(r"^[\ufeff\u200b]+", re.MULTILINE)
_INVISIBLE = re.compile(r"[\ufeff\u200b]")
#: CRLF and a lone CR each end a line: deleting a lone CR would join two lines (review of #1917).
_CR = re.compile(r"\r\n?")
_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
#: Check 9 (lesson #1788): a PR with this label must stay inside its card's Owned list.
PROGRAM = "program"
SCOPE_OK = "scope-ok"
#: Every PR adds a changelog fragment, so no card has to own one. Nothing else is implicit.
IMPLICIT_ALLOWED = ("changelog.d/*.md",)
#: The card sections whose backticked paths a slice PR may change (slice_card.md).
_OWNED_HEADING = re.compile(r"^#{2,4}\s*(owned files?|append-only|allowed)\b", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6}\s")
#: A ruling-style card names its list after a label that opens a line or a sentence:
#: "Owned: `tests/test_loc_ratchet.py`" (#1851), "**D1.** Owned files:". "A file not
#: owned: `x`" and "unowned:" are not labels (review of #1858).
_OWNED_LABEL = re.compile(r"(?:^\s*(?:[-*+]\s+)?|[.!?]\**\s+)\**owned(?: files?)?\**:", re.IGNORECASE)
#: A bold pseudo-heading ("**Do not read:** ...") ends an Owned section as a `#` heading does.
_BOLD_LABEL = re.compile(r"^\s*\*\*[^*\n]+?(?::\*\*|\*\*:)")
#: An Owned line that negates ("`a.py` (never touch `b.py`)") cannot be read safely: refuse it.
#: Letter boundaries, not `\b`, so "Do _not_ touch" negates (#1872); any n't contraction
#: (straight or curly apostrophe) counts. Keep the list in slice_card.md in sync.
_NEGATION = re.compile(
    r"(?<![a-z])(?:never|not|cannot|[a-z]*n['\u2019]t|neither|except|exclud(?:e|es|ing)|avoid(?:s|ing)?"
    r"|untouched|forbidden|out\s+of\s+scope|removed\s+from|leave(?![a-z]).*?(?<![a-z])alone"
    r"|no\s+(?:changes?|edits?)\s+to|no\s*(?=`))(?![a-z])",
    re.IGNORECASE,
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LIST_LINE = re.compile(r"\s*(?:[-*+]|\d+[.)])\s|\s")
_BULLET = re.compile(r"\s*(?:[-*+]|\d+[.)])\s")
#: A list item that opens with a backticked span ("- `FORK_PIN` (four keys)"): only that
#: leading span may be an extensionless file name (T's ruling on #1944, review F1).
_LEADING_SPAN = re.compile(r"\s*(?:[-*+]|\d+[.)])\s+`")
#: Words that may sit around a negation in a backticked span that is only a negation
#: phrase (`_quoted_negation`): "`do not` touch", "`must never edit`", "`never, ever`",
#: "`absolutely not`", "`can not`", "`not under any circumstances`" (#1891).
_FILLER = frozenset(
    "do does did must should shall will may can please touch edit edits change changes modify add use it this"
    " that these those them file files ever absolutely under any circumstances".split()
)
#: Editing verbs: in a quoted negation phrase, whatever follows one is its object, so
#: "`do not touch b`" negates whatever `b` is (#1891), while "`cannot join:`" (#1857) does not.
_VERBS = frozenset({"touch", "edit", "edits", "change", "changes", "modify", "add", "use"})
_DOT_EXT = re.compile(r"\.\w")
#: How far a negation in a card amendment reaches (`_amendment_paths`); wider is larger.
_FREE, _ITEM, _PARA, _LIST = range(4)
_AMENDMENT = re.compile(r"^\s*\**\s*card amendment", re.IGNORECASE)
_TICKED = re.compile(r"`([^`\n]+)`")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PATHLIKE =re.compile(r"[\w.\-/*?\[\]]+(?: [\w.\-/*?\[\]]+)*")
_GLOB = re.compile(r"[*?\[]")
COMPARE_TRIES = 5
COMPARE_WAIT_S = 3.0
#: Check 10 (lesson #1877, T's ruling on #1877): main's recent commits (head H first) and
#: its recent Tests push runs, in bounded reads (one page each, no --paginate, no wait).
#: Only tests.yml's push runs count: deps-canary's scheduled `suite` on the same commit does not.
MAIN_PAGE = 30
MAIN_COMMITS = f"repos/{{owner}}/{{repo}}/commits?sha=main&per_page={MAIN_PAGE}"
MAIN_COMMITS_JQ = "[.[].sha]"
MAIN_RUNS = f"repos/{{owner}}/{{repo}}/actions/workflows/tests.yml/runs?branch=main&event=push&per_page={MAIN_PAGE}"
MAIN_RUNS_JQ = "[.workflow_runs[] | {id, head_sha, status, conclusion}]"
#: The jobs of one main Tests run (the latest attempt); only REQUIRED_CHECKS among them judge main.
MAIN_JOBS = "repos/{{owner}}/{{repo}}/actions/runs/{run}/jobs?per_page=100"
MAIN_JOBS_JQ = "{total: .total_count, jobs: [.jobs[] | {name, status, conclusion}]}"
#: A required job that ended so is red; any other non-success (cancelled, skipped) asks for a re-run.
RED_CONCLUSIONS = ("failure", "timed_out")
FIXES_RED_MAIN = "fixes-red-main"
UNVERIFIED = (
    "Merge main and re-run, or wait for main's run; the fix or the revert of a red main lands "
    f"under the {FIXES_RED_MAIN!r} label (lesson #1877)."
)
RERUN = (
    "Re-run them on a fresh merge ref: merge main into the branch (or push an empty commit) "
    "and wait for the checks. The Re-run button reuses the old merge ref."
)
#: Check 11 (lesson #1877): a PR that touches tests/ or a BASELINE file refuses when its
#: required checks ran on a base more than STALE_COMMITS commits behind main, or when
#: main's tip landed more than STALE_HOURS after those checks started. #1698 was 68
#: commits and 43 h behind; a program day lands 40 PRs.
STALE_COMMITS = 10
STALE_HOURS = 24
_ACTIONS_JOB = re.compile(r"/actions/runs/(\d+)/job/(\d+)")
#: actions/checkout's line for a pull_request run: the merge ref is GitHub's test merge.
_CHECKOUT = re.compile(r"HEAD is now at [0-9a-f]+ Merge ([0-9a-f]{40}) into ([0-9a-f]{40})")
#: `compare/<base>...main`: commits on main the base lacks, and main's tip date (oldest first).
BEHIND_JQ = "[.ahead_by, (.commits | last | .commit.committer.date)]"


class LandError(Exception):
    """A check refused, or the merge could not be proven to have reached main."""


def run_gh(args: list[str]) -> str:
    # utf-8, not the locale codec: PR bodies and ADRs carry non-ASCII text.
    return subprocess.run(["gh", *args], check=True, stdout=subprocess.PIPE, encoding="utf-8").stdout


def run_git(args: list[str]) -> str:
    return subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE, encoding="utf-8").stdout


def run_python(argv: list[str], env: dict[str, str]) -> int:
    return subprocess.run([sys.executable, *argv], env=env).returncode


#: Set to the path of main's copy when it runs (ruling 2 on #1686); see `_run_mains_copy`.
MAIN_COPY_ENV = "APEGMSH_LAND_PR_MAIN_COPY"
SELF = "scripts/land_pr.py"


def _is_mains_copy() -> bool:
    return os.environ.get(MAIN_COPY_ENV) == os.path.abspath(__file__)


def _safe_io_encoding(current: str | None) -> str:
    """PYTHONIOENCODING for the child: keep its encoding, force errors to replace."""
    if not current:
        return ":replace"
    return current if current.endswith(":replace") else current.split(":")[0] + ":replace"


def _no_crash_on_text() -> None:
    """A gate script must never crash on the text it reports: card amendments and PR
    bodies carry non-ASCII (a sigma crashed check 9 on a cp1252 console, #2057). Keep
    each stream's encoding, so the console still decodes what it is sent, and replace
    what it cannot encode. A stream without `reconfigure` (a test's StringIO) is left."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def _run_mains_copy(argv: list[str]) -> int:
    """Run origin/main's land_pr.py, never the checkout's (maintainer ruling 2 on #1686).

    The checkout is the PR head (check 3), so the PR's own copy of this file could
    have dropped a check. Main's copy runs from a temp file with MAIN_COPY_ENV set
    to that file's path: the copy sees itself and runs in-process, which ends the
    re-exec; a stale MAIN_COPY_ENV in the shell names no other path, so it cannot
    switch the hop off for the checkout's copy."""
    run_git(["fetch", "origin", "main"])
    source = run_git(["show", f"origin/main:{SELF}"])
    with tempfile.TemporaryDirectory(prefix="land_pr_main_") as tmp:
        path = os.path.abspath(os.path.join(tmp, "land_pr.py"))
        with open(path, "w", encoding="utf-8", newline="") as out:
            out.write(source)
        print(f"land_pr: running origin/main's {SELF}, not the checkout's.", file=sys.stderr)
        # main's copy may predate the fix in `_no_crash_on_text`, so the child gets it by env.
        env = {**os.environ, MAIN_COPY_ENV: path, "PYTHONIOENCODING": _safe_io_encoding(os.environ.get("PYTHONIOENCODING"))}
        return run_python([path, *argv], env)


def _green(check: dict[str, Any]) -> bool:
    """A CheckRun carries status/conclusion; a commit StatusContext carries state."""
    if check.get("__typename") == "StatusContext":
        return check.get("state") == "SUCCESS"
    return check.get("status") == "COMPLETED" and check.get("conclusion") == "SUCCESS"


def _accepted(line: str) -> bool:
    """Whether a line is a Status line that `adr_index.parse_adr` reads as Accepted."""
    if not line.startswith("**Status:**"):
        return False
    text = line[len("**Status:**") :].strip().lstrip("*").strip()
    return re.match(r"accepted\b", text, re.IGNORECASE) is not None


def _sets_accepted(patch: str) -> bool:
    """Whether one ADR's unified-diff patch sets its Status to Accepted: an added
    header Status line reads Accepted and no removed header Status line did."""
    old = new = 0
    removed = added = False
    for line in patch.splitlines():
        hunk = _HUNK.match(line)
        if hunk:
            old, new = int(hunk.group(1)), int(hunk.group(2))
            continue
        if line.startswith("-"):
            removed |= old <= ADR_HEADER_LINES and _accepted(line[1:])
            old += 1
        elif line.startswith("+"):
            added |= new <= ADR_HEADER_LINES and _accepted(line[1:])
            new += 1
        elif not line.startswith("\\"):  # "\ No newline at end of file" is not a line
            old += 1
            new += 1
    return added and not removed


def _text(body: str | None) -> str:
    """A PR, card or comment body with its line ends normalised to LF, BOMs and zero-width
    spaces deleted at a line start and turned into a space elsewhere (#1891)."""
    text = _LEADING_INVISIBLE.sub("", _CR.sub("\n", body or ""))
    return _INVISIBLE.sub(" ", text)


def _check_freeze(number: int, pr: dict[str, Any], paths: list[str], frozen: list[str], gh: Runner) -> None:
    """Check 5. A changed file inside an open `freeze:<pattern>` window refuses, unless
    the PR holds that freeze: it carries the label itself, or names an issue that
    carries it (as `Slice: #<n>` or a closing keyword)."""
    held = {lab["name"] for lab in pr["labels"]}
    refs = {ref["number"] for ref in pr["closingIssuesReferences"]}
    refs |= {int(n) for n in SLICE_REF.findall(_text(pr["body"]))}
    issue_labels: dict[int, set[str]] = {}

    def holder(label: str) -> str | None:
        if label in held:
            return f"#{number}"
        for ref in sorted(refs):
            if ref not in issue_labels:
                view = json.loads(gh(["issue", "view", str(ref), "--json", "labels"]))
                issue_labels[ref] = {lab["name"] for lab in view["labels"]}
            if label in issue_labels[ref]:
                return f"#{ref}"
        return None

    for path in paths:
        for pattern in frozen:
            if not fnmatch.fnmatchcase(path, pattern):
                continue
            by = holder(f"freeze:{pattern}")
            if by is None:
                raise LandError(
                    f"check 5: {path} is inside the open freeze:{pattern} window "
                    f"(the board, #1203, names its expiry). Only the PR that holds the "
                    f"freeze lands under it: it carries the label, or names the issue "
                    f"that does as 'Slice: #<n>' or 'Closes #<n>'."
                )
            print(f"check 5: {path} is inside freeze:{pattern}, held by {by}; allowed.")


def _born(path: str, status: str, previous: str | None) -> bool:
    """Whether this row brings an ADR number into being: an added or copied file, or
    a rename from a non-ADR file or from another ADR number. Its whole header is
    judged, since a patch against the old file would hide an inherited Accepted."""
    if status in ("added", "copied"):
        return True
    if status != "renamed":
        return False
    if previous is None or not ADR_FILE.fullmatch(previous):
        return True
    return previous.rsplit("/", 1)[-1][:4] != path.rsplit("/", 1)[-1][:4]


def _changed_files(number: int, gh: Runner) -> list[FileRow]:
    """The PR's changed files. `pr view --json files` stops at 100 files, and `pr diff`
    is refused (HTTP 406) past 20000 diff lines (#1775); the paginated files endpoint
    has neither cap."""
    listing = gh(["api", f"repos/{{owner}}/{{repo}}/pulls/{number}/files", "--paginate", "--jq", FILES_JQ])
    return [json.loads(line) for line in listing.splitlines() if line.strip()]


def _check_adr_and_baseline(number: int, pr: dict[str, Any], rows: list[FileRow], git: Runner) -> None:
    """Checks 7 and 8 (ADR 0121 D4, invariant 8), read from the PR files listing."""
    for path, status, previous, changes, patch in rows:
        if not ADR_FILE.fullmatch(path) or status == "removed":
            continue
        if _born(path, status, previous):
            # Local HEAD is the PR head (check 3), so the head's file is readable here.
            header = git(["show", f"{pr['headRefOid']}:{path}"]).splitlines()[:ADR_HEADER_LINES]
            if any(_accepted(line) for line in header):
                how = "is a new ADR born with" if status != "renamed" else f"(renamed from {previous}) is a new ADR born with"
                raise LandError(
                    f"check 7: {path} {how} Status Accepted. Accepting an ADR is the "
                    f"maintainer's act (ADR 0121 D4 (a)): an agent must not accept the ADR "
                    f"that admits its own exception. Leave it Proposed; the maintainer "
                    f"accepts it, through the GitHub web merge if need be."
                )
            continue
        if patch is None:
            if changes:
                raise LandError(
                    f"check 7: GitHub returned no patch for {path} ({changes} changed lines), "
                    f"so whether it sets Status to Accepted cannot be read; refusing."
                )
            continue
        if _sets_accepted(patch):
            how = "sets"
            raise LandError(
                f"check 7: {path} {how} Status Accepted. Accepting an ADR is the "
                f"maintainer's act (ADR 0121 D4 (a)): an agent must not accept the ADR "
                f"that admits its own exception. Leave it Proposed; the maintainer "
                f"accepts it, through the GitHub web merge if need be."
            )
    labels = {lab["name"] for lab in pr["labels"]}
    for path, _status, previous, _changes, _patch in rows:
        for touched in (path, previous):
            if touched in BASELINE and HUMAN_GATE not in labels:
                raise LandError(
                    f"check 8: #{number} edits the baseline file {touched} without "
                    f"the {HUMAN_GATE!r} label (ADR 0121 D4 (b)). Never edit FROZEN or "
                    f"FREEZE_POINT_DIGEST to make room; the maintainer labels the PR when "
                    f"the edit is theirs to make."
                )


def _line_paths(line: str, real: Real | None = None) -> list[str]:
    """The paths a card line names in backticks. `path::symbol` is `path`; a bare
    file name listed right after a path ("`.../prototypes/contract_cov.py`, `legacy.py`,
    `clones.py`", #1802) is that path's sibling. Tokens that are not path-like
    (`lock-tests`, `show_web`, `<slug>` placeholders) and globs with no literal top
    directory (`*`, `**/*.py`) are dropped: dropping narrows the list.

    A token with neither `/` nor `.` (`FORK_PIN`, `LICENSE`, but also `foo`, `and`) is
    kept only when it opens a list item ("- `FORK_PIN` ...") and `real` says it is a file
    on origin/main (#1939, T's ruling on #1944). Such a token anywhere else in the line,
    in parenthetical prose or mid-line, is dropped, as is every one when there is no `real`."""
    out: list[str] = []
    folder, end = "", 0
    lead = _LEADING_SPAN.match(line)
    for match in _TICKED.finditer(line):
        leading = lead is not None and match.start() == lead.end() - 1
        if line[end : match.start()].strip(" ,+&") not in ("", "and"):
            folder = ""  # only a list ("`a/x.py`, `y.py`") makes siblings, not prose
        end = match.end()
        path = match.group(1).split("::", 1)[0].strip().removeprefix("./").lstrip("/")
        if not _PATHLIKE.fullmatch(path) or (" " in path and "/" not in path):
            folder = ""  # a space is allowed only inside a directory path (#2339)
            continue
        if "/" not in path and "." not in path:
            if real is None or not leading or _GLOB.search(path) or not real(path):
                folder = ""
                continue
            out.append(path)
            continue
        if "/" in path:
            folder = "" if path.endswith("/") else path.rsplit("/", 1)[0] + "/"
        elif folder:
            path = folder + path
        if _GLOB.search(path.split("/", 1)[0]):
            continue
        out.append(path)
    return out


def _quoted_negation(token: str) -> bool:
    """A backticked span that is only a negation phrase (`never`, `do not`, `don't
    touch`, `never ever`, `do not edit b.py`). A single-token span naming a path
    (`src/x.py::test_not_assembly`, `src/b.py::never`) is data, never a negation. In a
    multi-word span each path-like word is dropped first, so `never touch src/b.py`
    negates (#1891). What is left must hold a negation with only filler words before
    it, and only filler after it up to an editing verb, whose object may be any word
    (`do not touch b`). `cannot join:` and `not-yet` carry other words, so they are
    code, not negations; a span of only punctuation, paths or filler has no negation."""
    words = token.split()
    if len(words) > 1:
        token = " ".join(w for w in words if "/" not in w and not _DOT_EXT.search(w))
    elif "/" in token or _DOT_EXT.search(token):
        return False
    match = _NEGATION.search(token)
    if match is None:
        return False
    if any(w not in _FILLER for w in re.findall(r"[a-z]+", token[: match.start()].lower())):
        return False
    for word in re.findall(r"[a-z]+", _NEGATION.sub(" ", token[match.end() :]).lower()):
        if word in _VERBS:
            return True  # the rest is the verb's object
        if word not in _FILLER:
            return False
    return True


def _negation_word(line: str) -> str | None:
    """Whether a card line's prose negates. Each backticked span is emptied to "``"
    first: a path (`src/not_found.py`) or a quoted code string (#1857's "the two
    `cannot join:` templates") is data, not a negation, while "No `b.py`" still negates.
    A span that is only a negation phrase (`do not`, `never`) stays, so it negates
    (review of #1883, G1). Only single backticks in pairs are masked: a ``double`` run
    or an odd count pairs the ticks wrongly and would hide prose (#1250), so such a
    line is read whole. Returns the negation word found, or None."""
    if line.count("`") % 2 == 0 and "``" not in line:
        line = _TICKED.sub(lambda m: m.group(0) if _quoted_negation(m.group(1)) else "``", line)
    match = _NEGATION.search(line)
    return match.group(0) if match else None


def _negates(line: str) -> bool:
    """Whether a card line's prose negates (see `_negation_word`)."""
    return _negation_word(line) is not None


def _card_paths(body: str, real: Real | None = None) -> list[str] | None:
    """The backticked paths of a card's `### Owned files`, `### Append-only` and
    `### Allowed` sections, or of an "Owned:" label line and the list under it.
    None when the card has no such section at all. A section ends at any `#` heading
    or bold-label line; an Owned line with a backtick and a negation refuses (fail
    closed). HTML comments (the template's guidance) are not card text. `real` admits
    extensionless file names (`_line_paths`)."""
    paths: list[str] = []
    found = in_section = in_label = False
    for line in _HTML_COMMENT.sub("", body).splitlines():
        if _HEADING.match(line):
            in_section, in_label = _OWNED_HEADING.match(line) is not None, False
            found |= in_section
            continue
        if _BOLD_LABEL.match(line):
            in_section = in_label = False
        if in_label and line.strip() and not _LIST_LINE.match(line):
            in_label = False  # the label's list ends at its first line that is not a list item
        label = None if in_section or in_label else _OWNED_LABEL.search(line)
        if label:
            in_label = found = True
            line = line[label.end():]
        if (in_section or in_label) and "`" in line:
            if _negates(line) and not _line_paths(line) and not line.rstrip(" *").endswith(":"):
                # A prose note that names no path ("no `nodal_resisting_*`", #2298) is
                # neither read nor allowed to refuse the card; it is named, never silent.
                print(f"check 9: non-path line in Owned: {line.strip()!r}")
                continue
            if _negates(line):
                raise LandError(
                    f"check 9: an Owned line negates, so which paths it owns is ambiguous; "
                    f"refusing. Reword it (a '**Card amendment' comment): {line.strip()!r}"
                )
            paths.extend(_line_paths(line, real))
    return paths if found else None


def _amendment_paths(
    text: str, card: int, real: Real | None = None, skipped: list[tuple[str, str]] | None = None
) -> list[str]:
    """The paths a `**Card amendment` comment adds. A negated line adds none, nor do
    the lines it governs (#1872). Every exclusion ends at a heading or a bold label;
    the widest one in force wins, and a narrower negation inside it never shortens it.

    - LIST: a negated line ending in ":", or a negated bullet that names no path ("Do
      not touch the following"), opens a list. It runs across blank lines and ends at
      the first non-list line that follows a blank line.
    - PARA: any other negated line that is not a bullet excludes to the paragraph's
      end (a blank line).
    - ITEM: a negated bullet that names a path excludes itself, its sub-items and its
      unmarked continuation lines, up to a sibling or shallower bullet or a blank line,
      so one "do not" bullet does not drop the siblings of a prose amendment.

    `skipped`, when given, collects (line, negation word) for each skipped line that has
    a backtick (the lines check 9 prints); the word is the governing line's (`--check-card`)."""
    paths: list[str] = []
    mode, depth, after_blank = _FREE, -1, False
    word = ""
    for line in text.splitlines():
        indent = len(line) - len(line.lstrip())
        if _HEADING.match(line) or _BOLD_LABEL.match(line):
            mode = _FREE
        elif not line.strip():
            mode = _LIST if mode == _LIST else _FREE
        elif mode == _LIST and after_blank and not _LIST_LINE.match(line):
            mode = _FREE  # the list ended at the blank line; this is new prose
        elif mode == _ITEM and _BULLET.match(line) and indent <= depth:
            mode = _FREE  # a sibling or shallower bullet ends the negated item
        after_blank = not line.strip()
        found_word = _negation_word(line)
        if found_word is not None:
            bullet = _BULLET.match(line) is not None
            # Without `real`: "- Do not touch `FORK_PIN`" still names no path, so it keeps
            # excluding the list after it; admitting a file name never narrows an exclusion.
            # A trailing ":" opens a negated list only when the negation is in the sentence
            # that ends in it ("It never read globally. These paths join Owned:" does not, #2412).
            tail = _SENTENCE_END.split(line.strip())[-1]
            if (line.rstrip(" *").endswith(":") and _negates(tail)) or (bullet and not _line_paths(line)):
                new = _LIST
            else:
                new = _ITEM if bullet else _PARA
            if new > mode:
                mode, depth, word = new, indent, found_word
        if mode != _FREE:
            if "`" in line:
                if skipped is not None:
                    skipped.append((line.strip(), word))
                print(f"check 9: skipped a negated card-amendment line on #{card}: {line.strip()!r}")
            continue
        paths += _line_paths(line, real)
    return paths


def _real_files(git: Runner) -> Real:
    """Whether a path is a file on origin/main: a blob in `git ls-tree -r origin/main`,
    read once and only when a card needs it. A file the PR itself adds does not count, so
    a PR cannot make a prose word (`suite`) owned by adding a root file of that name; a
    card that creates an extensionless file needs `scope-ok` (T's ruling on #1944, review
    F1). A failed or empty read refuses (fail closed): it never admits a name (#1939)."""
    tree: list[frozenset[str]] = []

    def real(path: str) -> bool:
        if not tree:
            try:
                listing = git(["ls-tree", "-r", "-z", "--name-only", "origin/main"])
            except _UNREADABLE as exc:
                raise LandError(
                    f"check 9: could not list origin/main's files to read the card's extensionless "
                    f"Owned name {path!r} ({exc}); refusing."
                ) from exc
            files = frozenset(name for name in listing.split("\0") if name)
            if not files:
                raise LandError(
                    f"check 9: `git ls-tree origin/main` listed no files, so the card's extensionless "
                    f"Owned name {path!r} cannot be read; refusing."
                )
            tree.append(files)
        return path in tree[0]

    return real


def _owned(path: str, allowed: list[str]) -> bool:
    for entry in allowed:
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
        elif _GLOB.search(entry):
            if fnmatch.fnmatchcase(path, entry):
                return True
        elif path == entry:
            return True
    return False


def check_owned(number: int, pr: dict[str, Any], rows: list[FileRow], gh: Runner, git: Runner) -> None:
    """Check 9 (lesson #1788). A `program` PR changes only files its slice card owns:
    the card's Owned/Append-only/Allowed sections, plus `**Card amendment` comments,
    plus IMPLICIT_ALLOWED. Both names of a rename are judged. #1812 changed 52 files
    against 6 owned, from a stale tree, and would have reverted three merged PRs.
    An extensionless name (`FORK_PIN`) counts only when it opens a list item and is a
    file on origin/main (#1939)."""
    labels = {lab["name"] for lab in pr["labels"]}
    if PROGRAM not in labels:
        if SLICE_REF.search(_text(pr["body"])) and FIXES_RED_MAIN not in labels:  # red-main fixes keep the skip (#2041)
            # Fail-open before #2012: #2002 and #2170 carried `Slice:` but no label, so they
            # skipped this check and the program-wide WIP count. `scope-ok` does not apply.
            raise LandError(
                f"check 9: #{number}'s body has a 'Slice: #<n>' line but it is not labelled "
                f"{PROGRAM!r}, so it escapes the program-wide WIP count "
                f"(`gh pr list --label {PROGRAM}`) and the Owned-files check. Run "
                f"`gh pr edit {number} --add-label {PROGRAM}`; {SCOPE_OK!r} does not override this."
            )
        print(f"check 9: #{number} is not labelled {PROGRAM!r}; the Owned-files check does not apply.")
        return
    if SCOPE_OK in labels:
        print(f"check 9: OVERRIDDEN by the {SCOPE_OK!r} label; #{number}'s files are not checked against its card.")
        return
    slices = SLICE_REF.findall(_text(pr["body"]))
    if not slices:
        raise LandError(
            f"check 9: #{number} is labelled {PROGRAM!r} but its body has no 'Slice: #<n>' line, so "
            f"its Owned list cannot be read (PROGRAM.md section 7). Add the line, or the maintainer "
            f"applies {SCOPE_OK!r}."
        )
    card = int(slices[0])
    issue = json.loads(gh(["issue", "view", str(card), "--json", "body,comments"]))
    real = _real_files(git)
    allowed = _card_paths(_text(issue.get("body")), real)
    if allowed is None:
        raise LandError(
            f"check 9: slice card #{card} has no '### Owned files' section (nor an 'Owned:' "
            f"list), so #{number}'s scope cannot be checked; refusing. Amend the card, or the "
            f"maintainer applies {SCOPE_OK!r}."
        )
    for comment in issue.get("comments") or []:
        text = _text(comment.get("body"))
        if _AMENDMENT.match(text):
            allowed += _amendment_paths(text, card, real)
    if not allowed:
        raise LandError(
            f"check 9: slice card #{card}'s Owned section names no backticked path, nor does any "
            f"'**Card amendment' comment, so #{number}'s scope cannot be checked; refusing. "
            f"Record the list on the card, or the maintainer applies {SCOPE_OK!r}."
        )
    allowed += IMPLICIT_ALLOWED
    outside = sorted({name for row in rows for name in (row[0], row[2]) if name and not _owned(name, allowed)})
    if outside:
        listing = "\n".join(f"  {name}" for name in outside)
        raise LandError(
            f"check 9: #{number} changes {len(outside)} file(s) outside slice card #{card}'s Owned "
            f"list (lesson #1788: #1812 changed 52 files against 6 owned, from a stale tree):\n"
            f"{listing}\nRebuild the branch with only the owned files, or amend the card with a "
            f"'**Card amendment' comment naming them, or the maintainer applies {SCOPE_OK!r}."
        )
    print(f"check 9: #{number}'s {len(rows)} changed file(s) are inside slice card #{card}'s Owned list.")


def check_card(card: int, gh: Runner, git: Runner) -> int:
    """`--check-card <issue>` (lesson #1985): print the Owned set check 9 would read for a
    slice card, and each amendment line it skips with the negation word that caused it.
    Uses `_card_paths` and `_amendment_paths`, so it cannot drift from check 9. Reads only
    the issue (no PR endpoints, no merge). Exit 0 when the card names a path, else 1."""
    issue = json.loads(gh(["issue", "view", str(card), "--json", "body,comments"]))
    real = _real_files(git)
    owned = _card_paths(_text(issue.get("body")), real)
    if owned is None:
        print(f"check-card: slice card #{card} has no '### Owned files' section (nor an 'Owned:' list).")
        return 1
    allowed = list(owned)
    skipped: list[tuple[str, str]] = []
    for comment in issue.get("comments") or []:
        text = _text(comment.get("body"))
        if _AMENDMENT.match(text):
            allowed += _amendment_paths(text, card, real, skipped)
    print(f"check-card: slice card #{card}'s Owned set ({len(allowed)} path(s)):")
    for path in allowed:
        print(f"  {path}")
    if skipped:
        print(f"check-card: {len(skipped)} amendment line(s) skipped by a negation:")
        for line, word in skipped:
            print(f"  skipped (negation {word!r}): {line}")
        print("check-card: a skipped amendment line may hold paths the card meant to own; "
              "restate the amendment as a bold label plus a bullet list of paths (#2326).")
        return 1
    if not allowed:
        print(f"check-card: #{card} names no backticked path, so check 9 would refuse it.")
        return 1
    return 0


#: A failed or garbled read: refuse, never treat it as green (fail closed).
_UNREADABLE = (subprocess.CalledProcessError, OSError, ValueError, KeyError, TypeError, IndexError, AttributeError)
#: (where, base sha, earliest start) for each Actions run of a PR's required checks.
Base = tuple[str, str, datetime]
Bases = Callable[[int], list[Base]]


def _when(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _check_bases(number: int, pr: dict[str, Any], gh: Runner, check_no: int) -> list[Base]:
    """The merge-ref base each Actions run of the PR's required checks checked out, read
    from the job log's `Merge <head> into <base>` line (the only place the API exposes
    it). Every run is returned, as check 6 judges every run. Fails closed."""
    head = pr["headRefOid"]
    # Run id -> (one job id, start times). Every job of a run checks out the same merge
    # ref, so one log per run is read: lock-tests's, the shortest, when it is there.
    workflow_runs: dict[str, tuple[str, list[str]]] = {}
    required = [run for run in pr["statusCheckRollup"] if (run.get("name") or run.get("context")) in REQUIRED_CHECKS]
    for run in sorted(required, key=lambda run: REQUIRED_CHECKS.index(run.get("name") or run.get("context"))):
        name = run.get("name") or run.get("context")
        job = _ACTIONS_JOB.search(run.get("detailsUrl") or "")
        if job is None or not run.get("startedAt"):
            raise LandError(
                f"check {check_no}: required check {name!r} names no Actions job and start time, so the "
                f"base it ran on cannot be read; refusing. {RERUN}"
            )
        workflow_runs.setdefault(job.group(1), (job.group(2), []))[1].append(run["startedAt"])
    bases: list[Base] = []
    for run_id, (job_id, starts) in sorted(workflow_runs.items()):
        where = f"run {run_id} (job {job_id})"
        try:
            log = gh(["api", "--allow-escape-sequences", f"repos/{{owner}}/{{repo}}/actions/jobs/{job_id}/logs"])
            merge = _CHECKOUT.search(log)
            if merge is None or merge.group(1) != head:
                raise ValueError(f"no 'Merge {head[:12]} into <base>' checkout line")
            bases.append((where, merge.group(2), min(_when(stamp) for stamp in starts)))
        except _UNREADABLE as exc:
            raise LandError(
                f"check {check_no}: the base #{number}'s required checks ran on in {where} could not be "
                f"read ({exc!r}); refusing. {RERUN}"
            ) from exc
    return bases


def check_main_green(number: int, pr: dict[str, Any], gh: Runner, bases: Bases) -> None:
    """Check 10 (lesson #1877; T's ruling on #1877). main's latest completed Tests run is
    green, and a run still in progress on main's head H admits only a PR whose required
    checks ran on a base that contains H. #1858 and #1866 landed while 3af31061's run was
    in progress, on bases without it, and the latest completed run (6f66fd23) was green."""
    if FIXES_RED_MAIN in {lab["name"] for lab in pr["labels"]}:
        print(f"check 10: OVERRIDDEN by the {FIXES_RED_MAIN!r} label; main's state is not read.")
        return
    try:
        order = json.loads(gh(["api", MAIN_COMMITS, "--jq", MAIN_COMMITS_JQ]))
        if not order or not all(re.fullmatch(r"[0-9a-f]{40}", str(sha)) for sha in order):
            raise ValueError(f"main's commits read {order!r}")
        by_sha: dict[str, list[dict[str, Any]]] = {}
        for run in json.loads(gh(["api", MAIN_RUNS, "--jq", MAIN_RUNS_JQ])):
            by_sha.setdefault(str(run["head_sha"]), []).append(run)
        tip = order[0]
        # The latest completed run by main's commit order, not by completion time: the newest
        # commit whose five required jobs have all completed. Only those five judge main, so a
        # red or slow qt-window-tests / app-tests neither blocks nor leaves main unverified.
        latest: str | None = None
        jobs: list[tuple[int, str, str | None]] = []
        for sha in order:
            found: list[tuple[int, str, str, str | None]] = []
            for run in by_sha.get(sha, []):
                listing = json.loads(gh(["api", MAIN_JOBS.format(run=int(run["id"])), "--jq", MAIN_JOBS_JQ]))
                if int(listing["total"]) > len(listing["jobs"]):
                    raise ValueError(f"run {run['id']}'s jobs page is truncated")
                found += [(run["id"], job["name"], job["status"], job["conclusion"])
                          for job in listing["jobs"] if job["name"] in REQUIRED_CHECKS]
            if {job[1] for job in found} >= set(REQUIRED_CHECKS) and all(job[2] == "completed" for job in found):
                latest, jobs = sha, [(run_id, name, conclusion) for run_id, name, _status, conclusion in found]
                break
    except _UNREADABLE as exc:
        raise LandError(f"check 10: main unverified: its Tests runs could not be read ({exc!r}). {UNVERIFIED}") from exc
    if latest is None:
        raise LandError(
            f"check 10: main unverified at {tip[:12]}: none of main's last {len(order)} commits has a "
            f"completed Tests run. {UNVERIFIED}"
        )
    for run_id, name, conclusion in jobs:
        if conclusion in RED_CONCLUSIONS:
            raise LandError(
                f"check 10: main is red: required job {name!r} of its latest completed Tests run ({run_id}, on "
                f"{latest[:12]}) is {conclusion}. Landing now buries the break under another merge (#1858, "
                f"#1866). Only the fix or the revert lands, under the {FIXES_RED_MAIN!r} label (lesson #1877)."
            )
    for run_id, name, conclusion in jobs:
        if conclusion != "success":  # cancelled, skipped, ...: no verdict either way, so no landing
            raise LandError(
                f"check 10: main unverified at {latest[:12]}: required job {name!r} of its Tests run ({run_id}) "
                f"is {conclusion}, which proves nothing. Re-run main's Tests on {latest[:12]} "
                f"(gh run rerun {run_id}), then land."
            )
    if tip not in by_sha:
        raise LandError(f"check 10: main unverified at {tip[:12]}: no Tests run on main's head. {UNVERIFIED}")
    if latest == tip:
        print(f"check 10: main's Tests run on its head {tip[:12]} is green.")
        return
    checked = bases(10)
    if not checked:  # check 6 makes this unreachable; never let an empty list read as "contains H"
        raise LandError(f"check 10: main unverified at {tip[:12]}: no base for #{number}'s checks. {UNVERIFIED}")
    for where, base, _started in checked:
        try:
            status = gh(["api", f"repos/{{owner}}/{{repo}}/compare/{tip}...{base}", "--jq", ".status"]).strip()
        except _UNREADABLE as exc:
            raise LandError(f"check 10: main unverified at {tip[:12]}: compare with {base[:12]} failed ({exc!r}). {UNVERIFIED}") from exc
        if status not in ("ahead", "identical"):
            raise LandError(
                f"check 10: main unverified at {tip[:12]}: its Tests run is in progress and #{number}"
                f"'s required checks in {where} ran on base {base[:12]}, which does not contain it "
                f"(compare: {status}). {UNVERIFIED}"
            )
    print(
        f"check 10: main's Tests run on {tip[:12]} is in progress; the latest completed one "
        f"({latest[:12]}) is green, and the PR's checks ran on a base that contains {tip[:12]}."
    )


def check_fresh_base(number: int, rows: list[FileRow], gh: Runner, bases: Bases) -> None:
    """Check 11 (lesson #1877). A PR that touches tests/ or a BASELINE file had its
    required checks run on a base no more than STALE_COMMITS commits and STALE_HOURS
    behind main. #1698's checks ran on 10-11; X4-a added the retirement-ledger row for
    the xfail #1698 deleted on 10-12, so the merge 3af31061 turned lock-tests red."""
    touched = sorted({n for row in rows for n in (row[0], row[2]) if n and (n.startswith("tests/") or n in BASELINE)})
    if not touched:
        print(f"check 11: #{number} touches neither tests/ nor a baseline file; the base age does not apply.")
        return
    for where, base, started in bases(11):
        try:
            behind_raw, tip_date = json.loads(
                gh(["api", f"repos/{{owner}}/{{repo}}/compare/{base}...main", "--jq", BEHIND_JQ])
            )
            behind = int(behind_raw)
            hours = (_when(tip_date) - started) / timedelta(hours=1) if behind else 0.0
        except _UNREADABLE as exc:
            raise LandError(
                f"check 11: the base #{number}'s required checks ran on in {where} could not be "
                f"read ({exc!r}); refusing. {RERUN}"
            ) from exc
        if behind > STALE_COMMITS or hours > STALE_HOURS:
            raise LandError(
                f"check 11: #{number} touches {', '.join(touched[:3])}{' ...' if len(touched) > 3 else ''} "
                f"but its required checks in {where} ran on base {base[:12]}, {behind} commit(s) and "
                f"{hours:.0f} h behind main (limits: {STALE_COMMITS} commits, {STALE_HOURS} h). #1698 "
                f"was green on a 2-day-old base and its merge 3af31061 turned main red (lesson #1877). {RERUN}"
            )
        print(f"check 11: {where} ran on base {base[:12]}, {behind} commit(s) and {hours:.0f} h behind main.")


#: Check 12: the file a pin bump edits, and the fork lane that must have run on the new pin.
PIN_FILE = "FORK_PIN"
LIVE_FORK_RUNS_JSON = "databaseId,status,conclusion,headSha,url"


def check_fork_pin(number: int, pr: dict[str, Any], rows: list[FileRow], gh: Runner) -> None:
    """Check 12 (#2138 -> #2143). A PR that changes ``FORK_PIN`` needs a ``live-fork`` run with
    ``event == workflow_dispatch`` on its head SHA, completed and successful. Neither ``scope-ok``
    nor ``human-gate`` overrides it, and it applies to every PR. A PR that leaves the pin alone
    makes no run-list call."""
    if not any(PIN_FILE in (row[0], row[2]) for row in rows):
        print(f"check 12: #{number} does not change {PIN_FILE}; no live-fork dispatch is required.")
        return
    head = pr["headRefOid"]
    dispatch = f"gh workflow run live-fork.yml --ref {pr['headRefName']}"
    listing = gh(["run", "list", "--workflow", "live-fork.yml", "--event", "workflow_dispatch",
                  "--commit", head, "--json", LIVE_FORK_RUNS_JSON])
    runs = [run for run in json.loads(listing) if run.get("headSha") == head]
    if not runs:
        raise LandError(
            f"check 12: #{number} changes {PIN_FILE} but no live-fork workflow_dispatch run exists on {head[:12]}. "
            f"A new pin is proven by the fork lane, which is not a required check (#2138 -> #2143). Start one "
            f"with `{dispatch}`, wait for it to finish green, then land."
        )
    latest = max(runs, key=lambda run: run.get("databaseId") or 0)  # only the newest dispatch counts
    if latest.get("status") != "completed":
        raise LandError(
            f"check 12: the newest live-fork dispatch run on {head[:12]} is {latest.get('status')}, not completed "
            f"({latest.get('url')}). Pending is a refusal, not a wait."
        )
    if latest.get("conclusion") != "success":
        raise LandError(
            f"check 12: the newest live-fork dispatch run on {head[:12]} concluded {latest.get('conclusion')} "
            f"({latest.get('url')}); a {PIN_FILE} change needs it green. Re-run it with "
            f"`gh run rerun {latest.get('databaseId')}` or start a new one with `{dispatch}`."
        )
    print(f"check 12: {PIN_FILE} changed; live-fork dispatch run {latest.get('url')} is green on {head[:12]}.")


def check(number: int, gh: Runner, git: Runner) -> dict[str, Any]:
    """Checks 1-12, in order. Returns the PR record, or raises LandError."""
    pr = json.loads(gh(["pr", "view", str(number), "--json", PR_FIELDS]))
    head = pr["headRefOid"]
    if pr["baseRefName"] != "main":
        raise LandError(
            f"check 1: #{number} targets {pr['baseRefName']!r}, not main. A PR merged into "
            f"a stacked base never reaches main (#858 was missing for two months; #1057). "
            f"Retarget it (gh pr edit {number} --base main), then push a commit."
        )
    if pr["state"] == "MERGED":
        raise LandError(
            f"check 2: #{number} is already merged. A push to its branch now lands on an "
            f"orphaned branch (#335 -> #336); open a new PR for the change."
        )
    if pr["state"] != "OPEN" or pr["isDraft"]:
        state = "a draft" if pr["isDraft"] else pr["state"]
        raise LandError(f"check 2: #{number} is {state}, not an open PR.")
    local = git(["rev-parse", "HEAD"]).strip()
    if local != head:
        raise LandError(
            f"check 3: local HEAD {local[:12]} is not the PR head {head[:12]}. Push, then "
            f"wait until headRefOid equals your tip (#555 -> #556)."
        )
    if git(["status", "--porcelain"]).strip():
        raise LandError("check 4: the working tree is not clean (git status --porcelain).")
    labels = json.loads(gh(["label", "list", "--limit", "200", "--json", "name"]))
    frozen = [lab["name"][len("freeze:") :] for lab in labels if lab["name"].startswith("freeze:")]
    rows = _changed_files(number, gh)
    # A rename lists both names, so moving a file OUT of a frozen path is caught too.
    paths = [name for row in rows for name in (row[0], row[2]) if name]
    _check_freeze(number, pr, paths, frozen, gh)
    if pr["mergeable"] != "MERGEABLE":
        why = "merge main locally and push" if pr["mergeable"] == "CONFLICTING" else "retry once GitHub has computed it"
        raise LandError(f"check 6: #{number} is {pr['mergeable']}, not MERGEABLE; {why}.")
    runs: dict[str, list[dict[str, Any]]] = {}
    for run in pr["statusCheckRollup"]:  # every run per name: a rerun does not hide a failure
        runs.setdefault(run.get("name") or run.get("context"), []).append(run)
    for name in REQUIRED_CHECKS:
        if name not in runs:
            raise LandError(
                f"check 6: required check {name!r} has not reported on {head[:12]}. Zero "
                f"checks visible means a conflict or a stalled Actions, not green (#630)."
            )
        for run in runs[name]:
            if not _green(run):
                seen = run.get("conclusion") or run.get("status") or run.get("state")
                raise LandError(
                    f"check 6: required check {name!r} is {seen} on {head[:12]}, not SUCCESS "
                    f"(every run of a required check must be). Pending is a refusal, not a wait."
                )
    _check_adr_and_baseline(number, pr, rows, git)
    check_owned(number, pr, rows, gh, git)
    found: list[list[Base]] = []

    def bases(check_no: int) -> list[Base]:  # read once, by whichever check needs them first
        if not found:
            found.append(_check_bases(number, pr, gh, check_no))
        return found[0]

    check_main_green(number, pr, gh, bases)
    check_fresh_base(number, rows, gh, bases)
    check_fork_pin(number, pr, rows, gh)
    return pr


class UnverifiedError(LandError):
    """The merge call went through but the post-merge proof could not be completed."""


#: Transient failures of a read: a non-zero gh/git exit, a missing binary, a cut-off body.
class _NotYet(Exception):
    """A read that came back empty right after the merge (eventual consistency)."""


_TRANSIENT = (subprocess.CalledProcessError, OSError, json.JSONDecodeError, _NotYet)
READ_TRIES = 5
READ_WAIT_S = 1.0  # doubles each retry: 1, 2, 4, 8 s


def _retry(fn: Callable[[], Any], what: str, sleep: Callable[[float], None]) -> Any:
    """Run a post-merge READ with bounded backoff (never the merge call itself:
    it is not idempotent). Exhaustion raises UnverifiedError (#1682)."""
    wait = READ_WAIT_S
    for attempt in range(1, READ_TRIES + 1):
        try:
            return fn()
        except _TRANSIENT as exc:
            if attempt == READ_TRIES:
                raise UnverifiedError(
                    f"{what} failed {READ_TRIES} times ({exc}); the merge itself went through."
                ) from exc
            sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")  # pragma: no cover


def _unverified(number: int, cause: UnverifiedError) -> UnverifiedError:
    return UnverifiedError(
        f"MERGED BUT UNVERIFIED: #{number} is merged, but the proof it reached main did "
        f"not complete: {cause} Run: python scripts/land_pr.py --verify-only {number}"
    )


def prove(
    number: int,
    gh: Runner,
    git: Runner,
    sleep: Callable[[float], None],
    head: str | None = None,
    branch: str | None = None,
) -> int:
    """The post-merge proof: compare/main...<squash> reads behind or identical, and
    the branch tip is still the merged head. Every read is retried; raises
    UnverifiedError (message starting MERGED BUT UNVERIFIED) if a read keeps failing."""
    try:
        if head is None or branch is None:
            view = _retry(
                lambda: json.loads(gh(["pr", "view", str(number), "--json", "state,headRefName,headRefOid"])),
                f"reading #{number}", sleep,
            )
            if view["state"] != "MERGED":
                raise LandError(f"#{number} is {view['state']}, not merged; --verify-only proves a merged PR.")
            head, branch = view["headRefOid"], view["headRefName"]

        def merge_oid() -> str:
            merge = json.loads(gh(["pr", "view", str(number), "--json", "mergeCommit"])).get("mergeCommit")
            if not merge or not merge.get("oid"):  # null right after the merge: retry
                raise _NotYet("gh reports no mergeCommit yet")
            return str(merge["oid"])

        sha = _retry(merge_oid, f"the mergeCommit lookup for #{number}", sleep)
        status = ""
        for _ in range(COMPARE_TRIES):
            status = _retry(
                lambda: gh(["api", "repos/{owner}/{repo}/compare/main..." + sha, "--jq", ".status"]).strip(),
                f"compare/main...{sha[:12]}", sleep,
            )
            if status in ("behind", "identical"):
                break
            if status == "diverged":
                raise LandError(
                    f"merged #{number} as {sha[:12]}, but compare/main...{sha[:12]} reads "
                    f"'diverged': the squash commit is not on main (#858). Recover it by hand."
                )
            sleep(COMPARE_WAIT_S)
        else:
            raise LandError(
                f"compare/main...{sha[:12]} still reads {status!r} after {COMPARE_TRIES} tries; "
                f"confirm by hand before trusting the merge."
            )
        tip = _retry(
            lambda: git(["ls-remote", "origin", f"refs/heads/{branch}"]).split(),
            f"git ls-remote for {branch}", sleep,
        )
    except UnverifiedError as exc:
        if str(exc).startswith("MERGED BUT UNVERIFIED"):
            raise
        raise _unverified(number, exc) from exc
    if tip and tip[0] != head:
        raise LandError(
            f"merged #{number} as {sha[:12]}, but {branch} now reads {tip[0][:12]}, "
            f"not the merged head {head[:12]}: a push raced the merge and that commit is on "
            f"an orphaned branch, not on main. Open a new PR for it."
        )
    if not tip:
        print(f"{branch} is gone (auto-deleted); nothing raced the merge.")
    print(f"landed #{number} as {sha[:12]} on main (compare: {status}).")
    return 0


def land(
    number: int,
    gh: Runner = run_gh,
    git: Runner = run_git,
    *,
    dry_run: bool = False,
    verify_only: bool = False,
    owned_only: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    if verify_only:
        return prove(number, gh, git, sleep)
    if owned_only:  # check 9 alone, on any PR state (the dry run on merged PRs, #1855)
        pr = json.loads(gh(["pr", "view", str(number), "--json", "labels,body"]))
        check_owned(number, pr, _changed_files(number, gh), gh, git)
        return 0
    pr = check(number, gh, git)
    head = pr["headRefOid"]
    if dry_run:
        print(f"dry-run: #{number} passes checks 1-12 at {head[:12]}; not merging.")
        return 0
    gh(["pr", "merge", str(number), "--squash", "--match-head-commit", head])  # never retried
    return prove(number, gh, git, sleep, head, pr["headRefName"])


def main(argv: list[str] | None = None) -> int:
    _no_crash_on_text()
    if not _is_mains_copy():
        return _run_mains_copy(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("number", type=int, nargs="?", help="the PR number")
    parser.add_argument("--dry-run", action="store_true", help="run the checks, then stop before the merge")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="skip the checks and the merge; run only the post-merge proof for an already-merged PR",
    )
    parser.add_argument(
        "--owned-only", action="store_true",
        help="run only check 9 (the PR's files against its slice card's Owned list), on any PR",
    )
    parser.add_argument(
        "--check-card", type=int, metavar="ISSUE",
        help="print the Owned set check 9 parses from a slice issue and the amendment lines it skips; "
        "takes no PR number and no other mode flag",
    )
    args = parser.parse_args(argv)
    if args.check_card is not None:
        if args.number is not None or args.verify_only or args.dry_run or args.owned_only:
            parser.error("--check-card takes an issue number alone, with no PR number and no other mode flag")
        try:
            return check_card(args.check_card, run_gh, run_git)
        except LandError as exc:
            print(f"land_pr: {exc}", file=sys.stderr)
            return 1
    if args.number is None:
        parser.error("the PR number is required (or use --check-card ISSUE)")
    if args.verify_only + args.dry_run + args.owned_only > 1:
        parser.error("--verify-only, --dry-run and --owned-only are mutually exclusive")  # exit 2
    try:
        return land(
            args.number, run_gh, run_git, dry_run=args.dry_run, verify_only=args.verify_only,
            owned_only=args.owned_only, sleep=time.sleep,
        )
    except UnverifiedError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except LandError as exc:
        print(f"land_pr: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
