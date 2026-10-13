"""Land one PR on main: the "How work lands" checklist (AGENTS.md) as code.

    python scripts/land_pr.py <number>            # check, squash-merge, prove it landed
    python scripts/land_pr.py <number> --dry-run  # run the checks and stop before the merge
    python scripts/land_pr.py <number> --verify-only  # only the post-merge proof, for a merged PR
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
from typing import Any

Runner = Callable[[list[str]], str]

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
BASELINE = (
    "tests/freeze_census.py",
    "tests/test_freeze_lock.py",
    "internal_docs/program/retirement_ledger.toml",
    "tests/test_loc_ratchet.py",
)
HUMAN_GATE = "human-gate"
ADR_FILE = re.compile(r"architecture/decisions/\d{4}-[^/]*\.md")
#: `scripts/adr_index.py::parse_adr` reads the Status line from the first 30 lines.
ADR_HEADER_LINES = 30
#: A slice PR names its slice issue as `Slice: #<n>` (PROGRAM.md section 7).
SLICE_REF = re.compile(r"^Slice:\s*#(\d+)", re.MULTILINE)
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
#: Words that may sit around a negation in a backticked span that is only a negation
#: phrase (`_quoted_negation`): "`do not` touch", "`must never edit`".
_FILLER = frozenset(
    "do does did must should shall will may please touch edit edits change changes modify add use it this that"
    " these those them file files".split()
)
_DOT_EXT = re.compile(r"\.\w")
#: How far a negation in a card amendment reaches (`_amendment_paths`); wider is larger.
_FREE, _ITEM, _PARA, _LIST = range(4)
_AMENDMENT = re.compile(r"^\s*\**\s*card amendment", re.IGNORECASE)
_TICKED = re.compile(r"`([^`\n]+)`")
_PATHLIKE = re.compile(r"[\w.\-/*?\[\]]+")
_GLOB = re.compile(r"[*?\[]")
COMPARE_TRIES = 5
COMPARE_WAIT_S = 3.0


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
        return run_python([path, *argv], {**os.environ, MAIN_COPY_ENV: path})


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


def _check_freeze(number: int, pr: dict[str, Any], paths: list[str], frozen: list[str], gh: Runner) -> None:
    """Check 5. A changed file inside an open `freeze:<pattern>` window refuses, unless
    the PR holds that freeze: it carries the label itself, or names an issue that
    carries it (as `Slice: #<n>` or a closing keyword)."""
    held = {lab["name"] for lab in pr["labels"]}
    refs = {ref["number"] for ref in pr["closingIssuesReferences"]}
    refs |= {int(n) for n in SLICE_REF.findall(pr["body"] or "")}
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


def _line_paths(line: str) -> list[str]:
    """The paths a card line names in backticks. `path::symbol` is `path`; a bare
    file name listed right after a path ("`.../prototypes/contract_cov.py`, `legacy.py`,
    `clones.py`", #1802) is that path's sibling. Tokens that are not path-like
    (`lock-tests`, `show_web`, `<slug>` placeholders) and globs with no literal top
    directory (`*`, `**/*.py`) are dropped: dropping narrows the list."""
    out: list[str] = []
    folder, end = "", 0
    for match in _TICKED.finditer(line):
        if line[end : match.start()].strip(" ,+&") not in ("", "and"):
            folder = ""  # only a list ("`a/x.py`, `y.py`") makes siblings, not prose
        end = match.end()
        path = match.group(1).split("::", 1)[0].strip().removeprefix("./").lstrip("/")
        if not _PATHLIKE.fullmatch(path) or ("/" not in path and "." not in path):
            folder = ""
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
    touch`): no path character, and nothing but filler words around the negation.
    `cannot join:` and `not-yet` carry other words, so they are code, not negations."""
    if "/" in token or _DOT_EXT.search(token):
        return False
    rest = _NEGATION.sub(" ", token)
    return rest != token and all(word in _FILLER for word in re.findall(r"[a-z]+", rest.lower()))


def _negates(line: str) -> bool:
    """Whether a card line's prose negates. Each backticked span is emptied to "``"
    first: a path (`src/not_found.py`) or a quoted code string (#1857's "the two
    `cannot join:` templates") is data, not a negation, while "No `b.py`" still negates.
    A span that is only a negation phrase (`do not`, `never`) stays, so it negates
    (review of #1883, G1). Only single backticks in pairs are masked: a ``double`` run
    or an odd count pairs the ticks wrongly and would hide prose (#1250), so such a
    line is read whole."""
    if line.count("`") % 2 == 0 and "``" not in line:
        line = _TICKED.sub(lambda m: m.group(0) if _quoted_negation(m.group(1)) else "``", line)
    return _NEGATION.search(line) is not None


def _card_paths(body: str) -> list[str] | None:
    """The backticked paths of a card's `### Owned files`, `### Append-only` and
    `### Allowed` sections, or of an "Owned:" label line and the list under it.
    None when the card has no such section at all. A section ends at any `#` heading
    or bold-label line; an Owned line with a backtick and a negation refuses (fail
    closed). HTML comments (the template's guidance) are not card text."""
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
            if _negates(line):
                raise LandError(
                    f"check 9: an Owned line negates, so which paths it owns is ambiguous; "
                    f"refusing. Reword it (a '**Card amendment' comment): {line.strip()!r}"
                )
            paths.extend(_line_paths(line))
    return paths if found else None


def _amendment_paths(text: str, card: int) -> list[str]:
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
      so one "do not" bullet does not drop the siblings of a prose amendment."""
    paths: list[str] = []
    mode, depth, after_blank = _FREE, -1, False
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
        if _negates(line):
            bullet = _BULLET.match(line) is not None
            if line.rstrip(" *").endswith(":") or (bullet and not _line_paths(line)):
                new = _LIST
            else:
                new = _ITEM if bullet else _PARA
            if new > mode:
                mode, depth = new, indent
        if mode != _FREE:
            if "`" in line:
                print(f"check 9: skipped a negated card-amendment line on #{card}: {line.strip()!r}")
            continue
        paths += _line_paths(line)
    return paths


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


def check_owned(number: int, pr: dict[str, Any], rows: list[FileRow], gh: Runner) -> None:
    """Check 9 (lesson #1788). A `program` PR changes only files its slice card owns:
    the card's Owned/Append-only/Allowed sections, plus `**Card amendment` comments,
    plus IMPLICIT_ALLOWED. Both names of a rename are judged. #1812 changed 52 files
    against 6 owned, from a stale tree, and would have reverted three merged PRs."""
    labels = {lab["name"] for lab in pr["labels"]}
    if PROGRAM not in labels:
        print(f"check 9: #{number} is not labelled {PROGRAM!r}; the Owned-files check does not apply.")
        return
    if SCOPE_OK in labels:
        print(f"check 9: OVERRIDDEN by the {SCOPE_OK!r} label; #{number}'s files are not checked against its card.")
        return
    slices = SLICE_REF.findall(pr["body"] or "")
    if not slices:
        raise LandError(
            f"check 9: #{number} is labelled {PROGRAM!r} but its body has no 'Slice: #<n>' line, so "
            f"its Owned list cannot be read (PROGRAM.md section 7). Add the line, or the maintainer "
            f"applies {SCOPE_OK!r}."
        )
    card = int(slices[0])
    issue = json.loads(gh(["issue", "view", str(card), "--json", "body,comments"]))
    allowed = _card_paths(issue.get("body") or "")
    if allowed is None:
        raise LandError(
            f"check 9: slice card #{card} has no '### Owned files' section (nor an 'Owned:' "
            f"list), so #{number}'s scope cannot be checked; refusing. Amend the card, or the "
            f"maintainer applies {SCOPE_OK!r}."
        )
    for comment in issue.get("comments") or []:
        text = comment.get("body") or ""
        if _AMENDMENT.match(text):
            allowed += _amendment_paths(text, card)
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


def check(number: int, gh: Runner, git: Runner) -> dict[str, Any]:
    """Checks 1-9, in order. Returns the PR record, or raises LandError."""
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
    check_owned(number, pr, rows, gh)
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
        check_owned(number, pr, _changed_files(number, gh), gh)
        return 0
    pr = check(number, gh, git)
    head = pr["headRefOid"]
    if dry_run:
        print(f"dry-run: #{number} passes checks 1-9 at {head[:12]}; not merging.")
        return 0
    gh(["pr", "merge", str(number), "--squash", "--match-head-commit", head])  # never retried
    return prove(number, gh, git, sleep, head, pr["headRefName"])


def main(argv: list[str] | None = None) -> int:
    if not _is_mains_copy():
        return _run_mains_copy(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("number", type=int, help="the PR number")
    parser.add_argument("--dry-run", action="store_true", help="run the checks, then stop before the merge")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="skip the checks and the merge; run only the post-merge proof for an already-merged PR",
    )
    parser.add_argument(
        "--owned-only", action="store_true",
        help="run only check 9 (the PR's files against its slice card's Owned list), on any PR",
    )
    args = parser.parse_args(argv)
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
