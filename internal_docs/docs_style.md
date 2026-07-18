# Docs style — the ADR 0079 D5 contract

This page is the writing contract for everything published on the docs
site (`docs/`, Tier A). It exists because the old corpus read like an
archive generated in bulk: dozens of parallel pages of the same shape,
exhaustively symmetric, bullet-heavy, none telling you what to read
before or after it. The site is a course, and a course has an order and
a voice. If a page you are writing drifts toward inventory, stop and
rewrite — deduplicating a pile still leaves a pile.

## Order

The nav order **is** the reading order. Sections and pages are
sequenced pedagogically, never alphabetically; within Concepts the
pages follow the workflow spine (session → geometry → mesh → parts &
assembly → physics → bridge → results → persistence). The
[Learning path](../docs/tutorials/learning-path.md) page is the printed
table of contents of that order.

Every page knows its place. A page opens with one sentence stating its
job — what the reader will be able to do or understand when they reach
the bottom — and closes with an explicit pointer to the next page in
the sequence (tutorials and examples also name their prerequisite).
A page that cannot name its slot in the order does not belong on the
site. API reference pages are exempt: reference is looked up, not read
in sequence.

The site is capped at roughly **60 authored pages** (the auto-generated
API reference doesn't count). A new page must fill a named empty slot
in the order or displace a page it obsoletes. "Add another guide" is
not a move that exists.

## Voice

Write continuous prose in one human voice. The model registers are
[the first tutorial](../docs/tutorials/first-model.md) and
[the mental model page](../docs/concepts/mental-model.md) — read one
before writing, and match it. Bullets and tables are for genuinely
enumerable content (options, API grids, the learning-path ladder),
never the default texture of a page.

Banned, because they are the texture of generated docs:

- exhaustively symmetric coverage — covering a case because its sibling
  was covered, not because a reader needs it;
- a bold headline on every paragraph;
- the same summary stated at the top and again at the bottom;
- "Grounded in the current source: `<module paths>`" manifests and
  other maintainer bookkeeping in reader-facing pages;
- comprehensive tables as filler.

Say the thing once, in order, and stop.

## Rewrites, not splices

When consolidating old material (the P2 guide → concepts work), the old
pages are *source material*: read them, then write the new page fresh
in the site voice. Never assemble a page by splicing sections of old
pages together — splicing preserves exactly the archive texture this
contract exists to kill.

## Code and verification

Runnable snippets follow the reconciliation rules: verified green
against the current source, citation-carrying where the snippet mirrors
a test (`# verified: tests/<file>::<test>`). Tutorials and examples end
in a number checked against a known answer, and show their rendered
output — the published page must show the success, not bare code.

## Animations

Showcase animations (ADR 0079 D4) follow the same curation law as
pages: a small fixed set, each produced by a committed re-runnable
script from a real documented example, each captioned with the model,
the physics, and the line count of the script that made it. Web video
only (no GIFs), ~8 s muted loops, ≤ 3 MB per clip and ≤ 15 MB total. A
new animation displaces one or fills a named slot.

## The PR checklist questions

1. Which canonical page owns this fact, and did anything else restate
   it? (One home per Diátaxis mode; everything else links.)
2. Does the page name its slot in the reading order — job sentence at
   the top, next link at the bottom — and pass the voice rules above?
