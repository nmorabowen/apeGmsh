# ADR 0132 — The workspace file and the variants manifest: `<stem>.workspace.h5` and `<study>.variants.json`

**Status:** Proposed (2026-11-03). Carries `human-gate`: the owner ratifies both
formats on #2783 before any slice writes either file. The pointers in ADRs 0112,
0128 and 0130 land in the PR that accepts this one (ADR 0121 D5).

**Owner:** nmora

**Rulings:**

- "Separate .h5 (B)", the owner directly in the VU orchestrator session,
  recorded on #2693 (2026-11-03). It replaces the earlier "workspace.json (C)":
  what a user makes in the viewer goes to `<stem>.workspace.h5` beside the set,
  as definitions only, with its own version key and an atomic write. Sets are
  columnar; plot and view specs may be JSON strings inside the file. Values stay
  in `.user.h5`.
- "Yes, option A", the owner on #2753 (2026-11-03): a renderer-originated open
  needs a real user gesture, and main opens only paths it chose or the user
  picked.
- The slice card #2783 and its amendments: the variants manifest is completed
  here, and both formats go in a new ADR rather than an amendment of ADR 0130.

**Supersedes in part, on acceptance:**

- [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) D3, as ADR 0130 D2
  amended it: the app writes one zone of `.user.h5` **and** one whole file,
  `<stem>.workspace.h5` (D1, D6 here).
- [ADR 0130](0130-states-quantities-and-the-user-results-file.md) D2's "the app
  writes it only in the combination zone": the app writes that zone and this
  file. A6 does not govern workspace sets; they have a per-member verdict (D3).
- [ADR 0128](0128-agent-views-in-apegmshviewer.md) INV-1, narrowed to the agent
  modes (`--render`, `--describe`, `--snapshot`, `--present`): the human's window
  writes the workspace file when the human saves.

**Completes** ADR 0130 A7 item 1, the variants manifest: its list key, `path`,
`version`, `params`, and how main opens a second set (D8, D9). In this ADR's
reading D8 fills in what A7 left open, and D9 is a rule A7 did not need before
option A; the reviewer of the PR rules on whether either is a new decision.

The layouts are in [`h5-schema.md`](../h5-schema.md), in the sections
"`<stem>.workspace.h5`: what the user makes (ADR 0132)" and
"`<study>.variants.json`: the variants manifest (ADR 0132)". The manifest's JSON
Schema is [`schemas/variants.schema.json`](../schemas/variants.schema.json).

## Context

A user of apeGmshViewer makes things worth keeping: a set of nodes picked by
clicking, a plot of the roof drift, a view to come back to. Today none of them
survives the window. ADR 0130 gave the set a user file, `.user.h5`, for values
Python computes, and let the app write one zone of it. The scoping brief of
2026-11-02 (VU0, "what a user makes and keeps") compared a zone in `.user.h5`
(A), a separate `<stem>.workspace.h5` (B) and a JSON file (C). The owner first
chose C, then B.

Three facts shape the format:

- **A `fem_hash` change is not a verdict on a set.** On a toy tet box, renaming
  a label moved `fem_hash` while all 351 nodes kept their ids and coordinates;
  refining from h = 0.3 to 0.2 kept 40 of 351. A set by id alone can name a
  different node after a re-mesh, and a set judged by the hash goes stale on a
  rename (brief §3).
- **h5wasm 0.10.3 cannot delete a link or rewrite a vlen dataset**, so `.user.h5`
  keeps append-only generations and every save copies the whole file (ADR 0130
  A4), up to 3 GB on SLE. A plot is renamed far more often than a combination is
  authored, so it must not ride that file.
- **ADR 0130 A7 named the variants manifest and stopped**: `version` and a list
  of `{label, path, params}`. The VU10 builder stopped on it (#2694): the list
  key, what `path` names and from where, and the types were all open, and option
  A now requires main to justify every path it opens.

## Decision

### D1 — Two files beside the sets, definitions only

| File | Beside | Holds | Written by |
|---|---|---|---|
| `<stem>.workspace.h5` | `<stem>.h5`, same folder and stem | named sets, plot definitions, saved views | the app and Python |
| `<study>.variants.json` | the sets of one study | the study's variants | Python, a pipeline or a human; never the app |

Neither file holds a result value. A plot is evaluated when it is drawn; a user
who wants numbers exports CSV or asks Python, which writes values to `.user.h5`
`/results` (ADR 0130 D7). The workspace is a file of its own, not a zone of
`.user.h5`: deleting `.user.h5` to rebuild derived data loses no set, and a
notebook writing `/results` cannot make a user's save refuse. A preference
(theme, layout) belongs to the app's profile. A workspace belongs to one stem
and never names a member of another set.

### D2 — One version key per file, and the floor rule

`<stem>.workspace.h5` carries `/meta@workspace_schema_version`, and
`<study>.variants.json` carries `format` (`"apegmsh-variants"`) and `version`,
both a `MAJOR.MINOR.PATCH` string, never a number. One key versions the whole
workspace because every save rebuilds it whole (D6), and each plot and view row
carries its own grammar tag (D4, D5). Each file follows ADR 0113, current and
floor `1.0.0`: Python refuses a newer minor (D2 there), and the app opens a
same-major newer file read-only with one banner (D7 there). **No writer writes
over a file newer than it knows**, minor included, because it would drop what it
cannot read (the rule ADR 0130 A4 states for zones). A bad row refuses alone,
with a reason, and the file still opens; a malformed table or envelope refuses
the file. Names follow ADR 0130 A2.

### D3 — A set stores tags with their coordinates, and is judged per member

`/sets` is columnar, one row per set: `name`, `kind` (`nodes` or `elements`,
never both), `id_space`, `fem_hash`, and `member_offsets` into two member
columns, `ids` in the user's order and `xyz`, each node's coordinates or each
element's centroid. Order is kept so that a profile plot needs no new format.

`id_space` is `opensees` in 1.0.0: the tags the deck and the `.ladruno` use. A
node's tag is its FEM id, because the bridge emits every node under its FEM id.
An element's tag is an `/opensees/element_meta` entry, so a bridge-minted element
can be saved, and a `.ladruno` element row needs no translation to be plotted.

On open, each set gets a verdict against the paired model: `current` (same
`fem_hash`), `moved` (another hash, every member found at its coordinates within
`1e-9` of the model's bounding-box diagonal; the next save updates the hash),
`partial` ("k of n"; plots that read it refuse until a human accepts the
survivors) or `lost` (none; listed, never dropped silently). A nearest-coordinate
match is only an offer.

The card offered `rf1` for this. It does not fit: a set is a fact about the mesh,
and `results_hash` moves on every re-run of an unchanged mesh, so an `rf1` per set
would stale every set after a re-run and say nothing after a re-mesh. The
fingerprint per set is the model's `fem_hash`, with the per-member check above
(Open question 3).

### D4 — A plot is a JSON definition with a sweep address

`/plots` holds `name`, `grammar` (`apegmsh-plot/1`) and `body`, one plot as JSON
text. A plot is `history`, `xy` or `profile`. A series names a quantity (the
`results` or `user_results` group, a field and a component, as the legend shows
them), a target (a set of this file, or a physical group, label or
`/mesh_selections` entry of the model, read-only), a reduction (`each`, `sum`,
`mean`, `min`, `max`, `absmax`), and optionally `minus` (a second reduced target,
for a drift) and `scale`. 1.0.0 plots nodal quantities only.

The sweep address `over` is separate from the state address: `{kind: "stage",
source, steps?, axis}`, where `steps` is `[from, to]`, `-1` is the last step, and
no `steps` is every step. A sweep over modes, a spectrum case or a combination is
not in 1.0.0. A `profile` reads one state, named by ADR 0130 A3's
`(kind, source, step, mode)` verbatim. A plot whose source is missing or stale, or
that reads a `partial` or `lost` set, refuses and draws nothing; it never falls
back. JSON suits a plot: it is nested and small, and a new plot kind is a new key
in a row, not a new table.

### D5 — A saved view is one `apegmsh-view/1` shot

`/views` holds `name`, `grammar` (`apegmsh-view/1`) and `body`, one shot object of
that spec as JSON text, unchanged. The app validates it with the view-spec
validator, wrapped in a spec whose `files` are this set, and applies it by the
same compile as an offer (ADR 0128 INV-2). A view another writer added is listed
and never applied by itself (ADR 0128 D5). A view that names a workspace set
needs `apegmsh-view/2`, which is a later slice.

### D6 — The write rebuilds the file: a partial per writer, a revision check, a replace

A save writes a **whole new file**, never the old one in append mode. The file
is small, so this costs little, and it needs neither of the two operations h5wasm
lacks; so the workspace has no generations. The rest is ADR 0130 A2's rule as
`h5-schema.md` ("Writing") states it for `.user.h5`: each writer writes
`<stem>.workspace.h5.<host>-<pid>-<random>.partial`, created exclusively in the
same folder, reopens and verifies it, and immediately before the replace compares
`/meta@revision` on disk with the one it read first. If it moved, the write
refuses, and the second of two writers loses rather than overwrites. `@save_id`
lets the app recognise its own save when the watcher reports a change. Every row
a writer did not change is copied verbatim, refused rows included.

A file that is not a readable workspace (not HDF5, or no version key) is
**renamed aside** to `<stem>.workspace.h5.unreadable-<UTC timestamp>` before the
new file takes its name, never overwritten, and the app says so with a banner. A
file that reads but is newer is not unreadable: the write refuses.

That h5wasm can create such a file, zero-length vlen columns included, is not yet
probed; VU9a probes it first and stops if it fails.

### D7 — Who reads and writes what

The app reads every table, gives the set verdicts, evaluates plots and validates
views, and writes every table. Python reads every table and the verdicts, and
writes sets (taking coordinates and centroids from `<stem>.h5`, and refusing a tag
the model lacks), plots and views. It does not evaluate plots in 1.0.0. Its
reader and writer are a thin module under `apeGmsh.results.user`, named by VU9a.

### D8 — The variants manifest: `variants`, a model path, flat params

`<study>.variants.json` holds `format`, `version` and `variants`, a non-empty list
of `{label, path, params}`.

- `label` is a name, unique in the file.
- `path` names the variant's model file `<stem>.h5`; the set pairs from it by the
  app's pairing rules. It is relative to the manifest's folder, uses `/`, may
  enter subfolders, and never leaves the folder: no `..`, `.` or empty segment,
  no absolute path, drive or backslash, and after links and junctions resolve it
  must still lie inside the manifest's folder (Open question 5).
- `params` maps names to a string, a number or a boolean. Rows may have
  different keys.

A bad row refuses alone (`bad-path`, `escapes`, `missing`, `duplicate`,
`duplicate-path`, `bad-param`). A7 stands otherwise: a combination term holds no
path or hash, so a cross-variant combination cannot be written. The manifest
stays JSON: it is small, hand-written as often as generated, and never written by
the app.

### D9 — Main opens a variant only from a manifest the user opened

Under option A, a variant path is main-chosen because main read it from a
manifest the user opened. So:

1. Main reads a manifest only from a path the user gave: a dialog, a drop (the
   path Electron derives) or argv. Main does not search a folder for one (Open
   question 4).
2. Main parses and validates the manifest itself and keeps the rows that pass,
   each with its resolved path. The page names a variant by its row index and
   label; main refuses a request whose label does not match its current row.
   **The page never sends a variant path.**
3. When the watched manifest changes, main re-reads the same path; the rows come
   from that read.
4. Limits: only a row that passed D8, confined to the manifest's folder; only a
   model file, paired by the usual rules (its `.ladruno`, `.user.h5` and
   workspace come with it); at most one compare set beside the open set. A
   compare set is read-only: the app writes nothing into it, neither
   `/combinations` nor its workspace. Switching to a variant makes it the open
   set, which the app may then write.

## Invariants

- **INV-1 (definitions only).** Neither file holds a result value.
- **INV-2 (no silent loss).** No writer drops a row it did not change, a refused
  row included, and none overwrites an unreadable or newer file.
- **INV-3 (no silent stale).** A `partial` or `lost` set is listed, and every plot
  that reads it refuses until a human acts.
- **INV-4 (one compile).** A saved view is applied by the compile that applies an
  offer and takes a still.
- **INV-5 (main-chosen paths).** Every variant path main opens came from a row of
  a manifest the user opened, and lies inside that manifest's folder.

## Alternatives rejected

- **A zone in `.user.h5` (A).** Each save copies a file of up to 3 GB and adds a
  generation h5wasm cannot reclaim; a `/results` write by a notebook would make a
  user's save refuse; deleting `.user.h5` would lose the sets.
- **A JSON workspace (C).** The owner's first ruling, replaced by B. It would have
  given direct reading and diffing, at about 50% more bytes for sets (43 MB
  against 28 MB at 1M members with coordinates, brief §2).
- **Generations in the workspace, as in `.user.h5`.** Rebuilding a small file
  is simpler and leaves nothing to reclaim.
- **Members as one group per set.** A name would become a link (ADR 0130 A2
  forbids it), and K groups cost more than two columns and an offset.
- **Staleness by `fem_hash` alone, or by `rf1`.** A label rename would stale
  every set; a re-run would stale sets on an unchanged mesh (D3).
- **A variant path sent by the page**, checked by main. Option A refuses a free
  string from the page; a row index cannot name a file the user never opened.

## Consequences

- The app writes a second file. The write is small, atomic and per writer, and
  renaming a plot never touches the multi-gigabyte `.user.h5`.
- A re-meshed model keeps its sets readable: what survived is shown, and what did
  not is listed.
- The view spec's grammar stays the one external contract for views; the
  workspace only stores shots.
- A study is opened by opening its manifest, and VU10 can be re-scoped against
  D8 and D9.
- The workspace joins the zone registry and the schema corpus when VU9a adds its
  writer constant.

## Slices

| Slice | Work | Gate |
|---|---|---|
| VU9a-0 | This ADR, the two `h5-schema.md` sections, the manifest's JSON Schema and examples | owner ratifies on #2783 |
| VU9a | The h5wasm create probe; node pick and naming; the workspace reader and writer in the app and in Python; the set verdict | this ADR accepted |
| VU9b and later | Plot window (`history`, then `xy`, `profile`); saved views | VU9a |
| VU10 | Variant manifest, switch and compare, under D8 and D9 | this ADR accepted |

## Open questions

Each takes the recommended reading unless the owner rules otherwise on #2783.

1. **Variant compare and workspaces.** Rec: a workspace belongs to one stem; a
   comparison matches sets of the same name in each variant's workspace, each
   with its own verdict. The alternative is a study-level workspace beside the
   manifest.
2. **Sets are ordered and hold one kind.** Rec: yes.
3. **The set fingerprint.** The card named `rf1`; D3 uses the model's `fem_hash`
   with the per-member check, a departure from ADR 0130 A6 for this file. Rec:
   accept D3.
4. **Finding a manifest.** Rec: main reads one only when the user opens it. The
   alternative, main listing the folder of an opened set for a manifest that
   names it, is also main-chosen but opens a file the user did not pick.
5. **Variant paths in subfolders.** Rec: allowed, confined to the manifest's
   folder. The narrower rule is the manifest's folder only.
6. **Renaming an unreadable workspace aside without asking.** Rec: rename and
   show a banner; the alternative is to refuse the save until the user moves it.

## References

- Scoping brief, VU0, 2026-11-02: "apeGmshViewer: what a user makes and keeps"
  (orchestrator scratch; §2 and §3 measurements quoted above).
- #2693 (VU9, rulings C then B), #2694 (VU10, the manifest gap), #2753 (option
  A), #2783 (this slice).
- `apeGmshViewer/src/spec/spec.ts` (`apegmsh-view/1`),
  `apeGmshViewer/src/main/pairing.ts` (the pairing rules).
