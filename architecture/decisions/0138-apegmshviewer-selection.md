# ADR 0138 — apeGmshViewer selection: one selection per target kind, picked from an ID image

**Status:** Proposed (2026-11-19). The owner accepts it by merging the PR that carries it.

**Owner:** nmora

**Builds on** [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) (one store, events in, a read-only app),
[ADR 0132](0132-workspace-file-and-variants-manifest.md) (named sets in the workspace file) and
[ADR 0137](0137-apegmshviewer-large-models-and-tool-hosting.md) (tools receive a `ToolContext` and request data).
**Makes concrete** ADR 0137 D10's "targets (picks)" (D10 here). **Will amend** ADR 0132 D3 when the first slice that saves
Gauss point, hinge or fibre sets lands (D12). **Supersedes** nothing.

**Evidence:** [`internal_docs/spike_viewer_selection.md`](../../internal_docs/spike_viewer_selection.md).

## Context

Selection is what a user meets first in the viewer, and what plots, sets, hide and isolate all start from. Today it is
`State.selection`: a list of declaration paths (elements, groups, materials) with the click that put each one there. It
was built for the inspector, and it does not carry what comes next.

- **Only elements can be picked.** A node set is guessed afterwards from the clicked element's node nearest the click, so a
  selection made in Search gives no node, and a click on the deformed shape is refused. Gauss points, hinges and fibres
  cannot be selected. The hinges card (#3038) keeps its own picked element and end.
- **A group and its members mix.** With a group selected, Shift+click on one of its elements adds that element's path. The
  element stays selected through the group.
- **It does not scale.** Adding one element to a selection of 26,970 took 2.8 s in the reducer, and 86 s at 130,000. Today's
  highlight took 175 to 217 ms from selecting every element of a 27,000-element model to the painted frame.
- **No region selection, and sets do not come back.** Left drag does nothing. A saved set can be renamed and deleted, but not
  loaded into the selection.

The maintainer chose one selection per target kind (design session, 2026-11-18) over a single mixed selection and over
extending today's list.

## Decision

**D1. Five target kinds, each with an address.**

| Kind | Address | Exists |
|---|---|---|
| node | node tag | always |
| element | OpenSees element tag (an element with no OpenSees tag: its declaration path) | always |
| Gauss point | element + point index in that element's Gauss group | on a results stage that has the group |
| hinge | element + end (`i`, `j`) + axis (`z`, `y` or coupled) | on a model with Ladruno hinges |
| fibre | element + section point + fibre index | on a results stage with fibre output |

A beam's section station is its Gauss point, so a fibre's section point is a Gauss point index. Addresses use the tags of
ADR 0132 D3, so a selection can become a set without translation. In memory, each kind uses a dense index into its own
table and keeps the address beside it.

**D2. One selection per kind, and one active kind.** Each kind keeps its own ordered selection. Switching kinds keeps the
others. Clicks, regions and the keys act on the active kind only. Each kind also has a **current target**, the last one
picked, which single-target tools follow (an inspector card, a Mohr circle, a moment-rotation curve).

**D3. Four separate things.** The selection (what the user picked), the current target (what a single-target tool follows),
visibility (what is drawn) and sets (what is saved) are kept apart in the store. None of them is derived by reading another's
list.

**D4. Visibility stays per element.**

- Hide, Isolate and Show all act on elements. On another kind they act on the elements that own the selected targets: a
  node's elements, or the element of a Gauss point, hinge or fibre.
- Hidden elements, elements not active at the stage shown, and anything cut away by a section plane are never picked, by a
  click or by a region, in any mode (as the click does today, ADR 0083 C-PICK).
- A target that becomes hidden stays selected. It is not drawn, and the selection bar counts it as hidden.

**D5. Picking reads an ID image, and regions read projected positions.**

- On a press, and on the first hover after the view changes, the viewport draws the active kind's ID image into an
  off-screen target and reads it back. Each pixel holds a 24-bit ID, 0 meaning nothing, which matches the 2^24 limit of
  ADR 0137 D1. The image uses the drawn positions, section planes and hidden set, so it shows what the user sees.
- Point-like kinds (nodes, Gauss points, hinges) are drawn as discs over the faces as depth only, so a target behind a face is
  not visible.
- Through selection projects the kind's positions to the screen at the press, and leaves out what section planes cut away.
- The image and the projection are kept until the view, the drawn set or the active kind changes. Nothing is drawn per
  frame.
- Today's ray cast stays until SEL-1 shows the two agree on both test models, or explains each difference. On a model with
  beam lines they disagreed on 34 of 40 clicks, for a reason the spike has not yet established.

**D6. Mouse and keys.**

- Click replaces the selection. Shift+click adds. Ctrl+click removes. A click on empty space clears the active kind.
- Hover marks the target under the cursor and names it, before any click.
- Esc cancels a drag in progress, and otherwise clears the active kind.
- Selection changes have their own undo, so a click on empty space never loses a large selection for good.
- New keys (one per kind, hide, isolate, show all, invert, undo) are chosen in SEL-1 against `KEY_BINDINGS`. `F`, `E` and
  `Esc` keep their meaning.

**D7. Regions.**

- A left drag past `CLICK_SLOP_PX` draws a box. Lasso is a mode of the selection bar. Polygon and brush may follow.
- The modifiers are those of a click: none replaces, Shift adds, Ctrl removes.
- For elements, a drag from left to right takes elements fully inside; from right to left, elements touching. For
  point-like kinds, inside is inside.
- Visible only by default. A toggle on the selection bar selects through the model.
- The live count updates during the drag. The highlight updates at release.
- A region is a gesture. What a set stores is its members.

**D8. The highlight is an index over the drawn geometry.** The selected faces are an index over the faces' own position
buffer, so the highlight follows the deformed shape without a copy. Selected lines are a subset of the drawn lines. The halo
pass draws the outline. Building it for every element of a 27,000-element model took under 2 ms.

**D9. Selection state is typed and linear.** Each kind holds its members as a dense-index array with a membership mask. No
operation compares member lists pairwise, and no list of strings is kept per member. The store keeps immutable values: an
edit replaces the arrays, and a revision number changes with every edit.

**D10. What a tool sees.** ADR 0137 D10's "targets (picks)" is, per kind: the count, the current target's address, and the
selection's revision. A tool that needs the members asks for them by revision. A tool picks by sending an event (ADR 0137
D10): the section view sends fibre picks this way, and fibres are picked there, not in the 3D view. The hinges card's own
pick moves into the hinge kind.

**D11. Search selects any kind.** Queries return targets of a named kind: by tag or range, group, type, material or section,
coordinates, and result value at the step shown. Conversions move a selection across kinds: nodes of elements, elements of
nodes, Gauss points of elements, and back.

**D12. Sets follow the kinds.** A set holds one kind, as ADR 0132 decided. From the selection bar a set can be loaded,
added, subtracted, hidden and isolated. Node and element sets use the schema that exists. Gauss point, hinge and fibre sets
need new member columns and their identity rule, in an amendment to ADR 0132 D3 and the workspace schema, made by the slice
that first saves them.

**D13. Implementation order.**

| Order | Slice | Covers |
|---|---|---|
| 0 | SEL-0: this ADR and the spike | |
| 1 | SEL-1: the selection core, the selection bar, click, hover and box for elements and nodes, the indexed highlight, and the agreement check of D5 | D2 to D9 for two kinds |
| 2 | SEL-2: lasso and through selection, selection undo, sets loaded into the selection | D6, D7, D12 |
| 3 | SEL-3: Gauss points and hinges, and the hinges card's pick moved into the hinge kind | D1, D10 |
| 4 | SEL-4: search across kinds and conversions | D11 |
| 5 | SEL-5: fibres, picked in the section view | D1, D10 |

Each slice with a screen change carries the maintainer's approval stills (ADR 0112 D7).

## Consequences

- Inspector, Sets, Search, Spotlight, the line diagrams and the hinges card stop reading `State.selection.decls`. Each moves to
  the selection of its kind in the slice that touches it.
- Node sets stop depending on where a click landed, so the refusal on the deformed shape goes away with SEL-1.
- A press now draws one off-screen pass. It shares the GPU process with tool windows (ADR 0137 D8), and it is drawn only on a
  press or the first hover after a change.
- Every new target kind needs an ID pass, an address and a set rule before it can be selected.

## Not established

- `yoo_building_SLE_T2` (128,923 cells) was not measured. The machine never had the free memory set for it during the spike.
- The cause of the slab click disagreement (D5).
- Device pixel ratio above 1, a discrete GPU, other machines, and the deformed shape.
- Real Gauss point layouts, hinge markers and fibres. The spike used 8 stand-in points per element.
- The pick radius that feels right for lines and points. The spike used 2 pixels.

## Alternatives considered

- **Extend today's declaration list.** The quickest start, but it keeps the mixing of groups and members, the quadratic
  updates and the click-derived nodes.
- **One mixed selection with a pick filter.** Rejected by the maintainer: every tool, set and command would have to handle
  mixed kinds, and sets already hold one kind.
- **Ray casting only.** It has no answer for a box that must respect what is visible, and it tests every triangle per click.
- **A bounding-volume tree or a worker for picking.** Not needed at the sizes measured: every region update fit in one frame
  on the page thread.
