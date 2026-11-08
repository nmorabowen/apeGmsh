# apeGmshViewer — what the desktop app shows

apeGmshViewer is the installed desktop app the human uses to look at a
run. This page says what it opens and what it draws, so you can tell the
human where to look and read what they report back. It is not for checking
a solve yourself: that is `fem.assess()` / `results.assess()`
(`references/assess.md`). The app's agent commands (`--render`,
`--describe`, `--present`, `--snapshot`) are in `references/results.md`
§5, "apeGmshViewer from the command line"; the `.user.h5` you write from
Python is in its §4, "Combinations and the user file".

Never start the app to look for yourself. Its window belongs to the human.

## 1. Opening a set

A run leaves its files side by side under one stem, and opening any one of
them opens the whole set:

```text
<stem>.h5              the model
<stem>.geometry.h5     the /geometry sibling
<stem>.ladruno         results (a partitioned run: <stem>.part-<N>.ladruno)
<stem>.results.h5      native results
<stem>.user.h5         combinations, user results, modes, spectra, entities
```

- **One results file per set.** When both a `.ladruno` and a
  `.results.h5` sit beside the model, the one the human opened wins if it
  is one of them; otherwise the `.ladruno`. The other is never read, and
  the banner names it. Opening any part of a partitioned run opens the
  whole part set, and an incomplete part set is refused by name.
- **Names compare case-insensitively** on Windows and macOS. Any other
  suffix is refused with the list of the five above.
- **The set is watched.** A rewritten file reloads once it has stopped
  changing. A sibling that appears or vanishes reopens the set. The
  `.user.h5` appearing, being replaced or being deleted re-reads it, and so
  does a rewrite of the model or the results, because its staleness depends
  on them.
- **A study of variants** opens from its `<study>.variants.json`; to switch, compare and show a difference, see `variants.md`.

### What the app will open

The app opens only files a person or the app itself chose, never a path the
page asks for:

- a file the human dropped on the window;
- the file a double-click or a command line names, including a second
  `apeGmshViewer run.h5` while a window is open;
- the files of an offer you sent with `--present`, once the human clicks it
  (and the set that offer left, through its Back).

Every read of geometry, results or the `.user.h5` is confined to the files
of the set on screen. A path outside it is refused with a reason that
names the door and the path; nothing opens.

**Go to source** opens your script at the line that declared something,
in `$APEGMSH_EDITOR` (a template with `{file}` and `{line}`, such as
`code -g {file}:{line}`), else VS Code's `code`, else the OS text editor.
It opens only a source file the model's `/provenance` names. A row that
names another machine (a path starting `\\` or `//`) is never looked up and
never handed to an editor; go-to-source on it answers with the reason. On
Linux, when neither `$APEGMSH_EDITOR` nor `code` is found and the app falls
back to `xdg-open`, only a regular `.py`, `.pyw`, `.pyi`, `.pyx` or `.tcl`
file opens; with either one, any local file `/provenance` names opens.

## 2. The `.user.h5`

The app reads the set's `.user.h5` after the first frame, never before.
While it reads, the phase bar says **reading user file**; a file it cannot
open says **user file refused** (tooltip: why); a file that reads with
warnings says **user file: N warnings**. The app reads the file and does
not write it.

Refusal has three levels, as in Python: a bad `/meta` refuses the file; a
malformed zone refuses that zone and the others still read; a bad row
refuses alone, with its reason, and the rest of the zone stands.

**Staleness** is judged against the set's own files, never against what a
request names:

- against the model: the `fem_hash` and `model_hash` in the file against
  the model's `/meta/lineage`;
- against the results: the stored `results_hash` against a fingerprint of
  the results file on disk;
- each mode row: its lambda against the recorded LAMBDA of that mode.

Whatever cannot be compared is stale. A stale item is listed and says why
in its tooltip; it is never drawn as current.

## 3. The phase bar

The bar holds the phases (geometry, mesh, …) and then one button per state
family:

| Button | From | A click |
|---|---|---|
| **Stages** | the results file's stages | opens the first stage that records a step |
| **Modes** | the results file's mode shapes | opens the first mode; the modal table shows beside it |
| **Spectra** | `/spectra` of the `.user.h5` | nothing: listed only |
| **User Combinations** | `/combinations` of the `.user.h5`, then the model's `/opensees/combinations` | shows the first combination as a contour |

Each button shows its count, then ` · N stale` (or ` · stale` when every
member is), then ` · N refused` (or ` · refused` for a whole zone). The
tooltip names every member and every reason. A family with nothing to draw
is a disabled button that still lists. The bar shows even beside a single
phase when the `.user.h5` has something to say.

On a stage or a mode the bar is also the time scrubber: a picker listing
that family only, step −1 / play / step +1, a slider, and a readout such as
`step 3/5 · t = 0.6` (`mode shape` on a mode).

## 4. Modes and spectra

While a mode is on screen, the **modal table** card lists the
`/modes` rows of the `.user.h5`: mode, period, effective mass ratio and
cumulative ratio per component (percent), and a verdict, **current** or
**stale**, with the reason in the row's tooltip. The row of the mode on
screen is marked; a row matches only when its mode number and its lambda
both match the recorded mode. A mode with no row is shown without one,
not refused.

**Spectra** are listed in the button's tooltip, one line per case:
direction, scale, number of modes, damping (or "damping per mode"), and the
results it holds. A case draws through a User Combination that has a
`spectrum` term over it.

## 5. User Combinations

A click on User Combinations shows the first combination as a contour over
the model, and switches the colouring to contour. A picker in the bar then
chooses another. A combination is one state, so there is no scrubber; the
colour range is the combination's own values unless you fix it.

- It needs a mesh, and a nodal field and component picked in the Contour
  card.
- The readout beside the picker gives the rule and the sign:
  `linear · signed`, `srss · unsigned: contour only`, `sign unknown`, with
  ` · stale` when a source of it is stale.
- **An unsigned combination is refused as a deformed shape** (a modal rule,
  an envelope, or a `linear` over one): it lost the signs a shape needs.
  The refusal gives that reason. A signed one is not drawn deformed either: the
  app draws the deformed shape of a results step only.
- A row from the model's `/opensees/combinations` says `from model.h5` in
  the tooltip.

## 6. The User Results group

The Contour card's field picker lists **Results** (what the recorder
wrote, by source), then **User Results** (the `/results` entries of the
`.user.h5`). Each row gives the entry's target (nodal, element, entity,
whole model), whether it is per state or state-free, and whether it is
stale. User results are listed only: the app does not draw them yet.

## 7. Entities

The `/entities` of the `.user.h5` draw where they are, with no switch:

- a **section cut** as its bounding polygon, or as its plane where it
  crosses the model's box, with a tick toward the kept side and a marker at
  the point its moments are about;
- a **storey** as the plan outline (convex hull) of its nodes at its
  level, faintly filled, with a marker at its reference point;
- a **point set** as a marker at each point.

They draw only while the zone reads and is current and the model is on
screen (not on the geometry phase). Refused rows never draw. A storey
naming a node the model does not hold is not drawn; the reason is a warning
in the developer console (DevTools), not in the window.

## 8. Loads & SPs, Reactions, Vectors, Principal and supports

On the Loads & SPs, Reactions, Vectors and Principal cards every arrow is sized on screen: the
head is twice as long as it is wide and
`head` pixels long (12 by default), and a shaft stays between 15 and 80
pixels times the card's `scale`. Zooming rebuilds nothing. **thin out** (on
by default) draws one arrow, the largest, where several would overlap;
zoom in to see the rest.

**Loads & SPs** shows when the model has loads, off the geometry phase.
**3D loads** (off by default) draws them; **F**, **M** (double-headed),
**elem** (beam and surface element loads) and **SP** (non-zero prescribed
displacements and rotations) filter the kinds. **scale by λ(t)** scales
each pattern by its time series at the step on screen, and **support
holds** adds the support-reaction hold patterns. The colour menu offers one
colour, by load type, or by magnitude (jet). Each pattern is a row; a
click toggles it, and **active stage** resets to the stage's patterns.

**Reactions** shows when the step on screen records reaction forces or
moments. **3D reactions** (off by default) draws them: forces as arrows
pointing at the supports with a 45° tick across the tail, moments as half
circles about their axes (right-hand rule). **F** and **M** appear only for
a kind the step records. The menu picks the resultant, the X, Y, Z
components, or one axis (Z only when it is recorded). It draws at the
supports; **all nodes** adds the reactions recorded elsewhere (a free
node's residual, a constrained node's constraint force).

**Vectors** shows when the stage on screen records a nodal vector field:
each nodal field with two or more axes, such as `DISPLACEMENT`, `VELOCITY`
and `ACCELERATION` in a `.ladruno`. **3D vectors** (off by default) draws
the picked field as arrows that start at the nodes and point along the
vector, and they follow the deformed shape. A rotational field
(`ROTATION`, `ANGULAR_VELOCITY`, `moment`) draws a half circle about each
node's axis (right-hand rule). A 2-D model's lone `Rz` is not offered: the
fork's schema allows it to hold a pore pressure. The menu picks the
resultant, the X, Y, Z components, or one axis (Z only when the field
records it). An axis the field does not record falls back to the
resultant. Lengths are relative to the step's largest vector, so the card
shows direction and relative size, not a value. Vectors under 1e-6 of
the largest are not drawn. An `apegmsh-view/1` spec has no key for the
card, so `--render` cannot draw it.

**Principal** shows when the stage on screen records a stress or strain
tensor at the Gauss points of a `.ladruno` whose element ids are verified
(a group needs its xx, yy and xy components). **3D principal** (off by
default) draws, at each Gauss point, an arrow along each principal
direction: the length follows |p|, the colour is the signed value on a
diverging map over −max..max of the step (RdBu_r by default: compression
blue, tension red). A direction is an axis, not a sense: **axis** (the
default) draws a head each way from the point, **arrow** one head along
the sign-canonical direction, as the Qt diagram does. **p1**, **p2**,
**p3** pick the principal values (p1 ≥ p2 ≥ p3); the menu picks the field
(`stress`, `strain`). The arrows follow the deformed shape, and **thin
out** hides a point's arrows together. The tensor is Python's
`principal_frame`'s: a strain's shear is read as engineering and halved,
and a zz the element does not record is taken as 0 (plane stress for a
stress, plane strain for a strain); the card says both. Principal values
under 1e-6 of the largest are not drawn, so a 2-D element's zz draws
nothing. A field the card cannot draw is listed as "(not drawn)", with its
reasons. Like the Vectors card it shows direction and relative size, not
values, and `--render` cannot draw it.

**Supports**: the **BC** switch in the Display card (off by default) draws
the restraints from the model alone: an arrow into the node per restrained
translation, a half circle per restrained rotation.
