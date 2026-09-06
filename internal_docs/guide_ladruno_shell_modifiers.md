# Ladruno shell stiffness modifiers — what ADR 91 adds, and the import bug it exposes

Working memory for adopting the fork's `LadrunoShellModifier` section. Two separable
pieces of work, and **the second one is a correctness bug that exists today**, independent
of whether we ever add the primitive.

Fork-side sources, if you need to check a claim:
`Ladruno_implementation/91_ladruno_shell_stiffness_modifiers_adr.md` (spec),
`LadrunoShellModifier_guide.md` (consumer guide, with the measured frame-vs-shell table),
`91_ladruno_shell_modifier_apegmsh_emitter_guide.md` (the fork-side ask), fork PRs
**#793** (shipped) and **#796** (the frame-equivalence gates).

## 0. One-paragraph summary

The fork shipped a section **decorator**: wrap any order-8 plate section, scale its eight
stiffness terms and its mass independently. It is the OpenSees equivalent of ETABS area
section property modifiers, and it exists so elastic shell models of RC walls can carry
cracked-section stiffness (ACI 318-25 §6.6.3.1.1). Adding the primitive is small — a frozen
dataclass with one dependency. The part that matters more is that
`interop/etabs_import.py` currently **drops** ETABS area modifiers on the floor, which
silently builds cracked walls at gross stiffness.

## 1. What shipped on the fork

```
section LadrunoShellModifier $tag $innerSecTag \
    [-f11 v] [-f22 v] [-f12 v] [-m11 v] [-m22 v] [-m12 v] [-v13 v] [-v23 v] [-mass v]
```

All nine optional, default `1.0`, order-independent. An all-defaults wrap is **byte-identical**
to the inner section (their G1 gate), so wrapping unconditionally in a generator is free.

`SEC_TAG_LadrunoShellModifier = 33000`. Wraps `ElasticMembranePlateSection`,
`LayeredShell`, `LayeredShellFiberSection`, `PlateFiber`, the Nunez membrane sections —
anything order-8 whose response codes are exactly `{FXX,FYY,FXY,MXX,MYY,MXY,VXZ,VYZ}`.

There is **no `weight` modifier**, deliberately — see §5.

## 2. THE BUG: `etabs_import` silently drops cracked-section stiffness

`src/apeGmsh/interop/etabs_import.py` (571 lines) contains no handling of area section
property modifiers. Grep it for `modifier`, `f11`, `f22`, `m11` — nothing.

So importing a real ETABS building, where the shear walls are almost certainly assigned
`f11 = f22 = f12 = 0.35` because that is standard practice, builds those walls at **gross**
stiffness. Roughly 3× too stiff in plane. Nothing warns; the model converges and produces
confidently wrong drifts, periods, and wall-vs-frame shear distribution.

**This is worth fixing even if we never add the primitive.** Priority order:

1. **Refuse loudly.** If the imported model carries any area modifier ≠ 1.0 and the importer
   cannot represent it, raise. A silently-3×-stiff wall is worse than a failed import.
2. **Then** represent it (§3, §4).

First thing to check: whether the modifiers survive into `StructuralModel` at all, or are
lost upstream in the apeETABS reader (`cAreaObj.GetModifiers` / `cPropArea.GetModifiers`,
a 10-entry array). The data may never reach us.

## 3. The trap: area bucketing is keyed on section name alone

`etabs_import.py:189-198`:

```python
area_buckets.setdefault(ar.section, []).append(surf)
```

ETABS modifiers can be assigned **per area object**, overriding the section property. Two
walls sharing section `"W30"` can legitimately carry different modifiers — a pier next to a
coupling beam cracked harder than its neighbour is normal.

So once modifiers exist, `ar.section` is no longer a sufficient bucket key. It has to become
`(ar.section, modifier_tuple)`, one emitted section per distinct combination, with a stable
human-readable PG name.

Get it wrong and every wall in the group inherits whichever area happened to be first — a
subset of walls silently gets another wall's cracking. Same failure class as §2: no error.

## 4. The primitive — three sites, following the existing shapes

**(a) `src/apeGmsh/opensees/section/plate.py`** — the dataclass. Unlike
`ElasticMembranePlateSection` it is a decorator, so it has a real `dependencies()`:

```python
@dataclass(frozen=True)
class LadrunoShellModifier(Section):
    """ETABS-style stiffness modifiers on any order-8 plate section (fork ADR 91)."""

    inner: Section
    f11: float = 1.0
    f22: float = 1.0
    f12: float = 1.0
    m11: float = 1.0
    m22: float = 1.0
    m12: float = 1.0
    v13: float = 1.0
    v23: float = 1.0
    mass: float = 1.0

    _NAMES = ("f11", "f22", "f12", "m11", "m22", "m12", "v13", "v23", "mass")

    def __post_init__(self) -> None:
        for n in self._NAMES:
            if getattr(self, n) < 0.0:
                raise ValueError(f"{n} must be >= 0.0, got {getattr(self, n)}")

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        inner_tag = resolve_section_tag(emitter, self.inner)
        args: list = []
        for n in self._NAMES:
            v = getattr(self, n)
            if v != 1.0:                      # emit only what differs from default
                args += [f"-{n}", v]
        emitter.section("LadrunoShellModifier", tag, inner_tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.inner,)
```

Tag resolution for `inner`: reuse whatever `LayeredShell` does for its layer materials
(`plate.py:179`, the closure-captured `resolve_mat_tag` on the emitter). `plate.py:165-168`
already flags that as an open coordinator question — do not invent a second mechanism.

**(b) `src/apeGmsh/opensees/section/__init__.py`** — export it and add it to the
`.plate` bullet in the module docstring.

**(c) `src/apeGmsh/opensees/_internal/ns/section.py`** — the typed namespace method, in the
"Plate / shell sections" block beside `ElasticMembranePlateSection` (~line 118), plus the
import at line 25:

```python
def LadrunoShellModifier(
    self,
    *,
    inner: Section,
    f11: float = 1.0,
    ...
    mass: float = 1.0,
    name: str | None = None,
) -> LadrunoShellModifier:
    """``section LadrunoShellModifier`` — ETABS-style stiffness modifiers."""
    return self._bridge._register(
        LadrunoShellModifier(inner=inner, f11=f11, ..., mass=mass), name=name
    )
```

## 5. Mapping ETABS → the emitter, and the tenth entry

The OAPI area modifier array is 10 entries:

```
[f11, f22, f12, m11, m22, m12, v13, v23, mass, weight]
```

The first nine map straight through. **The tenth does not exist on the fork side.** In
OpenSees the shell self-weight body force is derived from the same `getRho()` that builds
the mass matrix (`ShellMITC4.cpp:1725-1755`), so a weight modifier could only alias the mass
modifier; ADR 91 §5 declined to ship an argument that quietly does something other than its
name.

So the importer must **refuse or warn explicitly on `weight != 1.0`**, and say to scale
self-weight at the load level instead. Dropping it silently is §2 all over again.

## 6. `f` vs `m` — the thing that will actually bite a user

**In-plane bending of a shell is MEMBRANE action.** A wall or deep beam loaded in its own
plane is cracked with `f11`, not `m11`. The `m` modifiers are out-of-plane plate bending,
which such a wall does not have.

The fork measured this (PR #796). A flexure-controlled cantilever, L/d = 10, built as a frame
and as a shell mesh of the same member:

| case | tip deflection | ratio vs own gross |
|---|---|---|
| FRAME gross | 0.053333 m | 1.0000 |
| FRAME `A,I × 0.35` | 0.152381 m | 2.8571 |
| SHELL gross | 0.051911 m | 1.0000 |
| SHELL `f11,f22,f12 × 0.35` | 0.148318 m | 2.8571 |
| SHELL `m11,m22,m12 × 0.35` | 0.051911 m | **1.0000 — no change** |

`1/0.35 = 2.8571`. The frame route and the shell membrane route land on the same softening to
8 significant figures; the `m` route does **nothing**, silently. `v13`/`v23` likewise.

Two consequences for us:

- Any authoring helper or import mapping we write must put ETABS `f*` on membrane and `m*` on
  plate bending, never "helpfully" merge them.
- If we ever add a convenience like `crack_wall(0.35)`, it must set the **membrane** trio.

The residual −2.7% frame-vs-shell in that table is bilinear membrane locking, not the
modifier: it converges away under refinement (−12.6% at 20×2 → −0.5% at 120×24) while the
softening ratio stays 2.8571 at every mesh density.

## 7. `Ep_mod`, while we are in `plate.py`

`plate.py:94-99` emits `ElasticMembranePlateSection` as `E, nu, h, rho` — four arguments. The
upstream section takes a **fifth**, `Ep_mod` (an out-of-plane modifier, upstream Degenkolb
contribution). We cannot express it at all today.

Optional and defaulting to `1.0`, so nothing is broken. But once `LadrunoShellModifier` exists
there are two ways to reduce out-of-plane stiffness and users will find both. Suggested: add
the optional field for round-trip fidelity, but route the ETABS import and any authoring API
through `LadrunoShellModifier`, which is strictly more expressive. `Ep_mod = r` is exactly
equivalent to `m11 = m22 = m12 = v13 = v23 = r` — the fork's G5 gate pins that to round-off,
so it is a fact to rely on, not an approximation.

## 8. Acceptance

- [ ] `LadrunoShellModifier` in `plate.py`, exported, typed namespace method
- [ ] All-defaults wrap emits and round-trips as a no-op
- [ ] Sparse emit: only non-default flags reach the deck
- [ ] `etabs_import` represents area modifiers **or refuses loudly** (§2)
- [ ] Area bucket key includes the modifier tuple (§3)
- [ ] `weight != 1.0` refused with an actionable message (§5)
- [ ] Optional `Ep_mod` on `ElasticMembranePlateSection` (§7)
- [ ] A numeric twin: an apeGmsh-emitted cracked-wall deck vs a hand-written one

## 9. An open question we are unusually well placed to settle

Fork ADR 91 **OQ-1**: the modifiers are applied as a congruence, `D' = S·D·S` with
`S = diag(√f11 … √v23)`, so the Poisson coupling term moves as `√(f11·f22)` rather than being
left alone. Diagonal-only scaling was rejected because it destroys positive definiteness at
exactly the cracked-wall modifiers this serves. CSI does not document which convention ETABS
uses, and when `f11 == f22` — every standard recipe — the two are indistinguishable.

**An apeGmsh ETABS round-trip test with strongly unequal `f11`/`f22` would settle it.** We have
both an ETABS reader and the fork emitter; nobody else in the stack does. Worth reporting back
to the fork if we build it.

## 10. Does this want an apeGmsh ADR?

Probably not on its own — §4 follows existing primitive shapes and adds no new architecture.
But §3 (the bucket-key change) alters the import path's grouping contract, and §2 changes an
import from silently-lossy to refusing. If that lands as behaviour change on
`import_structural_model`, an ADR in `src/apeGmsh/opensees/architecture/decisions/`
(next free after `0102`) is the honest place for it.
