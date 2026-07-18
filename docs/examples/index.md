# Examples

Recognizable structural problems, built end to end through the **typed
`apeSees` bridge** and checked against a known answer.

<div class="grid cards" markdown>

-   __[Portal frame (2D)](portal-frame.md)__ — two columns + a beam under
    gravity and lateral load; multiple element groups, drift, and a base
    shear that checks to exactly the applied load.

-   __[Modal analysis (cantilever)](modal-analysis.md)__ — `g.masses` +
    `ops.eigen`, the first three natural frequencies read back through
    `Results` and matched to the Euler–Bernoulli closed form.

-   __[Fiber sections & moment–curvature](fiber-moment-curvature.md)__ — a
    `W14×90` fibre section (dimensions from **apeSteel**) driven through
    yield with `ZeroLengthSection`; the M–κ curve passes through
    $M_y=F_yS_x$ and plateaus at $M_p=F_yZ_x$, shape factor 1.10.

-   __[Multi-part assembly](multipart-assembly.md)__ — build a column once
    as a reusable `Part`, stamp it three times with `g.parts.add`, and read
    each copy's deflection back by its own label (each exactly $PL^3/3EI$).

-   __[Tie non-matching meshes](tie-non-matching-meshes.md)__ — two solid
    blocks meshed at different sizes, joined by a `g.constraints.tie` the
    bridge **auto-emits**; the load transmits exactly and the column matches
    the monolithic bar to ~3%.

-   __[STEP import: plate with a hole](step-plate-with-hole.md)__ — import
    a CAD part, heal it, name edges by geometric query, refine at the
    hole, and recover the classic stress concentration $K_t\approx3$.

-   __[Choosing a results strategy](results-strategies.md)__ — the same
    portal solved and read back via `from_native` and `from_mpco` (STKO);
    the read code is identical and both agree on the 8.39 mm drift to zero.

-   __[Compose modules](compose-modules.md)__ — build the portal once, save
    it, and `g.compose` it into two bays; PGs come back label-prefixed
    (`bay2.Columns`) and each uncoupled bay drifts the exact E1 8.39 mm.

-   __[Pushover of a steel moment frame](pushover-steel-frame.md)__ — a
    `W14×90` **fibre** section (from **apeSteel**) in a `forceBeamColumn`
    pushed to a column-sway mechanism; the $V\!-\!\Delta$ capacity curve
    matches $K=2\cdot12EI_c/H^3$ to 2.7 % and $V_p=4M_p/H$ to 1.9 % at
    mechanism.

-   __[Plane-wave SSI: absorbing soil column](plane-wave-ssi.md)__ — a
    soil column wrapped by an `ASDAbsorbingBoundary3D` skin
    (`g.parts.add_plane_wave_box`), shaken by a base shear pulse and
    flipped to absorbing with `s.activate_absorbing`; the wave arrives at
    the surface at $H/V_s$ and then radiates out (late motion < 1 % of
    peak) instead of reflecting.

-   __[Staged SSI: gravity then the absorbing flip](staged-gravity-ssi.md)__
    — the canonical sequence: a gravity stage settles the column (5.6 cm)
    while the skin holds, `loadConst` freezes it, then `s.activate_absorbing`
    flips and the base wavelet shakes the equilibrated state. Shows why
    continuum `body_force` (always-on) is the wrong gravity route for a
    staged model and the `g.loads.gravity` + `from_model` route is right.

</div>

More rungs (shell-on-solid, transient) are landing wave by wave.

## Beyond the curated set

The repo's [`examples/`](https://github.com/nmorabowen/apeGmsh/tree/main/examples)
directory holds many more models — buckling, contact springs, tunnel
meshes, embedded rebars, soil–structure interaction. They are working
material rather than teaching material; start with the worked examples
above, then browse the repo when you need a starting point closer to
your own problem.
