"""Generate static Suzanne-mesh SVGs from the design suite's React component.

Ports suzanne.jsx's mulberry32 RNG + Bowyer-Watson Delaunay triangulation
so we produce the same `SuiteMark` visual (palette='flat', showLines=false)
without needing a browser runtime. Outputs two variants for light/dark
themes under docs/assets/.

Run once when the mark design changes; output is committed.
"""
from __future__ import annotations

import math
from pathlib import Path

MASK32 = 0xFFFFFFFF


# ---------- mulberry32 (matches suzanne.jsx:mulbSz) ----------
def mulberry32(seed: int):
    state = [seed & MASK32]

    def _next() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & MASK32
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK32
        t = (t ^ ((t + ((t ^ (t >> 7)) * (t | 61))) & MASK32)) & MASK32
        return ((t ^ (t >> 14)) & MASK32) / 4294967296

    return _next


# ---------- Bowyer-Watson (matches suzanne.jsx:delaunaySz) ----------
def _circumcircle(a, b, c):
    d = 2 * (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
    if abs(d) < 1e-10:
        return (0.0, 0.0, float("inf"))
    ax2 = a[0]*a[0] + a[1]*a[1]
    bx2 = b[0]*b[0] + b[1]*b[1]
    cx2 = c[0]*c[0] + c[1]*c[1]
    ux = (ax2*(b[1]-c[1]) + bx2*(c[1]-a[1]) + cx2*(a[1]-b[1])) / d
    uy = (ax2*(c[0]-b[0]) + bx2*(a[0]-c[0]) + cx2*(b[0]-a[0])) / d
    return (ux, uy, (ux-a[0])**2 + (uy-a[1])**2)


def delaunay(points):
    minx = min(p[0] for p in points); maxx = max(p[0] for p in points)
    miny = min(p[1] for p in points); maxy = max(p[1] for p in points)
    dmax = max(maxx - minx, maxy - miny)
    mx, my = (minx + maxx) / 2, (miny + maxy) / 2
    st = [(mx - 20 * dmax, my - dmax),
          (mx, my + 20 * dmax),
          (mx + 20 * dmax, my - dmax)]
    pts = points + st
    si = len(points)
    tris = [[si, si + 1, si + 2]]

    for i, p in enumerate(points):
        bad, keep = [], []
        for t in tris:
            c = _circumcircle(pts[t[0]], pts[t[1]], pts[t[2]])
            if (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < c[2]:
                bad.append(t)
            else:
                keep.append(t)
        edges = []
        for t in bad:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                shared = False
                for t2 in bad:
                    if t2 is t:
                        continue
                    for c, d in ((t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])):
                        if (c == a and d == b) or (c == b and d == a):
                            shared = True
                            break
                    if shared:
                        break
                if not shared:
                    edges.append((a, b))
        tris = keep
        for a, b in edges:
            tris.append([a, b, i])

    return [t for t in tris if all(x < len(points) for x in t)]


# ---------- Point generator (matches SuzanneMesh feature + grid) ----------
SZ_EAR_L = (52, 108)
SZ_EAR_R = (188, 108)

SZ_HEAD_D = (
    "M 80 42 C 72 42, 66 48, 66 58 L 66 80 C 58 84, 54 94, 56 108 "
    "C 54 122, 58 136, 68 146 L 68 160 C 70 174, 82 182, 96 184 "
    "C 106 194, 134 194, 144 184 C 158 182, 170 174, 172 160 L 172 146 "
    "C 182 136, 186 122, 184 108 C 186 94, 182 84, 174 80 L 174 58 "
    "C 174 48, 168 42, 160 42 C 148 36, 124 34, 120 34 "
    "C 116 34, 92 36, 80 42 Z"
)


def gen_points(seed: int = 11, density: int = 10):
    rand = mulberry32(seed)
    p = [
        (98, 88), (142, 88), (120, 82),
        (102, 100), (138, 100),
        (114, 152), (126, 152), (120, 148),
        (106, 162), (134, 162),
        (112, 170), (128, 170), (120, 173),
        SZ_EAR_L, SZ_EAR_R,
        (SZ_EAR_L[0] - 8, SZ_EAR_L[1] - 8),
        (SZ_EAR_R[0] + 8, SZ_EAR_R[1] - 8),
        (SZ_EAR_L[0] - 8, SZ_EAR_L[1] + 8),
        (SZ_EAR_R[0] + 8, SZ_EAR_R[1] + 8),
    ]
    bx, by, bw, bh = 30, 30, 180, 170
    cols = math.ceil(bw / density)
    rows = math.ceil(bh / density)
    for j in range(rows + 1):
        for i in range(cols + 1):
            jx = (rand() - 0.5) * density * 0.55
            jy = (rand() - 0.5) * density * 0.55
            p.append((bx + (i * bw) / cols + jx,
                      by + (j * bh) / rows + jy))
    return p


# ---------- SVG emitter ----------
def make_svg(fill: str, stroke: str, stroke_width: float = 0.5,
             seed: int = 11, density: int = 10) -> str:
    pts = gen_points(seed=seed, density=density)
    tris = delaunay(pts)

    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" '
        'aria-label="apeGmsh mark">',
        '<defs><clipPath id="sz-clip">',
        f'<path d="{SZ_HEAD_D}"/>',
        '<ellipse cx="52" cy="108" rx="18" ry="22"/>',
        '<ellipse cx="188" cy="108" rx="18" ry="22"/>',
        '</clipPath></defs>',
        f'<g clip-path="url(#sz-clip)" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linejoin="round">',
    ]
    for t in tris:
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        out.append(
            f'<polygon points="'
            f'{a[0]:.1f},{a[1]:.1f} {b[0]:.1f},{b[1]:.1f} {c[0]:.1f},{c[1]:.1f}'
            f'"/>'
        )
    out.append('</g></svg>')
    return '\n'.join(out)


def main():
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Light-mode mark: dark fill on warm paper (PAL_LIGHT)
    light = make_svg(fill="#101820", stroke="rgba(244,240,230,0.4)")
    (out_dir / "logo.svg").write_text(light, encoding="utf-8")

    # Dark-mode mark: pale fill on deep navy (PAL_DARK)
    dark = make_svg(fill="rgba(180,210,240,0.95)",
                    stroke="rgba(11,37,64,0.75)")
    (out_dir / "logo-dark.svg").write_text(dark, encoding="utf-8")

    print(f"Wrote {out_dir / 'logo.svg'}")
    print(f"Wrote {out_dir / 'logo-dark.svg'}")


if __name__ == "__main__":
    main()
