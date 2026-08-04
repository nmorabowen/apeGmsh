"""Compose the ADR 0090 evidence sheets from the rendered panels."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PANELS = HERE / "panels"

BG = (24, 24, 30)
FG = (205, 210, 225)
MUT = (150, 155, 170)

try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    FONT_S = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 12)
    FONT_H = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 18)
except Exception:
    FONT = FONT_S = FONT_H = ImageFont.load_default()


def panel(name):
    return Image.open(PANELS / f"{name}.png")


def sheet(out, title, rows, scale=2):
    """rows: list of list[(panel_name, caption)]."""
    row_imgs = []
    for row in rows:
        ims = []
        for name, cap in row:
            im = panel(name)
            im = im.resize((im.width * scale, im.height * scale),
                           Image.LANCZOS)
            ims.append((im, name, cap))
        row_imgs.append(ims)
    pad = 14
    cap_h = 44
    width = max(
        sum(im.width for im, _, _ in row) + pad * (len(row) + 1)
        for row in row_imgs
    )
    height = 56 + sum(
        max(im.height for im, _, _ in row) + cap_h + pad
        for row in row_imgs
    )
    s = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(s)
    d.text((pad, 14), title, fill=FG, font=FONT_H)
    y = 56
    for row in row_imgs:
        rh = max(im.height for im, _, _ in row)
        x = pad
        for im, name, cap in row:
            s.paste(im, (x, y))
            d.text((x, y + rh + 4), name, fill=FG, font=FONT_S)
            for i, line in enumerate(cap.split("\n")):
                d.text((x, y + rh + 18 + 13 * i), line, fill=MUT,
                       font=FONT_S)
            x += im.width + pad
        y += rh + cap_h + pad
    s.save(HERE / out)
    print("saved", out, s.size)


sheet(
    "SHEET_10_typography_and_ticks.png",
    "ADR 0090 / sheet 10 — typography, format, ticks (catppuccin_mocha, "
    "vertical, font_scale 1.0 unless noted)",
    [[
        ("T0_current", "current: black text on dark bg,\nraw key title"),
        ("T1_text_role", "palette `text` role\n(colour only)"),
        ("T2_mono_title", "+ mono labels,\ndesigned title"),
        ("T6_trailing_zeros", "fmt %#.3g —\naligned trailing zeros"),
        ("T4_sci_fmt", "fmt %.2e —\nnoisy, rejected"),
        ("T3_seven_ticks", "7 ticks — ragged values,\nrejected"),
        ("T5_scale_125", "font_scale 1.25\n(user knob, not default)"),
    ]],
)

sheet(
    "SHEET_11_container_chip.png",
    "ADR 0090 / sheet 11 — container: naked ramp vs backdrop chip",
    [[
        ("C0_mocha_naked", "mocha, no chip"),
        ("C1_mocha_chip", "VTK DrawBackground\nmantle 85% + surface0"),
        ("C1b_mocha_chip92", "same at 92%"),
        ("V2_chip_backdrop", "PROPOSED: backdrop actor,\n8 px outer pad"),
    ], [
        ("C2_paper_current", "paper, current"),
        ("C3_paper_chip", "paper, VTK chip\n(surface0 border too faint)"),
        ("C3b_paper_backdrop_s2", "PROPOSED: backdrop,\nsurface2 border"),
    ]],
)

sheet(
    "SHEET_12_ramp_and_swatches.png",
    "ADR 0090 / sheet 12 — ramp presentation + out-of-range/NaN swatches",
    [[
        ("R0_continuous", "continuous (default)"),
        ("R1_banded10_crop", "banded x10 — field and\nbar share the LUT"),
        ("R2b_nan_only", "NaN swatch: VTK relayouts\ninside the box — clips"),
        ("R2_banded_nan", "+ above/below swatches —\nbox explodes; deferred"),
    ]],
)

sheet(
    "SHEET_13_horizontal_and_multi.png",
    "ADR 0090 / sheet 13 — horizontal variant + two legends",
    [[
        ("H0_horizontal_current", "horizontal, current:\ntitle lost on field"),
        ("H1_horizontal_proposed", "VTK chip: title band\nNOT covered — defect"),
    ], [
        ("H2_horizontal_backdrop", "backdrop chip covers\ntitle band, 85%"),
        ("H3_horizontal_backdrop92", "PROPOSED: 92% —\nglass-card alpha"),
    ], [
        ("M0_two_legends_current", "two legends, current"),
        ("M1_two_legends_proposed", "two legends, proposed\n(per-legend chips)"),
    ]],
)


# Full-context sheet — no upscale, 0.62 downscale, stacked.
imgs = [
    (f"F_{n}_proposed_context",
     n) for n in ("catppuccin_mocha", "neutral_studio", "paper")
]
scale = 0.62
loaded = []
for name, label in imgs:
    im = panel(name)
    im = im.resize((int(im.width * scale), int(im.height * scale)),
                   Image.LANCZOS)
    loaded.append((im, label))
pad = 14
w = max(im.width for im, _ in loaded) + 2 * pad
h = 56 + sum(im.height + 30 + pad for im, _ in loaded)
s = Image.new("RGB", (w, h), BG)
d = ImageDraw.Draw(s)
d.text((pad, 14),
       "ADR 0090 / sheet 14 — proposed legend in context, three canonical "
       "palettes", fill=FG, font=FONT_H)
y = 56
for im, label in loaded:
    s.paste(im, (pad, y))
    d.text((pad, y + im.height + 6), label, fill=FG, font=FONT_S)
    y += im.height + 30 + pad
s.save(HERE / "SHEET_14_proposed_context.png")
print("saved SHEET_14_proposed_context.png", s.size)
