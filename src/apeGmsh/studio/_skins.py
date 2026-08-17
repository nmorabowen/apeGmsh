"""HTML and canvas skins of a ReportBundle (ADR 0095 Amendment 2).

Markdown stays the archive (INV-12). HTML is a shareable/print skin under
``docs/``. Canvas is a live Cursor projection under the IDE ``canvases/``
directory (not git-portable). Pure: no ``gmsh``, no Qt, no viewers.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

_CANVAS_NAME = "studio-report.canvas.tsx"


def cursor_project_slug(root: Path | str) -> str:
    """Cursor's ``~/.cursor/projects/<slug>`` mapping for an absolute path."""
    text = str(Path(root).resolve())
    text = text.replace(":", "")
    text = text.replace("\\", "-").replace("/", "-")
    text = text.replace(".", "")
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return text


def default_canvas_path(root: Path) -> Path | None:
    """IDE ``canvases/studio-report.canvas.tsx``, or ``CURSOR_CANVAS_DIR``.

    Returns ``None`` when the directory does not exist — callers must pass
    ``output=`` rather than write a non-live file under ``docs/``.
    """
    env = os.environ.get("CURSOR_CANVAS_DIR")
    if env:
        return Path(env) / _CANVAS_NAME
    folder = (
        Path.home() / ".cursor" / "projects" / cursor_project_slug(root) / "canvases"
    )
    if folder.is_dir():
        return folder / _CANVAS_NAME
    return None


def render_html(
    bundle: dict[str, Any],
    *,
    figures: list[tuple[str, str]],
    archive: str,
) -> str:
    """Self-contained print skin. Figures are relative ``docs/figures/`` paths."""
    names = bundle.get("names") or {}
    sel = bundle.get("selection") or {}
    last = bundle.get("last_run") or {}
    pin = bundle.get("pin") or {}
    assess = bundle.get("assess") or {}
    assess_error = bundle.get("assess_error")
    esc = html.escape

    def csv(items: Any) -> str:
        seq = [str(x) for x in (items or []) if x]
        return esc(", ".join(seq)) if seq else "(none)"

    def counts(obj: Any) -> str:
        if not isinstance(obj, dict) or not obj:
            return "(none)"
        return esc(" ".join(f"{k}={v}" for k, v in obj.items()))

    phase = esc(str(names.get("phase") or last.get("phase") or "(none)"))
    body: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        "<title>Studio report</title>",
        "<style>",
        "body{font:16px/1.45 system-ui,sans-serif;max-width:52rem;"
        "margin:2rem auto;padding:0 1rem;color:#111}",
        "h1,h2{font-weight:600} img{max-width:100%;height:auto}",
        "code{font-size:.92em} .note{color:#555}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Studio report</h1>",
        f'<p class="note">ReportBundle schema '
        f"{esc(str(bundle.get('schema')))}. "
        "Shareable skin of the habitat payload; "
        f"the archive is <code>{esc(archive)}</code>.</p>",
    ]
    if pin:
        body.append("<h2>Pin</h2><ul>")
        body.append(
            f"<li><strong>id:</strong> <code>{esc(str(pin.get('id') or ''))}</code>"
            f" ({esc(str(pin.get('ts') or ''))})</li>"
        )
        model = pin.get("model_h5") or {}
        results = pin.get("results") or {}
        if model.get("path"):
            body.append(
                f"<li><strong>model.h5:</strong> <code>{esc(str(model['path']))}</code>"
                f" (sha256 <code>{esc(str(model.get('sha256', '')[:12]))}</code>)</li>"
            )
        if results.get("path"):
            body.append(
                f"<li><strong>results:</strong> <code>{esc(str(results['path']))}</code>"
                f" (sha256 <code>{esc(str(results.get('sha256', '')[:12]))}</code>)</li>"
            )
        body.append("</ul>")
    body.extend([
        "<h2>Names</h2><ul>",
        f"<li><strong>phase:</strong> {phase}</li>",
        f"<li><strong>labels:</strong> {csv(names.get('labels') or last.get('labels'))}</li>",
        f"<li><strong>physical groups:</strong> "
        f"{csv(names.get('physical_groups') or last.get('physical_groups'))}</li>",
        f"<li><strong>entities:</strong> "
        f"{counts((names.get('counts') or last.get('counts') or {}).get('entities'))}</li>",
        f"<li><strong>elements:</strong> "
        f"{counts((names.get('counts') or last.get('counts') or {}).get('elements'))}</li>",
        "</ul>",
        "<h2>Pick</h2>",
    ])
    if not sel:
        body.append("<p>(none)</p>")
    else:
        body.extend([
            "<ul>",
            f"<li><strong>phase:</strong> {esc(str(sel.get('phase') or '(none)'))}</li>",
            f"<li><strong>labels:</strong> {csv(sel.get('labels'))}</li>",
            f"<li><strong>physical groups:</strong> {csv(sel.get('physical_groups'))}</li>",
            f"<li><strong>unnamed:</strong> {len(sel.get('unnamed') or [])}</li>",
            "</ul>",
        ])
    body.append("<h2>Last run</h2>")
    if not last:
        body.append("<p>(none)</p>")
    else:
        body.extend([
            "<ul>",
            f"<li><strong>ts:</strong> {esc(str(last.get('ts') or '(none)'))}</li>",
            f"<li><strong>script:</strong> <code>{esc(str(last.get('script') or '(none)'))}</code></li>",
            f"<li><strong>phase:</strong> {esc(str(last.get('phase') or '(none)'))}</li>",
            f"<li><strong>ok:</strong> {esc(str(last.get('ok')))}</li>",
            f"<li><strong>hash:</strong> <code>{esc(str(last.get('hash') or '(none)'))}</code></li>",
            "</ul>",
        ])
    body.append("<h2>Assess</h2>")
    if assess:
        body.append(_assess_meta_html(assess, esc))
        text = assess.get("text")
        if text:
            body.append(f"<pre>{esc(str(text).rstrip())}</pre>")
        else:
            findings = assess.get("findings") or []
            body.append(f"<p>{len(findings)} finding(s).</p>")
            if findings:
                body.append("<ul>")
                for row in findings[:24]:
                    if isinstance(row, dict):
                        body.append(
                            f"<li><code>{esc(str(row.get('code') or ''))}</code> "
                            f"({esc(str(row.get('severity') or ''))}): "
                            f"{esc(str(row.get('message') or ''))}</li>"
                        )
                body.append("</ul>")
            skipped = assess.get("skipped") or []
            if skipped:
                body.append("<p>Skipped checks:</p><ul>")
                for row in skipped:
                    if isinstance(row, dict):
                        body.append(
                            f"<li><code>{esc(str(row.get('code') or ''))}</code> "
                            f"— {esc(str(row.get('reason') or ''))}</li>"
                        )
                body.append("</ul>")
    elif assess_error:
        body.append(f"<p>(assess snapshot unreadable: {esc(str(assess_error))})</p>")
    else:
        body.append("<p>(none — run assess first)</p>")
    body.append("<h2>Figures</h2>")
    if not figures:
        body.append(
            "<p>(none — working visors stay under <code>.apegmsh/visors/</code>)</p>"
        )
    else:
        for caption, rel in figures:
            body.append(
                f'<p><img src="{esc(rel, quote=True)}" alt="{esc(caption)}"/></p>'
            )
    body.extend(["</body>", "</html>", ""])
    return "\n".join(body)


def _assess_meta_html(assess: dict[str, Any], esc: Any) -> str:
    """Target + timestamp line for the Assess section (disclosure, not policing)."""
    targets = assess.get("targets") or {}
    bits: list[str] = []
    if targets.get("path"):
        bits.append(f"target: <code>{esc(str(targets['path']))}</code>")
    if targets.get("model_h5"):
        bits.append(f"model_h5: <code>{esc(str(targets['model_h5']))}</code>")
    if assess.get("ts"):
        bits.append(f"assessed: {esc(str(assess['ts']))}")
    if not bits:
        return ""
    return f'<p class="note">{" &middot; ".join(bits)}</p>'


def render_canvas(
    bundle: dict[str, Any],
    *,
    figures: list[tuple[str, str]],
    archive: str,
) -> str:
    """Cursor ``.canvas.tsx`` projection. Data is inlined; no fetch."""
    payload = {
        "schema": bundle.get("schema"),
        "names": bundle.get("names") or {},
        "selection": bundle.get("selection") or {},
        "last_run": bundle.get("last_run") or {},
        "pin": bundle.get("pin") or {},
        "assess": bundle.get("assess") or {},
        "assess_error": bundle.get("assess_error"),
        "figures": [{"caption": cap, "src": src} for cap, src in figures],
        "archive": archive,
    }
    blob = json.dumps(payload, indent=2, ensure_ascii=False)
    return _CANVAS_PREFIX + blob + _CANVAS_SUFFIX


_CANVAS_PREFIX = """import {
  Callout,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const BUNDLE = """

_CANVAS_SUFFIX = """;

function csv(items) {
  const seq = (items || []).map(String).filter(Boolean);
  return seq.length ? seq.join(", ") : "(none)";
}

function counts(obj) {
  if (!obj || typeof obj !== "object") return "(none)";
  const keys = Object.keys(obj);
  if (!keys.length) return "(none)";
  return keys.map((k) => k + "=" + obj[k]).join(" ");
}

export default function StudioReport() {
  const names = BUNDLE.names || {};
  const sel = BUNDLE.selection || {};
  const last = BUNDLE.last_run || {};
  const pin = BUNDLE.pin || {};
  const assess = BUNDLE.assess || {};
  const figures = BUNDLE.figures || [];
  const findings = Array.isArray(assess.findings) ? assess.findings : [];
  const lastCounts = names.counts || last.counts || {};
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <Row gap={8} align="center">
          <Pill size="sm" active>
            ReportBundle
          </Pill>
          <Pill size="sm">schema {String(BUNDLE.schema)}</Pill>
          <Text tone="tertiary" size="small">
            live skin — markdown is the archive
          </Text>
        </Row>
        <H1>Studio report</H1>
        <Text tone="secondary">
          Generated from habitat files. Not a substitute for the script.
        </Text>
      </Stack>

      <Callout tone="info" title="Markdown is the archive">
        This canvas is a review projection beside chat. The git-portable
        chapter is {String(BUNDLE.archive)}.
      </Callout>

      <Grid columns={4} gap={16}>
        <Stat
          value={String(names.phase || last.phase || "—")}
          label="Phase"
        />
        <Stat
          value={String((names.labels || last.labels || []).length)}
          label="Labels"
        />
        <Stat
          value={String(
            (names.physical_groups || last.physical_groups || []).length
          )}
          label="Physical groups"
        />
        <Stat
          value={last.ok === true ? "ok" : last.ok === false ? "fail" : "—"}
          label="Last run"
          tone={
            last.ok === true
              ? "success"
              : last.ok === false
                ? "danger"
                : undefined
          }
        />
      </Grid>

      <H2>Names</H2>
      <Text>labels: {csv(names.labels || last.labels)}</Text>
      <Text>
        physical groups: {csv(names.physical_groups || last.physical_groups)}
      </Text>
      <Text>entities: {counts(lastCounts.entities)}</Text>
      <Text>elements: {counts(lastCounts.elements)}</Text>

      <H2>Pick</H2>
      {sel && Object.keys(sel).length ? (
        <Stack gap={8}>
          <Text>phase: {String(sel.phase || "(none)")}</Text>
          <Text>labels: {csv(sel.labels)}</Text>
          <Text>physical groups: {csv(sel.physical_groups)}</Text>
          <Text>unnamed: {String((sel.unnamed || []).length)}</Text>
        </Stack>
      ) : (
        <Text>(none)</Text>
      )}

      <H2>Pin</H2>
      {pin && pin.id ? (
        <Text>
          {String(pin.id)} ({String(pin.ts || "")})
        </Text>
      ) : (
        <Text>(none)</Text>
      )}

      <H2>Assess</H2>
      {assess && (assess.text || findings.length) ? (
        <Stack gap={8}>
          {assess.targets && (assess.targets.path || assess.ts) ? (
            <Text tone="tertiary" size="small">
              {assess.targets.path ? "target: " + assess.targets.path + "  " : ""}
              {assess.ts ? "assessed: " + assess.ts : ""}
            </Text>
          ) : null}
          {assess.text ? (
            <Text>{String(assess.text)}</Text>
          ) : (
            <Stack gap={8}>
              <Table
                headers={["Code", "Severity", "Message"]}
                rows={findings.slice(0, 24).map((row) => [
                  String(row.code || ""),
                  String(row.severity || ""),
                  String(row.message || ""),
                ])}
              />
              {(assess.skipped || []).length ? (
                <Table
                  headers={["Skipped", "Reason"]}
                  rows={(assess.skipped || []).map((row) => [
                    String(row.code || ""),
                    String(row.reason || ""),
                  ])}
                />
              ) : null}
            </Stack>
          )}
        </Stack>
      ) : BUNDLE.assess_error ? (
        <Text tone="danger">
          assess snapshot unreadable: {String(BUNDLE.assess_error)}
        </Text>
      ) : (
        <Text>(none — run assess first)</Text>
      )}

      <H2>Figures</H2>
      {figures.length ? (
        <Table
          headers={["Caption", "Path"]}
          rows={figures.map((row) => [row.caption, row.src])}
        />
      ) : (
        <Text>
          Pictures live in the markdown chapter. Working visors stay under
          .apegmsh/visors/.
        </Text>
      )}
    </Stack>
  );
}
"""
