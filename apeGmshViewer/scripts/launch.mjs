// Start the Electron app in one of three modes.
//
//   node scripts/launch.mjs view    [model.h5]
//   node scripts/launch.mjs measure <model.h5>          prints one MEASUREMENTS row
//   node scripts/launch.mjs capture <model.h5> <out.png>
//
// The only process started is the Electron binary from node_modules.

import electronBinary from "electron";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// `--uncapped` (measure only) lifts the vsync / frame-rate cap so the orbit
// fps shows GPU throughput instead of the display refresh.
// `--pick=<OpenSees type>` (capture only) selects an element of that type
// instead of the default target (a beam-column if any).
const flags = process.argv.slice(2).filter((a) => a.startsWith("--"));
const [mode, file, outArg] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const uncapped = flags.includes("--uncapped");
const pick = flags.find((f) => f.startsWith("--pick="))?.slice("--pick=".length);
const badFlag = flags.find(
  (f) =>
    !(f === "--uncapped" && mode === "measure") &&
    !(f.startsWith("--pick=") && f.length > "--pick=".length && mode === "capture"),
);
if (badFlag) {
  console.error(`unknown flag ${badFlag}; only "measure ... --uncapped" and "capture ... --pick=<type>" take one`);
  process.exit(2);
}

if (!["view", "measure", "capture"].includes(mode ?? "")) {
  console.error("usage: launch.mjs view|measure|capture [model.h5] [out.png]");
  process.exit(2);
}
if (mode !== "view" && !file) {
  console.error(`npm run ${mode} -- <model.h5>${mode === "capture" ? " <out.png>" : ""}`);
  process.exit(2);
}
if (file && !existsSync(file)) {
  console.error(`no such file: ${file}`);
  process.exit(2);
}
if (mode === "capture" && !outArg) {
  console.error("npm run capture -- <model.h5> <out.png>");
  process.exit(2);
}

const out =
  mode === "measure"
    ? join(mkdtempSync(join(tmpdir(), "agv-measure-")), "result.json")
    : mode === "capture"
      ? resolve(outArg)
      : undefined;
if (mode === "capture") mkdirSync(dirname(out), { recursive: true });

const args = [root, `--mode=${mode}`, `--t0=${Date.now()}`];
if (file) args.push(`--file=${resolve(file)}`);
if (uncapped) args.push("--uncapped=1");
if (pick) args.push(`--pick=${pick}`);
if (out) args.push(`--out=${out}`);

const child = spawn(electronBinary, args, { stdio: "inherit" });
child.on("exit", (code) => {
  if (mode === "view") process.exit(code ?? 0);
  if (mode === "capture") {
    if (existsSync(out)) {
      console.log(`capture: wrote ${out} (${statSync(out).size} bytes)`);
      const end = out.replace(/\.png$/i, "") + ".chain-end.png";
      if (existsSync(end)) console.log(`capture: wrote ${end} (${statSync(end).size} bytes)`);
      process.exit(0);
    }
    const why = existsSync(`${out}.error.txt`) ? readFileSync(`${out}.error.txt`, "utf8") : `electron exited ${code}`;
    console.error(`capture failed: ${why}`);
    process.exit(1);
  }
  if (!existsSync(out)) {
    console.error(`measure failed: electron exited ${code} without a result`);
    process.exit(1);
  }
  printRow(JSON.parse(readFileSync(out, "utf8")));
});

function headCommit() {
  // Read HEAD from the repository files (no git process).
  let dir = root;
  for (;;) {
    const dotgit = join(dir, ".git");
    if (existsSync(dotgit)) {
      let gitdir = dotgit;
      if (statSync(dotgit).isFile()) gitdir = resolve(dir, readFileSync(dotgit, "utf8").replace(/^gitdir:\s*/, "").trim());
      const head = readFileSync(join(gitdir, "HEAD"), "utf8").trim();
      if (!head.startsWith("ref:")) return head.slice(0, 8);
      const ref = head.slice(4).trim();
      const common = existsSync(join(gitdir, "commondir"))
        ? resolve(gitdir, readFileSync(join(gitdir, "commondir"), "utf8").trim())
        : gitdir;
      for (const base of [gitdir, common]) {
        if (existsSync(join(base, ref))) return readFileSync(join(base, ref), "utf8").trim().slice(0, 8);
      }
      const packed = join(common, "packed-refs");
      if (existsSync(packed)) {
        const line = readFileSync(packed, "utf8").split("\n").find((l) => l.endsWith(` ${ref}`));
        if (line) return line.slice(0, 8);
      }
      return "unknown";
    }
    const up = dirname(dir);
    if (up === dir) return "unknown";
    dir = up;
  }
}

function printRow(r) {
  const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : "n/a");
  const f0 = (x) => (Number.isFinite(x) ? x.toFixed(0) : "n/a");
  const name = r.file.split(/[\\/]/).pop();
  const elements = r.opsOnlyElements ? `${r.cells} (+${r.opsOnlyElements} OpenSees-only)` : `${r.cells}`;
  const mem = `${f0(r.memory.mainMB + r.memory.rendererMB)} (${f0(r.memory.mainMB)} + ${f0(r.memory.rendererMB)})`;
  const insp = r.inspector ? `${f1(r.inspector.fillMs)}${r.inspector.pickedTarget ? "" : " (picked a neighbour)"}` : "n/a";
  const day = new Date().toISOString().slice(0, 10);
  if (uncapped) {
    // Uncapped frames are shorter than the timer resolution, so the median
    // interval is quantised: report the mean over the whole orbit instead.
    // The other columns are not comparable here (the loop saturates the
    // renderer), so this mode prints its own row.
    console.log("| date | commit | model | size (MB) | elements | orbit frames | orbit ms | mean fps (uncapped) |");
    console.log("|---|---|---|---|---|---|---|---|");
    console.log(
      `| ${day} | ${headCommit()} | ${name} | ${(r.sizeBytes / 1048576).toFixed(2)} | ${elements} | ` +
        `${r.orbit.frames} | ${f0(r.orbit.durationMs)} | ${f0((1000 * r.orbit.frames) / r.orbit.durationMs)} |`,
    );
    console.log(`GPU "${r.gpu}"; canvas ${r.viewport.join("x")} @ ${r.pixelRatio}x`);
    return;
  }
  console.log(
    "| date | commit | model | size (MB) | nodes | elements | first frame (ms) | median fps (orbit) | render CPU (ms) | memory MB main+renderer | inspector fill (ms) |",
  );
  console.log("|---|---|---|---|---|---|---|---|---|---|---|");
  console.log(
    `| ${day} | ${headCommit()} | ${name} | ${(r.sizeBytes / 1048576).toFixed(2)} | ${r.nodes} | ${elements} | ` +
      `${f0(r.firstFrameMs)} | ${f1(r.orbit.medianFps)} | ${r.orbit.medianRenderCpuMs.toFixed(2)} | ${mem} | ${insp} |`,
  );
  console.log("");
  console.log(
    `detail: Electron ready at ${f0(r.startup.electronReadyMs)} ms, renderer up at ${f0(r.startup.rendererUpMs)} ms; ` +
      `read ${f0(r.readMs)} ms; orbit ${r.orbit.frames} frames over ${f0(r.orbit.durationMs)} ms; ` +
      `drawn ${r.drawn.segments} segments + ${r.drawn.triangles} triangles; GPU process ${f0(r.memory.gpuProcessMB)} MB; ` +
      `GPU "${r.gpu}"; canvas ${r.viewport.join("x")} @ ${r.pixelRatio}x; three r${r.three}`,
  );
  if (r.inspector) {
    console.log(
      `inspector: ${r.inspector.root} (${r.inspector.type}), ${r.inspector.links} linked objects; ` +
        `fill = pick ${f1(r.inspector.split.pickMs)} + state/DOM ${f1(r.inspector.split.stateAndDomMs)} + paint ${f1(r.inspector.split.paintMs)} ms` +
        (r.inspector.problems.length ? `; problems: ${r.inspector.problems.join(" | ")}` : "; no unresolved links"),
    );
  }
  for (const w of r.warnings) console.log(`warning: ${w}`);
}
