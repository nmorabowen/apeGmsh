// Go to source (ADR 0112 D3): open the user's script at a line, in the user's
// editor. This is the one process the app starts besides its own Electron
// binary, and it only ever starts an editor:
//
//   1. $APEGMSH_EDITOR, a command template with {file} and {line}
//      (e.g. `code -g {file}:{line}`, `"C:\Tools\notepad++.exe" -n{line} {file}`);
//      when it is set and broken, that is the answer: no silent fallback.
//   2. VS Code, `code -g file:line`, when `code` is on PATH.
//   3. The OS text editor, without the line: Notepad on Windows, `open -t` on
//      macOS, `xdg-open` elsewhere. Not the OS default *opener*: on Windows
//      the default action for a .py file runs it.
//
// No Electron import: the tests run in Node.

import { spawn } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { delimiter, extname, isAbsolute, join } from "node:path";

export type SourceResult = { ok: true } | { ok: false; reason: string };

/** What decides the editor; `sourceEnv()` reads it from this process. */
export interface SourceEnv {
  editor: string | undefined;
  path: string | undefined;
  pathext: string | undefined;
  comspec: string | undefined;
  platform: NodeJS.Platform;
}

export function sourceEnv(): SourceEnv {
  const env = process.env;
  return {
    editor: env["APEGMSH_EDITOR"],
    path: env["PATH"],
    pathext: env["PATHEXT"],
    comspec: env["ComSpec"],
    platform: process.platform,
  };
}

/** One process to start. `verbatim`: the args are already quoted for cmd.exe. */
export interface Launch {
  command: string;
  args: string[];
  verbatim: boolean;
}

const isFile = (p: string) => existsSync(p) && statSync(p).isFile();

/** The executable `name` names: a path as given, else the first match on PATH. */
export function which(name: string, env: SourceEnv): string | null {
  const win = env.platform === "win32";
  const exts = win && extname(name) === "" ? (env.pathext ?? ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean) : [""];
  const tryAt = (base: string) => exts.map((e) => base + e).find(isFile) ?? null;
  if (/[\\/]/.test(name)) return tryAt(name);
  for (const dir of (env.path ?? "").split(delimiter)) {
    const d = dir.trim().replace(/^"(.*)"$/, "$1");
    if (!d) continue;
    const hit = tryAt(join(d, name));
    if (hit) return hit;
  }
  return null;
}

/** Split a command template on whitespace; double quotes group. */
export function parseTemplate(template: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuote = false;
  let started = false;
  for (const ch of template) {
    if (ch === '"') {
      inQuote = !inQuote;
      started = true;
    } else if (!inQuote && /\s/.test(ch)) {
      if (started) out.push(cur);
      cur = "";
      started = false;
    } else {
      cur += ch;
      started = true;
    }
  }
  if (inQuote) throw new Error(`APEGMSH_EDITOR has an unclosed quote: ${template}`);
  if (started) out.push(cur);
  return out;
}

/** cmd.exe has no escape for `"` and expands `%` inside quotes: refuse both. */
function cmdQuote(arg: string): string {
  if (/["%\r\n]/.test(arg)) throw new Error(`cannot pass ${JSON.stringify(arg)} through cmd.exe (it holds " or %)`);
  return `"${arg}"`;
}

/** A batch file starts through cmd.exe; anything else starts directly. */
function launchOf(exe: string, args: string[], env: SourceEnv): Launch {
  if (env.platform === "win32" && /\.(cmd|bat)$/i.test(exe)) {
    const line = [exe, ...args].map(cmdQuote).join(" ");
    return { command: env.comspec ?? "cmd.exe", args: ["/d", "/s", "/c", `"${line}"`], verbatim: true };
  }
  return { command: exe, args, verbatim: false };
}

/** The process that opens `file` at `line`; throws with the reason it cannot. */
export function editorLaunch(file: string, line: number, env: SourceEnv): Launch {
  if (env.editor !== undefined && env.editor.trim() !== "") {
    const tokens = parseTemplate(env.editor);
    const unknown = env.editor.match(/\{(?!file\}|line\})[^}]*\}/);
    if (unknown) throw new Error(`APEGMSH_EDITOR: unknown placeholder ${unknown[0]}; use {file} and {line}`);
    if (!tokens.slice(1).some((t) => t.includes("{file}"))) {
      throw new Error(`APEGMSH_EDITOR must pass {file} as an argument: ${env.editor}`);
    }
    // A replacer function, so a `$&` in the path is not a replacement pattern.
    const filled = tokens.map((t) => t.replaceAll("{file}", () => file).replaceAll("{line}", () => String(line)));
    const exe = which(filled[0]!, env);
    if (!exe) throw new Error(`APEGMSH_EDITOR: ${filled[0]} not found`);
    return launchOf(exe, filled.slice(1), env);
  }
  const code = which("code", env);
  if (code) return launchOf(code, ["-g", `${file}:${line}`], env);
  const fallback =
    env.platform === "win32" ? "notepad" : env.platform === "darwin" ? "open" : "xdg-open";
  const exe = which(fallback, env);
  if (!exe) throw new Error(`no editor found: set APEGMSH_EDITOR, or put VS Code's \`code\` on PATH`);
  return launchOf(exe, env.platform === "darwin" ? ["-t", file] : [file], env);
}

/** Open `file` at `line`. Resolves once the editor process has started. */
export async function goToSource(file: unknown, line: unknown, env: SourceEnv = sourceEnv()): Promise<SourceResult> {
  if (typeof file !== "string" || !isAbsolute(file)) {
    return { ok: false, reason: `goToSource needs an absolute file path; got ${JSON.stringify(file)}` };
  }
  if (typeof line !== "number" || !Number.isInteger(line) || line < 1) {
    return { ok: false, reason: `goToSource needs a line number >= 1; got ${JSON.stringify(line)}` };
  }
  if (!isFile(file)) return { ok: false, reason: `missing: ${file}` };
  let launch: Launch;
  try {
    launch = editorLaunch(file, line, env);
  } catch (err) {
    return { ok: false, reason: err instanceof Error ? err.message : String(err) };
  }
  return new Promise((done) => {
    const child = spawn(launch.command, launch.args, {
      detached: true,
      stdio: "ignore",
      windowsHide: launch.verbatim,
      windowsVerbatimArguments: launch.verbatim,
    });
    child.once("error", (err) => done({ ok: false, reason: `${launch.command}: ${err.message}` }));
    child.once("spawn", () => {
      child.unref();
      done({ ok: true });
    });
  });
}
