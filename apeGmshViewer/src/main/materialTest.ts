// The material tester's runner (ADR 0135 D1 to D3): the one module besides
// source.ts that starts a process, and it only ever starts the fork's OpenSees.
//
// The executable comes from $APEGMSH_OPENSEES or the default install, never from
// a model or the page. Each run gets a fresh temporary directory, no shell, a
// time limit and an output limit. A new run supersedes the one going.
//
// No Electron import: the tests run in Node.

import { spawn } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, dirname, isAbsolute, join } from "node:path";
import { MaterialTestRefused, parseRun, scriptOf } from "./materialTestScript.ts";

export const DEFAULT_SOLVER = "C:\\Program Files\\Ladruno\\OpenSees\\bin\\OpenSees.exe";
const TIME_LIMIT_MS = 10_000;
const OUTPUT_LIMIT_BYTES = 32 * 1024 * 1024;

export interface SolverLaunch {
  command: string;
  /** placed before the script's path */
  args: string[];
  env: NodeJS.ProcessEnv;
}

export type SolverAnswer = { ok: true; launch: SolverLaunch } | { ok: false; error: string };

export type MaterialTestAnswer =
  | { ok: true; seq: number; build: string; strain: Float64Array; stress: Float64Array; tangent: Float64Array }
  | { ok: false; seq: number; build: string | null; error: string; superseded?: true };

const isFile = (p: string): boolean => existsSync(p) && statSync(p).isFile();

/** What `opensees_ladruno.bat` sets, relative to the executable. */
function solverEnv(command: string, base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const root = dirname(command);
  const env = { ...base };
  const key = Object.keys(env).find((k) => k.toLowerCase() === "path") ?? "PATH";
  const dirs = [root, join(root, "opensees-bin"), join(root, "oneapi"), join(root, "plugins")];
  env[key] = [...dirs, ...(env[key] ? [env[key]] : [])].join(delimiter);
  const tcl = join(root, "lib", "tcl8.6");
  if (existsSync(tcl)) env["TCL_LIBRARY"] = tcl;
  return env;
}

export function resolveSolver(env: NodeJS.ProcessEnv = process.env, fallback: string = DEFAULT_SOLVER): SolverAnswer {
  const chosen = env["APEGMSH_OPENSEES"];
  if (chosen) {
    // When it is set and broken, that is the answer: no silent fallback (as $APEGMSH_EDITOR).
    if (!isAbsolute(chosen) || !isFile(chosen)) return { ok: false, error: `$APEGMSH_OPENSEES must be the full path of an executable; got ${chosen}` };
    return { ok: true, launch: { command: chosen, args: [], env: solverEnv(chosen, env) } };
  }
  if (!isFile(fallback)) return { ok: false, error: `no OpenSees executable: set $APEGMSH_OPENSEES, or install the Ladruno fork at ${fallback}` };
  return { ok: true, launch: { command: fallback, args: [], env: solverEnv(fallback, env) } };
}

interface Active {
  stop(reason: "superseded" | "cancelled"): void;
}

export class MaterialTestRunner {
  private active: Active | null = null;
  /** counts runs and cancels, so a run overtaken before its process starts never starts one */
  private generation = 0;

  private readonly solver: () => SolverAnswer;
  private readonly timeLimitMs: number;
  private readonly outputLimit: number;

  constructor(solver: () => SolverAnswer = resolveSolver, timeLimitMs: number = TIME_LIMIT_MS, outputLimit: number = OUTPUT_LIMIT_BYTES) {
    this.solver = solver;
    this.timeLimitMs = timeLimitMs;
    this.outputLimit = outputLimit;
  }

  cancel(): void {
    this.generation++;
    this.active?.stop("cancelled");
  }

  async run(job: unknown, seq: number): Promise<MaterialTestAnswer> {
    const generation = ++this.generation;
    this.active?.stop("superseded");
    const overtaken: MaterialTestAnswer = { ok: false, seq, build: null, error: "a newer run replaced this one", superseded: true };
    const failed = (error: string, build: string | null = null): MaterialTestAnswer => ({ ok: false, seq, build, error });
    let built: ReturnType<typeof scriptOf>;
    try {
      built = scriptOf(job);
    } catch (e) {
      if (e instanceof MaterialTestRefused) return failed(`the run was refused: ${e.message}`);
      throw e;
    }
    const solver = this.solver();
    if (!solver.ok) return failed(solver.error);
    const dir = await mkdtemp(join(tmpdir(), "apegmsh-material-test-"));
    try {
      const path = join(dir, "test.tcl");
      await writeFile(path, built.script, "utf8");
      if (generation !== this.generation) return overtaken;
      const out = await this.execute(solver.launch, path, dir);
      if ("stopped" in out) {
        if (out.stopped === "superseded") return overtaken;
        return failed(out.stopped === "cancelled" ? "the run was cancelled" : out.stopped);
      }
      const parsed = parseRun(out.stderr, built.strain.length);
      if (!parsed.ok) return failed(parsed.error, parsed.build);
      return { ok: true, seq, build: parsed.build, strain: Float64Array.from(built.strain), stress: parsed.stress, tangent: parsed.tangent };
    } finally {
      await rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 });
    }
  }

  private execute(launch: SolverLaunch, script: string, cwd: string): Promise<{ stderr: string } | { stopped: string }> {
    return new Promise((resolve) => {
      const child = spawn(launch.command, [...launch.args, script], { cwd, env: launch.env, shell: false, windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
      const chunks: Buffer[] = [];
      let size = 0;
      let stopped: string | null = null;
      const stop = (why: string): void => {
        stopped ??= why;
        child.kill();
      };
      const entry: Active = { stop };
      this.active = entry;
      const timer = setTimeout(() => stop(`the run took longer than ${this.timeLimitMs / 1000} s and was stopped`), this.timeLimitMs);
      child.stderr.on("data", (b: Buffer) => {
        size += b.length;
        if (size > this.outputLimit) stop(`the solver wrote more than ${this.outputLimit / (1024 * 1024)} MB and was stopped`);
        else chunks.push(b);
      });
      const finish = (answer: { stderr: string } | { stopped: string }): void => {
        clearTimeout(timer);
        if (this.active === entry) this.active = null;
        resolve(answer);
      };
      child.on("error", (e) => finish({ stopped: stopped ?? `the solver could not start: ${e.message}` }));
      child.on("close", () => finish(stopped ? { stopped } : { stderr: Buffer.concat(chunks).toString("utf8") }));
    });
  }
}
