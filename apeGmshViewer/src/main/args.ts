// The main process's command line, read once when this module loads (VH2,
// #2866). Every mode the app runs in, and every file it is pointed at, comes
// from here, so a module of main/ reads the mode without importing main.ts.
// The modes and their arguments are described at the top of ./main.ts.
//
// This module only reads argv. A refused command line (`badArgs`, mode
// `"refused"`) is said on stderr and exits 2 in main.ts, where it always was.

import { fileFromArgv } from "./pairing.ts";
import { agentArgsOf } from "../spec/snapshot.ts";

// Unpackaged, argv[1] is the app path; packaged, the file follows the exe.
export const argvSkip = process.defaultApp ? 2 : 1;
/** The ADR's spellings as `--mode=` arguments; null when the command line uses none. */
const spelled = agentArgsOf(process.argv.slice(argvSkip));
const argv = spelled && "argv" in spelled ? [...spelled.argv, ...process.argv] : process.argv;

export function arg(name: string): string | undefined {
  const pre = `--${name}=`;
  return argv.find((a) => a.startsWith(pre))?.slice(pre.length);
}

export const MODES = ["view", "measure", "capture", "measure-results", "render", "describe", "present", "snapshot"] as const;
/** Why this command line is refused before anything starts: said on stderr, exit 2. Never a thrown error (a modal), never view mode. */
export const badArgs =
  spelled && "refused" in spelled
    ? spelled.refused
    : !(MODES as readonly string[]).includes(arg("mode") ?? "view")
      ? `unknown --mode=${arg("mode")}; expected ${MODES.join(", ")}`
      : arg("hang-ms") !== undefined && !/^[1-9][0-9]*$/.test(arg("hang-ms")!)
        ? `--hang-ms=${arg("hang-ms")}: expected a whole number of milliseconds above 0`
        : arg("capture-tries") !== undefined && !(/^[1-9][0-9]*$/.test(arg("capture-tries")!) && Number(arg("capture-tries")) >= 2)
          ? `--capture-tries=${arg("capture-tries")}: expected a whole number of captures, 2 or more`
          : null;
export const mode = badArgs === null ? (arg("mode") ?? "view") : "refused";
/** ADR 0128 D4: the two agent modes; `out` is a folder for them, and they write nothing outside it (INV-1). */
export const agentMode = mode === "render" || mode === "describe";
/** ADR 0128 D5, D6: these reach the window already running, and never open one. */
export const reachMode = mode === "present" || mode === "snapshot";
export const file = mode === "view" ? fileFromArgv(process.argv, argvSkip, process.cwd()) : arg("file");
export const out = arg("out");
export const t0 = Number(arg("t0") ?? Date.now());
/** Every mode but view runs for a caller that waits on its exit code (VA2h, #2665: see main.ts). */
export const headless = mode !== "view";
