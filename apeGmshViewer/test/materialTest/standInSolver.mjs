// A stand-in for OpenSees.exe (test/materialTest/runner.test.ts): it reads the
// script it is given and answers as $STAND_IN says, on stderr, as the fork does.
// Every ending exits 0, because the fork exits 0 after a script error too.

import { readFileSync, writeFileSync } from "node:fs";

const script = readFileSync(process.argv[2], "utf8");
const strains = /foreach mtStrain \{([^}]*)\}/.exec(script)[1].trim().split(/\s+/);
const mode = process.env.STAND_IN ?? "ok";
if (process.env.STAND_IN_CWD) writeFileSync(process.env.STAND_IN_CWD, process.cwd());
const say = (s) => process.stderr.write(`${s}\n`);

say("banner of the solver");
if (mode === "hang") setInterval(() => undefined, 1000);
else if (mode === "flood") {
  const line = `${"x".repeat(1023)}\n`;
  const pump = () => {
    while (process.stderr.write(line));
    process.stderr.once("drain", pump);
  };
  pump();
} else if (mode === "nofork") say("@@MT nofork");
else {
  say("@@MT build abc123");
  if (mode === "material-error") say('WARNING bad material\n    while executing\n"uniaxialMaterial Steel02 1"');
  else {
    const rows = mode === "short" ? strains.slice(0, 1) : strains;
    rows.forEach((e, i) => say(`@@MT row ${i} ${mode === "nan" && i === 1 ? "-nan(ind)" : (2 * Number(e)).toExponential(10)} 2.0000000000e+00`));
    if (mode !== "no-end") say(`@@MT end ${rows.length}`);
  }
}
