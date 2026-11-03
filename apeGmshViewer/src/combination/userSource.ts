// One user result as an evaluation source (ADR 0130 A5): its `STATES` rows and
// its `@linear` flag. The app's twin of `_UserResultSource` in
// src/apeGmsh/results/user/_states.py.
//
// A row whose address is a `combination` is that combination's stored row, and
// wins over the rule. A state the result stores no row for refuses
// `unresolved`; so does a whole case, because a user result is combined over a
// case only through a stored row. A state-free result (a design check such as
// D/C) is shown as it is and never combined: it refuses `state-free`.

import type { StateAddress } from "../reader/userRules.ts";
import { CombinationRefused, type Case, type Source } from "./evaluate.ts";
import { FieldError, type Field } from "./field.ts";

/** One user result, decoded: one field per `STATES` row, or `states: null` when it is state-free. */
export interface UserResultRows {
  readonly name: string;
  readonly linear: boolean;
  readonly states: readonly StateAddress[] | null;
  readonly fields: readonly Field[];
}

/** The recorded side a user result consults: it vets a `stage` or `mode` address and gives eigenvalues. */
export interface RecordedSide {
  check(address: StateAddress): void;
  eigenvalue(address: StateAddress): number;
}

const addressKey = (a: StateAddress) => JSON.stringify([a.kind, a.source, a.step, a.mode]);

/** `result` as a `Source`; `label` names the evaluation in a refusal. */
export function userResultSource(result: UserResultRows, recorded: RecordedSide | null, label: string): Source {
  if (result.states === null) {
    throw new CombinationRefused(label, "state-free", `user result ${JSON.stringify(result.name)} is state-free: it is shown as it is, never combined`);
  }
  if (result.states.length !== result.fields.length) {
    throw new FieldError(`user result ${JSON.stringify(result.name)} has ${result.states.length} STATES rows and ${result.fields.length} fields`);
  }
  const rows = new Map<string, Field>();
  result.states.forEach((a, i) => {
    const key = addressKey(a);
    if (rows.has(key)) throw new FieldError(`user result ${JSON.stringify(result.name)} stores two rows for ${key}`);
    rows.set(key, result.fields[i]!);
  });
  return {
    linear: result.linear,
    stored: (name: string) => rows.get(addressKey({ kind: "combination", source: name, step: -1, mode: 0 })) ?? null,
    field(address: StateAddress): Field {
      recorded?.check(address);
      const found = rows.get(addressKey(address));
      if (found === undefined) throw new CombinationRefused(label, "unresolved", `user result ${JSON.stringify(result.name)} stores no row for ${addressKey(address)}`);
      return found;
    },
    eigenvalue(address: StateAddress): number {
      if (recorded === null) throw new CombinationRefused(label, "unresolved", "no recorded results to read the eigenvalue from");
      return recorded.eigenvalue(address);
    },
    case(name: string): Case {
      throw new CombinationRefused(label, "unresolved", `case ${JSON.stringify(name)}: a user result is combined over a case only through a stored row (ADR 0130 A5)`);
    },
  };
}
