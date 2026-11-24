// #3130: the page side of a bulk answer (src/bulk.ts). The preload posts the
// answer to its window and the bridge resolves to its id; the two arrive in
// either order, and only the page's own window is heard.

import assert from "node:assert/strict";
import { test } from "node:test";
import { BULK_KEY, BulkInbox, type BulkWindow } from "../src/bulk.ts";

type FakeWindow = BulkWindow & { post(data: unknown, source?: unknown): void };

function fakeWindow(): FakeWindow {
  const listeners: ((e: { source: unknown; data: unknown }) => void)[] = [];
  const win: FakeWindow = {
    addEventListener: (_type: "message", l: (e: { source: unknown; data: unknown }) => void) => void listeners.push(l),
    post: (data: unknown, source: unknown = win) => {
      for (const l of listeners) l({ source, data });
    },
  };
  return win;
}

test("#3130: the answer is taken by its id when the message arrives first", async () => {
  const win = fakeWindow();
  const inbox = new BulkInbox(win);
  win.post({ [BULK_KEY]: 1, value: { ok: true, n: 1 } });
  assert.deepEqual(await inbox.take(Promise.resolve(1)), { ok: true, n: 1 });
});

test("#3130: the answer is taken by its id when the id arrives first", async () => {
  const win = fakeWindow();
  const inbox = new BulkInbox(win);
  const taken = inbox.take(Promise.resolve(2));
  await Promise.resolve();
  win.post({ [BULK_KEY]: 3, value: "another answer" });
  win.post({ [BULK_KEY]: 2, value: "the answer" });
  assert.equal(await taken, "the answer");
  assert.equal(await inbox.take(Promise.resolve(3)), "another answer", "an answer nobody waits for yet is held, not dropped");
});

test("#3130: another window's message, or one without a numeric id, is not an answer", async () => {
  const win = fakeWindow();
  const inbox = new BulkInbox(win);
  const taken = inbox.take(Promise.resolve(4));
  await Promise.resolve();
  win.post({ [BULK_KEY]: 4, value: "from a frame" }, {});
  win.post({ [BULK_KEY]: "4", value: "a string id" });
  win.post(null);
  win.post({ [BULK_KEY]: 4, value: "ours" });
  assert.equal(await taken, "ours");
});

test("#3130: a refused invoke rejects the take", async () => {
  const inbox = new BulkInbox(fakeWindow());
  await assert.rejects(inbox.take(Promise.reject(new Error("model:open: no such file"))), /no such file/);
});
