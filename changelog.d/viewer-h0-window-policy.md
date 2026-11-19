### CHANGED — apeGmshViewer: no page opens a window or navigates away (ADR 0137 H0)

Every `webContents` the app creates now denies every new-window request
(`window.open`, a `target=_blank` link) and prevents `will-navigate` and
`will-redirect` unless the target is the page already loaded (ADR 0137 D13
items 1 and 2). Only the main process creates windows. The policy is
`apeGmshViewer/src/main/windowPolicy.ts`, applied from
`app.on("web-contents-created")` so any later window inherits it. Main's own
loads and the watch reload (`webContents.reload()`) are untouched. The model
page's content security policy also gains `object-src 'none'`,
`frame-src 'none'`, `base-uri 'none'` and `form-action 'none'`. Nothing changes
on screen. New test: `test/main/windowPolicy.test.ts`.
