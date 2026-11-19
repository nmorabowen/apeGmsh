### FIXED — go-to-source opens only a source file the open set's /provenance names; drops and offers stop counting once replaced (#2760)

apeGmshViewer's go-to-source used to start the user's editor on any absolute
path the page named. It now opens only a source file that the open set's model
names in its `/provenance`. Main reads that zone itself when the request
arrives, and never uses the page's copy. A row relative to `@base_dir` and an
absolute row outside the set's folder both open. Any other file is refused
before an editor is chosen, and so is every request when no set is open.
Refusal messages are now built safely, so a BigInt or a circular request gets
a reason instead of an error.

Only the latest drop now counts as a gesture. A file the user dropped earlier,
and has since replaced with another drop, no longer opens without a gesture.
The latest drop still lets the page re-read its model. Offers now expire: main
keeps the same list the page shows, the newest offers up to the page's limit.
An offer that falls off that list can no longer be opened, and neither can the
set it left behind. A Back to that set is refused.
