### ADDED — `verify_move --class-map OLD=NEW[,NEW...]`: prove a method-to-mixin move (program slice S1-m, #1473)

`scripts/verify_move.py` keys defs by in-module qualname, so `apeSees.eigen` moving to
`_ModalMixin.eigen` read as removed plus added. The opt-in, repeatable `--class-map` matches a removed
`OLD.m` with an added `NEW.m` of an identical dump and reports `moved: OLD.m -> NEW.m`; a name added
under two mapped classes, or a changed body, still fails. Class-header entries (added NEW classes, a
changed OLD class) stop failing only when named in the map, and their full dump or diff prints under
`class headers (review by hand):`. Without the flag, behaviour and output are unchanged.

A move is accepted only when NEW is a direct head-side base of OLD. Class-sensitive bodies (private
`__x`, zero-argument `super()`, `__class__`) are refused. Any exempted class header makes the run exit 2
(`--json`: `"needs_review": true`) instead of 0, so a header change always gets a human pass.

Map mode fails closed: any matched move or exempted header exits 2 (MRO and class wiring are left to review). OLD's bases are read per file, and a non-def binding of the moved name in OLD's or NEW's body refuses the move.
