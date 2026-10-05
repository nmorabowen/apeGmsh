### ADDED — `verify_move --class-map OLD=NEW[,NEW...]`: prove a method-to-mixin move (program slice S1-m, #1473)

`scripts/verify_move.py` keys defs by in-module qualname, so `apeSees.eigen` moving to
`_ModalMixin.eigen` read as removed plus added. The opt-in, repeatable `--class-map` matches a removed
`OLD.m` with an added `NEW.m` of an identical dump and reports `moved: OLD.m -> NEW.m`; a name added
under two mapped classes, or a changed body, still fails. Class-header entries (added NEW classes, a
changed OLD class) stop failing only when named in the map, and their full dump or diff prints under
`class headers (review by hand):`. Without the flag, behaviour and output are unchanged.
