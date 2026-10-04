### FIXED — `acad/` sample files stored with LF

`acad/Drawing1.iges.opt`, `acad/frame_2D.bak` and `acad/frame_2D.geo` were committed with CRLF in the repository, so `git add --renormalize .` flagged them on every Windows clone. They are now stored with LF like the rest of the tree. Their content is unchanged apart from line endings.
