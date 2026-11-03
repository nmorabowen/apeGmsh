### CHANGED — `scripts/refresh_user_skill.py` refreshes every repo skill, not only `apegmsh` (#2811)

The script now also copies each `.claude/skills/<name>/` on origin/main
(the dev skills, `apegmsh-program` and the `apegmsh-helper` mirror
included) into `~/.claude/skills/<name>/`, reading the origin/main tree
object like before. `--check` reports every skill and exits 1 if any is
stale, comparing content with line endings ignored. A user-level skill
that is not in the repo is never touched. Run it after `git pull` on each
PC.
