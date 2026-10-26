### CHANGED — the program WIP count no longer counts draft PRs (#2519)

`scripts/program_probe.py` `wip_count` excludes draft PRs, per the maintainer
ruling of 2026-10-23 on #1202, so `scripts/open_program_pr.py` no longer
refuses at 8 counted PRs when some of them are drafts.
