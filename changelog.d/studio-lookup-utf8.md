### FIXED — `python -m apeGmsh.studio.lookup` no longer crashes on a Windows cp1252 console (#3074)

Looking up a symbol whose signature or docstring carries a non-Latin-1
character, such as the `ρ` of `g.loads.gravity`, raised `UnicodeEncodeError`
on a console using cp1252. The command now switches stdout and stderr to
UTF-8 before it prints. A stream that
cannot be reconfigured, such as a captured test stream, is left alone.
