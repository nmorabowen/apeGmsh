### FIXED — apeGmshViewer: app-chrome stills show open arrows over open cards

An app-chrome still draws every card and the warnings banner open, but a card collapsed in the app still showed its
closed arrow (▸) above the contents shown under it, and the banner read "▸ 2 warnings" above its list. The still now
shows the open arrow (▾) on both, and the banner keeps the open banner's height limit, so a long list scrolls inside
it instead of running off the still. In the app itself nothing changes.
