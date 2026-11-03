### FIXED — apeGmshViewer: only the current page can start offers; a User Combination contour offer says why it is refused (#2776)

The window sends `--present` offers to its page only after the page says it is
listening. Main now accepts that message only from the window's own page, in the
load that is on screen now. A late message from a page that is being replaced,
from another frame, or from another window no longer starts the offers. Before
this, such a message could send offers to a page that was about to go, and the
new page never listed them.

An offer that shows a User Combination as a contour is listed as refused with
the reason "combination offers are not supported yet". It used to say the view
had no user file, even when the set's user file was open. The render mode
(`--render`) still draws a combination contour.
