### CHANGED — apeGmshViewer draws its x-y plots with one shared chart (ADR 0137 H2)

The Hinges card's moment-rotation plot and the material tester's two plots are now drawn by one
shared chart on a canvas, which the tool windows will use too. It looks as before. Each curve is
reduced to the canvas's pixel columns along its path before it is drawn, keeping every column's
lowest and highest point and every turn of a hysteresis loop, so a long run draws as quickly as a
short one.
