### ADDED — the Step readout tool charts the picked node's history (ADR 0137 H2)

With a node picked in the model window (the selection bar's Nodes), the Step readout tool window
charts that node's chosen nodal component over the results stage, with the step on screen marked.
Tool windows now ask the app for the data they draw through one broker: the app reads the open
results file for them, in its loader process, reads data asked for twice at once only once, and
answers only a tool's newest request. A tool never names a file.
