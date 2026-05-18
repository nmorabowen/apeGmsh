/* ParaView Flows viewer — loads flows.json, renders a static package map in Cytoscape,
   and animates edge traversal for whichever action the user clicks. */

const PACKAGES = [
  // Manual positions — left-to-right layered diagram (data → UI → app).
  // Top row: scripting / plugins. Middle row: core layers. Bottom: clients.
  { id: "vtk",        label: "VTK",                          group: "data",   x:  100, y: 320 },
  { id: "remcore",    label: "Remoting/Core",                group: "server", x:  300, y: 480 },
  { id: "remsm",      label: "Remoting/ServerManager",       group: "server", x:  300, y: 320 },
  { id: "remviews",   label: "Remoting/Views",               group: "server", x:  300, y: 160 },
  { id: "qtcore",     label: "Qt/Core",                      group: "qt",     x:  540, y: 320 },
  { id: "qtcomp",     label: "Qt/Components",                group: "qt",     x:  760, y: 320 },
  { id: "qtapp",      label: "Qt/AppComponents",             group: "qt",     x:  980, y: 320 },
  { id: "clients",    label: "Clients/ParaView",             group: "app",    x: 1180, y: 320 },
  { id: "plugins",    label: "Plugins",                      group: "app",    x:  760, y: 120 },
  { id: "wrap",       label: "Wrapping/Python",              group: "app",    x:  760, y: 520 },
];

const GROUP_COLORS = {
  data:   "#8fbcbb",
  server: "#d08770",
  qt:     "#b48ead",
  app:    "#a3be8c",
};

let cy = null;
let flows = [];
let activeFlow = null;
let animationTimer = null;

async function init() {
  const resp = await fetch("flows.json");
  flows = await resp.json();
  buildGraph();
  buildActionList();
  if (flows.length > 0) selectFlow(flows[0].id);
}

function buildGraph() {
  cy = cytoscape({
    container: document.getElementById("cy"),
    elements: PACKAGES.map(pkg => ({
      data: { id: pkg.id, label: pkg.label, group: pkg.group },
      position: { x: pkg.x, y: pkg.y },
    })),
    style: [
      {
        selector: "node",
        style: {
          "background-color": ele => GROUP_COLORS[ele.data("group")] || "#888",
          "label": "data(label)",
          "color": "#0f1419",
          "text-valign": "center",
          "text-halign": "center",
          "font-size": 11,
          "font-weight": 600,
          "shape": "round-rectangle",
          "width": 150,
          "height": 44,
          "border-width": 2,
          "border-color": "#2a323d",
          "text-wrap": "wrap",
          "text-max-width": 140,
          "transition-property": "background-color, border-color, border-width",
          "transition-duration": "150ms",
        },
      },
      {
        selector: "node.dim",
        style: { "opacity": 0.25 },
      },
      {
        selector: "node.involved",
        style: {
          "border-color": "#7eb3ff",
          "border-width": 3,
        },
      },
      {
        selector: "node.hot",
        style: {
          "border-color": "#ffb454",
          "border-width": 4,
          "background-blacken": -0.15,
        },
      },
      {
        selector: "edge",
        style: {
          "width": 2,
          "line-color": "#3a6aaa",
          "target-arrow-color": "#3a6aaa",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "control-point-step-size": 60,
          "opacity": 0.45,
          "label": "data(step)",
          "font-size": 10,
          "color": "#7eb3ff",
          "text-background-color": "#0f1419",
          "text-background-opacity": 0.9,
          "text-background-padding": 2,
          "transition-property": "line-color, target-arrow-color, width, opacity",
          "transition-duration": "150ms",
        },
      },
      {
        selector: "edge.hot",
        style: {
          "line-color": "#ffb454",
          "target-arrow-color": "#ffb454",
          "width": 4,
          "opacity": 1.0,
        },
      },
    ],
    layout: { name: "preset" },
    minZoom: 0.4,
    maxZoom: 2.0,
    wheelSensitivity: 0.2,
    autoungrabify: true,
  });

  // Fit on load, then constrain panning so the graph doesn't fly off-screen.
  cy.fit(undefined, 40);
}

function buildActionList() {
  const ul = document.getElementById("action-list");
  ul.innerHTML = "";
  flows.forEach(flow => {
    const li = document.createElement("li");
    li.dataset.id = flow.id;
    li.innerHTML = `
      <span class="action-title">${escapeHtml(flow.title)}</span>
      <span class="action-grounds">${escapeHtml(flow.grounds || "")}</span>
    `;
    li.addEventListener("click", () => selectFlow(flow.id));
    ul.appendChild(li);
  });
}

function selectFlow(flowId) {
  const flow = flows.find(f => f.id === flowId);
  if (!flow) return;
  activeFlow = flow;

  // Update sidebar active state.
  document.querySelectorAll("#action-list li").forEach(li => {
    li.classList.toggle("active", li.dataset.id === flowId);
  });

  // Update right panel.
  document.getElementById("flow-title").textContent = flow.title;
  document.getElementById("flow-grounds").textContent = flow.grounds
    ? `→ ${flow.grounds}`
    : "";
  document.getElementById("flow-summary").textContent = flow.summary || "";

  const stepsList = document.getElementById("flow-steps");
  stepsList.innerHTML = "";
  flow.steps.forEach((step, idx) => {
    const li = document.createElement("li");
    li.dataset.idx = idx;
    li.innerHTML = `
      <div class="step-edge">
        <code>${escapeHtml(step.from)}</code>
        <span class="arrow">→</span>
        <code>${escapeHtml(step.to)}</code>
      </div>
      <div class="step-call">${escapeHtml(step.call)}</div>
      ${step.passes ? `<div class="step-passes"><b>passes</b> ${escapeHtml(step.passes)}</div>` : ""}
      ${step.file ? `<div class="step-file">${escapeHtml(step.file)}</div>` : ""}
      ${step.note ? `<div class="step-note">${escapeHtml(step.note)}</div>` : ""}
    `;
    li.addEventListener("click", () => highlightStep(idx));
    stepsList.appendChild(li);
  });

  renderFlowOnGraph(flow);
  animateFlow(flow);
}

function renderFlowOnGraph(flow) {
  // Remove all previous flow edges.
  cy.edges().remove();
  cy.nodes().removeClass("involved hot dim");

  const involved = new Set();
  flow.steps.forEach(s => { involved.add(s.from); involved.add(s.to); });

  cy.nodes().forEach(n => {
    if (involved.has(n.id())) n.addClass("involved");
    else n.addClass("dim");
  });

  flow.steps.forEach((step, idx) => {
    cy.add({
      group: "edges",
      data: {
        id: `e${idx}`,
        source: step.from,
        target: step.to,
        step: String(idx + 1),
        stepIdx: idx,
      },
    });
  });
}

function animateFlow(flow) {
  if (animationTimer) { clearInterval(animationTimer); animationTimer = null; }
  cy.edges().removeClass("hot");
  cy.nodes().removeClass("hot");

  let i = 0;
  const tick = () => {
    cy.edges().removeClass("hot");
    cy.nodes().removeClass("hot");
    if (i >= flow.steps.length) {
      clearInterval(animationTimer);
      animationTimer = null;
      return;
    }
    const step = flow.steps[i];
    cy.getElementById(`e${i}`).addClass("hot");
    cy.getElementById(step.from).addClass("hot");
    cy.getElementById(step.to).addClass("hot");
    highlightStepRow(i);
    i += 1;
  };
  tick();
  animationTimer = setInterval(tick, 900);
}

function highlightStep(idx) {
  if (!activeFlow) return;
  if (animationTimer) { clearInterval(animationTimer); animationTimer = null; }
  cy.edges().removeClass("hot");
  cy.nodes().removeClass("hot");

  const step = activeFlow.steps[idx];
  if (!step) return;
  cy.getElementById(`e${idx}`).addClass("hot");
  cy.getElementById(step.from).addClass("hot");
  cy.getElementById(step.to).addClass("hot");
  highlightStepRow(idx);
}

function highlightStepRow(idx) {
  document.querySelectorAll("#flow-steps li").forEach(li => {
    li.classList.toggle("highlighted", Number(li.dataset.idx) === idx);
  });
  const el = document.querySelector(`#flow-steps li[data-idx="${idx}"]`);
  if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

init();
