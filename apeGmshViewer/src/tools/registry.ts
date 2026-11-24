// The tools the app can open, by id (ADR 0137 H1). Main opens a tool window only for an id
// registered here, and the tool window mounts the definition. A definition touches the DOM only
// when it is mounted, so main and the model window may import this for the ids and titles.

import type { ToolDefinition } from "./contract.ts";
import { historyTool } from "./history.ts";
import { stepTool } from "./step.ts";

export const TOOLS: Readonly<Record<string, ToolDefinition>> = { [stepTool.id]: stepTool, [historyTool.id]: historyTool };
