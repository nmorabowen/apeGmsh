### CHANGED — ADR 0128 Amendment 3: the agent view shim is `apeGmsh.agent_view`, outside `viewers` (#2726)

The agent-view shim lives at `src/apeGmsh/agent_view.py`, so its public name is
`apeGmsh.agent_view`. The maintainer moved the module rather than raise the
`viewers` line-count ratchet (ruling of 2026-11-02 on #2713). ADR 0128 D7 is read
through the new amendment; its original text is unchanged.
