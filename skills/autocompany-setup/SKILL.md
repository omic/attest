---
name: autocompany-setup
description: Set up or reconnect a company's selected Fabric reference implementations using the user's existing agent client and business tools. Use for AutoCompany onboarding and connection readiness, rather than ordinary work inside an already configured Fabric.
---

# AutoCompany setup

Help a person put their company on Fabric through their existing Dot, Claude Code,
Codex or other capable agent client. The client is the operator; each Fabric owns
its domain workflows and approval gates. Do not introduce a new model-provider loop.

## Start with the company, not infrastructure

Ask for the company goal, tools already in use, and the first workflow to automate
in one short exchange. Infer answers already available in the conversation. Offer
an initial choice of engineering, security/compliance, or sales/marketing; memory
supports each. Start with one workflow rather than requiring every department.

Read only the relevant entry in [fabric-catalog.md](references/fabric-catalog.md).
Explain which implementation is runnable, a pilot, or proposed. Reuse existing
services and integrations before creating new ones. Do not invent endpoints,
repositories, installation commands or Dot skill directories.

## Create a resumable workspace

With the user-selected location and company name, run the bundled
`scripts/prepare_workspace.py` to create `company.json`. It records goals, selected
Fabrics, owners, references to connections and bounded automation policy. It never
stores credentials. Existing files are preserved; rerunning with changed inputs
requires reviewing the difference rather than silently overwriting the workspace.

## Connect through the current client

Discover the current client's actual tools and MCP integrations. A CLI being
installed, a config entry existing or a standalone transport test does not prove
that this session has a working connection. Use the client's supported integration
flow and existing business account. Keep credentials in its secret store or a
private credential file, never in chat, Git or `company.json`.

For each selected workflow, identify the minimum real source access needed. Guide
sign-in only for missing connections. Verify service reachability, the intended
account/project, required tool discovery and a successful scoped read. Record the
observation and time in the private company workspace. If access cannot be checked,
report `not_checked` or `blocked`, with a concrete next action. Never call it empty,
connected or healthy without evidence. Preserve other client settings.

## Make the first useful artifact

Engineering: turn the first monthly/quarterly goal into a milestone and a proposed
feature/issue plan with acceptance checks. Security/compliance: identify the first
obligation/evidence gap. Sales/marketing: prepare the first sourced account brief
and draft. Use the selected service's actual tools, not fabricated local stand-ins.
If only an offline reference is available, demonstrate it with clearly synthetic
data and identify the live implementation still needed.

Show the artifact before requesting any approval required by that service. Respect
existing authorization; do not add repeated confirmations to reversible setup work.
Human approval cannot be supplied by an agent credential. Starting a scheduler,
sending messages, changing production or spending model credits requires applicable
user authorization and configured limits. Default new automation to paused and a
small explicit invocation budget; do not infer unlimited subscription capacity.

## Finish with a usable handoff

Show the first artifact, the connection status of each selected Fabric, remaining
blocks, the next scheduled or manual action, and the command/prompt to resume.
Distinguish `workspace_created`, `connected`, `workflow_verified` and
`automation_enabled`. Only report the latter states after observing their actual
prerequisites. A generated skill or manifest is not a running company.
