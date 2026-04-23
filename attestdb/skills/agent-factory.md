# Agent Factory: Auto-Discovery and Building of Trusted Agents

## Quick Start

```python
from attestdb import AttestDB

db = AttestDB("org.db")

# Full pipeline: discover workflows → generate specs → build evals → assemble agents
agents = db.run_agent_factory(min_frequency=3, max_agents=10)

# Or step by step:
workflows = db.discover_workflows(min_frequency=3)
spec = db.generate_agent_spec(workflows[0])
eval_set = db.build_agent_eval(spec, n=30)
agent = db.assemble_agent(spec, eval_set)
report = db.validate_agent_trust(agent.agent_id)
```

## Pipeline Stages

### 1. Workflow Discovery (`discover_workflows`)

Mines the claim graph for recurring patterns:
- **Predicate chains**: 2-hop and 3-hop patterns (e.g. `authored → merged → deployed`)
- **Entity type co-occurrence**: which entity types appear together repeatedly
- **Source clustering**: which connectors contribute to the same workflows

Key parameters:
- `min_frequency`: minimum occurrences to qualify as a workflow (default 3)
- `max_workflows`: cap on returned workflows (default 50)
- `source_filter`: restrict to specific connectors (e.g. `["github", "slack"]`)

### 2. Spec Generation (`generate_agent_spec`)

Converts a discovered workflow into an `AgentSpec`:
- Maps predicates to capabilities (`assigned_to` → `task_routing`, `reviewed` → `review_management`)
- Maps connectors to capabilities (`github` → `code_management`, `slack` → `messaging`)
- Builds I/O schemas from entity types
- Sets trust requirements (min confidence, min sources, eval pass threshold)
- Collects grounding claim IDs for provenance

### 3. Eval Building (`build_agent_eval`)

Generates domain-specific eval sets from real organizational data:
- **Workflow chain questions** (40%): test understanding of the predicate chain
- **Entity knowledge questions** (30%): test entity type awareness
- **Provenance questions** (30%): test source diversity awareness

### 4. Agent Assembly (`assemble_agent`)

Registers the agent in the existing `AgentRegistry` and persists assembly metadata:
- Agent ID format: `factory:{spec_id}`
- Links spec, eval, and registration in the claim graph
- Status: `assembled` → `registered` → `validated` → `degraded`

### 5. Trust Validation (`validate_agent_trust`)

Continuous trustworthiness checking:
- **Eval drift**: has performance degraded from baseline?
- **Data freshness**: are grounding claims still current?
- **Grounding health**: do original claims still hold?
- Returns status: `healthy`, `drifting`, `degraded`, or `stale`

## MCP Tools

| Tool | Purpose |
|------|---------|
| `factory_discover_workflows` | Mine claim graph for workflow patterns |
| `factory_generate_spec` | Generate agent spec from a workflow |
| `factory_build_eval` | Build eval set for an agent spec |
| `factory_assemble_agent` | Register and persist an assembled agent |
| `factory_validate_trust` | Check agent trustworthiness |
| `factory_run_pipeline` | End-to-end: discover → spec → eval → assemble |
| `factory_list_workflows` | List all discovered workflows |
| `factory_list_agents` | List all factory-assembled agents |

## Claim Storage

All state is claim-native:

| Predicate | Subject | Object | Purpose |
|-----------|---------|--------|---------|
| `has_workflow` | workflow_id | workflow_pattern | Discovered workflow |
| `has_spec` | spec_id | workflow_id | Agent specification |
| `assembled_from` | agent_id | spec_id | Assembly record |
| `trust_validated` | agent_id | trust_status | Trust validation |

## Gotchas

- Discovery requires data from connectors first — run connectors before discovering workflows
- `min_frequency=1` will return many noisy patterns; start with 3+ for meaningful workflows
- Predicate chain extraction scans up to 20K claims; on very large DBs the cap limits discovery scope
- Trust validation checks freshness against `created_at`; agents assembled long ago will show as `stale`
- The `source_filter` in `factory_discover_workflows` limits which claims are analyzed, not which workflows are returned
