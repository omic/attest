# Release notes

## 0.2.2

- Organize the application roadmap as Fabric reference implementations with explicit readiness labels.
- Add synthetic, offline Software Engineering and Sales and Marketing examples with real database tests and human-review gates.
- Add the portable AutoCompany setup skill, a private workspace preparation script, and guided onboarding documentation. Setup starts paused; it does not provision a running company.
- Store and recall Brain memories with long subjects safely within LMDB key limits, and match subjects with surrounding spaces consistently.

The examples and setup kit make no live model calls. Separate private applications, enterprise features, models, training data, and private connectors are not included in this update.

## 0.2.1

- Restore five local brain memory tools, test real MCP client sessions across server restarts, and preserve client settings during install/uninstall.
- Configure Codex through TOML, remove its obsolete Attest notify hook, and stop installing unavailable lifecycle hooks.
- Keep MCP startup local rather than enabling paid embeddings merely because an API key is present.
- Add a synthetic compliance evidence example, application architecture diagrams, and a readiness-labelled reference roadmap.
- Avoid rewriting unchanged embedding indexes; persist caller-supplied embedding model labels.
- Mark canonical `interacts_with` relationships as symmetric, including claims ingested using `interacts`.
- Declare the HTTP and fuzzy-matching dependencies used by the public API.
- Correct package versions and Apache-2.0 metadata; include the license in native distributions.
- Provide a native source distribution for Python environments without a matching wheel.
- Remove calibration hooks that depended on unavailable private modules, their integration tests, and unused evaluation generators.
- Run standalone tests and artifact checks privately; remove automatic public CI and publication.

The release starts from the public 0.2.0 tree. It contains no newly exported private implementation. Optional integrations already present in that public tree may still require separately installed components.
