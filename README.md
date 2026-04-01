# AttestDB

Claim-native knowledge graph for AI agents. Every fact has a source, a confidence score, and gets flagged when the source is wrong.

[![PyPI](https://img.shields.io/pypi/v/attestdb)](https://pypi.org/project/attestdb/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: BSL-1.1](https://img.shields.io/badge/license-BSL--1.1-green)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/omic/attest)](https://github.com/omic/attest)

## 60-Second Quickstart

```bash
pip install attestdb
attestdb quickstart
```

## Why Not a Vector DB?

Vector DBs find similar text. AttestDB tracks whether claims are true -- with sources, confidence scores, and automatic retraction cascades when a source is wrong. Think of it as git for knowledge: every fact has provenance, and contradictions are surfaced, not hidden.

## Features

- **85M+ claims in production**, microsecond queries (~12µs indexed lookups)
- **30 connectors** (Slack, GitHub, Gmail, Postgres, etc.)
- **106 MCP tools** -- give your AI agent persistent memory
- **Rust engine (LMDB)** -- 1.3M claims/sec insert
- **Automatic retraction cascades** when sources are wrong
- **Gap detection** -- finds what your knowledge graph is missing

## Give Your AI Agent a Brain

```bash
pip install attestdb
attestdb mcp-config
# Restart Claude Code -- your agent now remembers across sessions
```

## 7-Day Pro Trial

```bash
attestdb trial start    # No credit card required
```

Pro plan includes: 1M cloud API tokens, hosted queries, team features.

## Pricing

| | Free | Pro ($99/mo) | Growth ($249/mo) | Team ($499/mo) | Enterprise |
|---|---|---|---|---|---|
| Local DB | Unlimited | Unlimited | Unlimited | Unlimited | Unlimited |
| Cloud tokens | -- | 1M/mo | 5M/mo | 10M/mo | Custom |
| Connectors | All 30 | All 30 | All 30 | All 30 | All 30 |
| MCP tools | 106 | 106 | 106 | 106 | 106 |
| Team features | -- | -- | -- | Yes | Yes |
| SOC 2 | -- | -- | -- | -- | In progress |

## Links

- [Documentation](https://attestdb.com/quickstart.html)
- [Live Demo](https://attestdb.com/demo/)
- [Pricing](https://attestdb.com/pricing.html)
- [GitHub](https://github.com/omic/attest)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

BSL-1.1 -- free for non-production use. Production use requires a license. See [LICENSE](LICENSE) for details.
