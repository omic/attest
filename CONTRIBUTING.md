# Contributing to Attest

Thank you for your interest in contributing to Attest.

## Getting Started

```bash
git clone https://github.com/omic/attest.git
cd attest
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,mcp]"

# Build Rust extension
cd rust/attest-py && maturin develop --release && cd ../..

# Run tests
pytest tests/unit/ tests/integration/ -v
cd rust && cargo test --all
```

## What to Contribute

We welcome contributions in these areas:

- **Bug fixes** with a test that reproduces the issue
- **Performance improvements** with benchmarks showing the improvement
- **Documentation** improvements and corrections
- **Test coverage** for untested code paths

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes, ensuring all tests pass
3. Run `ruff check attestdb/` and fix any lint issues
4. Submit a PR with a clear description of the change

### PR Requirements

- All existing tests must pass (`pytest tests/unit/ tests/integration/`)
- Rust tests must pass (`cd rust && cargo test --all`)
- No new lint errors (`ruff check attestdb/`)
- New features or bug fixes should include tests

## Architecture

Attest is a claim-native database. Before contributing, read:

- `docs/02_architecture.md` -- Full technical architecture
- `docs/07_design_decisions.md` -- Critical invariants (normalization, hashing, provenance)

**Key invariants that must never change:**
- Entity normalization algorithm (Python and Rust must produce identical results)
- Claim ID and Content ID computation (SHA-256 based)
- The 13 validation rules on every write
- Append-only claim log (claims are never mutated)

## Code Style

- Python: follow `ruff` defaults (100 char line length)
- Rust: follow `cargo fmt` defaults
- No filler tests -- every test must verify real behavior
- Mock only external boundaries (HTTP APIs, LLM calls), never internal components

## Intelligence Layer

The `intelligence/` and `connectors/` modules are not part of the open-source engine. They are available in `attestdb-enterprise`. If your contribution requires intelligence features, discuss in an issue first.

## License

By contributing, you agree that your contributions will be licensed under the [Business Source License 1.1](LICENSE).
