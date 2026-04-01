# Entity Normalization (LOCKED)

## Status: LOCKED

Normalization is identical in Python and Rust. Do NOT modify one without updating the other. 118 golden vectors in `tests/cross_lang/` verify cross-language consistency.

## Pipeline

Every entity ID goes through this exact pipeline:

1. **NFKD** Unicode normalization (compatibility decomposition)
2. **Strip Unicode CF characters** (format characters, zero-width joiners, etc.)
3. **Lowercase** the entire string
4. **Collapse whitespace** (multiple spaces/tabs → single space, trim)
5. **Greek letters** spelled out (α → alpha, β → beta, γ → gamma, etc.)

## Examples

| Input | Normalized |
|-------|-----------|
| `"  TP53  "` | `"tp53"` |
| `"NF-κB"` | `"nf-kappab"` |
| `"IL\u200B6"` | `"il6"` (zero-width space stripped) |

## Rules

1. **Never modify normalization in Python without updating Rust** (or vice versa). Run `tests/cross_lang/` after any change.
2. **Always use the normalization function** — don't manually lowercase or strip. The pipeline order matters.
3. **Greek letter expansion is part of normalization**, not a separate step. `α` becomes `alpha` everywhere.

## Files

- Python: `attestdb/core/normalization.py`
- Rust: `rust/attest-core/src/normalization.rs`
- Golden vectors: `tests/cross_lang/`
