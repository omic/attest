"""Entity normalization — locked, never changes.

Deterministic normalization function identical in Phase 1 (Python) and Phase 2 (Rust).
"""

import unicodedata

GREEK_MAP: dict[str, str] = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "τ": "tau",
    "ω": "omega",
}


def normalize_entity_id(raw: str) -> str:
    """Deterministic entity ID normalization. Once shipped, never changes."""
    # 1. Unicode normalize (NFKD decomposition)
    s = unicodedata.normalize("NFKD", raw)
    # 2. Strip zero-width / invisible format characters (Unicode category Cf)
    s = "".join(c for c in s if unicodedata.category(c) != "Cf")
    # 3. Lowercase
    s = s.lower()
    # 4. Collapse whitespace
    s = " ".join(s.split())
    # 5. Strip leading/trailing whitespace
    s = s.strip()
    # 6. Replace common Greek letters with spelled-out forms
    for char, name in GREEK_MAP.items():
        s = s.replace(char, name)
    return s
