"""Entity quality filtering: distinguish real named entities from
placeholders, boilerplate, redactions, and OCR-mangled tokens that
litter many real-world corpora.

The mechanism is generic — exclude entities by name pattern (prefix,
substring, regex) or by composition (single-token gibberish,
all-generic-tokens). The *patterns* are domain-specific. A court-doc
corpus excludes "juror_26" and "08-cv-80736-kam"; a customer-support
corpus excludes "user_12345" and "ticket-#…"; a code corpus excludes
"<anonymous>" and "node_modules".

Use:
    from attestdb.core.entity_quality import EntityQualityFilter
    f = EntityQualityFilter.journalism_court_docs()
    if f.is_substantive(entity_summary):
        ...

Compose your own:
    f = EntityQualityFilter()
    f.add_noise_prefix("ticket-")
    f.add_noise_regex(r"^user_\\d+$")
    f.add_generic_token("ticket")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


# Universal generic tokens — words that, on their own or as the entire
# entity name, almost certainly mean a placeholder rather than a named
# entity. Domain filters can extend or override.
DEFAULT_GENERIC_TOKENS: set[str] = {
    "victim", "victims", "witness", "witnesses", "minor", "minors",
    "user", "users", "person", "people", "the", "a", "an",
    "anyone", "someone", "everyone", "no one",
    "unknown", "anonymous", "redacted", "n/a",
}


@dataclass
class EntityQualityFilter:
    """Configuration for "is this entity a real named thing?"
    All sets/lists default empty; build up via the add_* methods or
    use one of the @classmethod presets.
    """
    noise_prefixes: set[str] = field(default_factory=set)
    noise_substrings: set[str] = field(default_factory=set)
    noise_exact_matches: set[str] = field(default_factory=set)
    noise_regexes: list[re.Pattern] = field(default_factory=list)
    generic_tokens: set[str] = field(default_factory=lambda: set(DEFAULT_GENERIC_TOKENS))
    reject_single_token_gibberish: bool = True
    reject_underscore_synthetic: bool = True
    min_name_chars: int = 1
    max_name_chars: int = 200

    # ── Mutators ─────────────────────────────────────────────────────

    def add_noise_prefix(self, prefix: str) -> "EntityQualityFilter":
        self.noise_prefixes.add(prefix.lower())
        return self

    def add_noise_prefixes(self, prefixes: Iterable[str]) -> "EntityQualityFilter":
        for p in prefixes:
            self.add_noise_prefix(p)
        return self

    def add_noise_substring(self, sub: str) -> "EntityQualityFilter":
        self.noise_substrings.add(sub.lower())
        return self

    def add_noise_exact(self, name: str) -> "EntityQualityFilter":
        self.noise_exact_matches.add(name.lower())
        return self

    def add_noise_exacts(self, names: Iterable[str]) -> "EntityQualityFilter":
        for n in names:
            self.add_noise_exact(n)
        return self

    def add_noise_regex(self, pattern: str) -> "EntityQualityFilter":
        self.noise_regexes.append(re.compile(pattern, re.IGNORECASE))
        return self

    def add_generic_token(self, token: str) -> "EntityQualityFilter":
        self.generic_tokens.add(token.lower())
        return self

    def add_generic_tokens(self, tokens: Iterable[str]) -> "EntityQualityFilter":
        for t in tokens:
            self.add_generic_token(t)
        return self

    # ── Predicate ────────────────────────────────────────────────────

    def is_substantive(self, entity_or_name) -> bool:
        """Accept either an EntitySummary-like object (with .name and .id)
        or a raw string. Returns True if the entity passes all filters.
        """
        if entity_or_name is None:
            return False
        if isinstance(entity_or_name, str):
            name = entity_or_name
        else:
            name = (getattr(entity_or_name, "name", None)
                    or getattr(entity_or_name, "id", None)
                    or "")
        nm = (name or "").strip().lower()
        if not nm or len(nm) < self.min_name_chars or len(nm) > self.max_name_chars:
            return False
        if nm in self.noise_exact_matches:
            return False
        for prefix in self.noise_prefixes:
            if nm.startswith(prefix):
                return False
        for sub in self.noise_substrings:
            if sub in nm:
                return False
        for rx in self.noise_regexes:
            if rx.search(nm):
                return False
        # Single-token gibberish: short string with no spaces that mixes
        # letters and digits (e.g. "n4", "g-yab", "n404se", "21cv5"). Pure
        # alphabetic short tokens like "Bob" or "Eve" are real names and
        # must be kept.
        if self.reject_single_token_gibberish and " " not in nm:
            stripped = nm.replace("-", "").replace("_", "")
            has_alpha = any(c.isalpha() for c in stripped)
            has_digit = any(c.isdigit() for c in stripped)
            if len(nm) <= 6 and has_alpha and has_digit:
                return False
            # Single hyphenated short token (no digits): "g-yab", "x-y"
            if "-" in nm and len(nm) <= 6:
                return False
        # Synthetic ids: short_underscore_joined tokens like "epstein_planes",
        # "user_session_42". Doesn't reject longer multi-word names that
        # happen to use underscores (rare in real names).
        if self.reject_underscore_synthetic and "_" in nm and len(nm.split("_")) >= 2 and len(nm) < 25:
            return False
        # All tokens are generic → not a real entity.
        toks = set(nm.split())
        if toks and toks <= self.generic_tokens:
            return False
        return True

    # ── Presets ──────────────────────────────────────────────────────

    @classmethod
    def journalism_court_docs(cls) -> "EntityQualityFilter":
        """Filter tuned for U.S. federal-court document corpora: jurors,
        case captions, redacted ids, victim pseudonyms, OCR mush.
        """
        f = cls()
        f.add_noise_prefixes([
            "juror_", "juror ", "exhibit_", "exhibit ", "table of",
            "redacted", "user_", "doc_",
            "judge ", "agent ", "officer ", "marshal ", "detective ",
            "minor victim", "victim-",
            "doe ", "doe-", "jane doe", "john doe",
        ])
        # Exact-match pseudonyms used in court docs instead of real names
        f.add_noise_exacts([
            "carolyn", "kate", "annie", "jane", "jane doe", "john doe",
            "predator", "stepbrother", "stepfather", "stepmother", "uncle",
            "abuser", "trafficker",
            "individual i", "individual ii", "individual iii",
            "individual iv", "individual v",
            "individual a", "individual b", "individual c",
            "co-conspirator", "co-conspirators",
            "the defendant", "defendant", "the plaintiff", "plaintiff",
        ])
        # Docket numbers like "08-cv-80736-kam" or "21:cr:00100-mkv"
        f.add_noise_regex(r"^\d+([-:]\w+){2,}$")
        # Single-letter aliases or anonymized witness ids ("n4", "w1")
        f.add_noise_regex(r"^[a-z]\d+[a-z]?$")
        # Generic tokens specific to litigation
        f.add_generic_tokens({
            "girls", "girl", "women", "woman", "men", "man",
            "youths", "juror", "jurors", "government",
            "defendants", "plaintiffs", "attorneys", "counsel", "experts",
            "reporter", "reporters", "the witness", "the victim",
            "young women", "young girls",
            "accuser", "accusers",
        })
        return f

    @classmethod
    def permissive(cls) -> "EntityQualityFilter":
        """Minimal filter — only rejects empty/None and pure whitespace.
        Useful when the corpus is already curated."""
        f = cls(generic_tokens=set())
        f.reject_single_token_gibberish = False
        f.reject_underscore_synthetic = False
        return f
