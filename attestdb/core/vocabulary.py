"""Built-in vocabulary constants and special predicate sets."""

BUILT_IN_ENTITY_TYPES: set[str] = {
    "entity", "event", "metric", "document", "agent", "system",
}

BUILT_IN_PREDICATE_TYPES: set[str] = {
    "relates_to", "caused", "observed", "derived_from", "contradicts",
    "same_as", "not_same_as", "retracted", "contradiction_resolved", "inquiry",
    "no_evidence_for", "superseded_by",
}

# Research predicates — used by multi-agent collaborative research
RESEARCH_PREDICATES: set[str] = {
    "has_finding", "has_gap", "investigating",
    "no_evidence_for", "superseded_by", "has_strategy",
}

# Opposing research predicates — evidence vs absence
OPPOSITE_RESEARCH_PREDICATES: dict[str, str] = {
    "no_evidence_for": "supports",
    "supports": "no_evidence_for",
}

BUILT_IN_SOURCE_TYPES: set[str] = {
    "observation", "computation", "document_extraction",
    "llm_inference", "human_annotation", "chat_extraction",
    "agent",
}

# Predicates with special engine semantics
CONTRADICTION_PREDICATES: set[str] = {"contradicts", "contradiction_resolved"}
SYMMETRIC_PREDICATES: set[str] = {"interacts", "binds", "resembles", "homolog_of", "same_as"}

# Opposing predicate pairs — used for contradiction detection
OPPOSITE_PREDICATES: dict[str, str] = {
    "activates": "inhibits",
    "inhibits": "activates",
    "upregulates": "downregulates",
    "downregulates": "upregulates",
    "promotes": "suppresses",
    "suppresses": "promotes",
    "increases": "decreases",
    "decreases": "increases",
    "causes": "prevents",
    "prevents": "causes",
    "enables": "blocks",
    "blocks": "enables",
    "enhances": "reduces",
    "reduces": "enhances",
    "stabilizes": "destabilizes",
    "destabilizes": "stabilizes",
    "phosphorylates": "dephosphorylates",
    "dephosphorylates": "phosphorylates",
    "methylates": "demethylates",
    "demethylates": "methylates",
}

# Predicate composition rules — (pred_A_to_C, pred_C_to_B) → predicted_A_to_B
# Causal logic: inhibits + promotes → inhibits,
# inhibits + inhibits → activates (double negative), etc.
PREDICATE_COMPOSITION: dict[tuple[str, str], str] = {
    # activates chains
    ("activates", "activates"): "activates",
    ("activates", "inhibits"): "inhibits",
    ("activates", "promotes"): "promotes",
    ("activates", "downregulates"): "downregulates",
    ("activates", "upregulates"): "upregulates",
    # inhibits chains (double negative)
    ("inhibits", "activates"): "inhibits",
    ("inhibits", "inhibits"): "activates",
    ("inhibits", "promotes"): "inhibits",
    ("inhibits", "downregulates"): "upregulates",
    ("inhibits", "upregulates"): "downregulates",
    # promotes chains
    ("promotes", "activates"): "promotes",
    ("promotes", "inhibits"): "suppresses",
    ("promotes", "promotes"): "promotes",
    ("promotes", "suppresses"): "suppresses",
    # suppresses chains (double negative)
    ("suppresses", "activates"): "suppresses",
    ("suppresses", "inhibits"): "promotes",
    ("suppresses", "promotes"): "suppresses",
    ("suppresses", "suppresses"): "promotes",
    # upregulates / downregulates
    ("upregulates", "activates"): "upregulates",
    ("upregulates", "inhibits"): "downregulates",
    ("upregulates", "upregulates"): "upregulates",
    ("upregulates", "downregulates"): "downregulates",
    ("downregulates", "activates"): "downregulates",
    ("downregulates", "inhibits"): "upregulates",
    ("downregulates", "downregulates"): "upregulates",
    ("downregulates", "upregulates"): "downregulates",
    # causes / prevents chains
    ("causes", "activates"): "activates",
    ("causes", "inhibits"): "inhibits",
    ("causes", "causes"): "causes",
    ("causes", "prevents"): "prevents",
    ("prevents", "activates"): "inhibits",
    ("prevents", "inhibits"): "activates",
    ("prevents", "causes"): "prevents",
    ("prevents", "prevents"): "causes",
    # increases / decreases (full cross with up/downregulates)
    ("increases", "activates"): "activates",
    ("increases", "inhibits"): "inhibits",
    ("increases", "upregulates"): "upregulates",
    ("increases", "downregulates"): "downregulates",
    ("decreases", "activates"): "inhibits",
    ("decreases", "inhibits"): "activates",
    ("decreases", "upregulates"): "downregulates",
    ("decreases", "downregulates"): "upregulates",
    # enables / blocks (full cross)
    ("enables", "activates"): "activates",
    ("enables", "inhibits"): "inhibits",
    ("enables", "upregulates"): "upregulates",
    ("enables", "downregulates"): "downregulates",
    ("blocks", "activates"): "inhibits",
    ("blocks", "inhibits"): "activates",
    ("blocks", "upregulates"): "downregulates",
    ("blocks", "downregulates"): "upregulates",
    # promotes / suppresses cross with up/downregulates
    ("promotes", "upregulates"): "upregulates",
    ("promotes", "downregulates"): "downregulates",
    ("suppresses", "upregulates"): "downregulates",
    ("suppresses", "downregulates"): "upregulates",
    # stabilizes / destabilizes
    ("stabilizes", "activates"): "activates",
    ("stabilizes", "inhibits"): "inhibits",
    ("destabilizes", "activates"): "inhibits",
    ("destabilizes", "inhibits"): "activates",
    # regulates — non-directional, preserves the other predicate's direction
    ("regulates", "activates"): "regulates",
    ("regulates", "inhibits"): "regulates",
    ("regulates", "upregulates"): "regulates",
    ("regulates", "downregulates"): "regulates",
    ("regulates", "regulates"): "regulates",
    ("activates", "regulates"): "regulates",
    ("inhibits", "regulates"): "regulates",
    ("upregulates", "regulates"): "regulates",
    ("downregulates", "regulates"): "regulates",
}

# Predicates that don't compose meaningfully — produce "associated_with" in compositions
_WEAK_PREDICATES: set[str] = {
    "associated_with", "relates_to", "resembles", "interacts_with",
    "interacts", "coexpressed_with", "same_as", "associates",
    "participates_in", "investigated_in", "expresses",
}

# Causal predicates — directional, compose via PREDICATE_COMPOSITION rules
CAUSAL_PREDICATES: set[str] = {
    "activates", "inhibits", "upregulates", "downregulates",
    "promotes", "suppresses", "increases", "decreases",
    "causes", "prevents", "enables", "blocks",
    "enhances", "reduces", "stabilizes", "destabilizes",
    "phosphorylates", "dephosphorylates", "methylates", "demethylates",
    "regulates",  # non-directional but still indicates regulatory relationship
}

# Predicate equivalence — semantically close predicates that should match
# when comparing predictions against expected outcomes.
# "upregulates" and "activates" both mean positive regulation;
# "downregulates" and "inhibits" both mean negative regulation.
PREDICATE_EQUIVALENCE: dict[str, str] = {
    "activates": "positive",
    "upregulates": "positive",
    "promotes": "positive",
    "increases": "positive",
    "enables": "positive",
    "enhances": "positive",
    "stabilizes": "positive",
    "inhibits": "negative",
    "downregulates": "negative",
    "suppresses": "negative",
    "decreases": "negative",
    "blocks": "negative",
    "reduces": "negative",
    "destabilizes": "negative",
    "prevents": "negative",
}


def predicates_agree(pred_a: str, pred_b: str) -> bool:
    """Check if two predicates are semantically equivalent (same direction).

    Returns True if both are positive regulation or both are negative.
    Returns False if they oppose or either is unknown.
    """
    class_a = PREDICATE_EQUIVALENCE.get(pred_a)
    class_b = PREDICATE_EQUIVALENCE.get(pred_b)
    if class_a and class_b:
        return class_a == class_b
    return pred_a == pred_b


def directional_confidence(
    supporting: int,
    opposing: int,
    min_evidence: int = 3,
    source_weights: list[float] | None = None,
) -> tuple[float, str]:
    """Score how much to trust a directional prediction.

    Based on evidence count, agreement ratio, and source reliability.
    NLP-extracted predications have ~30% directional error rate per claim —
    with 1-2 claims, errors dominate. With 10+, signal overwhelms noise.

    Args:
        supporting: Number of claims supporting this direction.
        opposing: Number of claims supporting the opposite direction.
        min_evidence: Minimum total claims before trusting any direction.
        source_weights: Optional reliability weights for each supporting
            claim (from confidence.source_reliability). If provided,
            weighted_supporting replaces raw count for agreement scoring.
            One curated claim (weight 1.0) counts more than one NLP claim
            (weight 0.5).

    Returns (confidence, verdict):
    - confidence: 0.0-1.0 score
    - verdict: "strong", "moderate", "weak", or "insufficient"

    Usage:
        conf, verdict = directional_confidence(supporting=124, opposing=10)
        # → (0.92, "strong") — 92% agreement, 134 total claims

        # With source weights: 5 curated claims > 10 NLP claims
        conf, verdict = directional_confidence(5, 1,
            source_weights=[1.0, 1.0, 0.95, 0.9, 0.85])
    """
    total = supporting + opposing
    if total < min_evidence:
        return (0.0, "insufficient")

    if source_weights:
        weighted_support = sum(source_weights)
        weighted_total = weighted_support + opposing * 0.5  # opposing claims get default weight
        agreement = weighted_support / weighted_total if weighted_total > 0 else 0
    else:
        agreement = supporting / total if total > 0 else 0
    # Evidence strength: log scale, saturates around 50 claims
    import math
    evidence_factor = min(1.0, math.log(total + 1) / math.log(50))
    confidence = agreement * evidence_factor

    if confidence >= 0.7 and total >= 10:
        verdict = "strong"
    elif confidence >= 0.5 and total >= min_evidence:
        verdict = "moderate"
    elif agreement > 0.5:
        verdict = "weak"
    else:
        verdict = "insufficient"

    return (round(confidence, 3), verdict)


def compose_predicates(pred_ac: str, pred_cb: str) -> str:
    """Compose two predicates along A→C→B into predicted A→B predicate.

    Lookup in composition table, fall back to same-predicate transitivity,
    then "associated_with".
    """
    if pred_ac in _WEAK_PREDICATES or pred_cb in _WEAK_PREDICATES:
        return "associated_with"
    result = PREDICATE_COMPOSITION.get((pred_ac, pred_cb))
    if result:
        return result
    # Same-predicate transitivity
    if pred_ac == pred_cb:
        return pred_ac
    return "associated_with"


def predict_predicate_from_paths(
    evidence_paths: list[dict],
) -> tuple[str, float]:
    """Weighted majority vote across bridging paths.

    Each path dict should have:
        - ac_predicates: set[str] — predicates from A to bridge C
        - cb_predicates: set[str] — predicates from bridge C to B
        - path_weight: float — weight (e.g. min(confidence_ac, confidence_cb))

    Returns (predicted_predicate, vote_fraction).
    """
    votes: dict[str, float] = {}
    total_weight = 0.0
    for path in evidence_paths:
        ac_preds = path.get("ac_predicates", set())
        cb_preds = path.get("cb_predicates", set())
        weight = path.get("path_weight", 1.0)
        for ac_p in ac_preds:
            for cb_p in cb_preds:
                composed = compose_predicates(ac_p, cb_p)
                votes[composed] = votes.get(composed, 0.0) + weight
                total_weight += weight
    if not votes:
        return ("associated_with", 0.0)
    best = max(votes, key=lambda k: votes[k])
    fraction = votes[best] / total_weight if total_weight > 0 else 0.0
    return (best, fraction)


# ---------------------------------------------------------------------------
# Predicate normalization — map LLM-generated variants to controlled vocabulary
# ---------------------------------------------------------------------------

# Standard predicates — the canonical set that AttestDB accepts.
# LLM-generated predicates are normalized to these at ingestion time.
STANDARD_PREDICATES: set[str] = {
    # Causal / regulatory
    "inhibits", "activates", "upregulates", "downregulates",
    "promotes", "suppresses", "increases", "decreases",
    "causes", "prevents", "enables", "blocks",
    "enhances", "reduces", "regulates",
    # Interaction
    "interacts_with", "interacts", "binds", "targets",
    # Association
    "associated_with", "correlates_with", "contributes_to",
    # Spatial / structural
    "expressed_in", "participates_in", "involved_in",
    # Therapeutic / clinical
    "treats", "biomarker_for",
    # Post-translational
    "phosphorylates", "dephosphorylates", "methylates", "demethylates",
    "stabilizes", "destabilizes",
    # Semantic
    "induces", "modulates", "mediates", "encodes",
}

# Map common LLM-generated predicate variants to standard forms.
# Covers: verb tense normalization, wordy phrasings, synonyms.
PREDICATE_ALIAS_MAP: dict[str, str] = {
    # Verb stem / tense variants
    "activate": "activates", "inhibit": "inhibits",
    "reduce": "reduces", "treat": "treats",
    "cause": "causes", "prevent": "prevents",
    "regulate": "regulates", "bind": "binds",
    "increase": "increases", "decrease": "decreases",
    "enhance": "enhances", "promote": "promotes",
    "suppress": "suppresses", "induce": "induces",
    "block": "blocks", "modulate": "modulates",
    "target": "targets", "mediate": "mediates",
    "detect": "detects", "predict": "predicts",
    "encode": "encodes", "upregulate": "upregulates",
    "downregulate": "downregulates", "stabilize": "stabilizes",
    "destabilize": "destabilizes",
    "phosphorylate": "phosphorylates",
    "dephosphorylate": "dephosphorylates",
    "methylate": "methylates", "demethylate": "demethylates",
    # Wordy / LLM phrasings
    "is_associated_with": "associated_with",
    "is_linked_to": "associated_with",
    "is_correlated_with": "correlates_with",
    "is_involved_in": "involved_in",
    "is_used_for": "treats",
    "is_a_biomarker_for": "biomarker_for",
    "facilitate_the_spread_of": "promotes",
    "facilitates": "promotes",
    "concerned_with": "associated_with",
    "affects": "regulates",
    "influences": "regulates",
    "complicated_by": "associated_with",
    "is_major_mechanism_for": "mediates",
    "serves_as": "biomarker_for",
    "associates": "associated_with",
    "relates_to": "associated_with",
    "linked_to": "associated_with",
    "connected_to": "associated_with",
    "implicated_in": "associated_with",
    "resistant_to": "associated_with",
    "sensitive_to": "associated_with",
}


def normalize_predicate(pred: str) -> str:
    """Normalize an LLM-generated predicate to the controlled vocabulary.

    4-stage normalization:
    1. Direct match to standard set
    2. Alias map lookup
    3. Strip common prefixes/suffixes and retry
    4. Fallback to 'associated_with' for unknown long predicates

    Usage:
        normalize_predicate("is_associated_with")  → "associated_with"
        normalize_predicate("activate")             → "activates"
        normalize_predicate("inhibit")              → "inhibits"
        normalize_predicate("may_potentially_cause") → "causes"
    """
    cleaned = pred.lower().strip().rstrip(".").replace(" ", "_")

    # Stage 1: direct match
    if cleaned in STANDARD_PREDICATES:
        return cleaned

    # Stage 2: alias map
    mapped = PREDICATE_ALIAS_MAP.get(cleaned)
    if mapped:
        return mapped

    # Stage 3: strip common prefixes/suffixes and retry
    stripped = cleaned
    for prefix in ("is_", "can_", "may_", "potentially_", "directly_"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
    for suffix in ("_in", "_of", "_by", "_to", "_for", "_with"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]

    if stripped in STANDARD_PREDICATES:
        return stripped
    mapped = PREDICATE_ALIAS_MAP.get(stripped)
    if mapped:
        return mapped

    # Stage 4: fallback for unknown long predicates
    if len(cleaned) > 30:
        return "associated_with"

    return cleaned


# ---------------------------------------------------------------------------
# Knowledge predicates — used by learning layer, recall, and retrieval
# ---------------------------------------------------------------------------

# All predicates that represent recorded knowledge (learnings, bugs, fixes, etc.)
KNOWLEDGE_PREDICATES: set[str] = {
    "has_warning", "has_vulnerability",
    "has_pattern", "has_decision",
    "has_tip",
    "had_bug", "has_issue", "failed_on",
    "has_fix", "resolved",
    "has_status",
    "had_outcome", "produced_by",
    "has_next_steps",
    # Strategic predicates
    "has_goal", "has_strategy", "has_architecture",
    "has_trade_off", "has_convention", "depends_on",
    "has_requirement", "supersedes",
}

# Priority ordering for knowledge predicates (lower = surface first).
# Warnings and vulnerabilities are most actionable and should always appear
# before historical bug reports.
KNOWLEDGE_PRIORITY: dict[str, int] = {
    # Tier 0: Urgent — surface immediately
    "has_warning": 0,
    "has_vulnerability": 0,
    # Tier 1: Strategic — goals, architecture, decisions
    "has_goal": 1,
    "has_strategy": 1,
    "has_architecture": 1,
    "has_decision": 1,
    "has_pattern": 1,
    # Tier 2: Constraints — requirements, trade-offs, conventions, dependencies
    "has_requirement": 2,
    "has_trade_off": 2,
    "has_convention": 2,
    "depends_on": 2,
    "has_tip": 2,
    # Tier 3: Problem history
    "had_bug": 3,
    "has_issue": 3,
    "failed_on": 3,
    # Tier 4-5: Resolutions
    "has_fix": 4,
    "supersedes": 4,
    "resolved": 5,
    "has_status": 5,
    # Tier 6+: Session metadata
    "had_outcome": 6,
    "produced_by": 7,
    "has_next_steps": 8,
}


# Display labels for knowledge predicates (replaces string-hacking with replace())
KNOWLEDGE_LABEL: dict[str, str] = {
    "has_warning": "warning",
    "has_vulnerability": "vulnerability",
    "has_goal": "goal",
    "has_strategy": "strategy",
    "has_architecture": "architecture",
    "has_pattern": "pattern",
    "has_decision": "decision",
    "has_requirement": "requirement",
    "has_trade_off": "trade-off",
    "has_convention": "convention",
    "depends_on": "dependency",
    "has_tip": "tip",
    "had_bug": "bug",
    "has_issue": "issue",
    "failed_on": "failed",
    "has_fix": "fix",
    "supersedes": "supersedes",
    "resolved": "resolved",
    "has_status": "status",
    "had_outcome": "outcome",
    "produced_by": "source",
    "has_next_steps": "next_steps",
}

# Priority threshold: predicates at or below this are "high priority"
# (warnings, patterns, decisions, tips) vs "history" (bugs, fixes, status)
KNOWLEDGE_HIGH_PRIORITY_THRESHOLD = 2


def knowledge_label(predicate_id: str) -> str:
    """Human-readable label for a knowledge predicate."""
    return KNOWLEDGE_LABEL.get(predicate_id, predicate_id)


def knowledge_sort_key(predicate_id: str, confidence: float = 0.5) -> tuple:
    """Sort key for knowledge claims: priority first, then confidence descending."""
    return (KNOWLEDGE_PRIORITY.get(predicate_id, 9), -confidence)


# Quantitative payload schemas — schemas whose data has {value, unit}
# ---------------------------------------------------------------------------
# Entity alias map — map common synonyms to canonical entity names
# ---------------------------------------------------------------------------

# Maps normalized aliases to canonical entity IDs.  Applied AFTER
# normalize_entity_id() (NFKD, lowercase, whitespace collapse, Greek) so all
# keys must already be in normalized form.
#
# This is the built-in map.  Users can extend it at runtime via
# AttestDB.add_entity_alias() which merges into the pipeline's alias table.
ENTITY_ALIAS_MAP: dict[str, str] = {
    # TP53 family
    "p53": "tp53",
    "tumour protein p53": "tp53",
    "tumor protein p53": "tp53",
    "trp53": "tp53",
    # PD-L1 / CD274
    "pd-l1": "cd274",
    "pdl1": "cd274",
    "pd l1": "cd274",
    "programmed death-ligand 1": "cd274",
    "programmed death ligand 1": "cd274",
    # BRCA
    "brca": "brca1",
    # Amyloid beta
    "abeta": "amyloid beta",
    "a-beta": "amyloid beta",
    "amyloid-beta": "amyloid beta",
    "beta-amyloid": "amyloid beta",
    "beta amyloid": "amyloid beta",
    "abeta42": "amyloid beta 42",
    "abeta-42": "amyloid beta 42",
    "a-beta-42": "amyloid beta 42",
    "abeta40": "amyloid beta 40",
    "abeta-40": "amyloid beta 40",
    # TNF-alpha
    "tnf-alpha": "tnf",
    "tnf alpha": "tnf",
    "tnfalpha": "tnf",
    "tumor necrosis factor alpha": "tnf",
    "tumour necrosis factor alpha": "tnf",
    "tumor necrosis factor": "tnf",
    # IL-6
    "interleukin-6": "il6",
    "interleukin 6": "il6",
    "il-6": "il6",
    # EGFR
    "erbb1": "egfr",
    "her1": "egfr",
    "epidermal growth factor receptor": "egfr",
    # HER2
    "erbb2": "her2",
    "neu": "her2",
    "cd340": "her2",
    # VEGF
    "vascular endothelial growth factor": "vegfa",
    "vegf": "vegfa",
    "vegf-a": "vegfa",
    # Common disease aliases
    "alzheimer's disease": "alzheimer disease",
    "alzheimer's": "alzheimer disease",
    "alzheimers": "alzheimer disease",
    "alzheimers disease": "alzheimer disease",
    "ad": "alzheimer disease",
    "parkinson's disease": "parkinson disease",
    "parkinson's": "parkinson disease",
    "parkinsons": "parkinson disease",
    "parkinsons disease": "parkinson disease",
    "pd": "parkinson disease",
    "als": "amyotrophic lateral sclerosis",
    "lou gehrig's disease": "amyotrophic lateral sclerosis",
    "lou gehrigs disease": "amyotrophic lateral sclerosis",
    # TREM2
    "triggering receptor expressed on myeloid cells 2": "trem2",
    # APOE
    "apolipoprotein e": "apoe",
    "apolipoprotein e4": "apoe4",
    "apoe epsilon4": "apoe4",
    "apoe-epsilon4": "apoe4",
}


def normalize_entity_name(name: str, extra_aliases: dict[str, str] | None = None) -> str:
    """Normalize an entity name, resolving known synonyms to canonical IDs.

    Two-stage normalization:
    1. Run normalize_entity_id() (NFKD, lowercase, whitespace collapse, Greek)
    2. Check alias maps (extra_aliases first, then built-in ENTITY_ALIAS_MAP)

    The extra_aliases parameter is for runtime aliases registered via
    AttestDB.add_entity_alias().

    Does NOT modify normalize_entity_id() — that function is locked.
    """
    from attestdb.core.normalization import normalize_entity_id

    normalized = normalize_entity_id(name)

    # Check runtime aliases first (user-registered take precedence)
    if extra_aliases:
        canonical = extra_aliases.get(normalized)
        if canonical is not None:
            return canonical

    # Check built-in alias map
    canonical = ENTITY_ALIAS_MAP.get(normalized)
    if canonical is not None:
        return canonical

    return normalized


# ---------------------------------------------------------------------------
# Regex-based triple extraction from prose
# ---------------------------------------------------------------------------

import re

# Verbs recognized by the regex extractor (base + optional 's' suffix).
# Ordered by specificity — more specific verbs first.
_VERB_STEMS: list[str] = [
    "downregulate", "upregulate", "dephosphorylate", "phosphorylate",
    "demethylate", "methylate", "destabilize", "stabilize",
    "inhibit", "activate", "suppress", "promote",
    "increase", "decrease", "enhance", "reduce",
    "cause", "prevent", "enable", "block",
    "treat", "target", "bind", "induce",
    "modulate", "mediate", "encode", "regulate",
]

# Build alternation: each stem matches with optional trailing 's' or 'es'
_VERB_ALT = "|".join(
    rf"{stem}(?:e?s)?" for stem in _VERB_STEMS
)

# Trailing clause patterns to strip from the object
_TRAILING_CLAUSE_RE = re.compile(
    r"\s+(?:in vitro|in vivo|via\b|through\b|by\b|leading to\b|resulting in\b"
    r"|which\b|that\b|, which\b|, leading\b|, resulting\b|, via\b|, through\b"
    r"|, by\b).*$",
    re.IGNORECASE,
)

# Pattern 1: "X <verb> Y"
_PAT_VERB = re.compile(
    rf"^([\w][\w\s/()-]{{1,60}}?)\s+({_VERB_ALT})\s+(.{{2,80}})$",
    re.IGNORECASE,
)

# Pattern 2: "X is associated/linked/correlated with Y"
_PAT_ASSOC = re.compile(
    r"^([\w][\w\s/()-]{1,60}?)\s+(?:is\s+)?"
    r"(?:associated|linked|correlated)\s+with\s+(.{2,80})$",
    re.IGNORECASE,
)

# Pattern 3: "X is expressed/found/detected/located in Y"
_PAT_EXPRESSED = re.compile(
    r"^([\w][\w\s/()-]{1,60}?)\s+(?:is\s+)?"
    r"(?:expressed|found|detected|located)\s+in\s+(.{2,80})$",
    re.IGNORECASE,
)


def _clean_entity(s: str) -> str:
    """Strip leading articles and trailing whitespace/punctuation from an entity."""
    s = s.strip().rstrip(".")
    # Strip leading article
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()


def extract_triples_from_prose(text: str) -> list[dict]:
    """Extract (subject, predicate, object) triples from a prose sentence.

    Uses regex patterns to identify simple declarative sentences containing
    known biological/scientific predicates. No LLM call required.

    Returns a list of dicts with keys: subject, predicate, object.
    All predicates are normalized via normalize_predicate().
    Subjects and objects are lowercased and stripped of articles.

    Returns an empty list if no patterns match.
    """
    text = text.strip()
    if not text:
        return []

    # Skip questions
    if text.endswith("?") or re.match(
        r"^(?:does|do|is|are|was|were|can|could|would|should|how|what|why|which|who)\s",
        text,
        re.IGNORECASE,
    ):
        return []

    results: list[dict] = []

    # Try Pattern 1: "X <verb> Y"
    m = _PAT_VERB.match(text)
    if m:
        subj = _clean_entity(m.group(1))
        pred_raw = m.group(2).strip().lower()
        obj_raw = m.group(3)
        # Strip trailing clauses from object
        obj_raw = _TRAILING_CLAUSE_RE.sub("", obj_raw)
        obj = _clean_entity(obj_raw)
        pred = normalize_predicate(pred_raw)
        results.append({
            "subject": subj.lower(),
            "predicate": pred,
            "object": obj.lower(),
        })
        return results

    # Try Pattern 2: "X is associated/linked/correlated with Y"
    m = _PAT_ASSOC.match(text)
    if m:
        subj = _clean_entity(m.group(1))
        obj_raw = m.group(2)
        obj_raw = _TRAILING_CLAUSE_RE.sub("", obj_raw)
        obj = _clean_entity(obj_raw)
        results.append({
            "subject": subj.lower(),
            "predicate": "associated_with",
            "object": obj.lower(),
        })
        return results

    # Try Pattern 3: "X is expressed/found/detected in Y"
    m = _PAT_EXPRESSED.match(text)
    if m:
        subj = _clean_entity(m.group(1))
        obj_raw = m.group(2)
        obj_raw = _TRAILING_CLAUSE_RE.sub("", obj_raw)
        obj = _clean_entity(obj_raw)
        results.append({
            "subject": subj.lower(),
            "predicate": "expressed_in",
            "object": obj.lower(),
        })
        return results

    return results


QUANTITATIVE_SCHEMAS: set[str] = {
    "binding_affinity",
    "expression_data",
    "differential_expression",
    "clinical_outcome",
    "enzyme_activity",
    "latency_measurement",
}
