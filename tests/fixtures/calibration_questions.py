"""25 hand-crafted ResearchQuestion objects for calibration/abstention evaluation.

Categories:
- ANSWERABLE_QUESTIONS (10): Well-established bio relationships
- FICTIONAL_QUESTIONS (5): Non-existent entities the LLM should refuse
- REAL_BUT_WRONG_QUESTIONS (5): Real entities with incorrect/implausible relationships
- AMBIGUOUS_QUESTIONS (5): Debatable or uncertain relationships
"""

from __future__ import annotations

from attestdb.core.types import ResearchQuestion


# --- 10 answerable questions: well-established bio facts ---
ANSWERABLE_QUESTIONS: list[ResearchQuestion] = [
    ResearchQuestion(
        entity_id="BRCA1",
        entity_type="gene",
        gap_type="single_source",
        question="What genes and proteins does BRCA1 interact with in DNA repair pathways?",
    ),
    ResearchQuestion(
        entity_id="BRCA2",
        entity_type="gene",
        gap_type="single_source",
        question="What is the relationship between BRCA2 and breast cancer susceptibility?",
    ),
    ResearchQuestion(
        entity_id="TP53",
        entity_type="gene",
        gap_type="missing_predicate",
        question="What proteins does TP53 interact with to regulate apoptosis?",
        predicate_hint="interacts",
    ),
    ResearchQuestion(
        entity_id="insulin",
        entity_type="compound",
        gap_type="single_source",
        question="What receptors and proteins does insulin bind to in glucose metabolism?",
    ),
    ResearchQuestion(
        entity_id="EGFR",
        entity_type="protein",
        gap_type="missing_predicate",
        question="What compounds are known to inhibit EGFR in cancer therapy?",
        predicate_hint="inhibits",
    ),
    ResearchQuestion(
        entity_id="KRAS",
        entity_type="gene",
        gap_type="single_source",
        question="What downstream signaling pathways does KRAS activate?",
    ),
    ResearchQuestion(
        entity_id="Metformin",
        entity_type="compound",
        gap_type="missing_predicate",
        question="What molecular targets does Metformin act on?",
        predicate_hint="inhibits",
    ),
    ResearchQuestion(
        entity_id="VEGF",
        entity_type="protein",
        gap_type="single_source",
        question="What is the role of VEGF in angiogenesis and which receptors does it bind?",
    ),
    ResearchQuestion(
        entity_id="JAK2",
        entity_type="protein",
        gap_type="missing_predicate",
        question="What proteins does JAK2 phosphorylate in the JAK-STAT pathway?",
        predicate_hint="activates",
    ),
    ResearchQuestion(
        entity_id="Imatinib",
        entity_type="compound",
        gap_type="single_source",
        question="What kinases does Imatinib inhibit and what diseases does it treat?",
    ),
]


# --- 5 fictional questions: non-existent entities ---
FICTIONAL_QUESTIONS: list[ResearchQuestion] = [
    ResearchQuestion(
        entity_id="ZZFAKE99",
        entity_type="gene",
        gap_type="single_source",
        question="What proteins does gene ZZFAKE99 interact with?",
    ),
    ResearchQuestion(
        entity_id="XQ-7734",
        entity_type="compound",
        gap_type="single_source",
        question="What diseases does the drug XQ-7734 treat?",
    ),
    ResearchQuestion(
        entity_id="PHANTOKIN3",
        entity_type="protein",
        gap_type="missing_predicate",
        question="What substrates does PHANTOKIN3 phosphorylate?",
        predicate_hint="phosphorylates",
    ),
    ResearchQuestion(
        entity_id="Neuridazolam-X",
        entity_type="compound",
        gap_type="single_source",
        question="What is the mechanism of action of Neuridazolam-X on the nervous system?",
    ),
    ResearchQuestion(
        entity_id="QWRT7",
        entity_type="gene",
        gap_type="single_source",
        question="What pathways involve the gene QWRT7?",
    ),
]


# --- 5 real-but-wrong questions: real entities, implausible relationships ---
REAL_BUT_WRONG_QUESTIONS: list[ResearchQuestion] = [
    ResearchQuestion(
        entity_id="BRCA1",
        entity_type="gene",
        gap_type="missing_predicate",
        question="Does BRCA1 treat asthma?",
        predicate_hint="treats",
    ),
    ResearchQuestion(
        entity_id="influenza",
        entity_type="disease",
        gap_type="missing_predicate",
        question="What compounds does influenza phosphorylate?",
        predicate_hint="phosphorylates",
    ),
    ResearchQuestion(
        entity_id="aspirin",
        entity_type="compound",
        gap_type="missing_predicate",
        question="Does aspirin upregulate telomerase reverse transcriptase to extend lifespan?",
        predicate_hint="upregulates",
    ),
    ResearchQuestion(
        entity_id="hemoglobin",
        entity_type="protein",
        gap_type="missing_predicate",
        question="Does hemoglobin inhibit photosynthesis in plant cells?",
        predicate_hint="inhibits",
    ),
    ResearchQuestion(
        entity_id="Metformin",
        entity_type="compound",
        gap_type="missing_predicate",
        question="Does Metformin activate prion protein aggregation in neurons?",
        predicate_hint="activates",
    ),
]


# --- 5 ambiguous questions: debatable or context-dependent ---
AMBIGUOUS_QUESTIONS: list[ResearchQuestion] = [
    ResearchQuestion(
        entity_id="TP53",
        entity_type="gene",
        gap_type="single_source",
        question="Is TP53 associated with type 2 diabetes?",
    ),
    ResearchQuestion(
        entity_id="Metformin",
        entity_type="compound",
        gap_type="missing_predicate",
        question="Does Metformin inhibit EGFR?",
        predicate_hint="inhibits",
    ),
    ResearchQuestion(
        entity_id="aspirin",
        entity_type="compound",
        gap_type="single_source",
        question="Does aspirin prevent Alzheimer disease?",
    ),
    ResearchQuestion(
        entity_id="VEGF",
        entity_type="protein",
        gap_type="single_source",
        question="Is VEGF associated with depression?",
    ),
    ResearchQuestion(
        entity_id="caffeine",
        entity_type="compound",
        gap_type="single_source",
        question="Does caffeine protect against Parkinson disease?",
    ),
]
