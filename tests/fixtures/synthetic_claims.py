"""Synthetic 100-claim fixture covering all edge cases."""

from __future__ import annotations

import random

import numpy as np

from attestdb.core.types import ClaimInput


def _ts(offset: int) -> int:
    """Generate a timestamp with an offset from a base."""
    return 1700000000000000000 + offset * 1000000000


def generate_synthetic_claims() -> list[ClaimInput]:
    """Generate 100 synthetic claims covering all important cases."""
    rng = random.Random(42)
    claims: list[ClaimInput] = []

    # --- Standard bio claims (protein-protein, compound-protein, gene-disease) ---
    bio_triples = [
        (("TREM2", "protein"), ("binds", "relates_to"), ("APOE", "protein")),
        (("TREM2", "protein"), ("binds", "relates_to"), ("DAP12", "protein")),
        (("BRAF", "protein"), ("activates", "relates_to"), ("MEK1", "protein")),
        (("MEK1", "protein"), ("activates", "relates_to"), ("ERK2", "protein")),
        (("TP53", "protein"), ("inhibits", "relates_to"), ("MDM2", "protein")),
        (("EGFR", "protein"), ("binds", "relates_to"), ("Erlotinib", "entity")),
        (("Imatinib", "entity"), ("inhibits", "relates_to"), ("BCR-ABL", "protein")),
        (("BRCA1", "entity"), ("associated_with", "relates_to"), ("Breast Cancer", "entity")),
        (("TP53", "protein"), ("associated_with", "relates_to"), ("Li-Fraumeni Syndrome", "entity")),
        (("KRAS", "protein"), ("activates", "relates_to"), ("RAF1", "protein")),
        (("PIK3CA", "protein"), ("activates", "relates_to"), ("AKT1", "protein")),
        (("AKT1", "protein"), ("inhibits", "relates_to"), ("TSC2", "protein")),
        (("mTOR", "protein"), ("activates", "relates_to"), ("S6K1", "protein")),
        (("VEGFA", "entity"), ("binds", "relates_to"), ("VEGFR2", "protein")),
        (("HER2", "protein"), ("activates", "relates_to"), ("PI3K", "protein")),
        (("CDK4", "protein"), ("binds", "relates_to"), ("Cyclin D1", "entity")),
        (("RB1", "protein"), ("inhibits", "relates_to"), ("E2F1", "protein")),
        (("BCL2", "protein"), ("inhibits", "relates_to"), ("BAX", "protein")),
        (("PARP1", "protein"), ("relates_to", "relates_to"), ("DNA Repair", "entity")),
        (("ALK", "protein"), ("binds", "relates_to"), ("Crizotinib", "entity")),
    ]

    sources = [
        ("PMID:12345678", "observation"),
        ("PMID:23456789", "observation"),
        ("ChEMBL_DB", "computation"),
        ("AlphaFold_v2", "computation"),
        ("Expert_Review_2024", "human_annotation"),
    ]

    for i, (subj, pred, obj) in enumerate(bio_triples):
        src_id, src_type = sources[i % len(sources)]
        claims.append(ClaimInput(
            subject=subj, predicate=pred, object=obj,
            provenance={"source_type": src_type, "source_id": src_id},
            timestamp=_ts(i),
        ))

    # --- Corroborating claims (same content_id, different sources) ---
    # TREM2 binds APOE from multiple independent sources
    for i, (src_id, src_type) in enumerate([
        ("PMID:99999991", "human_annotation"),
        ("PMID:99999992", "observation"),
        ("Docking_Pipeline_v3", "computation"),
    ]):
        claims.append(ClaimInput(
            subject=("TREM2", "protein"),
            predicate=("binds", "relates_to"),
            object=("APOE", "protein"),
            provenance={"source_type": src_type, "source_id": src_id},
            timestamp=_ts(100 + i),
        ))

    # --- Quantitative payloads (binding affinity) ---
    claims.append(ClaimInput(
        subject=("TREM2", "protein"),
        predicate=("binds", "relates_to"),
        object=("APOE", "protein"),
        provenance={"source_type": "observation", "source_id": "SPR_Experiment_42"},
        payload={
            "schema": "binding_affinity",
            "data": {"metric": "Kd", "value": 5.2, "unit": "nM"},
        },
        timestamp=_ts(200),
    ))

    claims.append(ClaimInput(
        subject=("Erlotinib", "entity"),
        predicate=("inhibits", "relates_to"),
        object=("EGFR", "protein"),
        provenance={"source_type": "observation", "source_id": "Assay_Lab_17"},
        payload={
            "schema": "binding_affinity",
            "data": {"metric": "IC50", "value": 2.0, "unit": "nM"},
        },
        timestamp=_ts(201),
    ))

    # --- Provenance chains (claim B references claim A) ---
    # We'll add claim A first, then B references A's claim_id
    # (claim_ids are computed on ingest, so we use a marker approach)
    claims.append(ClaimInput(
        subject=("TREM2", "protein"),
        predicate=("observed", "observed"),
        object=("Microglial Activation", "entity"),
        provenance={
            "source_type": "observation",
            "source_id": "scRNAseq_Study_1",
            "method": "single_cell_rna_seq",
        },
        timestamp=_ts(300),
    ))
    # The derived claim will reference the above — handled in test code
    # since claim_id is only known after ingestion

    # --- Alias claims (same_as) ---
    claims.append(ClaimInput(
        subject=("Q9NZC2", "protein"),
        predicate=("same_as", "same_as"),
        object=("TREM2", "protein"),
        provenance={"source_type": "human_annotation", "source_id": "curator_v1"},
        confidence=1.0,
        timestamp=_ts(400),
    ))

    claims.append(ClaimInput(
        subject=("P02649", "protein"),
        predicate=("same_as", "same_as"),
        object=("APOE", "protein"),
        provenance={"source_type": "human_annotation", "source_id": "curator_v1"},
        confidence=1.0,
        timestamp=_ts(401),
    ))

    # --- Contradiction claims ---
    claims.append(ClaimInput(
        subject=("TREM2", "protein"),
        predicate=("activates", "relates_to"),
        object=("NF-kB", "entity"),
        provenance={"source_type": "observation", "source_id": "PMID:11111111"},
        timestamp=_ts(500),
    ))
    claims.append(ClaimInput(
        subject=("TREM2", "protein"),
        predicate=("inhibits", "relates_to"),
        object=("NF-kB", "entity"),
        provenance={"source_type": "observation", "source_id": "PMID:22222222"},
        timestamp=_ts(501),
    ))

    # --- Greek letter edge cases ---
    claims.append(ClaimInput(
        subject=("α-synuclein", "protein"),
        predicate=("associated_with", "relates_to"),
        object=("Parkinson Disease", "entity"),
        provenance={"source_type": "human_annotation", "source_id": "Expert_1"},
        timestamp=_ts(600),
    ))

    claims.append(ClaimInput(
        subject=("β-amyloid", "protein"),
        predicate=("associated_with", "relates_to"),
        object=("Alzheimer Disease", "entity"),
        provenance={"source_type": "human_annotation", "source_id": "Expert_1"},
        timestamp=_ts(601),
    ))

    claims.append(ClaimInput(
        subject=("γ-secretase", "protein"),
        predicate=("relates_to", "relates_to"),
        object=("β-amyloid", "protein"),
        provenance={"source_type": "observation", "source_id": "PMID:33333333"},
        timestamp=_ts(602),
    ))

    # --- Unicode edge cases ---
    claims.append(ClaimInput(
        subject=("Ångström Protein", "entity"),
        predicate=("relates_to", "relates_to"),
        object=("Crystal Structure", "entity"),
        provenance={"source_type": "observation", "source_id": "PDB_DB"},
        timestamp=_ts(700),
    ))

    claims.append(ClaimInput(
        subject=("  Extra   Spaces  ", "entity"),
        predicate=("relates_to", "relates_to"),
        object=("Normal Entity", "entity"),
        provenance={"source_type": "computation", "source_id": "Test"},
        timestamp=_ts(701),
    ))

    # --- Embedding claims (random 768-dim) ---
    np_rng = np.random.RandomState(42)
    for i in range(10):
        emb = np_rng.randn(768).tolist()
        subj_name = f"Protein_{i}"
        obj_name = f"Pathway_{i}"
        claims.append(ClaimInput(
            subject=(subj_name, "entity"),
            predicate=("relates_to", "relates_to"),
            object=(obj_name, "entity"),
            provenance={"source_type": "computation", "source_id": f"Pipeline_{i}"},
            embedding=emb,
            timestamp=_ts(800 + i),
        ))

    # --- More standard claims to reach 100 ---
    extra_triples = [
        (("JAK2", "protein"), ("activates", "relates_to"), ("STAT3", "protein")),
        (("STAT3", "protein"), ("relates_to", "relates_to"), ("Cell Proliferation", "entity")),
        (("NOTCH1", "protein"), ("activates", "relates_to"), ("HES1", "protein")),
        (("WNT3A", "entity"), ("activates", "relates_to"), ("β-catenin", "entity")),
        (("SHH", "entity"), ("binds", "relates_to"), ("PTCH1", "protein")),
        (("PTCH1", "protein"), ("inhibits", "relates_to"), ("SMO", "protein")),
        (("FGF2", "entity"), ("binds", "relates_to"), ("FGFR1", "protein")),
        (("FGFR1", "protein"), ("activates", "relates_to"), ("RAS", "protein")),
        (("RAS", "protein"), ("activates", "relates_to"), ("RAF", "protein")),
        (("RAF", "protein"), ("activates", "relates_to"), ("MEK", "protein")),
        (("MEK", "protein"), ("activates", "relates_to"), ("ERK", "protein")),
        (("ERK", "protein"), ("relates_to", "relates_to"), ("Cell Growth", "entity")),
        (("Tamoxifen", "entity"), ("inhibits", "relates_to"), ("ESR1", "protein")),
        (("ESR1", "protein"), ("relates_to", "relates_to"), ("Breast Cancer", "entity")),
        (("Metformin", "entity"), ("inhibits", "relates_to"), ("Complex I", "entity")),
        (("Complex I", "entity"), ("relates_to", "relates_to"), ("AMPK", "protein")),
        (("AMPK", "protein"), ("inhibits", "relates_to"), ("mTOR", "protein")),
        (("Rapamycin", "entity"), ("inhibits", "relates_to"), ("mTOR", "protein")),
        (("TREM2", "protein"), ("relates_to", "relates_to"), ("Alzheimer Disease", "entity")),
        (("APOE", "protein"), ("associated_with", "relates_to"), ("Alzheimer Disease", "entity")),
        (("APOE", "protein"), ("binds", "relates_to"), ("Lipid Particle", "entity")),
        (("Insulin", "entity"), ("binds", "relates_to"), ("INSR", "protein")),
        (("INSR", "protein"), ("activates", "relates_to"), ("IRS1", "protein")),
        (("IRS1", "protein"), ("activates", "relates_to"), ("PI3K", "protein")),
        (("Aspirin", "entity"), ("inhibits", "relates_to"), ("COX2", "protein")),
        (("COX2", "protein"), ("relates_to", "relates_to"), ("Inflammation", "entity")),
        (("TNF", "protein"), ("activates", "relates_to"), ("NF-kB", "entity")),
        (("IL6", "protein"), ("activates", "relates_to"), ("STAT3", "protein")),
        (("Dexamethasone", "entity"), ("inhibits", "relates_to"), ("NF-kB", "entity")),
        (("PD1", "protein"), ("binds", "relates_to"), ("PDL1", "protein")),
        (("PDL1", "protein"), ("relates_to", "relates_to"), ("Immune Evasion", "entity")),
        (("Pembrolizumab", "entity"), ("inhibits", "relates_to"), ("PD1", "protein")),
        (("CD8 T Cell", "entity"), ("relates_to", "relates_to"), ("Tumor Killing", "entity")),
        (("CTLA4", "protein"), ("inhibits", "relates_to"), ("CD28", "protein")),
        (("Ipilimumab", "entity"), ("inhibits", "relates_to"), ("CTLA4", "protein")),
        (("VEGFA", "entity"), ("relates_to", "relates_to"), ("Angiogenesis", "entity")),
        (("Bevacizumab", "entity"), ("inhibits", "relates_to"), ("VEGFA", "entity")),
        (("HIF1A", "protein"), ("activates", "relates_to"), ("VEGFA", "entity")),
        (("VHL", "protein"), ("inhibits", "relates_to"), ("HIF1A", "protein")),
        (("PTEN", "protein"), ("inhibits", "relates_to"), ("PI3K", "protein")),
        (("PI3K", "protein"), ("activates", "relates_to"), ("PIP3", "entity")),
        (("PIP3", "entity"), ("activates", "relates_to"), ("AKT1", "protein")),
        (("Staurosporine", "entity"), ("inhibits", "relates_to"), ("PKC", "protein")),
        (("PKC", "protein"), ("activates", "relates_to"), ("NF-kB", "entity")),
        (("MYC", "protein"), ("relates_to", "relates_to"), ("Cell Proliferation", "entity")),
        (("MYC", "protein"), ("activates", "relates_to"), ("CDK4", "protein")),
        (("SMAD4", "protein"), ("inhibits", "relates_to"), ("MYC", "protein")),
        (("TGFβ", "entity"), ("activates", "relates_to"), ("SMAD4", "protein")),
        (("RB1", "protein"), ("associated_with", "relates_to"), ("Retinoblastoma", "entity")),
        (("APC", "protein"), ("inhibits", "relates_to"), ("β-catenin", "entity")),
        (("GSK3β", "protein"), ("inhibits", "relates_to"), ("β-catenin", "entity")),
        (("Lithium", "entity"), ("inhibits", "relates_to"), ("GSK3β", "protein")),
        (("EGFR", "protein"), ("activates", "relates_to"), ("RAS", "protein")),
        (("Cetuximab", "entity"), ("inhibits", "relates_to"), ("EGFR", "protein")),
        (("Sorafenib", "entity"), ("inhibits", "relates_to"), ("RAF", "protein")),
    ]

    for i, (subj, pred, obj) in enumerate(extra_triples):
        src_id, src_type = sources[i % len(sources)]
        claims.append(ClaimInput(
            subject=subj, predicate=pred, object=obj,
            provenance={"source_type": src_type, "source_id": src_id},
            timestamp=_ts(900 + i),
        ))

    return claims[:100]  # Ensure exactly 100
