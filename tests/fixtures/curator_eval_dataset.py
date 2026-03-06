"""Generate 200+ expert-labeled claims for curator accuracy evaluation.

Each entry is (ClaimInput, expert_label) where label is "store"/"skip"/"flag".

The dataset covers:
- Novel findings from various source types (store)
- Corroborating evidence from independent sources (store)
- Contradictions against established knowledge (flag)
- Low-confidence LLM inferences (skip)
- Redundant/trivially known claims (skip)
- Claims from high-quality sources (store)
- Suspicious provenance patterns (flag)
"""

from __future__ import annotations

from attestdb.core.types import ClaimInput


def _ts(offset: int) -> int:
    return 1700000000000000000 + offset * 1000000000


# Known relationships that will be seeded as background knowledge
BACKGROUND_KNOWLEDGE = [
    (("TREM2", "protein"), ("activates", "relates_to"), ("NF-kB", "protein"), "observation", "PMID:10001"),
    (("BRAF", "gene"), ("activates", "relates_to"), ("MEK1", "protein"), "observation", "PMID:10002"),
    (("TP53", "gene"), ("inhibits", "relates_to"), ("MDM2", "protein"), "observation", "PMID:10003"),
    (("EGFR", "protein"), ("activates", "relates_to"), ("RAS", "protein"), "observation", "PMID:10004"),
    (("JAK2", "protein"), ("activates", "relates_to"), ("STAT3", "protein"), "observation", "PMID:10005"),
    (("KRAS", "gene"), ("activates", "relates_to"), ("RAF1", "protein"), "observation", "PMID:10006"),
    (("BRCA1", "gene"), ("inhibits", "relates_to"), ("cell_cycle", "entity"), "observation", "PMID:10007"),
    (("Imatinib", "compound"), ("inhibits", "relates_to"), ("BCR-ABL", "protein"), "observation", "PMID:10008"),
    (("Metformin", "compound"), ("inhibits", "relates_to"), ("mTOR", "protein"), "observation", "PMID:10009"),
    (("VEGF", "protein"), ("activates", "relates_to"), ("angiogenesis", "entity"), "observation", "PMID:10010"),
]


def generate_curator_eval_dataset() -> tuple[list[tuple], list[tuple[ClaimInput, str]]]:
    """Generate background knowledge and 250 labeled claims for curator eval.

    Returns:
        (background_claims, labeled_data) where:
        - background_claims: list of (subject, predicate, object, source_type, source_id) tuples to seed
        - labeled_data: list of (ClaimInput, expert_label) pairs
    """
    labeled: list[tuple[ClaimInput, str]] = []
    ts = 20000  # Starting timestamp offset

    # =========================================================================
    # STORE: Novel findings from credible sources (80 claims)
    # =========================================================================

    novel_findings = [
        # New protein-protein interactions
        (("TREM2", "protein"), ("binds", "relates_to"), ("APOE", "protein"), "observation", "PMID:20001"),
        (("BRCA1", "gene"), ("activates", "relates_to"), ("RAD51", "protein"), "observation", "PMID:20002"),
        (("AKT1", "protein"), ("phosphorylates", "relates_to"), ("GSK3B", "protein"), "observation", "PMID:20003"),
        (("PTEN", "gene"), ("inhibits", "relates_to"), ("PI3K", "protein"), "observation", "PMID:20004"),
        (("MYC", "gene"), ("activates", "relates_to"), ("CDK4", "protein"), "observation", "PMID:20005"),
        (("RB1", "gene"), ("inhibits", "relates_to"), ("E2F1", "protein"), "observation", "PMID:20006"),
        (("NOTCH1", "protein"), ("activates", "relates_to"), ("HES1", "protein"), "observation", "PMID:20007"),
        (("WNT3A", "protein"), ("activates", "relates_to"), ("beta-catenin", "protein"), "observation", "PMID:20008"),
        (("SHH", "protein"), ("activates", "relates_to"), ("GLI1", "protein"), "observation", "PMID:20009"),
        (("TNF-alpha", "protein"), ("activates", "relates_to"), ("NF-kB", "protein"), "observation", "PMID:20010"),
        # Compound-target interactions
        (("Erlotinib", "compound"), ("inhibits", "relates_to"), ("EGFR", "protein"), "experimental", "ChEMBL:30001"),
        (("Gefitinib", "compound"), ("inhibits", "relates_to"), ("EGFR", "protein"), "experimental", "ChEMBL:30002"),
        (("Sorafenib", "compound"), ("inhibits", "relates_to"), ("BRAF", "gene"), "experimental", "ChEMBL:30003"),
        (("Vemurafenib", "compound"), ("inhibits", "relates_to"), ("BRAF", "gene"), "experimental", "ChEMBL:30004"),
        (("Trastuzumab", "compound"), ("inhibits", "relates_to"), ("HER2", "protein"), "experimental", "ChEMBL:30005"),
        (("Bevacizumab", "compound"), ("inhibits", "relates_to"), ("VEGF", "protein"), "experimental", "ChEMBL:30006"),
        (("Rituximab", "compound"), ("binds", "relates_to"), ("CD20", "protein"), "experimental", "ChEMBL:30007"),
        (("Nivolumab", "compound"), ("binds", "relates_to"), ("PD-1", "protein"), "experimental", "ChEMBL:30008"),
        (("Pembrolizumab", "compound"), ("binds", "relates_to"), ("PD-1", "protein"), "experimental", "ChEMBL:30009"),
        (("Olaparib", "compound"), ("inhibits", "relates_to"), ("PARP1", "protein"), "experimental", "ChEMBL:30010"),
        # Gene-disease associations
        (("BRCA2", "gene"), ("associated_with", "relates_to"), ("breast cancer", "disease"), "observation", "PMID:20011"),
        (("CFTR", "gene"), ("associated_with", "relates_to"), ("cystic fibrosis", "disease"), "observation", "PMID:20012"),
        (("HTT", "gene"), ("associated_with", "relates_to"), ("Huntington disease", "disease"), "observation", "PMID:20013"),
        (("APP", "gene"), ("associated_with", "relates_to"), ("Alzheimer disease", "disease"), "observation", "PMID:20014"),
        (("LRRK2", "gene"), ("associated_with", "relates_to"), ("Parkinson disease", "disease"), "observation", "PMID:20015"),
        # Signaling pathway claims
        (("MAPK1", "protein"), ("activates", "relates_to"), ("ELK1", "protein"), "observation", "PMID:20016"),
        (("PI3K", "protein"), ("activates", "relates_to"), ("AKT1", "protein"), "observation", "PMID:20017"),
        (("mTOR", "protein"), ("activates", "relates_to"), ("S6K1", "protein"), "observation", "PMID:20018"),
        (("AMPK", "protein"), ("inhibits", "relates_to"), ("mTOR", "protein"), "observation", "PMID:20019"),
        (("p38", "protein"), ("activates", "relates_to"), ("ATF2", "protein"), "observation", "PMID:20020"),
        # Expression data
        (("TP53", "gene"), ("upregulates", "relates_to"), ("p21", "protein"), "observation", "PMID:20021"),
        (("MYC", "gene"), ("downregulates", "relates_to"), ("p27", "protein"), "observation", "PMID:20022"),
        (("HIF1A", "gene"), ("upregulates", "relates_to"), ("VEGF", "protein"), "observation", "PMID:20023"),
        (("FOXO3", "gene"), ("upregulates", "relates_to"), ("BIM", "protein"), "observation", "PMID:20024"),
        (("NRF2", "gene"), ("upregulates", "relates_to"), ("HO-1", "protein"), "observation", "PMID:20025"),
        # Computational predictions (lower confidence but still store-worthy)
        (("CDK6", "protein"), ("binds", "relates_to"), ("CCND1", "protein"), "computation", "AlphaFold:40001"),
        (("PLK1", "protein"), ("activates", "relates_to"), ("CDC25C", "protein"), "computation", "AlphaFold:40002"),
        (("Aurora-B", "protein"), ("phosphorylates", "relates_to"), ("H3", "protein"), "computation", "ML:40003"),
        (("HDAC1", "protein"), ("inhibits", "relates_to"), ("p21_promoter", "entity"), "computation", "ML:40004"),
        (("DNMT1", "protein"), ("inhibits", "relates_to"), ("CDKN2A", "gene"), "computation", "ML:40005"),
        # Human annotations
        (("TREM2", "protein"), ("associated_with", "relates_to"), ("neuroinflammation", "entity"), "human_annotation", "expert_1"),
        (("TDP-43", "protein"), ("associated_with", "relates_to"), ("ALS", "disease"), "human_annotation", "expert_2"),
        (("SOD1", "gene"), ("associated_with", "relates_to"), ("ALS", "disease"), "human_annotation", "expert_3"),
        (("GBA", "gene"), ("associated_with", "relates_to"), ("Parkinson disease", "disease"), "human_annotation", "expert_4"),
        (("FUS", "protein"), ("associated_with", "relates_to"), ("ALS", "disease"), "human_annotation", "expert_5"),
        # Database imports (always store)
        (("insulin", "compound"), ("binds", "relates_to"), ("INSR", "protein"), "database_import", "DrugBank:50001"),
        (("aspirin", "compound"), ("inhibits", "relates_to"), ("COX2", "protein"), "database_import", "DrugBank:50002"),
        (("methotrexate", "compound"), ("inhibits", "relates_to"), ("DHFR", "protein"), "database_import", "DrugBank:50003"),
        (("tamoxifen", "compound"), ("inhibits", "relates_to"), ("ESR1", "protein"), "database_import", "DrugBank:50004"),
        (("warfarin", "compound"), ("inhibits", "relates_to"), ("VKORC1", "protein"), "database_import", "DrugBank:50005"),
        # More novel interactions
        (("SMAD3", "protein"), ("activates", "relates_to"), ("PAI1", "protein"), "observation", "PMID:20026"),
        (("TGFB1", "protein"), ("activates", "relates_to"), ("SMAD3", "protein"), "observation", "PMID:20027"),
        (("IL-6", "protein"), ("activates", "relates_to"), ("STAT3", "protein"), "observation", "PMID:20028"),
        (("IL-1B", "protein"), ("activates", "relates_to"), ("NF-kB", "protein"), "observation", "PMID:20029"),
        (("CXCL12", "protein"), ("binds", "relates_to"), ("CXCR4", "protein"), "observation", "PMID:20030"),
        (("FGF2", "protein"), ("activates", "relates_to"), ("FGFR1", "protein"), "observation", "PMID:20031"),
        (("PDGF", "protein"), ("activates", "relates_to"), ("PDGFRA", "protein"), "observation", "PMID:20032"),
        (("EGF", "protein"), ("activates", "relates_to"), ("EGFR", "protein"), "observation", "PMID:20033"),
        (("IGF1", "protein"), ("activates", "relates_to"), ("IGF1R", "protein"), "observation", "PMID:20034"),
        (("NGF", "protein"), ("activates", "relates_to"), ("TRKA", "protein"), "observation", "PMID:20035"),
        # Literature extraction claims
        (("SIRT1", "protein"), ("inhibits", "relates_to"), ("p53_acetylation", "entity"), "document_extraction", "DOI:60001"),
        (("ATM", "protein"), ("activates", "relates_to"), ("CHK2", "protein"), "document_extraction", "DOI:60002"),
        (("ATR", "protein"), ("activates", "relates_to"), ("CHK1", "protein"), "document_extraction", "DOI:60003"),
        (("RAD51", "protein"), ("associated_with", "relates_to"), ("DNA_repair", "entity"), "document_extraction", "DOI:60004"),
        (("XRCC1", "protein"), ("associated_with", "relates_to"), ("base_excision_repair", "entity"), "document_extraction", "DOI:60005"),
        # More computational
        (("STAT1", "protein"), ("activates", "relates_to"), ("IRF1", "protein"), "computation", "ML:40006"),
        (("IRF3", "protein"), ("activates", "relates_to"), ("IFNB1", "gene"), "computation", "ML:40007"),
        (("RIPK1", "protein"), ("activates", "relates_to"), ("MLKL", "protein"), "computation", "ML:40008"),
        (("CASP8", "protein"), ("activates", "relates_to"), ("CASP3", "protein"), "computation", "ML:40009"),
        (("BCL2", "protein"), ("inhibits", "relates_to"), ("BAX", "protein"), "computation", "ML:40010"),
        # More human annotations
        (("APOE4", "protein"), ("associated_with", "relates_to"), ("Alzheimer disease", "disease"), "human_annotation", "expert_6"),
        (("MAPT", "gene"), ("associated_with", "relates_to"), ("frontotemporal dementia", "disease"), "human_annotation", "expert_7"),
        (("SNCA", "gene"), ("associated_with", "relates_to"), ("Parkinson disease", "disease"), "human_annotation", "expert_8"),
        (("PSEN1", "gene"), ("associated_with", "relates_to"), ("Alzheimer disease", "disease"), "human_annotation", "expert_9"),
        (("C9orf72", "gene"), ("associated_with", "relates_to"), ("ALS", "disease"), "human_annotation", "expert_10"),
    ]

    for i, (subj, pred, obj, src_type, src_id) in enumerate(novel_findings):
        ts += 1
        labeled.append((
            ClaimInput(subject=subj, predicate=pred, object=obj,
                       provenance={"source_type": src_type, "source_id": src_id},
                       timestamp=_ts(ts)),
            "store",
        ))

    # =========================================================================
    # STORE: Corroborating evidence (30 claims)
    # These repeat known relationships but from different sources
    # =========================================================================

    corroborating = [
        (("TREM2", "protein"), ("activates", "relates_to"), ("NF-kB", "protein"), "computation", "ML:70001"),
        (("TREM2", "protein"), ("activates", "relates_to"), ("NF-kB", "protein"), "document_extraction", "DOI:70002"),
        (("BRAF", "gene"), ("activates", "relates_to"), ("MEK1", "protein"), "experimental", "ChEMBL:70003"),
        (("BRAF", "gene"), ("activates", "relates_to"), ("MEK1", "protein"), "human_annotation", "expert_11"),
        (("TP53", "gene"), ("inhibits", "relates_to"), ("MDM2", "protein"), "computation", "ML:70005"),
        (("TP53", "gene"), ("inhibits", "relates_to"), ("MDM2", "protein"), "document_extraction", "DOI:70006"),
        (("EGFR", "protein"), ("activates", "relates_to"), ("RAS", "protein"), "experimental", "ChEMBL:70007"),
        (("EGFR", "protein"), ("activates", "relates_to"), ("RAS", "protein"), "human_annotation", "expert_12"),
        (("JAK2", "protein"), ("activates", "relates_to"), ("STAT3", "protein"), "experimental", "ChEMBL:70009"),
        (("JAK2", "protein"), ("activates", "relates_to"), ("STAT3", "protein"), "document_extraction", "DOI:70010"),
        (("KRAS", "gene"), ("activates", "relates_to"), ("RAF1", "protein"), "computation", "ML:70011"),
        (("KRAS", "gene"), ("activates", "relates_to"), ("RAF1", "protein"), "human_annotation", "expert_13"),
        (("Imatinib", "compound"), ("inhibits", "relates_to"), ("BCR-ABL", "protein"), "database_import", "DrugBank:70013"),
        (("Imatinib", "compound"), ("inhibits", "relates_to"), ("BCR-ABL", "protein"), "human_annotation", "expert_14"),
        (("Metformin", "compound"), ("inhibits", "relates_to"), ("mTOR", "protein"), "experimental", "ChEMBL:70015"),
        (("Metformin", "compound"), ("inhibits", "relates_to"), ("mTOR", "protein"), "document_extraction", "DOI:70016"),
        (("VEGF", "protein"), ("activates", "relates_to"), ("angiogenesis", "entity"), "human_annotation", "expert_15"),
        (("VEGF", "protein"), ("activates", "relates_to"), ("angiogenesis", "entity"), "document_extraction", "DOI:70018"),
        (("BRCA1", "gene"), ("inhibits", "relates_to"), ("cell_cycle", "entity"), "computation", "ML:70019"),
        (("BRCA1", "gene"), ("inhibits", "relates_to"), ("cell_cycle", "entity"), "document_extraction", "DOI:70020"),
        # Additional corroborations
        (("TREM2", "protein"), ("activates", "relates_to"), ("NF-kB", "protein"), "experimental", "ChEMBL:70021"),
        (("BRAF", "gene"), ("activates", "relates_to"), ("MEK1", "protein"), "computation", "ML:70022"),
        (("TP53", "gene"), ("inhibits", "relates_to"), ("MDM2", "protein"), "human_annotation", "expert_16"),
        (("EGFR", "protein"), ("activates", "relates_to"), ("RAS", "protein"), "computation", "ML:70024"),
        (("JAK2", "protein"), ("activates", "relates_to"), ("STAT3", "protein"), "computation", "ML:70025"),
        (("KRAS", "gene"), ("activates", "relates_to"), ("RAF1", "protein"), "experimental", "ChEMBL:70026"),
        (("Imatinib", "compound"), ("inhibits", "relates_to"), ("BCR-ABL", "protein"), "experimental", "ChEMBL:70027"),
        (("Metformin", "compound"), ("inhibits", "relates_to"), ("mTOR", "protein"), "human_annotation", "expert_17"),
        (("VEGF", "protein"), ("activates", "relates_to"), ("angiogenesis", "entity"), "computation", "ML:70029"),
        (("BRCA1", "gene"), ("inhibits", "relates_to"), ("cell_cycle", "entity"), "human_annotation", "expert_18"),
    ]

    for i, (subj, pred, obj, src_type, src_id) in enumerate(corroborating):
        ts += 1
        labeled.append((
            ClaimInput(subject=subj, predicate=pred, object=obj,
                       provenance={"source_type": src_type, "source_id": src_id},
                       timestamp=_ts(ts)),
            "store",
        ))

    # =========================================================================
    # FLAG: Contradictions against established knowledge (40 claims)
    # These use opposite predicates (activates↔inhibits, upregulates↔downregulates)
    # =========================================================================

    contradictions = [
        # Direct contradictions to background knowledge
        (("TREM2", "protein"), ("inhibits", "relates_to"), ("NF-kB", "protein"), "observation", "PMID:80001"),
        (("TREM2", "protein"), ("inhibits", "relates_to"), ("NF-kB", "protein"), "computation", "ML:80002"),
        (("BRAF", "gene"), ("inhibits", "relates_to"), ("MEK1", "protein"), "observation", "PMID:80003"),
        (("BRAF", "gene"), ("inhibits", "relates_to"), ("MEK1", "protein"), "computation", "ML:80004"),
        (("TP53", "gene"), ("activates", "relates_to"), ("MDM2", "protein"), "observation", "PMID:80005"),
        (("TP53", "gene"), ("activates", "relates_to"), ("MDM2", "protein"), "computation", "ML:80006"),
        (("EGFR", "protein"), ("inhibits", "relates_to"), ("RAS", "protein"), "observation", "PMID:80007"),
        (("EGFR", "protein"), ("inhibits", "relates_to"), ("RAS", "protein"), "computation", "ML:80008"),
        (("JAK2", "protein"), ("inhibits", "relates_to"), ("STAT3", "protein"), "observation", "PMID:80009"),
        (("JAK2", "protein"), ("inhibits", "relates_to"), ("STAT3", "protein"), "computation", "ML:80010"),
        (("KRAS", "gene"), ("inhibits", "relates_to"), ("RAF1", "protein"), "observation", "PMID:80011"),
        (("KRAS", "gene"), ("inhibits", "relates_to"), ("RAF1", "protein"), "computation", "ML:80012"),
        (("BRCA1", "gene"), ("activates", "relates_to"), ("cell_cycle", "entity"), "observation", "PMID:80013"),
        (("BRCA1", "gene"), ("activates", "relates_to"), ("cell_cycle", "entity"), "computation", "ML:80014"),
        (("Imatinib", "compound"), ("activates", "relates_to"), ("BCR-ABL", "protein"), "observation", "PMID:80015"),
        (("Imatinib", "compound"), ("activates", "relates_to"), ("BCR-ABL", "protein"), "computation", "ML:80016"),
        (("Metformin", "compound"), ("activates", "relates_to"), ("mTOR", "protein"), "observation", "PMID:80017"),
        (("Metformin", "compound"), ("activates", "relates_to"), ("mTOR", "protein"), "computation", "ML:80018"),
        (("VEGF", "protein"), ("inhibits", "relates_to"), ("angiogenesis", "entity"), "observation", "PMID:80019"),
        (("VEGF", "protein"), ("inhibits", "relates_to"), ("angiogenesis", "entity"), "computation", "ML:80020"),
        # Upregulates/downregulates contradictions
        (("TREM2", "protein"), ("downregulates", "relates_to"), ("NF-kB", "protein"), "observation", "PMID:80021"),
        (("BRAF", "gene"), ("downregulates", "relates_to"), ("MEK1", "protein"), "observation", "PMID:80022"),
        (("EGFR", "protein"), ("downregulates", "relates_to"), ("RAS", "protein"), "observation", "PMID:80023"),
        (("JAK2", "protein"), ("downregulates", "relates_to"), ("STAT3", "protein"), "observation", "PMID:80024"),
        (("KRAS", "gene"), ("downregulates", "relates_to"), ("RAF1", "protein"), "observation", "PMID:80025"),
        # More contradictions from different sources
        (("TREM2", "protein"), ("inhibits", "relates_to"), ("NF-kB", "protein"), "llm_inference", "agent_80001"),
        (("BRAF", "gene"), ("inhibits", "relates_to"), ("MEK1", "protein"), "llm_inference", "agent_80002"),
        (("TP53", "gene"), ("activates", "relates_to"), ("MDM2", "protein"), "llm_inference", "agent_80003"),
        (("EGFR", "protein"), ("inhibits", "relates_to"), ("RAS", "protein"), "llm_inference", "agent_80004"),
        (("JAK2", "protein"), ("inhibits", "relates_to"), ("STAT3", "protein"), "llm_inference", "agent_80005"),
        (("KRAS", "gene"), ("inhibits", "relates_to"), ("RAF1", "protein"), "llm_inference", "agent_80006"),
        (("Imatinib", "compound"), ("activates", "relates_to"), ("BCR-ABL", "protein"), "llm_inference", "agent_80007"),
        (("Metformin", "compound"), ("activates", "relates_to"), ("mTOR", "protein"), "llm_inference", "agent_80008"),
        (("VEGF", "protein"), ("inhibits", "relates_to"), ("angiogenesis", "entity"), "llm_inference", "agent_80009"),
        (("BRCA1", "gene"), ("activates", "relates_to"), ("cell_cycle", "entity"), "llm_inference", "agent_80010"),
        # Contradictions from document extraction
        (("TREM2", "protein"), ("inhibits", "relates_to"), ("NF-kB", "protein"), "document_extraction", "DOI:80011"),
        (("BRAF", "gene"), ("inhibits", "relates_to"), ("MEK1", "protein"), "document_extraction", "DOI:80012"),
        (("TP53", "gene"), ("activates", "relates_to"), ("MDM2", "protein"), "document_extraction", "DOI:80013"),
        (("EGFR", "protein"), ("inhibits", "relates_to"), ("RAS", "protein"), "document_extraction", "DOI:80014"),
        (("JAK2", "protein"), ("inhibits", "relates_to"), ("STAT3", "protein"), "document_extraction", "DOI:80015"),
    ]

    for i, (subj, pred, obj, src_type, src_id) in enumerate(contradictions):
        ts += 1
        labeled.append((
            ClaimInput(subject=subj, predicate=pred, object=obj,
                       provenance={"source_type": src_type, "source_id": src_id},
                       timestamp=_ts(ts)),
            "flag",
        ))

    # =========================================================================
    # SKIP: Low-value and uninformative claims (50 claims)
    # =========================================================================

    skippable = [
        # Low-confidence LLM inferences
        *[(("entity_" + str(i), "entity"), ("relates_to", "relates_to"),
           ("entity_" + str(i + 100), "entity"), "llm_inference", f"agent_skip_{i}")
          for i in range(20)],
        # Vague/generic claims
        (("protein_X", "entity"), ("relates_to", "relates_to"), ("protein_Y", "entity"), "llm_inference", "agent_vague_1"),
        (("gene_A", "entity"), ("relates_to", "relates_to"), ("gene_B", "entity"), "llm_inference", "agent_vague_2"),
        (("compound_1", "entity"), ("relates_to", "relates_to"), ("target_1", "entity"), "llm_inference", "agent_vague_3"),
        (("molecule_X", "entity"), ("relates_to", "relates_to"), ("pathway_Y", "entity"), "llm_inference", "agent_vague_4"),
        (("factor_A", "entity"), ("relates_to", "relates_to"), ("process_B", "entity"), "llm_inference", "agent_vague_5"),
        (("receptor_1", "entity"), ("relates_to", "relates_to"), ("ligand_1", "entity"), "llm_inference", "agent_vague_6"),
        (("enzyme_X", "entity"), ("relates_to", "relates_to"), ("substrate_Y", "entity"), "llm_inference", "agent_vague_7"),
        (("kinase_A", "entity"), ("relates_to", "relates_to"), ("target_B", "entity"), "llm_inference", "agent_vague_8"),
        (("channel_1", "entity"), ("relates_to", "relates_to"), ("ion_1", "entity"), "llm_inference", "agent_vague_9"),
        (("transporter_X", "entity"), ("relates_to", "relates_to"), ("cargo_Y", "entity"), "llm_inference", "agent_vague_10"),
        # More low-confidence LLM
        *[(("unknown_" + str(i), "entity"), ("relates_to", "relates_to"),
           ("unknown_" + str(i + 200), "entity"), "llm_inference", f"agent_low_{i}")
          for i in range(20)],
    ]

    for i, (subj, pred, obj, src_type, src_id) in enumerate(skippable):
        ts += 1
        labeled.append((
            ClaimInput(subject=subj, predicate=pred, object=obj,
                       provenance={"source_type": src_type, "source_id": src_id},
                       confidence=0.1,  # Very low confidence
                       timestamp=_ts(ts)),
            "skip",
        ))

    # =========================================================================
    # STORE: High-quality claims that should always be stored (50 more)
    # =========================================================================

    high_quality = [
        # Clinical trial results
        (("Pembrolizumab", "compound"), ("treats", "relates_to"), ("melanoma", "disease"), "observation", "NCT:90001"),
        (("Nivolumab", "compound"), ("treats", "relates_to"), ("lung cancer", "disease"), "observation", "NCT:90002"),
        (("Atezolizumab", "compound"), ("treats", "relates_to"), ("bladder cancer", "disease"), "observation", "NCT:90003"),
        (("Ipilimumab", "compound"), ("treats", "relates_to"), ("melanoma", "disease"), "observation", "NCT:90004"),
        (("Durvalumab", "compound"), ("treats", "relates_to"), ("lung cancer", "disease"), "observation", "NCT:90005"),
        # Structural biology
        (("SARS-CoV-2_spike", "protein"), ("binds", "relates_to"), ("ACE2", "protein"), "experimental", "PDB:90006"),
        (("insulin", "compound"), ("binds", "relates_to"), ("INSR", "protein"), "experimental", "PDB:90007"),
        (("hemoglobin", "protein"), ("binds", "relates_to"), ("oxygen", "entity"), "experimental", "PDB:90008"),
        (("myosin", "protein"), ("binds", "relates_to"), ("actin", "protein"), "experimental", "PDB:90009"),
        (("tubulin", "protein"), ("binds", "relates_to"), ("colchicine", "compound"), "experimental", "PDB:90010"),
        # Metabolic pathways
        (("hexokinase", "protein"), ("activates", "relates_to"), ("glycolysis", "entity"), "database_import", "KEGG:90011"),
        (("pyruvate_kinase", "protein"), ("activates", "relates_to"), ("glycolysis", "entity"), "database_import", "KEGG:90012"),
        (("citrate_synthase", "protein"), ("activates", "relates_to"), ("TCA_cycle", "entity"), "database_import", "KEGG:90013"),
        (("Complex_I", "protein"), ("activates", "relates_to"), ("oxidative_phosphorylation", "entity"), "database_import", "KEGG:90014"),
        (("ATP_synthase", "protein"), ("activates", "relates_to"), ("ATP_production", "entity"), "database_import", "KEGG:90015"),
        # Epigenetics
        (("EZH2", "protein"), ("inhibits", "relates_to"), ("H3K27", "entity"), "observation", "PMID:90016"),
        (("BRD4", "protein"), ("activates", "relates_to"), ("MYC", "gene"), "observation", "PMID:90017"),
        (("KDM5A", "protein"), ("inhibits", "relates_to"), ("H3K4me3", "entity"), "observation", "PMID:90018"),
        (("SETD2", "protein"), ("activates", "relates_to"), ("H3K36me3", "entity"), "observation", "PMID:90019"),
        (("TET2", "protein"), ("activates", "relates_to"), ("DNA_demethylation", "entity"), "observation", "PMID:90020"),
        # Immune system
        (("IL-2", "protein"), ("activates", "relates_to"), ("T_cell", "entity"), "observation", "PMID:90021"),
        (("IFN-gamma", "protein"), ("activates", "relates_to"), ("macrophage", "entity"), "observation", "PMID:90022"),
        (("IL-10", "protein"), ("inhibits", "relates_to"), ("inflammation", "entity"), "observation", "PMID:90023"),
        (("TGF-beta", "protein"), ("inhibits", "relates_to"), ("T_cell", "entity"), "observation", "PMID:90024"),
        (("CTLA-4", "protein"), ("inhibits", "relates_to"), ("T_cell_activation", "entity"), "observation", "PMID:90025"),
        # Autophagy
        (("BECN1", "protein"), ("activates", "relates_to"), ("autophagy", "entity"), "observation", "PMID:90026"),
        (("ATG5", "protein"), ("activates", "relates_to"), ("autophagy", "entity"), "observation", "PMID:90027"),
        (("LC3", "protein"), ("activates", "relates_to"), ("autophagosome", "entity"), "observation", "PMID:90028"),
        (("ULK1", "protein"), ("activates", "relates_to"), ("autophagy", "entity"), "observation", "PMID:90029"),
        (("SQSTM1", "protein"), ("activates", "relates_to"), ("selective_autophagy", "entity"), "observation", "PMID:90030"),
        # More drug targets
        (("lenalidomide", "compound"), ("inhibits", "relates_to"), ("CRBN", "protein"), "experimental", "ChEMBL:90031"),
        (("bortezomib", "compound"), ("inhibits", "relates_to"), ("proteasome", "entity"), "experimental", "ChEMBL:90032"),
        (("thalidomide", "compound"), ("inhibits", "relates_to"), ("CRBN", "protein"), "experimental", "ChEMBL:90033"),
        (("ibrutinib", "compound"), ("inhibits", "relates_to"), ("BTK", "protein"), "experimental", "ChEMBL:90034"),
        (("venetoclax", "compound"), ("inhibits", "relates_to"), ("BCL2", "protein"), "experimental", "ChEMBL:90035"),
        # Neuroscience
        (("dopamine", "compound"), ("binds", "relates_to"), ("DRD2", "protein"), "observation", "PMID:90036"),
        (("serotonin", "compound"), ("binds", "relates_to"), ("HTR2A", "protein"), "observation", "PMID:90037"),
        (("glutamate", "compound"), ("binds", "relates_to"), ("GRIN1", "protein"), "observation", "PMID:90038"),
        (("GABA", "compound"), ("binds", "relates_to"), ("GABRA1", "protein"), "observation", "PMID:90039"),
        (("acetylcholine", "compound"), ("binds", "relates_to"), ("CHRNA7", "protein"), "observation", "PMID:90040"),
        # More signaling
        (("ERBB2", "protein"), ("activates", "relates_to"), ("PI3K", "protein"), "observation", "PMID:90041"),
        (("SRC", "protein"), ("activates", "relates_to"), ("FAK", "protein"), "observation", "PMID:90042"),
        (("ABL1", "protein"), ("activates", "relates_to"), ("CRKL", "protein"), "observation", "PMID:90043"),
        (("FYN", "protein"), ("activates", "relates_to"), ("PAG1", "protein"), "observation", "PMID:90044"),
        (("LCK", "protein"), ("activates", "relates_to"), ("ZAP70", "protein"), "observation", "PMID:90045"),
        # Metabolism
        (("PPARG", "protein"), ("activates", "relates_to"), ("adipogenesis", "entity"), "observation", "PMID:90046"),
        (("SREBP1", "protein"), ("activates", "relates_to"), ("lipogenesis", "entity"), "observation", "PMID:90047"),
        (("LXR", "protein"), ("activates", "relates_to"), ("cholesterol_efflux", "entity"), "observation", "PMID:90048"),
        (("FXR", "protein"), ("inhibits", "relates_to"), ("bile_acid_synthesis", "entity"), "observation", "PMID:90049"),
        (("HNF4A", "protein"), ("activates", "relates_to"), ("gluconeogenesis", "entity"), "observation", "PMID:90050"),
    ]

    for i, (subj, pred, obj, src_type, src_id) in enumerate(high_quality):
        ts += 1
        labeled.append((
            ClaimInput(subject=subj, predicate=pred, object=obj,
                       provenance={"source_type": src_type, "source_id": src_id},
                       timestamp=_ts(ts)),
            "store",
        ))

    return BACKGROUND_KNOWLEDGE, labeled
