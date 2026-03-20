"""Bulk loader for seeding databases from external sources."""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import random
import time

from attestdb.core.confidence import tier1_confidence
from attestdb.core.hashing import compute_claim_id, compute_content_id
from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import BatchResult, ClaimInput
from attestdb.infrastructure.id_mapper import GeneIDMapper
from attestdb.infrastructure.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

# LMDB max key size — entity IDs exceeding this are truncated + hashed.
_MAX_ENTITY_ID_LEN = 500  # leave headroom below LMDB's 511-byte limit


def _safe_entity_id(raw_id: str, *, ext_ids: dict | None = None) -> tuple[str, dict]:
    """Ensure entity ID fits in LMDB key limit.

    If the normalized ID exceeds _MAX_ENTITY_ID_LEN bytes, truncate and append
    a hash suffix.  The original ID is preserved in ext_ids["original_id"].

    Returns (safe_id, updated_ext_ids).
    """
    import hashlib
    if len(raw_id.encode("utf-8")) <= _MAX_ENTITY_ID_LEN:
        return raw_id, ext_ids or {}
    h = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
    # Truncate at a safe byte boundary
    truncated = raw_id.encode("utf-8")[:_MAX_ENTITY_ID_LEN - 18].decode("utf-8", errors="ignore")
    safe = f"{truncated}_{h}"
    updated = dict(ext_ids) if ext_ids else {}
    updated["original_id"] = raw_id
    logger.debug("Truncated oversized entity ID (%d bytes): %s... -> %s",
                 len(raw_id.encode("utf-8")), raw_id[:80], safe[:80])
    return safe, updated

# GitHub raw URL works for LFS files; raw.githubusercontent.com does not
HETIONET_EDGES_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
HETIONET_NODES_URL = "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv"

# External data source URLs
DISEASES_URL = "https://download.jensenlab.org/human_disease_knowledge_full.tsv"
STRING_LINKS_URL = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
REACTOME_URL = "https://reactome.org/download/current/NCBI2Reactome.txt"

# Additional data sources (auth may be required)
DISGENET_CURATED_URL = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/curated_gene_disease_associations.tsv.gz"
CHEMBL_SQLITE_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"

# Open data sources (no auth required)
GOA_HUMAN_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"
CTD_CHEM_GENE_URL = "https://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz"
CTD_GENE_DISEASE_URL = "https://ctdbase.org/reports/CTD_genes_diseases.tsv.gz"

# GO evidence code -> confidence mapping (experimental > curated computational > electronic)
GO_EVIDENCE_CONFIDENCE = {
    # Experimental (high confidence)
    "EXP": 0.95, "IDA": 0.95, "IPI": 0.93, "IMP": 0.92, "IGI": 0.90, "IEP": 0.88,
    # Curated computational (medium confidence)
    "ISS": 0.80, "ISO": 0.80, "ISA": 0.78, "ISM": 0.78, "RCA": 0.78, "IC": 0.75,
    # Automated computational (lower confidence)
    "IBA": 0.70, "IBD": 0.68, "IGC": 0.65, "IKR": 0.65, "IRD": 0.65,
    # Electronic annotation (lowest)
    "IEA": 0.55,
}

# GO aspect code -> entity type
GO_ASPECT_MAP = {
    "P": "biological_process",
    "F": "molecular_function",
    "C": "cellular_component",
}

# BioGRID and ClinVar URLs (fully open, no auth)
BIOGRID_ORGANISM_URL = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab3.zip"
CLINVAR_GENE_CONDITION_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id"

# New open data sources (no auth required)
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
DRKG_URL = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
SIDER_URL = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
HPO_GENES_URL = "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt"
INTACT_URL = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip"
PHARMGKB_URL = "https://api.pharmgkb.org/v1/download/file/data/relationships.zip"
MONDO_SSSOM_URL = "https://raw.githubusercontent.com/monarch-initiative/mondo/master/src/ontology/mappings/mondo.sssom.tsv"

# Phase 1+2 data sources
STITCH_URL = "http://stitch.embl.de/download/protein_chemical.links.v5.0/9606.protein_chemical.links.v5.0.tsv.gz"
CTD_CHEM_DISEASE_URL = "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
TISSUES_URL = "https://download.jensenlab.org/human_tissue_knowledge_full.tsv"
OPEN_TARGETS_ASSOC_URL = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/output/associationByOverallDirect/"
SEMMEDDB_PREDICATIONS_URL = "https://data.lhncbc.nlm.nih.gov/public/SemMedDB/semmedVER43_2024_R_PREDICATION.csv.gz"

# Composite knowledge graphs
KG2C_LITE_URL = "https://media.githubusercontent.com/media/ncats/translator-lfs-artifacts/main/files/kg2c_lite_2.10.0.json.gz"
MONARCH_KG_URL = "https://data.monarchinitiative.org/monarch-kg/latest/monarch-kg.tar.gz"
PHARMEBINET_URL = "https://zenodo.org/record/7011027/files/pharmebinet_tsv_2022_08_19_v2.tar.gz"

# BioGRID experimental system type -> confidence
BIOGRID_CONFIDENCE = {
    "physical": 0.85,
    "genetic": 0.75,
}

# CTD interaction action -> predicate mapping
CTD_ACTION_MAP = {
    "increases": "upregulates",
    "decreases": "downregulates",
    "affects": "regulates",
    "increases^expression": "upregulates",
    "decreases^expression": "downregulates",
    "increases^activity": "activates",
    "decreases^activity": "inhibits",
    "increases^phosphorylation": "upregulates",
    "decreases^phosphorylation": "downregulates",
    "affects^binding": "binds",
    "increases^secretion": "upregulates",
    "decreases^secretion": "downregulates",
    "affects^expression": "regulates",
    "affects^activity": "regulates",
    "increases^metabolic processing": "upregulates",
    "decreases^metabolic processing": "downregulates",
}

# DrugBank action -> predicate mapping
DRUGBANK_ACTION_MAP = {
    "inhibitor": "inhibits",
    "antagonist": "inhibits",
    "blocker": "inhibits",
    "negative modulator": "inhibits",
    "suppressor": "inhibits",
    "agonist": "activates",
    "activator": "activates",
    "inducer": "upregulates",
    "positive modulator": "activates",
    "stimulator": "activates",
    "potentiator": "activates",
    "substrate": "binds",
    "binder": "binds",
    "ligand": "binds",
    "modulator": "regulates",
    "cofactor": "binds",
    "chelator": "binds",
    "carrier": "binds",
}

# PrimeKG display_relation -> Attest predicate
PRIMEKG_PREDICATE_MAP = {
    "target": "interacts",
    "enzyme": "interacts",
    "carrier": "interacts",
    "transporter": "interacts",
    "indication": "treats",
    "contraindication": "contraindicates",
    "off-label use": "treats",
    "associated with": "associated_with",
    "interacts with": "interacts",
    "phenotype present": "associated_with",
    "phenotype absent": "associated_with",
    "pathway": "participates_in",
    "ppi": "interacts",
    "side effect": "causes_side_effect",
    "parent-child": "associated_with",
    "synergistic interaction": "interacts",
    "expression present": "expressed_in",
    "expression absent": "associated_with",
}

# DRKG relation prefix -> Attest predicate
DRKG_PREDICATE_MAP = {
    "DRUGBANK::target": "interacts",
    "DRUGBANK::enzyme": "interacts",
    "DRUGBANK::carrier": "interacts",
    "DRUGBANK::transporter": "interacts",
    "Hetionet::CtD": "treats",
    "Hetionet::CpD": "palliates",
    "Hetionet::CuG": "upregulates",
    "Hetionet::CdG": "downregulates",
    "Hetionet::GiG": "interacts",
    "Hetionet::DaG": "associated_with",
    "Hetionet::CbG": "binds",
    "Hetionet::DuG": "upregulates",
    "Hetionet::DdG": "downregulates",
    "Hetionet::GpBP": "participates_in",
    "Hetionet::GpCC": "participates_in",
    "Hetionet::GpMF": "participates_in",
    "Hetionet::GpPW": "participates_in",
    "GNBR::T": "treats",
    "GNBR::E": "associated_with",
    "GNBR::B": "binds",
    "GNBR::A+": "activates",
    "GNBR::A-": "inhibits",
    "GNBR::N": "inhibits",
    "GNBR::Sa": "associated_with",
    "GNBR::Pr": "associated_with",
    "GNBR::Pa": "associated_with",
    "GNBR::J": "associated_with",
    "GNBR::Mp": "associated_with",
    "GNBR::K": "associated_with",
    "GNBR::Z": "associated_with",
    "GNBR::L": "associated_with",
    "GNBR::G": "associated_with",
    "GNBR::Md": "associated_with",
    "GNBR::C": "associated_with",
    "GNBR::Te": "treats",
    "GNBR::D": "associated_with",
    "GNBR::X": "associated_with",
    "GNBR::Rg": "regulates",
    "GNBR::Q": "associated_with",
    "GNBR::I": "inhibits",
    "GNBR::Y": "associated_with",
    "GNBR::W": "associated_with",
    "GNBR::V+": "activates",
    "GNBR::U": "associated_with",
    "GNBR::O": "associated_with",
    "GNBR::H": "associated_with",
    "GNBR::in-trial": "treats",
    "bioarx::HumGenHumGen": "interacts",
    "bioarx::CcSe": "causes_side_effect",
    "bioarx::DrDi": "treats",
    "bioarx::GpDi": "associated_with",
    "STRING::": "interacts",
    "INTACT::": "interacts",
}

# PharmGKB Evidence level -> confidence
PHARMGKB_EVIDENCE_CONFIDENCE = {
    "4": 0.90,  # Clinical annotation
    "3": 0.85,  # Variant annotation
    "2": 0.80,  # VIP
    "1A": 0.90, # CPIC guideline
    "1B": 0.85, # PharmGKB guideline
    "1": 0.75,  # Pathway
}

# SemMedDB predicate -> Attest predicate
SEMMEDDB_PREDICATE_MAP = {
    "TREATS": "treats",
    "PREVENTS": "prevents",
    "CAUSES": "causes",
    "PREDISPOSES": "predisposes",
    "ASSOCIATED_WITH": "associated_with",
    "AFFECTS": "regulates",
    "AUGMENTS": "upregulates",
    "DISRUPTS": "inhibits",
    "INHIBITS": "inhibits",
    "STIMULATES": "activates",
    "INTERACTS_WITH": "interacts",
    "COEXISTS_WITH": "associated_with",
    "PRODUCES": "associated_with",
    "CONVERTS_TO": "associated_with",
    "ISA": "associated_with",
    "PART_OF": "participates_in",
    "LOCATION_OF": "expressed_in",
    "PROCESS_OF": "associated_with",
    "USES": "associated_with",
    "DIAGNOSES": "associated_with",
    "ADMINISTERED_TO": "treats",
    "METHOD_OF": "associated_with",
    "MEASURES": "associated_with",
    "OCCURS_IN": "associated_with",
    "PRECEDES": "associated_with",
    "COMPLICATES": "associated_with",
    "COMPARED_WITH": "associated_with",
    "HIGHER_THAN": "associated_with",
    "LOWER_THAN": "associated_with",
    "SAME_AS": "equivalent_to",
}

# SemMedDB semantic type -> Attest entity type
SEMMEDDB_SEMTYPE_MAP = {
    "gngm": "gene", "aapp": "protein", "enzy": "protein",
    "phsu": "compound", "orch": "compound", "antb": "compound",
    "clnd": "compound", "hops": "compound", "strd": "compound",
    "dsyn": "disease", "neop": "disease", "patf": "disease",
    "sosy": "phenotype", "fndg": "phenotype", "lbtr": "phenotype",
    "bpoc": "entity", "tisu": "entity", "cell": "entity",
    "moft": "biological_process", "biof": "biological_process",
    "celf": "biological_process", "genf": "biological_process",
    "orgf": "biological_process", "phsf": "biological_process",
}

# PharMeBINet edge type verb -> Attest predicate
# Edge types are like TREATS_CHtD, BINDS_CHbG — we match on the verb prefix
PHARMEBINET_VERB_MAP = {
    # Verb forms (from PharMeBINet edge type names)
    "TREATS": "treats",
    "PALLIATES": "palliates",
    "PREVENTS": "prevents",
    "CONTRAINDICATES": "contraindicates",
    "CAUSES": "causes_side_effect",
    "MIGHT_CAUSES": "causes_side_effect",
    "INDUCES": "causes",
    "MIGHT_INDUCES": "causes",
    "BINDS": "binds",
    "INHIBITS": "inhibits",
    "INTERACTS": "interacts",
    "ASSOCIATES": "associated_with",
    "COVARIES": "associated_with",
    "RESEMBLES": "associated_with",
    "PRESENTS": "associated_with",
    "PRODUCES": "associated_with",
    "EXPRESSES": "expresses",
    "LOCALIZES": "localizes",
    "UPREGULATES": "upregulates",
    "DOWNREGULATES": "downregulates",
    "REGULATES": "regulates",
    "INVOLVED_IN": "participates_in",
    "ENABLES": "participates_in",
    "CONTRIBUTES_TO": "participates_in",
    "PARTICIPATES_IN": "participates_in",
    "PART_OF": "participates_in",
    "IS_ACTIVE_IN": "participates_in",
    "LOCATED_IN": "localizes",
    "COLOCALIZES_WITH": "localizes",
    "ACTS_UPSTREAM_OF": "regulates",
    "IS_A": "associated_with",
    "EQUAL": "equivalent_to",
    "LEADS_TO": "causes",
    "DEGENERATES": "inhibits",
    "METABOLIZES": "interacts",
    "INCREASES": "upregulates",
    "DECREASES": "downregulates",
    "AFFECTS": "regulates",
    "HAS": "associated_with",
    "BELONGS_TO": "associated_with",
    "INCLUDES": "includes",
    "MAY_DIAGNOSES": "associated_with",
    "COMBINATION_CAUSES_ADR": "causes_side_effect",
    # Noun forms (PharMeBINet uses ASSOCIATION_GaD, INTERACTION_GiG, etc.)
    "ASSOCIATION": "associated_with",
    "INTERACTION": "interacts",
    "SIDE_EFFECT": "causes_side_effect",
    "TARGET": "interacts",
    "PARTICIPATES": "participates_in",
    "INDICATION": "treats",
    "CONTRAINDICATION": "contraindicates",
}

# PharMeBINet node label -> Attest entity type
PHARMEBINET_LABEL_MAP = {
    "Gene": "gene",
    "Protein": "protein",
    "Chemical": "compound",
    "Compound": "compound",
    "Drug": "compound",
    "Salt": "compound",
    "Disease": "disease",
    "Phenotype": "phenotype",
    "Symptom": "phenotype",
    "SideEffect": "phenotype",
    "BiologicalProcess": "biological_process",
    "MolecularFunction": "molecular_function",
    "CellularComponent": "cellular_component",
    "Pathway": "pathway",
    "Anatomy": "entity",
    "PharmacologicClass": "entity",
    "GeneVariant": "gene",
    "Variant": "gene",
    "Haplotype": "gene",
    "Reaction": "entity",
    "ReactionLikeEvent": "entity",
    "Treatment": "compound",
    "Product": "compound",
    "Interaction": "entity",
    "ClinicalAnnotation": "entity",
    "VariantAnnotation": "entity",
    "RNA": "gene",
    "Target": "protein",
    "Metabolite": "compound",
}

# Biolink category -> Attest entity type (shared by KG2c, Monarch KG)
BIOLINK_CATEGORY_MAP = {
    "biolink:Gene": "gene",
    "biolink:Protein": "protein",
    "biolink:Disease": "disease",
    "biolink:PhenotypicFeature": "phenotype",
    "biolink:ChemicalEntity": "compound",
    "biolink:SmallMolecule": "compound",
    "biolink:Drug": "compound",
    "biolink:MolecularMixture": "compound",
    "biolink:BiologicalProcess": "biological_process",
    "biolink:MolecularActivity": "molecular_function",
    "biolink:CellularComponent": "cellular_component",
    "biolink:Pathway": "pathway",
    "biolink:Cell": "cell_line",
    "biolink:AnatomicalEntity": "entity",
    "biolink:GrossAnatomicalStructure": "entity",
    "biolink:OrganismTaxon": "organism",
    "biolink:SequenceVariant": "gene",
    "biolink:Genotype": "gene",
    "biolink:ClinicalFinding": "phenotype",
    "biolink:NamedThing": "entity",
    "biolink:PhysiologicalProcess": "biological_process",
    "biolink:NucleicAcidEntity": "gene",
    "biolink:Treatment": "compound",
    "biolink:DrugExposure": "compound",
    "biolink:MolecularEntity": "entity",
    "biolink:InformationContentEntity": "entity",
    "biolink:OntologyClass": "entity",
    "biolink:LifeStage": "entity",
    "biolink:Case": "entity",
}

# Biolink predicate -> Attest predicate (shared by KG2c, Monarch KG)
BIOLINK_PREDICATE_MAP = {
    # Causal / mechanistic
    "biolink:treats": "treats",
    "biolink:treats_or_applied_or_studied_to_treat": "treats",
    "biolink:ameliorates_condition": "treats",
    "biolink:preventative_for_condition": "prevents",
    "biolink:causes": "causes",
    "biolink:caused_by": "causes",
    "biolink:contributes_to": "associated_with",
    "biolink:contraindicated_in": "contraindicates",
    "biolink:disrupts": "inhibits",
    # Regulatory
    "biolink:regulates": "regulates",
    "biolink:positively_regulates": "upregulates",
    "biolink:negatively_regulates": "downregulates",
    "biolink:affects": "regulates",
    "biolink:acts_upstream_of": "regulates",
    "biolink:acts_upstream_of_positive_effect": "upregulates",
    "biolink:acts_upstream_of_negative_effect": "downregulates",
    "biolink:acts_upstream_of_or_within": "regulates",
    "biolink:acts_upstream_of_or_within_positive_effect": "upregulates",
    "biolink:acts_upstream_of_or_within_negative_effect": "downregulates",
    # Interaction / physical
    "biolink:interacts_with": "interacts",
    "biolink:physically_interacts_with": "interacts",
    "biolink:binds": "binds",
    # Association
    "biolink:associated_with": "associated_with",
    "biolink:correlated_with": "associated_with",
    "biolink:gene_associated_with_condition": "associated_with",
    "biolink:associated_with_increased_likelihood_of": "predisposes",
    "biolink:genetically_associated_with": "associated_with",
    "biolink:has_phenotype": "associated_with",
    "biolink:has_disease": "associated_with",
    "biolink:has_gene": "associated_with",
    "biolink:disease_has_location": "localizes",
    "biolink:model_of": "associated_with",
    "biolink:has_mode_of_inheritance": "associated_with",
    # Functional / localization
    "biolink:enables": "participates_in",
    "biolink:actively_involved_in": "participates_in",
    "biolink:participates_in": "participates_in",
    "biolink:is_active_in": "participates_in",
    "biolink:expressed_in": "expressed_in",
    "biolink:located_in": "localizes",
    "biolink:colocalizes_with": "localizes",
    # Structural / equivalence
    "biolink:has_part": "includes",
    "biolink:part_of": "participates_in",
    "biolink:subclass_of": "associated_with",
    "biolink:superclass_of": "associated_with",
    "biolink:same_as": "equivalent_to",
    "biolink:close_match": "equivalent_to",
    "biolink:broad_match": "associated_with",
    "biolink:related_to": "associated_with",
    # Variant
    "biolink:is_sequence_variant_of": "variant_of",
    "biolink:has_sequence_variant": "variant_of",
    # Orthology
    "biolink:orthologous_to": "homolog_of",
    "biolink:homologous_to": "homolog_of",
    "biolink:has_participant": "interacts",
}

# Hetionet metaedge abbreviations -> readable predicates
METAEDGE_MAP = {
    "CbG": "binds",
    "CdG": "downregulates",
    "CuG": "upregulates",
    "CtD": "treats",
    "CpD": "palliates",
    "DrD": "resembles",
    "DaG": "associates",
    "DdG": "downregulates",
    "DuG": "upregulates",
    "DlA": "localizes",
    "GiG": "interacts",
    "Gr>G": "regulates",
    "GpBP": "participates_in",
    "GpCC": "participates_in",
    "GpMF": "participates_in",
    "GpPW": "participates_in",
    "PCiC": "includes",
    "AeG": "expresses",
    "AdG": "downregulates",
    "AuG": "upregulates",
}

# Hetionet Kind prefix -> entity_type
KIND_MAP = {
    "Gene": "gene",
    "Compound": "compound",
    "Disease": "disease",
    "Anatomy": "entity",
    "Biological Process": "entity",
    "Cellular Component": "entity",
    "Molecular Function": "entity",
    "Pathway": "entity",
    "Pharmacologic Class": "entity",
    "Side Effect": "entity",
    "Symptom": "entity",
}


def _parse_hetionet_id(raw: str) -> tuple[str, str]:
    """Parse 'Kind::ID' format into (display_name, entity_type).

    Example: 'Gene::9021' -> ('Gene_9021', 'gene')
             'Compound::DB00563' -> ('Compound_DB00563', 'compound')
    """
    raw = raw.strip()
    if "::" in raw:
        kind, eid = raw.split("::", 1)
        entity_type = KIND_MAP.get(kind, "entity")
        display_name = f"{kind}_{eid}"
        return display_name, entity_type
    return raw, "entity"


def _extract_external_ids(display_name: str, entity_type: str) -> dict[str, str]:
    """Extract external IDs from a Hetionet-style entity display name.

    Gene_9021 -> {"ncbi_gene": "9021"}
    Compound_DB00563 -> {"drugbank": "DB00563"}
    Disease_DOID:14330 -> {"doid": "DOID:14330"}
    """
    if "_" not in display_name:
        return {}
    kind, eid = display_name.split("_", 1)
    kind_lower = kind.lower()
    if kind_lower == "gene" and eid.isdigit():
        return {"ncbi_gene": eid}
    if kind_lower == "compound" and eid.startswith("DB"):
        return {"drugbank": eid}
    if kind_lower == "disease" and eid.startswith("DOID:"):
        return {"doid": eid}
    return {}


class BulkLoader:
    """Bulk loading utilities for Hetionet and ChEMBL."""

    def __init__(self, pipeline: IngestionPipeline):
        self._pipeline = pipeline

    def download_hetionet(self, dest_dir: str) -> str:
        """Download Hetionet edges SIF.gz. Returns path to decompressed TSV."""
        import requests

        dest = os.path.join(dest_dir, "hetionet-edges.tsv")
        if os.path.exists(dest):
            logger.info("Hetionet already downloaded: %s", dest)
            return dest

        logger.info("Downloading Hetionet edges from %s", HETIONET_EDGES_URL)
        resp = requests.get(HETIONET_EDGES_URL, timeout=120)
        resp.raise_for_status()

        # Decompress gzip and save as TSV
        data = gzip.decompress(resp.content).decode("utf-8")
        with open(dest, "w") as f:
            f.write(data)

        logger.info("Downloaded Hetionet to %s (%d lines)", dest, data.count("\n"))
        return dest

    def download_hetionet_nodes(self, dest_dir: str) -> str:
        """Download Hetionet nodes TSV. Returns path to downloaded file."""
        import requests

        dest = os.path.join(dest_dir, "hetionet-nodes.tsv")
        if os.path.exists(dest):
            return dest

        resp = requests.get(HETIONET_NODES_URL, timeout=120)
        resp.raise_for_status()
        with open(dest, "w") as f:
            f.write(resp.text)
        return dest

    def _parse_hetionet_edges(
        self, path: str,
    ) -> list[tuple[str, str, str, str, str, str]]:
        """Parse Hetionet SIF file into raw edge tuples.

        Returns list of (source_name, source_type, predicate, target_name, target_type, metaedge).
        """
        raw_edges: list[tuple[str, str, str, str, str, str]] = []
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # Skip header

            for row in reader:
                if len(row) < 3:
                    continue
                source_raw, metaedge, target_raw = row[0], row[1], row[2]
                source_name, source_type = _parse_hetionet_id(source_raw)
                target_name, target_type = _parse_hetionet_id(target_raw)
                predicate = METAEDGE_MAP.get(metaedge, metaedge.lower())
                raw_edges.append((source_name, source_type, predicate,
                                  target_name, target_type, metaedge))
        return raw_edges

    @staticmethod
    def _edges_to_claims(
        raw_edges: list[tuple[str, str, str, str, str, str]],
    ) -> list[ClaimInput]:
        """Convert raw edge tuples to ClaimInput objects."""
        return [
            ClaimInput(
                subject=(src, stype),
                predicate=(pred, "relates_to"),
                object=(tgt, ttype),
                provenance={"source_type": "computation", "source_id": "hetionet"},
            )
            for src, stype, pred, tgt, ttype, _ in raw_edges
        ]

    def load_hetionet(
        self,
        path: str,
        holdout_fraction: float = 0.1,
        max_edges: int | None = None,
        seed: int = 42,
    ) -> tuple[BatchResult, list[ClaimInput]]:
        """Load Hetionet edges, splitting a holdout set for evaluation.

        Parses SIF format: source<TAB>metaedge<TAB>target
        Where source/target are 'Kind::ID' strings.

        When max_edges is set, uses snowball sampling to extract a dense
        subgraph (high avg degree) rather than random edge sampling which
        produces extremely sparse subgraphs unsuitable for link prediction.

        Returns (BatchResult for ingested edges, list of withheld ClaimInput).
        """
        t0 = time.time()
        rng = random.Random(seed)
        raw_edges = self._parse_hetionet_edges(path)
        timestamp = int(time.time() * 1_000_000_000)

        if max_edges and len(raw_edges) > max_edges:
            raw_edges = self._snowball_sample(raw_edges, max_edges, rng)

        # Split holdout on raw edges before building claim rows
        rng.shuffle(raw_edges)
        holdout_count = int(len(raw_edges) * holdout_fraction)
        withheld_edges = raw_edges[:holdout_count]
        ingest_edges = raw_edges[holdout_count:]

        # Convert withheld to ClaimInput for eval harness
        withheld = self._edges_to_claims(withheld_edges)

        logger.info(
            "Hetionet: %d total edges, %d to ingest, %d withheld",
            len(raw_edges),
            len(ingest_edges),
            len(withheld),
        )

        # Build entities dict and claim_rows for fast Rust batch insert
        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []

        for src, stype, pred, tgt, ttype, _ in ingest_edges:
            subj_canonical = normalize_entity_id(src)
            obj_canonical = normalize_entity_id(tgt)

            ext_ids_src = _extract_external_ids(src, stype)
            ext_ids_tgt = _extract_external_ids(tgt, ttype)

            entities[subj_canonical] = (stype, src, json.dumps(ext_ids_src) if ext_ids_src else "{}")
            entities[obj_canonical] = (ttype, tgt, json.dumps(ext_ids_tgt) if ext_ids_tgt else "{}")

            source_id = f"hetionet:{src}:{pred}:{tgt}"
            claim_id = compute_claim_id(subj_canonical, pred, obj_canonical, source_id, "database_import", timestamp)
            content_id = compute_content_id(subj_canonical, pred, obj_canonical)
            confidence = tier1_confidence("database_import")

            claim_rows.append((
                subj_canonical, obj_canonical, claim_id, content_id,
                pred, "relates_to", confidence,
                "database_import", source_id, "", "[]", "", "", "",
                timestamp, "active",
            ))

        if not claim_rows:
            return BatchResult(ingested=0), withheld

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "Hetionet loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result, withheld

    @staticmethod
    def _snowball_sample(
        raw_edges: list[tuple[str, str, str, str, str, str]],
        max_edges: int,
        rng: random.Random,
    ) -> list[tuple[str, str, str, str, str, str]]:
        """Extract a dense subgraph via ego network selection.

        Finds the entity whose 1-hop ego network (entity + all neighbors)
        has the most internal edges close to max_edges, producing a naturally
        dense subgraph with high average degree.
        """
        # Build adjacency from all edges
        adj: dict[str, set[str]] = {}
        for src, _, _, tgt, _, _ in raw_edges:
            adj.setdefault(src, set()).add(tgt)
            adj.setdefault(tgt, set()).add(src)

        # Find candidate hub entities (degree 50-500) and score by ego density
        candidates = [e for e in adj if 50 <= len(adj[e]) <= 500]
        rng.shuffle(candidates)

        best_entity = None
        best_score = float("inf")

        for entity in candidates[:200]:  # Sample up to 200 candidates
            ego = {entity} | adj[entity]
            if len(ego) > 1000:
                continue
            # Count internal edges (fast approximation: sum degrees within ego / 2)
            n_internal = 0
            for e in ego:
                n_internal += len(adj[e] & ego)
            n_internal //= 2  # Each edge counted twice

            # Score: how close to max_edges (prefer slightly over)
            score = abs(n_internal - max_edges)
            if n_internal >= max_edges * 0.5 and score < best_score:
                best_score = score
                best_entity = entity

        if best_entity is None:
            # Fallback: just use the highest-degree entity with <1000 neighbors
            for e in sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True):
                if len(adj[e]) < 1000:
                    best_entity = e
                    break
            if best_entity is None:
                best_entity = next(iter(adj))

        # Collect ego network
        ego = {best_entity} | adj[best_entity]
        selected_edges = [
            e for e in raw_edges
            if e[0] in ego and e[3] in ego
        ]

        # Truncate if needed
        if len(selected_edges) > max_edges:
            rng.shuffle(selected_edges)
            selected_edges = selected_edges[:max_edges]

        # Compute stats for logging
        sub_adj: dict[str, set[str]] = {}
        for src, _, _, tgt, _, _ in selected_edges:
            sub_adj.setdefault(src, set()).add(tgt)
            sub_adj.setdefault(tgt, set()).add(src)
        n_ents = len(sub_adj)
        degrees = [len(v) for v in sub_adj.values()] if sub_adj else [0]
        avg_deg = sum(degrees) / len(degrees) if degrees else 0

        logger.info(
            "Ego network sample (hub=%s): %d entities, %d edges, avg degree %.1f",
            best_entity[:40], n_ents, len(selected_edges), avg_deg,
        )
        return selected_edges

    # --- Multi-source downloads ---

    def download_diseases(self, dest_dir: str) -> str:
        """Download DISEASES knowledge TSV. Returns path to downloaded file."""
        import requests

        dest = os.path.join(dest_dir, "human_disease_knowledge_full.tsv")
        if os.path.exists(dest):
            logger.info("DISEASES already downloaded: %s", dest)
            return dest

        logger.info("Downloading DISEASES from %s", DISEASES_URL)
        resp = requests.get(DISEASES_URL, timeout=120)
        resp.raise_for_status()
        with open(dest, "w") as f:
            f.write(resp.text)
        logger.info("Downloaded DISEASES to %s (%d lines)", dest, resp.text.count("\n"))
        return dest

    def download_string_links(self, dest_dir: str) -> str:
        """Download STRING PPI links (human, compressed). Returns path to decompressed file."""
        import requests

        dest = os.path.join(dest_dir, "9606.protein.links.txt")
        if os.path.exists(dest):
            logger.info("STRING already downloaded: %s", dest)
            return dest

        logger.info("Downloading STRING PPI from %s", STRING_LINKS_URL)
        resp = requests.get(STRING_LINKS_URL, timeout=300)
        resp.raise_for_status()

        data = gzip.decompress(resp.content).decode("utf-8")
        with open(dest, "w") as f:
            f.write(data)
        logger.info("Downloaded STRING to %s (%d lines)", dest, data.count("\n"))
        return dest

    def download_reactome(self, dest_dir: str) -> str:
        """Download Reactome NCBI2Reactome mapping. Returns path to downloaded file."""
        import requests

        dest = os.path.join(dest_dir, "NCBI2Reactome.txt")
        if os.path.exists(dest):
            logger.info("Reactome already downloaded: %s", dest)
            return dest

        logger.info("Downloading Reactome from %s", REACTOME_URL)
        resp = requests.get(REACTOME_URL, timeout=120)
        resp.raise_for_status()
        with open(dest, "w") as f:
            f.write(resp.text)
        logger.info("Downloaded Reactome to %s (%d lines)", dest, resp.text.count("\n"))
        return dest

    # --- Multi-source loaders ---

    def load_diseases(self, path: str, mapper: GeneIDMapper) -> BatchResult:
        """Load DISEASES gene-disease associations via bulk import.

        Format (no header): ENSP  symbol  DOID  disease_name  source_db  curation  score
        Maps ENSP -> Gene_{entrez} via mapper; falls back to symbol -> entrez.
        Disease entity: Disease_DOID:{id} (matches Hetionet format).
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        mapped = 0
        unmapped = 0

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 7:
                    continue
                ensp = row[0].strip()
                symbol = row[1].strip()
                doid = row[2].strip()
                disease_name = row[3].strip()

                if not doid or not ensp:
                    continue

                # Map ENSP -> Gene entity ID
                gene_id = mapper.ensp_to_gene_entity_id(ensp)
                if gene_id is None and symbol:
                    gene_id = mapper.symbol_to_gene_entity_id(symbol)
                if gene_id is None:
                    unmapped += 1
                    continue
                mapped += 1

                # Disease entity matches Hetionet: Disease_DOID:xxxx
                disease_entity = f"Disease_{doid}"

                gene_canonical = normalize_entity_id(gene_id)
                disease_canonical = normalize_entity_id(disease_entity)

                if gene_canonical not in entities:
                    gene_ext = _extract_external_ids(gene_id, "gene")
                    if ensp:
                        gene_ext["ensembl_protein"] = ensp
                    if symbol:
                        gene_ext["symbol"] = symbol
                    ext_json = json.dumps(gene_ext) if gene_ext else "{}"
                    entities[gene_canonical] = ("gene", gene_id, ext_json)
                if disease_canonical not in entities:
                    disease_ext = _extract_external_ids(disease_entity, "disease")
                    d_name = disease_name or disease_entity
                    d_ext = json.dumps(disease_ext) if disease_ext else "{}"
                    entities[disease_canonical] = ("disease", d_name, d_ext)

                pred_id = "associates"
                source_id = "diseases_knowledge"
                source_type = "database_import"

                claim_id = compute_claim_id(
                    gene_canonical, pred_id, disease_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        gene_canonical, pred_id, disease_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(gene_canonical, pred_id, disease_canonical)

                claim_rows.append((
                    gene_canonical, disease_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("DISEASES: %d mapped, %d unmapped genes", mapped, unmapped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "DISEASES loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    def load_string_ppi(
        self, path: str, mapper: GeneIDMapper, min_score: int = 700,
    ) -> BatchResult:
        """Load STRING protein-protein interactions via bulk import.

        Format (space-separated, header on line 1): protein1 protein2 combined_score
        Filters combined_score >= min_score.
        Maps both proteins via mapper.string_protein_to_gene_entity_id().
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        mapped = 0
        unmapped = 0

        with open(path, "r") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                protein1, protein2 = parts[0], parts[1]
                try:
                    score = int(parts[2])
                except ValueError:
                    continue

                if score < min_score:
                    continue

                gene_a = mapper.string_protein_to_gene_entity_id(protein1)
                gene_b = mapper.string_protein_to_gene_entity_id(protein2)
                if gene_a is None or gene_b is None:
                    unmapped += 1
                    continue
                mapped += 1

                gene_a_canonical = normalize_entity_id(gene_a)
                gene_b_canonical = normalize_entity_id(gene_b)

                # Skip self-loops
                if gene_a_canonical == gene_b_canonical:
                    continue

                if gene_a_canonical not in entities:
                    gene_a_ext = _extract_external_ids(gene_a, "gene")
                    gene_a_ext["ensembl_protein"] = protein1
                    entities[gene_a_canonical] = ("gene", gene_a, json.dumps(gene_a_ext))
                if gene_b_canonical not in entities:
                    gene_b_ext = _extract_external_ids(gene_b, "gene")
                    gene_b_ext["ensembl_protein"] = protein2
                    entities[gene_b_canonical] = ("gene", gene_b, json.dumps(gene_b_ext))

                pred_id = "interacts"
                source_id = "string_ppi"
                source_type = "database_import"

                claim_id = compute_claim_id(
                    gene_a_canonical, pred_id, gene_b_canonical, source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        gene_a_canonical, pred_id, gene_b_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(gene_a_canonical, pred_id, gene_b_canonical)

                claim_rows.append((
                    gene_a_canonical, gene_b_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("STRING: %d mapped, %d unmapped protein pairs", mapped, unmapped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "STRING loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    def load_reactome(self, path: str) -> BatchResult:
        """Load Reactome gene-pathway memberships via bulk import.

        Format (TSV, no header): GeneID  ReactomeID  URL  PathwayName  EvidenceCode  Species
        Filters Species == 'Homo sapiens'.
        Gene entity: Gene_{GeneID} (direct Entrez, no mapping needed).
        Pathway entity: Pathway_{ReactomeID}.
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 6:
                    continue
                gene_id_raw, reactome_id, _url, pathway_name, _evidence, species = (
                    row[0].strip(), row[1].strip(), row[2].strip(),
                    row[3].strip(), row[4].strip(), row[5].strip(),
                )

                if species != "Homo sapiens":
                    continue
                if not gene_id_raw or not reactome_id:
                    continue

                gene_entity = f"Gene_{gene_id_raw}"
                pathway_entity = f"Pathway_{reactome_id}"

                gene_canonical = normalize_entity_id(gene_entity)
                pathway_canonical = normalize_entity_id(pathway_entity)

                if gene_canonical not in entities:
                    entities[gene_canonical] = ("gene", gene_entity, "{}")
                if pathway_canonical not in entities:
                    entities[pathway_canonical] = ("pathway", pathway_name or pathway_entity, "{}")

                pred_id = "participates_in"
                source_id = "reactome"
                source_type = "pathway_database"

                claim_id = compute_claim_id(
                    gene_canonical, pred_id, pathway_canonical, source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        gene_canonical, pred_id, pathway_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(gene_canonical, pred_id, pathway_canonical)

                claim_rows.append((
                    gene_canonical, pathway_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "Reactome loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # --- New data source downloads ---

    def download_disgenet(self, dest_dir: str) -> str | None:
        """Download DisGeNET curated gene-disease associations.

        Requires DISGENET_EMAIL + DISGENET_PASSWORD env vars for authentication.
        Returns path to downloaded TSV file, or None if auth unavailable.
        """
        import requests

        dest = os.path.join(dest_dir, "curated_gene_disease_associations.tsv.gz")
        if os.path.exists(dest):
            logger.info("DisGeNET already downloaded: %s", dest)
            return dest

        email = os.environ.get("DISGENET_EMAIL")
        password = os.environ.get("DISGENET_PASSWORD")
        api_key = os.environ.get("DISGENET_API_KEY")

        if api_key:
            # Use API key directly
            headers = {"Authorization": f"Bearer {api_key}"}
        elif email and password:
            # Get auth token via API
            try:
                auth_resp = requests.post(
                    "https://www.disgenet.org/api/auth/",
                    data={"email": email, "password": password},
                    timeout=30,
                )
                auth_resp.raise_for_status()
                token = auth_resp.json().get("token")
                if not token:
                    logger.warning("DisGeNET auth returned no token")
                    return None
                headers = {"Authorization": f"Bearer {token}"}
            except Exception as e:
                logger.warning("DisGeNET auth failed: %s", e)
                return None
        else:
            logger.warning(
                "DisGeNET requires DISGENET_API_KEY or DISGENET_EMAIL+DISGENET_PASSWORD. "
                "Register at https://www.disgenet.org/ and set env vars."
            )
            return None

        logger.info("Downloading DisGeNET curated GDAs...")
        try:
            resp = requests.get(DISGENET_CURATED_URL, headers=headers, timeout=300)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            logger.info("Downloaded DisGeNET to %s (%d bytes)", dest, len(resp.content))
            return dest
        except Exception as e:
            logger.warning("DisGeNET download failed: %s", e)
            return None

    def download_drugbank_targets(self, dest_dir: str) -> str | None:
        """Download DrugBank target polypeptide CSV.

        Requires DRUGBANK_EMAIL + DRUGBANK_PASSWORD env vars.
        Returns path to downloaded CSV file, or None if auth unavailable.
        """
        import requests
        import zipfile
        import io

        dest = os.path.join(dest_dir, "drugbank_targets.csv")
        if os.path.exists(dest):
            logger.info("DrugBank targets already downloaded: %s", dest)
            return dest

        email = os.environ.get("DRUGBANK_EMAIL")
        password = os.environ.get("DRUGBANK_PASSWORD")
        if not email or not password:
            logger.warning(
                "DrugBank requires DRUGBANK_EMAIL + DRUGBANK_PASSWORD. "
                "Register at https://go.drugbank.com/academic_research and set env vars."
            )
            return None

        url = "https://go.drugbank.com/releases/latest/downloads/target-all-polypeptide-ids"
        logger.info("Downloading DrugBank targets...")
        try:
            resp = requests.get(url, auth=(email, password), timeout=120)
            resp.raise_for_status()

            # DrugBank downloads are zip files containing CSV
            if resp.headers.get("content-type", "").startswith("application/zip") or resp.content[:4] == b"PK\x03\x04":
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if csv_names:
                        with open(dest, "wb") as f:
                            f.write(zf.read(csv_names[0]))
                    else:
                        logger.warning("DrugBank zip contains no CSV files")
                        return None
            else:
                with open(dest, "wb") as f:
                    f.write(resp.content)

            logger.info("Downloaded DrugBank targets to %s", dest)
            return dest
        except Exception as e:
            logger.warning("DrugBank download failed: %s", e)
            return None

    def download_chembl(self, dest_dir: str) -> str | None:
        """Download ChEMBL SQLite database (no auth required, ~3-4GB).

        Returns path to extracted SQLite file, or None on failure.
        """
        import requests
        import tarfile

        dest = os.path.join(dest_dir, "chembl.db")
        if os.path.exists(dest):
            logger.info("ChEMBL already downloaded: %s", dest)
            return dest

        # Find the SQLite tar.gz URL from the latest directory
        tar_path = os.path.join(dest_dir, "chembl_sqlite.tar.gz")
        if not os.path.exists(tar_path):
            # Try to find the actual filename from the latest directory listing
            logger.info("Downloading ChEMBL SQLite (~3-4GB, this will take a while)...")
            # Try recent ChEMBL versions (latest first)
            urls_to_try = [
                f"{CHEMBL_SQLITE_BASE_URL}chembl_36_sqlite.tar.gz",
                f"{CHEMBL_SQLITE_BASE_URL}chembl_35_sqlite.tar.gz",
                f"{CHEMBL_SQLITE_BASE_URL}chembl_37_sqlite.tar.gz",
            ]
            downloaded = False
            for url in urls_to_try:
                try:
                    resp = requests.get(url, timeout=30, stream=True)
                    if resp.status_code == 200:
                        total = int(resp.headers.get("content-length", 0))
                        logger.info("Downloading %s (%.1f GB)...", url, total / 1e9)
                        with open(tar_path, "wb") as f:
                            downloaded_bytes = 0
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)
                                downloaded_bytes += len(chunk)
                                if total and downloaded_bytes % (100 * 1024 * 1024) < 1024 * 1024:
                                    logger.info("  %.0f%%", downloaded_bytes / total * 100)
                        downloaded = True
                        break
                except Exception as e:
                    logger.warning("ChEMBL download from %s failed: %s", url, e)
                    continue

            if not downloaded:
                logger.warning("Could not download ChEMBL SQLite from any URL")
                return None

        # Extract SQLite from tar.gz
        logger.info("Extracting ChEMBL SQLite...")
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.name.endswith(".db"):
                        member.name = "chembl.db"
                        tf.extract(member, dest_dir)
                        break
                else:
                    logger.warning("No .db file found in ChEMBL tar.gz")
                    return None
        except Exception as e:
            logger.warning("ChEMBL extraction failed: %s", e)
            return None

        logger.info("ChEMBL extracted to %s", dest)
        return dest

    # --- New data source loaders ---

    def load_disgenet(self, path: str) -> BatchResult:
        """Load DisGeNET curated gene-disease associations via bulk import.

        Format (TSV with header): geneId  geneSymbol  DSI  DPI  diseaseId  diseaseName
            diseaseType  diseaseClass  diseaseSemanticType  score  EI  YearInitial
            YearFinal  NofPmids  NofSnps  source

        Gene entities use direct Entrez ID mapping (same as Hetionet).
        Disease entities use UMLS CUI (Disease_UMLS:{CUI}) — separate namespace
        from Hetionet DOID entities, enabling cross-source corroboration.
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)  # Skip header
            if not header:
                return BatchResult(ingested=0)

            for row in reader:
                if len(row) < 10:
                    continue

                gene_id_raw = row[0].strip()
                gene_symbol = row[1].strip()
                disease_id = row[4].strip()  # UMLS CUI (e.g., C0014544)
                disease_name = row[5].strip()
                try:
                    score = float(row[9].strip())
                except (ValueError, IndexError):
                    score = 0.0

                if not gene_id_raw or not disease_id:
                    continue

                # Gene: direct Entrez ID (same as Hetionet)
                gene_entity = f"Gene_{gene_id_raw}"
                # Disease: UMLS CUI namespace
                disease_entity = f"Disease_UMLS:{disease_id}"

                gene_canonical = normalize_entity_id(gene_entity)
                disease_canonical = normalize_entity_id(disease_entity)

                if gene_canonical not in entities:
                    gene_ext = {"ncbi_gene": gene_id_raw}
                    if gene_symbol:
                        gene_ext["symbol"] = gene_symbol
                    entities[gene_canonical] = ("gene", gene_symbol or gene_entity,
                                                json.dumps(gene_ext))
                if disease_canonical not in entities:
                    disease_ext = {"umls_cui": disease_id}
                    entities[disease_canonical] = ("disease", disease_name or disease_entity,
                                                    json.dumps(disease_ext))

                pred_id = "associates"
                source_id = "disgenet_curated"
                source_type = "database_import"

                claim_id = compute_claim_id(
                    gene_canonical, pred_id, disease_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        gene_canonical, pred_id, disease_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(gene_canonical, pred_id, disease_canonical)

                claim_rows.append((
                    gene_canonical, disease_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

        logger.info("DisGeNET: %d gene-disease associations parsed", count)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "DisGeNET loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    def load_drugbank(self, path: str, mapper: GeneIDMapper) -> BatchResult:
        """Load DrugBank drug-target interactions via bulk import.

        Parses the target-all-polypeptide-ids CSV. Each row is a target protein
        with a semicolon-separated list of DrugBank IDs.

        Compound entities use direct DrugBank IDs (same as Hetionet).
        Gene targets mapped via gene symbol or UniProt ID.
        Action types mapped to bio predicates (inhibits, activates, binds, etc.).
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        mapped = 0
        unmapped = 0

        with open(path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Extract target info
                gene_name = (row.get("Gene Name") or row.get("gene_name") or "").strip()
                uniprot_id = (row.get("UniProt ID") or row.get("uniprot_id") or "").strip()
                species = (row.get("Species") or row.get("organism") or "").strip()
                actions_raw = (row.get("Actions") or row.get("action") or "").strip()
                known_action = (row.get("Known Action") or row.get("known_action") or "").strip()

                # Only human targets
                if species and "human" not in species.lower():
                    continue

                # Map target to gene entity
                gene_id = None
                if gene_name:
                    gene_id = mapper.symbol_to_gene_entity_id(gene_name)
                if gene_id is None and uniprot_id:
                    gene_id = mapper.uniprot_to_gene_entity_id(uniprot_id)
                if gene_id is None:
                    unmapped += 1
                    continue

                # Get drug IDs
                drug_ids_raw = (row.get("Drug IDs") or row.get("drug_ids") or "").strip()
                if not drug_ids_raw:
                    continue

                drug_ids = [d.strip() for d in drug_ids_raw.split(";") if d.strip()]
                if not drug_ids:
                    continue

                # Determine predicate from action
                actions = [a.strip().lower() for a in actions_raw.split(";") if a.strip()]
                pred_id = "binds"  # Default
                for action in actions:
                    if action in DRUGBANK_ACTION_MAP:
                        pred_id = DRUGBANK_ACTION_MAP[action]
                        break

                gene_canonical = normalize_entity_id(gene_id)
                if gene_canonical not in entities:
                    gene_ext = _extract_external_ids(gene_id, "gene")
                    if gene_name:
                        gene_ext["symbol"] = gene_name
                    if uniprot_id:
                        gene_ext["uniprot"] = uniprot_id
                    entities[gene_canonical] = ("gene", gene_name or gene_id,
                                                json.dumps(gene_ext))

                for db_id in drug_ids:
                    if not db_id.startswith("DB"):
                        continue

                    compound_entity = f"Compound_{db_id}"
                    compound_canonical = normalize_entity_id(compound_entity)

                    if compound_canonical not in entities:
                        comp_ext = {"drugbank": db_id}
                        entities[compound_canonical] = ("compound", compound_entity,
                                                        json.dumps(comp_ext))

                    source_id = "drugbank"
                    source_type = "database_import"

                    claim_id = compute_claim_id(
                        compound_canonical, pred_id, gene_canonical,
                        source_id, source_type, timestamp,
                    )
                    if claim_id in seen_claim_ids:
                        timestamp += 1
                        claim_id = compute_claim_id(
                            compound_canonical, pred_id, gene_canonical,
                            source_id, source_type, timestamp,
                        )
                    seen_claim_ids.add(claim_id)

                    content_id = compute_content_id(compound_canonical, pred_id, gene_canonical)

                    claim_rows.append((
                        compound_canonical, gene_canonical,
                        claim_id, content_id, pred_id, "relates_to",
                        tier1_confidence(source_type),
                        source_type, source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))
                    mapped += 1

        logger.info("DrugBank: %d drug-target pairs mapped, %d targets unmapped", mapped, unmapped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "DrugBank loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    def load_chembl(self, path: str, mapper: GeneIDMapper, min_pchembl: float = 5.0) -> BatchResult:
        """Load ChEMBL compound-target bioactivities via bulk import.

        Queries the ChEMBL SQLite database for compound-target pairs with
        activity data (IC50, Ki, Kd, EC50). Filters by pchembl_value >= min_pchembl.

        Compound entities: Compound_CHEMBL{id} (separate namespace from DrugBank).
        Gene targets: mapped via gene symbol from component_sequences.
        """
        import sqlite3

        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        mapped = 0
        unmapped = 0

        conn = sqlite3.connect(path)
        cursor = conn.cursor()

        # Extract compound-target pairs with bioactivity
        query = """
            SELECT DISTINCT
                md.chembl_id AS molecule_chembl_id,
                md.pref_name AS molecule_name,
                cseq.accession AS uniprot_id,
                COALESCE(
                    (SELECT cs2.component_synonym FROM component_synonyms cs2
                     WHERE cs2.component_id = cseq.component_id
                     AND cs2.syn_type = 'GENE_SYMBOL' LIMIT 1),
                    td.pref_name
                ) AS gene_name,
                a.pchembl_value,
                a.standard_type
            FROM activities a
            JOIN assays ass ON a.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN target_components tc ON td.tid = tc.tid
            JOIN component_sequences cseq ON tc.component_id = cseq.component_id
            JOIN molecule_dictionary md ON a.molregno = md.molregno
            WHERE a.pchembl_value >= ?
              AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
              AND td.target_type = 'SINGLE PROTEIN'
              AND td.organism = 'Homo sapiens'
        """

        logger.info("Querying ChEMBL SQLite for bioactivities (pchembl >= %.1f)...", min_pchembl)
        try:
            cursor.execute(query, (min_pchembl,))
        except sqlite3.OperationalError as e:
            # Handle schema differences between ChEMBL versions
            logger.warning("ChEMBL query failed (schema mismatch?): %s", e)
            # Simplified fallback query without component_synonyms
            query_simple = """
                SELECT DISTINCT
                    md.chembl_id,
                    md.pref_name,
                    cseq.accession,
                    td.pref_name,
                    a.pchembl_value,
                    a.standard_type
                FROM activities a
                JOIN assays ass ON a.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                JOIN target_components tc ON td.tid = tc.tid
                JOIN component_sequences cseq ON tc.component_id = cseq.component_id
                JOIN molecule_dictionary md ON a.molregno = md.molregno
                WHERE a.pchembl_value >= ?
                  AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
                  AND td.target_type = 'SINGLE PROTEIN'
                  AND td.organism = 'Homo sapiens'
            """
            cursor.execute(query_simple, (min_pchembl,))

        for row in cursor:
            chembl_id, mol_name, uniprot_id, gene_name, pchembl, std_type = row

            if not chembl_id or not gene_name:
                continue

            # Map target to gene entity
            gene_id = None
            if gene_name:
                gene_id = mapper.symbol_to_gene_entity_id(gene_name)
            if gene_id is None and uniprot_id:
                gene_id = mapper.uniprot_to_gene_entity_id(uniprot_id)
            if gene_id is None:
                unmapped += 1
                continue
            mapped += 1

            # Compound entity in ChEMBL namespace
            compound_entity = f"Compound_{chembl_id}"
            compound_canonical = normalize_entity_id(compound_entity)
            gene_canonical = normalize_entity_id(gene_id)

            if compound_canonical not in entities:
                comp_ext = {"chembl": chembl_id}
                display = mol_name or compound_entity
                entities[compound_canonical] = ("compound", display,
                                                json.dumps(comp_ext))
            if gene_canonical not in entities:
                gene_ext = _extract_external_ids(gene_id, "gene")
                if gene_name:
                    gene_ext["symbol"] = gene_name
                if uniprot_id:
                    gene_ext["uniprot"] = uniprot_id
                entities[gene_canonical] = ("gene", gene_name or gene_id,
                                            json.dumps(gene_ext))

            pred_id = "binds"
            source_id = "chembl"
            source_type = "experimental"

            claim_id = compute_claim_id(
                compound_canonical, pred_id, gene_canonical,
                source_id, source_type, timestamp,
            )
            if claim_id in seen_claim_ids:
                timestamp += 1
                claim_id = compute_claim_id(
                    compound_canonical, pred_id, gene_canonical,
                    source_id, source_type, timestamp,
                )
            seen_claim_ids.add(claim_id)

            content_id = compute_content_id(compound_canonical, pred_id, gene_canonical)

            claim_rows.append((
                compound_canonical, gene_canonical,
                claim_id, content_id, pred_id, "relates_to",
                tier1_confidence(source_type),
                source_type, source_id,
                "", "[]", "", "", "", timestamp, "active",
            ))

        conn.close()

        logger.info("ChEMBL: %d compound-target pairs mapped, %d unmapped", mapped, unmapped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "ChEMBL loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── GO Annotations ────────────────────────────────────────────────

    def download_goa(self, dest_dir: str) -> str:
        """Download human GO annotations (GAF format). Returns path."""
        import requests

        dest = os.path.join(dest_dir, "goa_human.gaf.gz")
        if os.path.exists(dest):
            logger.info("GOA already cached at %s", dest)
            return dest

        logger.info("Downloading GO annotations for human...")
        resp = requests.get(GOA_HUMAN_URL, timeout=120, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        logger.info("GOA downloaded: %.1f MB", size_mb)
        return dest

    def load_goa(
        self, path: str, mapper: GeneIDMapper, min_evidence: str = "all",
    ) -> BatchResult:
        """Load Gene Ontology annotations from GAF file.

        Maps UniProt accessions to Entrez Gene IDs via the mapper.
        Creates gene → participates_in → GO term claims.

        Args:
            path: Path to goa_human.gaf.gz
            mapper: GeneIDMapper with UniProt mapping loaded
            min_evidence: Filter level. 'experimental' = EXP/IDA/IPI/IMP/IGI/IEP only.
                          'curated' = experimental + ISS/ISO/ISA/ISM/RCA/IC.
                          'all' = include IEA (electronic). Default 'all'.
        """
        t0 = time.time()
        timestamp = int(t0)

        experimental_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}
        curated_codes = experimental_codes | {"ISS", "ISO", "ISA", "ISM", "RCA", "IC"}

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        mapped = 0
        unmapped = 0
        filtered = 0

        with gzip.open(path, "rt") as f:
            for line in f:
                if line.startswith("!"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 15:
                    continue

                db = parts[0]  # e.g. "UniProtKB"
                uniprot_id = parts[1]  # e.g. "P04637"
                gene_symbol = parts[2]  # e.g. "TP53"
                qualifier = parts[3]  # e.g. "NOT" or ""
                go_id = parts[4]  # e.g. "GO:0006915"
                evidence = parts[6]  # e.g. "IDA"
                aspect = parts[8]  # P, F, or C
                go_name = parts[9]  # full name
                taxon = parts[12]  # e.g. "taxon:9606"

                # Skip non-UniProtKB, non-human, negated annotations
                if db != "UniProtKB":
                    continue
                if "9606" not in taxon:
                    continue
                if "NOT" in qualifier.upper():
                    continue

                # Evidence filter
                if min_evidence == "experimental" and evidence not in experimental_codes:
                    filtered += 1
                    continue
                if min_evidence == "curated" and evidence not in curated_codes:
                    filtered += 1
                    continue

                # Map UniProt → Entrez Gene ID
                gene_eid = mapper.uniprot_to_gene_entity_id(uniprot_id)
                if gene_eid is None:
                    # Fallback: try gene symbol
                    gene_eid = mapper.symbol_to_gene_entity_id(gene_symbol)
                if gene_eid is None:
                    unmapped += 1
                    continue

                # Normalize IDs
                gene_canonical = normalize_entity_id(gene_eid)
                go_type = GO_ASPECT_MAP.get(aspect, "biological_process")
                go_canonical = normalize_entity_id(go_id)

                # Confidence from evidence code
                confidence = GO_EVIDENCE_CONFIDENCE.get(evidence, 0.60)

                # Register entities
                if gene_canonical not in entities:
                    display = gene_symbol if gene_symbol else gene_eid
                    if display == gene_eid and mapper is not None and gene_eid.startswith("Gene_"):
                        entrez_str = gene_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display = sym
                    entities[gene_canonical] = ("gene", display, "{}")
                if go_canonical not in entities:
                    entities[go_canonical] = (go_type, go_name or go_id, "{}")

                # Build claim
                predicate = "participates_in"
                source_id = f"goa:{uniprot_id}:{go_id}:{evidence}"
                content_id = compute_content_id(gene_canonical, predicate, go_canonical)
                claim_id = compute_claim_id(
                    gene_canonical, predicate, go_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    gene_canonical, go_canonical, claim_id, content_id,
                    predicate, "relation", confidence,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                mapped += 1

        logger.info(
            "GOA: %d annotations mapped, %d unmapped, %d filtered by evidence",
            mapped, unmapped, filtered,
        )

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "GOA loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── CTD (Comparative Toxicogenomics Database) ─────────────────────

    def download_ctd_chem_gene(self, dest_dir: str) -> str | None:
        """Download CTD chemical-gene interactions. Returns path or None."""
        import requests

        dest = os.path.join(dest_dir, "CTD_chem_gene_ixns.tsv.gz")
        if os.path.exists(dest):
            logger.info("CTD chem-gene already cached at %s", dest)
            return dest

        logger.info("Downloading CTD chemical-gene interactions...")
        try:
            resp = requests.get(CTD_CHEM_GENE_URL, timeout=300, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            logger.info("CTD chem-gene downloaded: %.1f MB", size_mb)
            return dest
        except Exception as e:
            logger.warning("CTD chem-gene download failed: %s", e)
            return None

    def download_ctd_gene_disease(self, dest_dir: str) -> str | None:
        """Download CTD gene-disease associations. Returns path or None."""
        import requests

        dest = os.path.join(dest_dir, "CTD_genes_diseases.tsv.gz")
        if os.path.exists(dest):
            logger.info("CTD gene-disease already cached at %s", dest)
            return dest

        logger.info("Downloading CTD gene-disease associations...")
        try:
            resp = requests.get(CTD_GENE_DISEASE_URL, timeout=300, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            logger.info("CTD gene-disease downloaded: %.1f MB", size_mb)
            return dest
        except Exception as e:
            logger.warning("CTD gene-disease download failed: %s", e)
            return None

    def load_ctd_chem_gene(self, path: str, human_only: bool = True,
                           mapper: GeneIDMapper | None = None) -> BatchResult:
        """Load CTD chemical-gene interactions.

        CTD uses NCBI Gene IDs and MeSH chemical IDs directly.
        Only loads human interactions by default (organism_id=9606).

        File format (tab-delimited, # comment headers):
            ChemicalName, ChemicalID(MESH), CasRN, GeneSymbol, GeneID,
            GeneForms, Organism, OrganismID, Interaction, InteractionActions,
            PubMedIDs
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        with gzip.open(path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 10:
                    continue

                chem_name = parts[0].strip()
                chem_mesh = parts[1].strip()
                gene_symbol = parts[3].strip()
                gene_id_str = parts[4].strip()
                organism_id = parts[7].strip()
                actions_raw = parts[9].strip() if len(parts) > 9 else ""

                # Filter human only
                if human_only and organism_id != "9606":
                    skipped += 1
                    continue

                if not gene_id_str or not gene_id_str.isdigit():
                    skipped += 1
                    continue

                # Map to entity IDs
                gene_eid = f"Gene_{gene_id_str}"
                gene_canonical = normalize_entity_id(gene_eid)
                chem_canonical = normalize_entity_id(f"Compound_{chem_mesh}")

                # Determine predicate from interaction actions
                predicate = "associated_with"  # default
                if actions_raw:
                    for action in actions_raw.split("|"):
                        action = action.strip().lower()
                        if action in CTD_ACTION_MAP:
                            predicate = CTD_ACTION_MAP[action]
                            break

                # Register entities
                if gene_canonical not in entities:
                    display = gene_symbol if gene_symbol else gene_eid
                    if display == gene_eid and mapper is not None and gene_eid.startswith("Gene_"):
                        entrez_str = gene_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display = sym
                    ext = json.dumps({"entrez": gene_id_str}) if gene_id_str else "{}"
                    entities[gene_canonical] = ("gene", display, ext)
                if chem_canonical not in entities:
                    entities[chem_canonical] = (
                        "compound", chem_name or f"Compound_{chem_mesh}",
                        json.dumps({"mesh": chem_mesh}) if chem_mesh else "{}",
                    )

                source_id = f"ctd:cg:{chem_mesh}:{gene_id_str}"
                content_id = compute_content_id(chem_canonical, predicate, gene_canonical)
                claim_id = compute_claim_id(
                    chem_canonical, predicate, gene_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    chem_canonical, gene_canonical, claim_id, content_id,
                    predicate, "relation", 0.85,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info(
            "CTD chem-gene: %d interactions parsed, %d skipped",
            len(claim_rows), skipped,
        )

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "CTD chem-gene loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    def load_ctd_gene_disease(self, path: str, direct_only: bool = True,
                              mapper: GeneIDMapper | None = None) -> BatchResult:
        """Load CTD gene-disease associations.

        CTD uses NCBI Gene IDs and MeSH/OMIM disease IDs.
        By default only loads direct evidence (not inferred).

        File format (tab-delimited, # comment headers):
            GeneSymbol, GeneID, DiseaseName, DiseaseID, DirectEvidence,
            InferenceChemicalName, InferenceScore, OmimIDs, PubMedIDs
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        with gzip.open(path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue

                gene_symbol = parts[0].strip()
                gene_id_str = parts[1].strip()
                disease_name = parts[2].strip()
                disease_id = parts[3].strip()  # e.g. "MESH:D000544" or "OMIM:104300"
                direct_evidence = parts[4].strip() if len(parts) > 4 else ""

                # Filter: only direct evidence if requested
                if direct_only and not direct_evidence:
                    skipped += 1
                    continue

                if not gene_id_str or not gene_id_str.isdigit():
                    skipped += 1
                    continue

                # Map to entity IDs
                gene_eid = f"Gene_{gene_id_str}"
                gene_canonical = normalize_entity_id(gene_eid)

                # Normalize disease ID: MESH:D000544 -> Disease_MESH:D000544
                disease_eid = f"Disease_{disease_id}"
                disease_canonical = normalize_entity_id(disease_eid)

                # Determine predicate from DirectEvidence field
                predicate = "associated_with"
                if direct_evidence:
                    ev_lower = direct_evidence.lower()
                    if "marker" in ev_lower:
                        predicate = "associated_with"
                    elif "therapeutic" in ev_lower:
                        predicate = "treats"

                # Register entities
                if gene_canonical not in entities:
                    display = gene_symbol if gene_symbol else gene_eid
                    if display == gene_eid and mapper is not None and gene_eid.startswith("Gene_"):
                        entrez_str = gene_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display = sym
                    ext = json.dumps({"entrez": gene_id_str}) if gene_id_str else "{}"
                    entities[gene_canonical] = ("gene", display, ext)
                if disease_canonical not in entities:
                    ext_ids = {}
                    if disease_id.startswith("MESH:"):
                        ext_ids["mesh"] = disease_id.split(":", 1)[1]
                    elif disease_id.startswith("OMIM:"):
                        ext_ids["omim"] = disease_id.split(":", 1)[1]
                    entities[disease_canonical] = (
                        "disease", disease_name or disease_eid,
                        json.dumps(ext_ids) if ext_ids else "{}",
                    )

                source_id = f"ctd:gd:{gene_id_str}:{disease_id}"
                content_id = compute_content_id(gene_canonical, predicate, disease_canonical)
                claim_id = compute_claim_id(
                    gene_canonical, predicate, disease_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    gene_canonical, disease_canonical, claim_id, content_id,
                    predicate, "relation", 0.85,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info(
            "CTD gene-disease: %d associations parsed, %d skipped",
            len(claim_rows), skipped,
        )

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "CTD gene-disease loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── BioGRID (Protein-Protein Interactions) ────────────────────────

    def download_biogrid(self, dest_dir: str) -> str | None:
        """Download BioGRID organism archive, extract human file. Returns path."""
        import requests
        import zipfile

        human_path = os.path.join(dest_dir, "biogrid_human.tab3.txt")
        if os.path.exists(human_path):
            logger.info("BioGRID human already cached at %s", human_path)
            return human_path

        zip_path = os.path.join(dest_dir, "biogrid_organism.tab3.zip")
        if not os.path.exists(zip_path):
            logger.info("Downloading BioGRID organism archive...")
            try:
                resp = requests.get(BIOGRID_ORGANISM_URL, timeout=300, stream=True)
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                logger.info("BioGRID downloaded: %.1f MB", size_mb)
            except Exception as e:
                logger.warning("BioGRID download failed: %s", e)
                return None

        # Extract human file from zip
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if "Homo_sapiens" in name and name.endswith(".tab3.txt"):
                        with zf.open(name) as src, open(human_path, "wb") as dst:
                            import shutil
                            shutil.copyfileobj(src, dst)
                        logger.info("Extracted %s", name)
                        return human_path
            logger.warning("No Homo sapiens file found in BioGRID archive")
            return None
        except Exception as e:
            logger.warning("BioGRID extraction failed: %s", e)
            return None

    def load_biogrid(self, path: str, low_throughput_only: bool = False,
                     mapper: GeneIDMapper | None = None) -> BatchResult:
        """Load BioGRID protein-protein interactions.

        BioGRID TAB3 format uses Entrez Gene IDs directly.
        Creates gene-gene interacts claims.

        Args:
            path: Path to human BioGRID TAB3 file.
            low_throughput_only: If True, skip high-throughput experiments
                (higher quality but fewer interactions).
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_pairs: set[tuple[str, str]] = set()
        skipped = 0

        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 19:
                    continue

                gene_a_str = parts[1].strip()  # Entrez Gene Interactor A
                gene_b_str = parts[2].strip()  # Entrez Gene Interactor B
                symbol_a = parts[7].strip() if len(parts) > 7 else ""  # Official Symbol A
                symbol_b = parts[8].strip() if len(parts) > 8 else ""  # Official Symbol B
                exp_system_type = parts[12].strip().lower() if len(parts) > 12 else ""
                throughput = parts[17].strip().lower() if len(parts) > 17 else ""
                organism_a = parts[15].strip() if len(parts) > 15 else ""
                organism_b = parts[16].strip() if len(parts) > 16 else ""

                # Human only
                if organism_a != "9606" or organism_b != "9606":
                    skipped += 1
                    continue

                # Skip self-interactions
                if gene_a_str == gene_b_str:
                    skipped += 1
                    continue

                if not gene_a_str.isdigit() or not gene_b_str.isdigit():
                    skipped += 1
                    continue

                if low_throughput_only and "high" in throughput:
                    skipped += 1
                    continue

                # Deduplicate (A,B) and (B,A)
                pair = (min(gene_a_str, gene_b_str), max(gene_a_str, gene_b_str))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                gene_a_eid = f"Gene_{gene_a_str}"
                gene_b_eid = f"Gene_{gene_b_str}"
                gene_a_canonical = normalize_entity_id(gene_a_eid)
                gene_b_canonical = normalize_entity_id(gene_b_eid)

                confidence = BIOGRID_CONFIDENCE.get(exp_system_type, 0.80)

                if gene_a_canonical not in entities:
                    display_a = symbol_a if symbol_a else gene_a_eid
                    if display_a == gene_a_eid and mapper is not None and gene_a_eid.startswith("Gene_"):
                        entrez_str = gene_a_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display_a = sym
                    ext_a = json.dumps({"entrez": gene_a_str}) if gene_a_str else "{}"
                    entities[gene_a_canonical] = ("gene", display_a, ext_a)
                if gene_b_canonical not in entities:
                    display_b = symbol_b if symbol_b else gene_b_eid
                    if display_b == gene_b_eid and mapper is not None and gene_b_eid.startswith("Gene_"):
                        entrez_str = gene_b_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display_b = sym
                    ext_b = json.dumps({"entrez": gene_b_str}) if gene_b_str else "{}"
                    entities[gene_b_canonical] = ("gene", display_b, ext_b)

                predicate = "interacts"
                source_id = f"biogrid:{gene_a_str}:{gene_b_str}"
                content_id = compute_content_id(gene_a_canonical, predicate, gene_b_canonical)
                claim_id = compute_claim_id(
                    gene_a_canonical, predicate, gene_b_canonical,
                    source_id, "experimental", timestamp,
                )

                claim_rows.append((
                    gene_a_canonical, gene_b_canonical, claim_id, content_id,
                    predicate, "relation", confidence,
                    "experimental", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info(
            "BioGRID: %d unique interactions parsed, %d skipped",
            len(claim_rows), skipped,
        )

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "BioGRID loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── ClinVar (Gene-Disease Associations) ───────────────────────────

    def download_clinvar(self, dest_dir: str) -> str:
        """Download ClinVar gene_condition_source_id file. Returns path."""
        import requests

        dest = os.path.join(dest_dir, "clinvar_gene_condition.tsv")
        if os.path.exists(dest):
            logger.info("ClinVar already cached at %s", dest)
            return dest

        logger.info("Downloading ClinVar gene-condition mapping...")
        resp = requests.get(CLINVAR_GENE_CONDITION_URL, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        size_kb = os.path.getsize(dest) / 1024
        logger.info("ClinVar downloaded: %.0f KB", size_kb)
        return dest

    def load_clinvar(self, path: str, mapper: GeneIDMapper | None = None) -> BatchResult:
        """Load ClinVar gene-disease associations.

        Uses gene_condition_source_id file which maps Entrez Gene IDs
        to disease conditions via UMLS concept IDs and OMIM MIM numbers.

        File format (tab-delimited, 7 columns):
            GeneID, GeneSymbol, ConceptID, SourceName, SourceID, DiseaseMIM, LastUpdated
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        with open(path, "r") as f:
            header = next(f, None)  # Skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue

                gene_id_str = parts[0].strip()
                gene_symbol = parts[1].strip()
                concept_id = parts[2].strip()  # UMLS CUI or CN
                source_name = parts[3].strip()  # e.g. "OMIM", "GeneReviews"
                source_id_val = parts[4].strip()
                disease_mim = parts[5].strip()

                if not gene_id_str or not gene_id_str.isdigit():
                    skipped += 1
                    continue
                if not concept_id:
                    skipped += 1
                    continue

                gene_eid = f"Gene_{gene_id_str}"
                gene_canonical = normalize_entity_id(gene_eid)

                # Build disease entity ID from concept ID
                disease_eid = f"Disease_{concept_id}"
                disease_canonical = normalize_entity_id(disease_eid)

                # External IDs for the disease
                ext_ids = {}
                if concept_id.startswith("C"):
                    ext_ids["umls"] = concept_id
                if disease_mim and disease_mim.isdigit():
                    ext_ids["omim"] = disease_mim

                if gene_canonical not in entities:
                    display = gene_symbol if gene_symbol else gene_eid
                    if display == gene_eid and mapper is not None and gene_eid.startswith("Gene_"):
                        entrez_str = gene_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display = sym
                    ext = json.dumps({"entrez": gene_id_str}) if gene_id_str else "{}"
                    entities[gene_canonical] = ("gene", display, ext)
                if disease_canonical not in entities:
                    # Use source_id_val as display name if available
                    display = source_id_val or concept_id
                    entities[disease_canonical] = (
                        "disease", display,
                        json.dumps(ext_ids) if ext_ids else "{}",
                    )

                predicate = "associated_with"
                src_id = f"clinvar:{gene_id_str}:{concept_id}:{source_name}"
                content_id = compute_content_id(gene_canonical, predicate, disease_canonical)
                claim_id = compute_claim_id(
                    gene_canonical, predicate, disease_canonical,
                    src_id, "database_import", timestamp,
                )

                claim_rows.append((
                    gene_canonical, disease_canonical, claim_id, content_id,
                    predicate, "relation", 0.88,
                    "database_import", src_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info(
            "ClinVar: %d gene-disease associations parsed, %d skipped",
            len(claim_rows), skipped,
        )

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "ClinVar loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── PrimeKG (Multi-relational Knowledge Graph) ────────────────────

    def download_primekg(self, dest_dir: str) -> str:
        """Download PrimeKG CSV from Harvard Dataverse. Returns path."""
        import requests

        dest = os.path.join(dest_dir, "primekg_edges.csv")
        if os.path.exists(dest):
            logger.info("PrimeKG already cached at %s", dest)
            return dest

        logger.info("Downloading PrimeKG (~200MB)...")
        resp = requests.get(PRIMEKG_URL, timeout=600, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded PrimeKG to %s", dest)
        return dest

    def load_primekg(self, path: str) -> BatchResult:
        """Load PrimeKG multi-relational edges.

        CSV columns: relation, display_relation, x_id, x_type, x_name,
                     x_source, y_id, y_type, y_name, y_source
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        # Map PrimeKG node types to entity type + ID prefix
        type_map = {
            "gene/protein": ("gene", "Gene"),
            "drug": ("compound", "Compound"),
            "disease": ("disease", "Disease"),
            "pathway": ("pathway", "Pathway"),
            "biological_process": ("biological_process", "BP"),
            "molecular_function": ("molecular_function", "MF"),
            "cellular_component": ("cellular_component", "CC"),
            "anatomy": ("entity", "Anatomy"),
            "exposure": ("entity", "Exposure"),
            "effect/phenotype": ("phenotype", "Phenotype"),
        }

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                display_rel = row.get("display_relation", "").strip()
                x_id = row.get("x_id", "").strip()
                x_type = row.get("x_type", "").strip().lower()
                x_name = row.get("x_name", "").strip()
                y_id = row.get("y_id", "").strip()
                y_type = row.get("y_type", "").strip().lower()
                y_name = row.get("y_name", "").strip()

                if not x_id or not y_id or not display_rel:
                    skipped += 1
                    continue

                predicate = PRIMEKG_PREDICATE_MAP.get(display_rel, "associated_with")

                # Build entity IDs
                x_info = type_map.get(x_type, ("entity", "Entity"))
                y_info = type_map.get(y_type, ("entity", "Entity"))

                x_eid = f"{x_info[1]}_{x_id}"
                y_eid = f"{y_info[1]}_{y_id}"
                x_canonical = normalize_entity_id(x_eid)
                x_canonical, x_ext = _safe_entity_id(x_canonical)
                y_canonical = normalize_entity_id(y_eid)
                y_canonical, y_ext = _safe_entity_id(y_canonical)

                if x_canonical == y_canonical:
                    skipped += 1
                    continue

                if x_canonical not in entities:
                    entities[x_canonical] = (x_info[0], x_name or x_eid, json.dumps(x_ext) if x_ext else "{}")
                if y_canonical not in entities:
                    entities[y_canonical] = (y_info[0], y_name or y_eid, json.dumps(y_ext) if y_ext else "{}")

                source_id = f"primekg:{x_id}:{display_rel}:{y_id}"
                content_id = compute_content_id(x_canonical, predicate, y_canonical)
                claim_id = compute_claim_id(
                    x_canonical, predicate, y_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    x_canonical, y_canonical, claim_id, content_id,
                    predicate, "relation", 0.85,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("PrimeKG: %d edges parsed, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "PrimeKG loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── DRKG (Drug Repurposing Knowledge Graph) ───────────────────────

    def download_drkg(self, dest_dir: str) -> str:
        """Download DRKG tar.gz from S3, extract TSV. Returns path."""
        import requests
        import tarfile as _tarfile

        dest = os.path.join(dest_dir, "drkg.tsv")
        if os.path.exists(dest):
            logger.info("DRKG already cached at %s", dest)
            return dest

        tar_path = os.path.join(dest_dir, "drkg.tar.gz")
        if not os.path.exists(tar_path):
            logger.info("Downloading DRKG (~100MB)...")
            resp = requests.get(DRKG_URL, timeout=600, stream=True)
            resp.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract the TSV file from the tar.gz
        logger.info("Extracting DRKG...")
        with _tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                # Skip macOS resource fork files (._prefix)
                basename = os.path.basename(member.name)
                if basename.startswith("._"):
                    continue
                if member.name == "drkg.tsv" or basename == "drkg.tsv":
                    extracted = tf.extractfile(member)
                    if extracted:
                        with open(dest, "wb") as f:
                            f.write(extracted.read())
                        break
            else:
                # Try extracting any large file
                for member in tf.getmembers():
                    basename = os.path.basename(member.name)
                    if basename.startswith("._"):
                        continue
                    if member.isfile() and member.size > 1_000_000:
                        extracted = tf.extractfile(member)
                        if extracted:
                            with open(dest, "wb") as f:
                                f.write(extracted.read())
                            break

        logger.info("DRKG extracted to %s", dest)
        return dest

    def load_drkg(self, path: str) -> BatchResult:
        """Load DRKG triples (head, relation, tail).

        Format: TSV, 3 columns: head\\trelation\\ttail
        Entity format: Type::ID (e.g., Compound::DB00945, Gene::5742)
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        # DRKG entity type -> (attest_type, id_prefix)
        drkg_type_map = {
            "Compound": ("compound", "Compound"),
            "Gene": ("gene", "Gene"),
            "Disease": ("disease", "Disease"),
            "Anatomy": ("entity", "Anatomy"),
            "Side Effect": ("phenotype", "SideEffect"),
        }
        # Types to skip entirely
        skip_types = {"Atc", "Tax"}

        def _parse_drkg_entity(raw: str) -> tuple[str, str, str] | None:
            """Parse 'Type::ID' into (entity_id, entity_type, display)."""
            parts = raw.split("::", 1)
            if len(parts) != 2:
                return None
            etype, eid = parts[0].strip(), parts[1].strip()
            if etype in skip_types:
                return None
            info = drkg_type_map.get(etype, ("entity", etype))
            entity_id = f"{info[1]}_{eid}"
            return entity_id, info[0], entity_id

        with open(path, "r", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    skipped += 1
                    continue

                head_raw, relation, tail_raw = parts[0], parts[1], parts[2]

                head = _parse_drkg_entity(head_raw)
                tail = _parse_drkg_entity(tail_raw)
                if head is None or tail is None:
                    skipped += 1
                    continue

                head_eid, head_type, head_display = head
                tail_eid, tail_type, tail_display = tail

                head_canonical = normalize_entity_id(head_eid)
                tail_canonical = normalize_entity_id(tail_eid)

                if head_canonical == tail_canonical:
                    skipped += 1
                    continue

                # Map relation to predicate
                predicate = DRKG_PREDICATE_MAP.get(relation, None)
                if predicate is None:
                    # Try prefix match (e.g., "STRING::REACTION" -> "interacts")
                    prefix = relation.rsplit("::", 1)[0] + "::" if "::" in relation else ""
                    predicate = DRKG_PREDICATE_MAP.get(prefix, "associated_with")

                if head_canonical not in entities:
                    entities[head_canonical] = (head_type, head_display, "{}")
                if tail_canonical not in entities:
                    entities[tail_canonical] = (tail_type, tail_display, "{}")

                source_id = f"drkg:{head_raw}:{relation}:{tail_raw}"
                content_id = compute_content_id(head_canonical, predicate, tail_canonical)
                claim_id = compute_claim_id(
                    head_canonical, predicate, tail_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    head_canonical, tail_canonical, claim_id, content_id,
                    predicate, "relation", 0.80,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("DRKG: %d triples parsed, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "DRKG loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── SIDER (Drug Side Effects) ─────────────────────────────────────

    def download_sider(self, dest_dir: str) -> str:
        """Download SIDER meddra_all_se.tsv.gz. Returns path to decompressed TSV."""
        import requests

        dest = os.path.join(dest_dir, "sider_meddra_all_se.tsv")
        if os.path.exists(dest):
            logger.info("SIDER already cached at %s", dest)
            return dest

        logger.info("Downloading SIDER (~10MB)...")
        resp = requests.get(SIDER_URL, timeout=120)
        resp.raise_for_status()

        data = gzip.decompress(resp.content).decode("utf-8")
        with open(dest, "w") as f:
            f.write(data)
        logger.info("SIDER downloaded to %s (%d lines)", dest, data.count("\n"))
        return dest

    def load_sider(self, path: str) -> BatchResult:
        """Load SIDER drug-side effect pairs.

        Format (TSV, no header): STITCH_flat, STITCH_stereo, UMLS_label,
               MedDRA_type, UMLS_MedDRA, side_effect_name
        STITCH flat IDs: CIDxxxxxxxxx -> subtract 100000000 for PubChem CID.
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_pairs: set[tuple[str, str]] = set()
        skipped = 0

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 6:
                    skipped += 1
                    continue

                stitch_flat = row[0].strip()
                umls_cui = row[4].strip()
                se_name = row[5].strip()

                if not stitch_flat or not umls_cui or not se_name:
                    skipped += 1
                    continue

                # Convert STITCH flat ID to PubChem CID
                # STITCH flat format: CIDxxxxxxxxx (9-digit, zero-padded)
                # Subtract 100000000 to get PubChem CID
                stitch_num = stitch_flat.replace("CID", "").lstrip("0")
                if not stitch_num:
                    skipped += 1
                    continue
                try:
                    cid = int(stitch_num)
                    if cid > 100000000:
                        cid -= 100000000
                except ValueError:
                    skipped += 1
                    continue

                compound_eid = f"Compound_CID:{cid}"
                se_eid = f"SideEffect_{umls_cui}"

                compound_canonical = normalize_entity_id(compound_eid)
                se_canonical = normalize_entity_id(se_eid)

                # Deduplicate compound-side_effect pairs
                pair = (compound_canonical, se_canonical)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if compound_canonical not in entities:
                    entities[compound_canonical] = (
                        "compound", compound_eid,
                        json.dumps({"pubchem_cid": str(cid)}),
                    )
                if se_canonical not in entities:
                    entities[se_canonical] = ("phenotype", se_name, "{}")

                source_id = f"sider:{stitch_flat}:{umls_cui}"
                predicate = "causes_side_effect"
                content_id = compute_content_id(compound_canonical, predicate, se_canonical)
                claim_id = compute_claim_id(
                    compound_canonical, predicate, se_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    compound_canonical, se_canonical, claim_id, content_id,
                    predicate, "relation", 0.82,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info(
            "SIDER: %d drug-SE pairs parsed, %d skipped",
            len(claim_rows), skipped,
        )

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "SIDER loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── HPO (Human Phenotype Ontology Gene-Phenotype) ─────────────────

    def download_hpo(self, dest_dir: str) -> str:
        """Download HPO genes_to_phenotype.txt. Returns path."""
        import requests

        dest = os.path.join(dest_dir, "hpo_genes_to_phenotype.txt")
        if os.path.exists(dest):
            logger.info("HPO already cached at %s", dest)
            return dest

        logger.info("Downloading HPO gene-phenotype annotations...")
        resp = requests.get(HPO_GENES_URL, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        logger.info("HPO downloaded to %s", dest)
        return dest

    def load_hpo(self, path: str, mapper: GeneIDMapper | None = None) -> BatchResult:
        """Load HPO gene-phenotype associations.

        Format (TSV, header): gene_id, gene_symbol, hpo_id, hpo_name,
                              frequency, disease_id
        Gene→Phenotype: associated_with predicate.
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_pairs: set[tuple[str, str]] = set()
        skipped = 0

        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    skipped += 1
                    continue

                gene_id_str = parts[0].strip()
                gene_symbol = parts[1].strip()
                hpo_id = parts[2].strip()
                hpo_name = parts[3].strip()

                if not gene_id_str or not hpo_id:
                    skipped += 1
                    continue

                # Some files have "ncbi_gene_id" header as first data row
                if not gene_id_str.replace("-", "").isdigit():
                    continue

                gene_eid = f"Gene_{gene_id_str}"
                # Normalize HPO ID: HP:0001250 -> Phenotype_HP:0001250
                phenotype_eid = f"Phenotype_{hpo_id}"

                gene_canonical = normalize_entity_id(gene_eid)
                phenotype_canonical = normalize_entity_id(phenotype_eid)

                pair = (gene_canonical, phenotype_canonical)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if gene_canonical not in entities:
                    display = gene_symbol if gene_symbol else gene_eid
                    if display == gene_eid and mapper is not None and gene_eid.startswith("Gene_"):
                        entrez_str = gene_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display = sym
                    ext = json.dumps({"entrez": gene_id_str}) if gene_id_str else "{}"
                    entities[gene_canonical] = ("gene", display, ext)
                if phenotype_canonical not in entities:
                    entities[phenotype_canonical] = ("phenotype", hpo_name or phenotype_eid, "{}")

                source_id = f"hpo:{gene_id_str}:{hpo_id}"
                predicate = "associated_with"
                content_id = compute_content_id(gene_canonical, predicate, phenotype_canonical)
                claim_id = compute_claim_id(
                    gene_canonical, predicate, phenotype_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    gene_canonical, phenotype_canonical, claim_id, content_id,
                    predicate, "relation", 0.85,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("HPO: %d gene-phenotype pairs, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "HPO loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── IntAct (Molecular Interactions) ───────────────────────────────

    def download_intact(self, dest_dir: str) -> str:
        """Download IntAct MITAB from EBI FTP. Returns path to extracted TSV."""
        import requests
        import zipfile

        dest = os.path.join(dest_dir, "intact.txt")
        if os.path.exists(dest):
            logger.info("IntAct already cached at %s", dest)
            return dest

        zip_path = os.path.join(dest_dir, "intact.zip")
        if not os.path.exists(zip_path):
            logger.info("Downloading IntAct (~150MB)...")
            resp = requests.get(INTACT_URL, timeout=600, stream=True)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        logger.info("Extracting IntAct...")
        with zipfile.ZipFile(zip_path) as zf:
            # Find the main MITAB file (usually intact.txt or similar)
            for name in zf.namelist():
                if name.endswith(".txt") and "negative" not in name.lower():
                    with open(dest, "wb") as f:
                        f.write(zf.read(name))
                    break
        logger.info("IntAct extracted to %s", dest)
        return dest

    def load_intact(self, path: str, mapper: GeneIDMapper | None = None) -> BatchResult:
        """Load IntAct molecular interactions (MITAB 2.7 format).

        Col 0: ID interactor A (e.g., uniprotkb:P12345)
        Col 1: ID interactor B
        Col 14: Confidence (e.g., intact-miscore:0.75)

        Maps UniProt IDs to Entrez gene IDs via mapper. Falls back to
        using UniProt accessions directly if mapper unavailable.
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_pairs: set[tuple[str, str]] = set()
        skipped = 0

        def _extract_uniprot(col: str) -> str | None:
            """Extract UniProt accession from MITAB ID column."""
            for entry in col.split("|"):
                entry = entry.strip()
                if entry.startswith("uniprotkb:"):
                    acc = entry.split(":", 1)[1].split("-")[0]  # strip isoform
                    if acc and len(acc) >= 6:
                        return acc
            return None

        def _extract_confidence(col: str) -> float:
            """Extract intact-miscore from confidence column."""
            for entry in col.split("|"):
                entry = entry.strip()
                if entry.startswith("intact-miscore:"):
                    try:
                        return float(entry.split(":", 1)[1])
                    except ValueError:
                        pass
            return 0.70  # default

        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    skipped += 1
                    continue

                uniprot_a = _extract_uniprot(parts[0])
                uniprot_b = _extract_uniprot(parts[1])
                if uniprot_a is None or uniprot_b is None:
                    skipped += 1
                    continue

                # Skip self-interactions
                if uniprot_a == uniprot_b:
                    skipped += 1
                    continue

                # Map to gene entity IDs
                if mapper is not None:
                    gene_a = mapper.uniprot_to_gene_entity_id(uniprot_a)
                    gene_b = mapper.uniprot_to_gene_entity_id(uniprot_b)
                else:
                    gene_a = f"Protein_{uniprot_a}"
                    gene_b = f"Protein_{uniprot_b}"

                if gene_a is None or gene_b is None:
                    skipped += 1
                    continue

                gene_a_canonical = normalize_entity_id(gene_a)
                gene_b_canonical = normalize_entity_id(gene_b)

                if gene_a_canonical == gene_b_canonical:
                    skipped += 1
                    continue

                # Deduplicate (A,B) and (B,A)
                pair = (min(gene_a_canonical, gene_b_canonical),
                        max(gene_a_canonical, gene_b_canonical))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                confidence = _extract_confidence(parts[14]) if len(parts) > 14 else 0.70

                etype = "gene" if mapper is not None else "protein"
                if gene_a_canonical not in entities:
                    ext_a = json.dumps({"uniprot": uniprot_a})
                    entities[gene_a_canonical] = (etype, gene_a, ext_a)
                if gene_b_canonical not in entities:
                    ext_b = json.dumps({"uniprot": uniprot_b})
                    entities[gene_b_canonical] = (etype, gene_b, ext_b)

                source_id = f"intact:{uniprot_a}:{uniprot_b}"
                predicate = "interacts"
                content_id = compute_content_id(gene_a_canonical, predicate, gene_b_canonical)
                claim_id = compute_claim_id(
                    gene_a_canonical, predicate, gene_b_canonical,
                    source_id, "experimental", timestamp,
                )

                claim_rows.append((
                    gene_a_canonical, gene_b_canonical, claim_id, content_id,
                    predicate, "relation", confidence,
                    "experimental", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("IntAct: %d interactions parsed, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "IntAct loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── PharmGKB (Pharmacogenomics Relationships) ─────────────────────

    def download_pharmgkb(self, dest_dir: str) -> str:
        """Download PharmGKB relationships zip. Returns path to extracted TSV."""
        import requests
        import zipfile

        dest = os.path.join(dest_dir, "pharmgkb_relationships.tsv")
        if os.path.exists(dest):
            logger.info("PharmGKB already cached at %s", dest)
            return dest

        zip_path = os.path.join(dest_dir, "pharmgkb_relationships.zip")
        if not os.path.exists(zip_path):
            logger.info("Downloading PharmGKB relationships...")
            resp = requests.get(PHARMGKB_URL, timeout=120)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(resp.content)

        logger.info("Extracting PharmGKB...")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".tsv") or name.endswith("relationships.tsv"):
                    with open(dest, "wb") as f:
                        f.write(zf.read(name))
                    break
        logger.info("PharmGKB extracted to %s", dest)
        return dest

    def load_pharmgkb(self, path: str, mapper: GeneIDMapper | None = None) -> BatchResult:
        """Load PharmGKB pharmacogenomics relationships.

        TSV columns: Entity1_id, Entity1_name, Entity1_type,
                     Entity2_id, Entity2_name, Entity2_type,
                     Evidence, Association, PK, PD
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        def _resolve_entity(eid: str, name: str, etype: str) -> tuple[str, str, str] | None:
            """Resolve PharmGKB entity to Attest entity ID."""
            etype_lower = etype.lower()
            if etype_lower == "gene":
                if mapper is not None and name:
                    gene_eid = mapper.symbol_to_gene_entity_id(name)
                    if gene_eid:
                        return gene_eid, "gene", name
                return f"Gene_PharmGKB:{eid}", "gene", name or eid
            elif etype_lower in ("drug", "chemical"):
                return f"Compound_PharmGKB:{eid}", "compound", name or eid
            elif etype_lower == "disease":
                return f"Disease_PharmGKB:{eid}", "disease", name or eid
            elif etype_lower == "phenotype":
                return f"Phenotype_PharmGKB:{eid}", "phenotype", name or eid
            elif etype_lower == "variant" or etype_lower == "haplotype":
                return f"Variant_PharmGKB:{eid}", "gene", name or eid
            return None

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None:
                return BatchResult(ingested=0)

            for row in reader:
                if len(row) < 7:
                    skipped += 1
                    continue

                e1_id = row[0].strip()
                e1_name = row[1].strip()
                e1_type = row[2].strip()
                e2_id = row[3].strip()
                e2_name = row[4].strip()
                e2_type = row[5].strip()
                evidence = row[6].strip()
                association = row[7].strip() if len(row) > 7 else ""

                if not e1_id or not e2_id:
                    skipped += 1
                    continue

                e1 = _resolve_entity(e1_id, e1_name, e1_type)
                e2 = _resolve_entity(e2_id, e2_name, e2_type)
                if e1 is None or e2 is None:
                    skipped += 1
                    continue

                e1_eid, e1_atype, e1_display = e1
                e2_eid, e2_atype, e2_display = e2

                e1_canonical = normalize_entity_id(e1_eid)
                e2_canonical = normalize_entity_id(e2_eid)

                if e1_canonical == e2_canonical:
                    skipped += 1
                    continue

                # Determine predicate based on entity types
                e1_lower = e1_type.lower()
                e2_lower = e2_type.lower()
                if (e1_lower in ("drug", "chemical") and e2_lower == "disease") or \
                   (e2_lower in ("drug", "chemical") and e1_lower == "disease"):
                    predicate = "treats"
                else:
                    predicate = "associated_with"

                # Map evidence to confidence
                confidence = PHARMGKB_EVIDENCE_CONFIDENCE.get(evidence, 0.75)

                if e1_canonical not in entities:
                    entities[e1_canonical] = (e1_atype, e1_display, "{}")
                if e2_canonical not in entities:
                    entities[e2_canonical] = (e2_atype, e2_display, "{}")

                source_id = f"pharmgkb:{e1_id}:{e2_id}"
                content_id = compute_content_id(e1_canonical, predicate, e2_canonical)
                claim_id = compute_claim_id(
                    e1_canonical, predicate, e2_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    e1_canonical, e2_canonical, claim_id, content_id,
                    predicate, "relation", confidence,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("PharmGKB: %d relationships parsed, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "PharmGKB loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── Mondo Disease Ontology (Cross-references) ─────────────────────

    def download_mondo(self, dest_dir: str) -> str:
        """Download Mondo SSSOM TSV. Returns path."""
        import requests

        dest = os.path.join(dest_dir, "mondo.sssom.tsv")
        if os.path.exists(dest):
            logger.info("Mondo already cached at %s", dest)
            return dest

        logger.info("Downloading Mondo disease ontology mappings...")
        resp = requests.get(MONDO_SSSOM_URL, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        logger.info("Mondo downloaded to %s", dest)
        return dest

    def load_mondo(self, path: str) -> BatchResult:
        """Load Mondo disease cross-reference mappings (SSSOM format).

        Format (TSV): subject_id, subject_label, predicate_id,
                      object_id, object_label, mapping_justification, ...
        Maps disease IDs across ontologies (MONDO, DOID, MESH, OMIM, etc.).
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        def _mondo_entity(raw_id: str, label: str) -> tuple[str, str] | None:
            """Convert ontology ID to Attest disease entity ID."""
            raw_id = raw_id.strip()
            if not raw_id:
                return None
            # Normalize prefix format: MONDO:0005148, DOID:9352, MESH:D003920
            # Some have full URI — extract just the CURIE
            if "/" in raw_id:
                raw_id = raw_id.rsplit("/", 1)[-1]
            # Replace underscore with colon if needed (OBO format: MONDO_0005148)
            raw_id = raw_id.replace("_", ":", 1)
            prefix = raw_id.split(":")[0].upper() if ":" in raw_id else ""
            if prefix in ("MONDO", "DOID", "MESH", "OMIM", "ORPHANET",
                          "HP", "NCIT", "ICD10CM", "ICD9", "SCTID",
                          "UMLS", "EFO", "COHD", "MEDDRA", "GARD"):
                return f"Disease_{raw_id}", label or raw_id
            return None

        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    # Check if this is the header
                    if "subject_id" in line:
                        continue
                    skipped += 1
                    continue

                subject_id = parts[0].strip()
                subject_label = parts[1].strip() if len(parts) > 1 else ""
                predicate_id = parts[2].strip() if len(parts) > 2 else ""
                object_id = parts[3].strip()
                object_label = parts[4].strip() if len(parts) > 4 else ""

                # Only process exact/close mappings
                if predicate_id and "exact" not in predicate_id.lower() and \
                   "close" not in predicate_id.lower() and \
                   "equivalent" not in predicate_id.lower():
                    skipped += 1
                    continue

                subj = _mondo_entity(subject_id, subject_label)
                obj = _mondo_entity(object_id, object_label)
                if subj is None or obj is None:
                    skipped += 1
                    continue

                subj_eid, subj_display = subj
                obj_eid, obj_display = obj

                subj_canonical = normalize_entity_id(subj_eid)
                obj_canonical = normalize_entity_id(obj_eid)

                if subj_canonical == obj_canonical:
                    skipped += 1
                    continue

                if subj_canonical not in entities:
                    entities[subj_canonical] = ("disease", subj_display, "{}")
                if obj_canonical not in entities:
                    entities[obj_canonical] = ("disease", obj_display, "{}")

                source_id = f"mondo:{subject_id}:{object_id}"
                predicate = "equivalent_to"
                content_id = compute_content_id(subj_canonical, predicate, obj_canonical)
                claim_id = compute_claim_id(
                    subj_canonical, predicate, obj_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    subj_canonical, obj_canonical, claim_id, content_id,
                    predicate, "relation", 0.95,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("Mondo: %d cross-references parsed, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "Mondo loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── STITCH (Chemical-Protein Interactions) ────────────────────────

    def download_stitch(self, dest_dir: str) -> str:
        """Download STITCH human chemical-protein links. Returns path to decompressed TSV."""
        import requests

        dest = os.path.join(dest_dir, "stitch_chemical_protein.tsv")
        if os.path.exists(dest):
            logger.info("STITCH already cached at %s", dest)
            return dest

        logger.info("Downloading STITCH chemical-protein links (~500MB)...")
        resp = requests.get(STITCH_URL, timeout=600, stream=True)
        resp.raise_for_status()

        gz_path = os.path.join(dest_dir, "stitch_chemical_protein.tsv.gz")
        with open(gz_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Decompress
        with gzip.open(gz_path, "rt") as gz_f, open(dest, "w") as out_f:
            for line in gz_f:
                out_f.write(line)

        logger.info("STITCH extracted to %s", dest)
        return dest

    def load_stitch(
        self, path: str, min_score: int = 700,
        mapper: GeneIDMapper | None = None,
    ) -> BatchResult:
        """Load STITCH chemical-protein interactions.

        Format (space/tab-separated, header): chemical protein combined_score
        Chemical IDs: CIDsXXXXXXXXX (stereo) or CIDmXXXXXXXXX (flat).
        Protein IDs: 9606.ENSPXXX (same as STRING).
        Maps chemicals to PubChem CID, proteins to Gene via ENSP.
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_pairs: set[tuple[str, str]] = set()
        skipped = 0

        with open(path, "r") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                chem_raw, protein_raw = parts[0], parts[1]
                try:
                    score = int(parts[2])
                except ValueError:
                    continue

                if score < min_score:
                    skipped += 1
                    continue

                # Parse chemical: CIDsXXXXXXXXX or CIDmXXXXXXXXX
                # Strip "CIDs" or "CIDm" prefix, convert to PubChem CID
                cid_str = chem_raw
                if cid_str.startswith("CIDs") or cid_str.startswith("CIDm"):
                    cid_str = cid_str[4:]
                elif cid_str.startswith("CID"):
                    cid_str = cid_str[3:]
                cid_str = cid_str.lstrip("0")
                if not cid_str:
                    skipped += 1
                    continue

                try:
                    cid = int(cid_str)
                    if cid > 100000000:
                        cid -= 100000000
                except ValueError:
                    skipped += 1
                    continue

                compound_eid = f"Compound_CID:{cid}"
                compound_canonical = normalize_entity_id(compound_eid)

                # Parse protein: strip "9606." prefix for ENSP lookup
                protein_id = protein_raw
                if protein_id.startswith("9606."):
                    protein_id = protein_id[5:]
                # Map ENSP → Gene entity via mapper if available
                gene_eid = None
                if mapper is not None:
                    gene_eid = mapper.ensp_to_gene_entity_id(protein_id)
                if gene_eid is None:
                    gene_eid = f"Gene_{protein_id}"
                gene_canonical = normalize_entity_id(gene_eid)

                if compound_canonical == gene_canonical:
                    skipped += 1
                    continue

                pair = (compound_canonical, gene_canonical)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if compound_canonical not in entities:
                    entities[compound_canonical] = (
                        "compound", compound_eid,
                        json.dumps({"pubchem_cid": str(cid)}),
                    )
                if gene_canonical not in entities:
                    # Resolve gene symbol via mapper if available
                    display = gene_eid
                    if mapper is not None and gene_eid.startswith("Gene_"):
                        entrez_str = gene_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display = sym
                    ext = json.dumps({"ensp": protein_id})
                    entities[gene_canonical] = ("gene", display, ext)

                predicate = "interacts"
                source_id = f"stitch:{chem_raw}:{protein_raw}"
                content_id = compute_content_id(compound_canonical, predicate, gene_canonical)
                claim_id = compute_claim_id(
                    compound_canonical, predicate, gene_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    compound_canonical, gene_canonical, claim_id, content_id,
                    predicate, "relation", 0.80,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("STITCH: %d interactions parsed, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "STITCH loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── CTD Chemical-Disease Associations ─────────────────────────────

    def download_ctd_chem_disease(self, dest_dir: str) -> str | None:
        """Download CTD chemical-disease associations. Returns path or None."""
        import requests

        dest = os.path.join(dest_dir, "CTD_chemicals_diseases.tsv.gz")
        if os.path.exists(dest):
            logger.info("CTD chem-disease already cached at %s", dest)
            return dest

        logger.info("Downloading CTD chemical-disease associations...")
        try:
            resp = requests.get(CTD_CHEM_DISEASE_URL, timeout=300, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            logger.info("CTD chem-disease downloaded: %.1f MB", size_mb)
            return dest
        except Exception as e:
            logger.warning("CTD chem-disease download failed: %s", e)
            return None

    def load_ctd_chem_disease(self, path: str, direct_only: bool = True) -> BatchResult:
        """Load CTD chemical-disease associations.

        File format (tab-delimited, # comment headers):
            ChemicalName, ChemicalID(MESH), CasRN, DiseaseName, DiseaseID,
            DirectEvidence, InferenceGeneSymbol, InferenceScore, OmimIDs, PubMedIDs

        DirectEvidence values: 'therapeutic', 'marker/mechanism', or empty (inferred).
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        with gzip.open(path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue

                chem_name = parts[0].strip()
                chem_mesh = parts[1].strip()
                disease_name = parts[3].strip()
                disease_id = parts[4].strip()  # e.g., "MESH:D003920" or "OMIM:222100"
                direct_evidence = parts[5].strip() if len(parts) > 5 else ""

                if direct_only and not direct_evidence:
                    skipped += 1
                    continue

                if not chem_mesh or not disease_id:
                    skipped += 1
                    continue

                chem_canonical = normalize_entity_id(f"Compound_{chem_mesh}")
                disease_canonical = normalize_entity_id(f"Disease_{disease_id}")

                # Determine predicate
                predicate = "associated_with"
                if direct_evidence:
                    ev_lower = direct_evidence.lower()
                    if "therapeutic" in ev_lower:
                        predicate = "treats"
                    elif "marker" in ev_lower:
                        predicate = "associated_with"

                if chem_canonical not in entities:
                    entities[chem_canonical] = (
                        "compound", chem_name or f"Compound_{chem_mesh}",
                        json.dumps({"mesh": chem_mesh}) if chem_mesh else "{}",
                    )
                if disease_canonical not in entities:
                    ext_ids = {}
                    if disease_id.startswith("MESH:"):
                        ext_ids["mesh"] = disease_id.split(":", 1)[1]
                    elif disease_id.startswith("OMIM:"):
                        ext_ids["omim"] = disease_id.split(":", 1)[1]
                    entities[disease_canonical] = (
                        "disease", disease_name or disease_id,
                        json.dumps(ext_ids) if ext_ids else "{}",
                    )

                source_id = f"ctd:cd:{chem_mesh}:{disease_id}"
                content_id = compute_content_id(chem_canonical, predicate, disease_canonical)
                claim_id = compute_claim_id(
                    chem_canonical, predicate, disease_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    chem_canonical, disease_canonical, claim_id, content_id,
                    predicate, "relation", 0.85,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("CTD chem-disease: %d associations parsed, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "CTD chem-disease loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── TISSUES (Jensen Lab Gene-Tissue Associations) ─────────────────

    def download_tissues(self, dest_dir: str) -> str:
        """Download TISSUES knowledge TSV (Jensen Lab). Returns path."""
        import requests

        dest = os.path.join(dest_dir, "human_tissue_knowledge_full.tsv")
        if os.path.exists(dest):
            logger.info("TISSUES already cached at %s", dest)
            return dest

        logger.info("Downloading TISSUES from %s", TISSUES_URL)
        resp = requests.get(TISSUES_URL, timeout=120)
        resp.raise_for_status()
        with open(dest, "w") as f:
            f.write(resp.text)
        logger.info("Downloaded TISSUES to %s (%d lines)", dest, resp.text.count("\n"))
        return dest

    def load_tissues(self, path: str, mapper: GeneIDMapper) -> BatchResult:
        """Load TISSUES gene-tissue associations (Jensen Lab).

        Same format as DISEASES: ENSP  symbol  BTO  tissue_name  source_db  curation  score
        Maps ENSP -> Gene_{entrez} via mapper.
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        mapped = 0
        unmapped = 0

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 7:
                    continue
                ensp = row[0].strip()
                symbol = row[1].strip()
                tissue_id = row[2].strip()   # BTO ID
                tissue_name = row[3].strip()

                if not tissue_id or not ensp:
                    continue

                # Map ENSP -> Gene entity ID
                gene_id = mapper.ensp_to_gene_entity_id(ensp)
                if gene_id is None and symbol:
                    gene_id = mapper.symbol_to_gene_entity_id(symbol)
                if gene_id is None:
                    unmapped += 1
                    continue
                mapped += 1

                tissue_entity = f"Tissue_{tissue_id}"
                gene_canonical = normalize_entity_id(gene_id)
                tissue_canonical = normalize_entity_id(tissue_entity)

                if gene_canonical not in entities:
                    gene_ext = _extract_external_ids(gene_id, "gene")
                    if symbol:
                        gene_ext["symbol"] = symbol
                    entities[gene_canonical] = ("gene", gene_id, json.dumps(gene_ext) if gene_ext else "{}")
                if tissue_canonical not in entities:
                    entities[tissue_canonical] = ("entity", tissue_name or tissue_entity, "{}")

                predicate = "expressed_in"
                source_id = "tissues_knowledge"
                source_type = "database_import"

                content_id = compute_content_id(gene_canonical, predicate, tissue_canonical)
                claim_id = compute_claim_id(
                    gene_canonical, predicate, tissue_canonical,
                    source_id, source_type, timestamp,
                )

                claim_rows.append((
                    gene_canonical, tissue_canonical, claim_id, content_id,
                    predicate, "relates_to", tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("TISSUES: %d mapped, %d unmapped genes", mapped, unmapped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "TISSUES loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── Open Targets (Target-Disease Associations) ────────────────────

    def download_open_targets(self, dest_dir: str) -> str:
        """Download Open Targets overall direct associations (Parquet). Returns path to dir."""
        import requests

        dest = os.path.join(dest_dir, "open_targets_associations")
        if os.path.exists(dest) and os.listdir(dest):
            logger.info("Open Targets already cached at %s", dest)
            return dest

        os.makedirs(dest, exist_ok=True)

        # List the Parquet parts from the FTP directory
        logger.info("Downloading Open Targets associations (Parquet)...")
        try:
            resp = requests.get(OPEN_TARGETS_ASSOC_URL, timeout=60)
            resp.raise_for_status()
            # Parse filenames from directory listing (HTML)
            import re
            parquet_files = re.findall(r'href="(part-[^"]+\.parquet)"', resp.text)
            if not parquet_files:
                # Try a simpler pattern
                parquet_files = re.findall(r'(part-\S+\.parquet)', resp.text)
            if not parquet_files:
                logger.warning("No Parquet files found at %s", OPEN_TARGETS_ASSOC_URL)
                return dest

            for pf in parquet_files:
                pf_path = os.path.join(dest, pf)
                if os.path.exists(pf_path):
                    continue
                url = OPEN_TARGETS_ASSOC_URL + pf
                pf_resp = requests.get(url, timeout=120)
                pf_resp.raise_for_status()
                with open(pf_path, "wb") as f:
                    f.write(pf_resp.content)
            logger.info("Downloaded %d Open Targets Parquet files to %s", len(parquet_files), dest)
        except Exception as e:
            logger.warning("Open Targets download failed: %s", e)

        return dest

    def load_open_targets(self, path: str, mapper: GeneIDMapper | None = None) -> BatchResult:
        """Load Open Targets overall direct associations from Parquet files.

        Each row: targetId (Ensembl ENSG), diseaseId (EFO/Mondo), score, evidenceCount.
        Maps Ensembl gene IDs to our Gene_{entrez} entities when mapper available.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("Open Targets requires pyarrow: pip install pyarrow")
            return BatchResult(ingested=0)

        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0

        # Read all Parquet files in the directory
        parquet_files = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(".parquet")
        )
        if not parquet_files:
            logger.warning("No Parquet files found in %s", path)
            return BatchResult(ingested=0)

        for pf in parquet_files:
            table = pq.read_table(pf, columns=["targetId", "diseaseId", "score"])
            for i in range(len(table)):
                target_id = str(table.column("targetId")[i])  # ENSG00000157764
                disease_id = str(table.column("diseaseId")[i])  # EFO_0000565
                score = float(table.column("score")[i])

                if not target_id or not disease_id:
                    skipped += 1
                    continue

                # Map Ensembl gene ID to our entity format
                # Try mapper first (Ensembl → Entrez), fall back to raw ENSG ID
                gene_eid = None
                if mapper is not None:
                    gene_eid = mapper.ensp_to_gene_entity_id(target_id)
                if gene_eid is None:
                    gene_eid = f"Gene_{target_id}"

                # Normalize disease ID: EFO_0000565 → Disease_EFO:0000565
                disease_eid = f"Disease_{disease_id.replace('_', ':', 1)}"

                gene_canonical = normalize_entity_id(gene_eid)
                disease_canonical = normalize_entity_id(disease_eid)

                if gene_canonical == disease_canonical:
                    skipped += 1
                    continue

                if gene_canonical not in entities:
                    # Resolve gene symbol via mapper if available
                    display = gene_eid
                    if mapper is not None and gene_eid.startswith("Gene_"):
                        entrez_str = gene_eid[5:]
                        if entrez_str.isdigit():
                            sym = mapper.entrez_to_symbol(int(entrez_str))
                            if sym:
                                display = sym
                    ext = json.dumps({"ensembl": target_id})
                    entities[gene_canonical] = ("gene", display, ext)
                if disease_canonical not in entities:
                    entities[disease_canonical] = ("disease", disease_eid, "{}")

                # Map OT score to confidence (OT scores are already 0-1)
                confidence = max(0.50, min(0.95, score))

                predicate = "associated_with"
                source_id = f"opentargets:{target_id}:{disease_id}"
                content_id = compute_content_id(gene_canonical, predicate, disease_canonical)
                claim_id = compute_claim_id(
                    gene_canonical, predicate, disease_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    gene_canonical, disease_canonical, claim_id, content_id,
                    predicate, "relation", confidence,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        logger.info("Open Targets: %d associations parsed, %d skipped", len(claim_rows), skipped)

        if not claim_rows:
            return BatchResult(ingested=0)

        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        elapsed = time.time() - t0
        logger.info(
            "Open Targets loaded: %d entities, %d claims in %.1fs",
            len(entities), len(claim_rows), elapsed,
        )
        return result

    # ── SemMedDB (Literature-Mined Predications) ──────────────────────

    def download_semmeddb(self, dest_dir: str) -> str | None:
        """Download SemMedDB PREDICATION CSV (requires UTS account or manual download).

        Returns path to the gzipped CSV, or None if not available.
        Users must download manually from:
        https://lhncbc.nlm.nih.gov/temp/SemRep_SemMedDB_SKR/SemMedDB_download.html
        and place the file in dest_dir as semmed_predications.csv.gz
        """
        dest = os.path.join(dest_dir, "semmed_predications.csv.gz")
        if os.path.exists(dest):
            logger.info("SemMedDB already cached at %s", dest)
            return dest

        # Try downloading directly (may require UTS authentication)
        import requests
        logger.info("Attempting SemMedDB download (~3GB)...")
        try:
            resp = requests.get(SEMMEDDB_PREDICATIONS_URL, timeout=600, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            logger.info("SemMedDB downloaded: %.1f MB", size_mb)
            return dest
        except Exception as e:
            logger.warning(
                "SemMedDB download failed: %s. "
                "Download manually from https://lhncbc.nlm.nih.gov/temp/SemRep_SemMedDB_SKR/SemMedDB_download.html "
                "and place as %s",
                e, dest,
            )
            return None

    def load_semmeddb(self, path: str) -> BatchResult:
        """Load SemMedDB predications from CSV.

        PREDICATION CSV columns (pipe-delimited or comma-delimited):
            PREDICATION_ID, SENTENCE_ID, PMID, PREDICATE,
            SUBJECT_CUI, SUBJECT_NAME, SUBJECT_SEMTYPE, SUBJECT_NOVELTY,
            OBJECT_CUI, OBJECT_NAME, OBJECT_SEMTYPE, OBJECT_NOVELTY

        130M+ predications, UMLS CUIs, 30 predicate types.
        Filters: skip NEG_ prefixed predicates, skip non-biomedical semantic types.
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0
        processed = 0
        total_ingested = 0

        # Determine if gzipped
        open_fn = gzip.open if path.endswith(".gz") else open

        with open_fn(path, "rt", errors="replace") as f:
            import csv as csv_mod
            reader = csv_mod.reader(f)
            for parts in reader:
                # SemMedDB CSV has 12+ columns (quoted, comma-delimited)
                if len(parts) < 12:
                    skipped += 1
                    continue

                predicate_raw = parts[3].strip()
                subject_cui = parts[4].strip()
                subject_name = parts[5].strip()
                subject_semtype = parts[6].strip()
                object_cui = parts[8].strip()
                object_name = parts[9].strip()
                object_semtype = parts[10].strip()
                pmid = parts[2].strip()

                # Skip negated predicates
                if predicate_raw.startswith("NEG_"):
                    skipped += 1
                    continue

                # Skip non-biomedical semantic types
                subj_type = SEMMEDDB_SEMTYPE_MAP.get(subject_semtype)
                obj_type = SEMMEDDB_SEMTYPE_MAP.get(object_semtype)
                if subj_type is None or obj_type is None:
                    skipped += 1
                    continue

                if not subject_cui or not object_cui:
                    skipped += 1
                    continue
                if subject_cui == object_cui:
                    skipped += 1
                    continue

                # Map predicate
                predicate = SEMMEDDB_PREDICATE_MAP.get(predicate_raw, "associated_with")

                # Build entity IDs from UMLS CUIs
                # SemMedDB CUI field may contain NCBI Gene ID after pipe:
                # "C0079419|7157" → use "gene_7157" to match existing entities
                def _semmed_entity_id(cui: str, etype: str) -> str:
                    prefix = {"gene": "gene", "protein": "protein",
                              "compound": "compound", "disease": "disease",
                              "phenotype": "phenotype"}.get(etype, "entity")
                    if "|" in cui and etype in ("gene", "protein"):
                        parts_cui = cui.split("|")
                        ncbi_id = parts_cui[-1]
                        if ncbi_id.isdigit():
                            return f"{prefix}_{ncbi_id}"
                    return f"{prefix}_{cui}"

                subj_eid = _semmed_entity_id(subject_cui, subj_type)
                obj_eid = _semmed_entity_id(object_cui, obj_type)

                subj_canonical = normalize_entity_id(subj_eid)
                obj_canonical = normalize_entity_id(obj_eid)

                if subj_canonical == obj_canonical:
                    skipped += 1
                    continue

                if subj_canonical not in entities:
                    entities[subj_canonical] = (subj_type, subject_name or subj_eid, "{}")
                if obj_canonical not in entities:
                    entities[obj_canonical] = (obj_type, object_name or obj_eid, "{}")

                source_id = f"semmeddb:{pmid}" if pmid else "semmeddb"
                content_id = compute_content_id(subj_canonical, predicate, obj_canonical)
                claim_id = compute_claim_id(
                    subj_canonical, predicate, obj_canonical,
                    source_id, "literature_extraction", timestamp,
                )

                claim_rows.append((
                    subj_canonical, obj_canonical, claim_id, content_id,
                    predicate, "relation", 0.65,
                    "literature_extraction", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

                processed += 1

                # Flush every 5M rows to keep memory bounded
                if processed % 5_000_000 == 0:
                    logger.info("SemMedDB: %dM rows — flushing chunk...", processed // 1_000_000)
                    self._ingest_append_direct(entities, claim_rows, timestamp)
                    total_ingested += len(claim_rows)
                    entities = {}
                    claim_rows = []

        # Final flush
        if claim_rows:
            self._ingest_append_direct(entities, claim_rows, timestamp)
            total_ingested += len(claim_rows)

        elapsed = time.time() - t0
        logger.info(
            "SemMedDB: %d parsed, %d ingested, %d skipped in %.1fs",
            processed, total_ingested, skipped, elapsed,
        )
        return BatchResult(ingested=total_ingested)

    # ---- RTX-KG2c (composite biomedical KG, 40-55M edges) ----

    def download_kg2c(self, dest_dir: str) -> str:
        """Download RTX-KG2c lite JSON. Returns path to gzipped JSON."""
        import requests

        dest = os.path.join(dest_dir, "kg2c_lite.json.gz")
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            if size_mb > 100:  # Valid download > 100MB
                logger.info("KG2c already cached at %s (%.0f MB)", dest, size_mb)
                return dest

        logger.info("Downloading RTX-KG2c lite (~538 MB, may take a while)...")
        resp = requests.get(KG2C_LITE_URL, timeout=3600, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        logger.info("KG2c downloaded: %.1f MB", size_mb)
        return dest

    def load_kg2c(self, path: str) -> BatchResult:
        """Load RTX-KG2c lite from gzipped JSON.

        JSON structure: {"nodes": [...], "edges": [...], "kg2_version": "...", ...}
        Node fields: id, name, category, all_categories
        Edge fields: id, subject, object, predicate, primary_knowledge_source,
                     qualified_predicate, qualified_object_aspect,
                     qualified_object_direction, domain_range_exclusion

        Two-phase loading:
        1. Stream nodes → register entities in LMDB, build curie_to_eid dict
        2. Stream edges → resolve CURIEs via dict → insert claims (Rust resolves
           display names from LMDB)

        Uses streaming JSON parsing (ijson) if available, falls back to
        full in-memory parsing.
        """
        t0 = time.time()
        timestamp = int(t0)

        claim_rows: list[tuple] = []
        skipped = 0
        total_ingested = 0

        def _kg2c_entity_id(curie: str, category: str) -> tuple[str, str] | None:
            """Convert KG2c CURIE to Attest entity ID + type."""
            if not curie:
                return None
            etype = BIOLINK_CATEGORY_MAP.get(category, "entity")
            # Map CURIE prefixes to our entity ID format
            prefix = curie.split(":")[0] if ":" in curie else ""
            if prefix == "NCBIGene":
                gid = curie.split(":", 1)[1]
                return f"Gene_{gid}", etype
            elif prefix == "UniProtKB":
                return f"Protein_{curie}", etype
            elif prefix in ("CHEBI", "DRUGBANK", "PUBCHEM.COMPOUND"):
                return f"Compound_{curie}", etype
            elif prefix in ("MONDO", "DOID", "MESH", "OMIM", "HP", "ORPHANET",
                            "EFO", "NCIT", "UMLS", "SNOMEDCT"):
                if etype == "phenotype" or prefix == "HP":
                    return f"Phenotype_{curie}", "phenotype"
                return f"Disease_{curie}", "disease"
            elif prefix in ("GO", "REACT", "KEGG.PATHWAY", "PANTHER.PATHWAY"):
                return f"Pathway_{curie}", "pathway"
            elif prefix == "HGNC":
                return f"Gene_{curie}", "gene"
            elif prefix in ("ENSEMBL", "ENSG"):
                return f"Gene_{curie}", "gene"
            elif prefix == "PR":
                return f"Protein_{curie}", "protein"
            else:
                # Keep the CURIE as-is for less common prefixes
                return f"Entity_{curie}", etype

        logger.info("Loading KG2c nodes and edges (two-phase)...")

        # Use ijson for streaming if available, else load in memory
        try:
            import ijson
            _has_ijson = True
        except ImportError:
            _has_ijson = False

        # Phase 1: Stream nodes → register entities in LMDB, build CURIE→eid dict.
        # The dict maps CURIE → normalized entity ID (~1GB for 6.7M entries).
        # Display names are stored in LMDB, not in the dict.
        curie_to_eid: dict[str, tuple[str, str]] = {}  # CURIE → (normalized_eid, etype)
        entity_batch: dict[str, tuple[str, str, str]] = {}
        ENTITY_BATCH_SIZE = 500_000
        node_count = 0

        def _register_node(nid: str, name: str, category: str) -> None:
            nonlocal node_count
            result = _kg2c_entity_id(nid, category)
            if not result:
                return
            eid, etype = result
            canonical = normalize_entity_id(eid)
            canonical, ext = _safe_entity_id(canonical)
            curie_to_eid[nid] = (canonical, etype)
            if canonical not in entity_batch:
                entity_batch[canonical] = (etype, name or eid, json.dumps(ext) if ext else "{}")
                flush_buf[canonical] = entity_batch[canonical]
            node_count += 1

        flush_buf: dict[str, tuple[str, str, str]] = {}

        def _flush_entity_batch() -> None:
            """Register buffered entities in LMDB. entity_batch stays in memory for Phase 2."""
            if flush_buf:
                self._register_entities(flush_buf, timestamp)
                flush_buf.clear()

        if _has_ijson:
            logger.info("KG2c Phase 1: streaming nodes to LMDB with ijson...")
            with gzip.open(path, "rb") as f:
                for node in ijson.items(f, "nodes.item"):
                    nid = node.get("id", "")
                    if not nid:
                        continue
                    _register_node(nid, node.get("name", ""), node.get("category", "biolink:NamedThing"))
                    if len(flush_buf) >= ENTITY_BATCH_SIZE:
                        _flush_entity_batch()
                        logger.info("KG2c Phase 1: %dK nodes registered...", node_count // 1_000)
            _flush_entity_batch()
            logger.info("KG2c Phase 1: %d nodes registered in LMDB", node_count)

            # Phase 2: stream edges
            logger.info("KG2c Phase 2: streaming edges...")
            with gzip.open(path, "rb") as f:
                for edge in ijson.items(f, "edges.item"):
                    subj_curie = edge.get("subject", "")
                    obj_curie = edge.get("object", "")
                    pred_raw = edge.get("predicate", "")
                    source = edge.get("primary_knowledge_source", "")

                    if not subj_curie or not obj_curie or not pred_raw:
                        skipped += 1
                        continue

                    subj_lookup = curie_to_eid.get(subj_curie)
                    obj_lookup = curie_to_eid.get(obj_curie)
                    if not subj_lookup or not obj_lookup:
                        skipped += 1
                        continue

                    subj_canonical, _ = subj_lookup
                    obj_canonical, _ = obj_lookup

                    if subj_canonical == obj_canonical:
                        skipped += 1
                        continue

                    predicate = BIOLINK_PREDICATE_MAP.get(pred_raw, "associated_with")
                    source_id = f"kg2c:{source}" if source else "kg2c"
                    content_id = compute_content_id(subj_canonical, predicate, obj_canonical)
                    claim_id = compute_claim_id(
                        subj_canonical, predicate, obj_canonical,
                        source_id, "database_import", timestamp,
                    )

                    claim_rows.append((
                        subj_canonical, obj_canonical, claim_id, content_id,
                        predicate, "relation", 0.80,
                        "database_import", source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))

                    if len(claim_rows) >= 500_000:
                        _r = self._ingest_append_direct(entity_batch, claim_rows, timestamp)
                        total_ingested += _r.ingested
                        logger.info("KG2c Phase 2: %dK edges ingested...", total_ingested // 1_000)
                        claim_rows.clear()
        else:
            # Fallback: load entire JSON into memory (needs ~4-8GB RAM)
            logger.info("KG2c: loading full JSON (no ijson, needs ~5GB RAM)...")
            with gzip.open(path, "rt") as f:
                data = json.load(f)

            logger.info("KG2c Phase 1: registering %d nodes...", len(data.get("nodes", [])))
            for node in data.get("nodes", []):
                nid = node.get("id", "")
                if not nid:
                    continue
                _register_node(nid, node.get("name", ""), node.get("category", "biolink:NamedThing"))
                if len(flush_buf) >= ENTITY_BATCH_SIZE:
                    _flush_entity_batch()
            _flush_entity_batch()
            logger.info("KG2c Phase 1: %d nodes registered", node_count)

            logger.info("KG2c Phase 2: processing %d edges...", len(data.get("edges", [])))
            for edge in data.get("edges", []):
                subj_curie = edge.get("subject", "")
                obj_curie = edge.get("object", "")
                pred_raw = edge.get("predicate", "")
                source = edge.get("primary_knowledge_source", "")

                if not subj_curie or not obj_curie or not pred_raw:
                    skipped += 1
                    continue

                subj_lookup = curie_to_eid.get(subj_curie)
                obj_lookup = curie_to_eid.get(obj_curie)
                if not subj_lookup or not obj_lookup:
                    skipped += 1
                    continue

                subj_canonical, _ = subj_lookup
                obj_canonical, _ = obj_lookup

                if subj_canonical == obj_canonical:
                    skipped += 1
                    continue

                predicate = BIOLINK_PREDICATE_MAP.get(pred_raw, "associated_with")
                source_id = f"kg2c:{source}" if source else "kg2c"
                content_id = compute_content_id(subj_canonical, predicate, obj_canonical)
                claim_id = compute_claim_id(
                    subj_canonical, predicate, obj_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    subj_canonical, obj_canonical, claim_id, content_id,
                    predicate, "relation", 0.80,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

                if len(claim_rows) >= 500_000:
                    _r = self._ingest_append_direct(entity_batch, claim_rows, timestamp)
                    total_ingested += _r.ingested
                    logger.info("KG2c Phase 2: %dK edges ingested...", total_ingested // 1_000)
                    claim_rows.clear()

            del data  # Free memory

        logger.info(
            "KG2c: parsing complete, %d skipped, %d ingested so far",
            skipped, total_ingested,
        )

        if claim_rows:
            _r = self._ingest_append_direct(entity_batch, claim_rows, timestamp)
            total_ingested += _r.ingested

        elapsed = time.time() - t0
        logger.info(
            "KG2c loaded: %d nodes, %d claims in %.1fs",
            node_count, total_ingested, elapsed,
        )
        return BatchResult(ingested=total_ingested)

    # ---- Monarch KG (multi-source biomedical KG, 15.3M edges) ----

    def download_monarch_kg(self, dest_dir: str) -> str:
        """Download Monarch KG tar.gz and extract edges TSV. Returns path to edges file."""
        import requests
        import tarfile

        edges_dest = os.path.join(dest_dir, "monarch-kg_edges.tsv")
        if os.path.exists(edges_dest):
            size_mb = os.path.getsize(edges_dest) / (1024 * 1024)
            if size_mb > 10:  # Valid extraction > 10MB
                logger.info("Monarch KG edges already cached at %s (%.0f MB)", edges_dest, size_mb)
                return edges_dest

        tar_path = os.path.join(dest_dir, "monarch-kg.tar.gz")
        if not os.path.exists(tar_path):
            logger.info("Downloading Monarch KG (~272 MB)...")
            resp = requests.get(MONARCH_KG_URL, timeout=1800, stream=True)
            resp.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            logger.info("Monarch KG downloaded: %.1f MB", os.path.getsize(tar_path) / (1024 * 1024))

        # Extract edges TSV from tar
        logger.info("Extracting Monarch KG edges...")
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                basename = os.path.basename(member.name)
                if basename.startswith("._"):
                    continue
                if "edges" in basename and basename.endswith(".tsv"):
                    member.name = "monarch-kg_edges.tsv"
                    tf.extract(member, dest_dir)
                    logger.info("Extracted edges: %s", edges_dest)
                    break
                if "nodes" in basename and basename.endswith(".tsv"):
                    member.name = "monarch-kg_nodes.tsv"
                    tf.extract(member, dest_dir)

        if not os.path.exists(edges_dest):
            raise FileNotFoundError(f"No edges TSV found in {tar_path}")
        return edges_dest

    def load_monarch_kg(self, path: str) -> BatchResult:
        """Load Monarch KG edges from TSV.

        Columns (15): subject, subject_label, subject_category, subject_taxon,
            subject_taxon_label, negated, predicate, object, object_label,
            object_category, qualifiers, publications, has_evidence,
            primary_knowledge_source, aggregator_knowledge_source

        Node IDs use CURIE format (HGNC:2593, MONDO:0008730, CHEBI:701, etc.).
        Predicates use Biolink model (biolink:has_phenotype, biolink:causes, etc.).
        """
        t0 = time.time()
        timestamp = int(t0)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        skipped = 0
        total_ingested = 0
        total_parsed = 0

        def _monarch_entity_id(curie: str, category: str, label: str) -> tuple[str, str, str] | None:
            """Convert Monarch CURIE to (attest_id, entity_type, display_name)."""
            if not curie:
                return None
            # Reject biolink: predicates leaked into label column (data quality issue)
            if label.startswith("biolink:"):
                label = ""
            etype = BIOLINK_CATEGORY_MAP.get(category, "entity")
            prefix = curie.split(":")[0] if ":" in curie else ""

            # Map well-known CURIE prefixes
            if prefix == "HGNC":
                return f"Gene_{curie}", "gene", label or curie
            elif prefix == "NCBIGene":
                gid = curie.split(":", 1)[1]
                return f"Gene_{gid}", "gene", label or curie
            elif prefix in ("MGI", "ZFIN", "RGD", "WB", "FB", "SGD"):
                # Model organism gene IDs
                return f"Gene_{curie}", "gene", label or curie
            elif prefix == "UniProtKB":
                return f"Protein_{curie}", "protein", label or curie
            elif prefix in ("CHEBI", "DrugCentral"):
                return f"Compound_{curie}", "compound", label or curie
            elif prefix in ("MONDO", "DOID", "OMIM", "Orphanet", "DECIPHER"):
                return f"Disease_{curie}", "disease", label or curie
            elif prefix in ("HP", "MP", "ZP", "WBPhenotype", "XPO"):
                return f"Phenotype_{curie}", "phenotype", label or curie
            elif prefix in ("GO",):
                if etype == "biological_process":
                    return f"BiologicalProcess_{curie}", "biological_process", label or curie
                elif etype == "molecular_function":
                    return f"MolecularFunction_{curie}", "molecular_function", label or curie
                elif etype == "cellular_component":
                    return f"CellularComponent_{curie}", "cellular_component", label or curie
                return f"Pathway_{curie}", "pathway", label or curie
            elif prefix in ("REACT", "KEGG"):
                return f"Pathway_{curie}", "pathway", label or curie
            elif prefix in ("UBERON", "CL", "CLO"):
                if prefix == "CL" or prefix == "CLO":
                    return f"CellType_{curie}", "cell_line", label or curie
                return f"Anatomy_{curie}", "entity", label or curie
            elif prefix == "NCBITaxon":
                return f"Organism_{curie}", "organism", label or curie
            else:
                return f"Entity_{curie}", etype, label or curie

        with open(path, "r", errors="replace") as f:
            header = f.readline().rstrip("\n").split("\t")
            # Find column indices
            col_map = {name: i for i, name in enumerate(header)}
            subj_idx = col_map.get("subject", 0)
            subj_label_idx = col_map.get("subject_label", 1)
            subj_cat_idx = col_map.get("subject_category", 2)
            neg_idx = col_map.get("negated", 5)
            pred_idx = col_map.get("predicate", 6)
            obj_idx = col_map.get("object", 7)
            obj_label_idx = col_map.get("object_label", 8)
            obj_cat_idx = col_map.get("object_category", 9)
            source_idx = col_map.get("primary_knowledge_source", 13)

            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(subj_idx, obj_idx, pred_idx):
                    skipped += 1
                    continue

                # Skip negated assertions
                negated = parts[neg_idx] if neg_idx < len(parts) else ""
                if negated.lower() in ("true", "1"):
                    skipped += 1
                    continue

                subj_curie = parts[subj_idx].strip()
                obj_curie = parts[obj_idx].strip()
                pred_raw = parts[pred_idx].strip()
                subj_label = parts[subj_label_idx].strip() if subj_label_idx < len(parts) else ""
                obj_label = parts[obj_label_idx].strip() if obj_label_idx < len(parts) else ""
                subj_cat = parts[subj_cat_idx].strip() if subj_cat_idx < len(parts) else "biolink:NamedThing"
                obj_cat = parts[obj_cat_idx].strip() if obj_cat_idx < len(parts) else "biolink:NamedThing"
                source = parts[source_idx].strip() if source_idx < len(parts) else ""

                if not subj_curie or not obj_curie or not pred_raw:
                    skipped += 1
                    continue

                subj_result = _monarch_entity_id(subj_curie, subj_cat, subj_label)
                obj_result = _monarch_entity_id(obj_curie, obj_cat, obj_label)
                if not subj_result or not obj_result:
                    skipped += 1
                    continue

                subj_eid, subj_type, subj_name = subj_result
                obj_eid, obj_type, obj_name = obj_result
                subj_canonical = normalize_entity_id(subj_eid)
                subj_canonical, subj_ext = _safe_entity_id(subj_canonical)
                obj_canonical = normalize_entity_id(obj_eid)
                obj_canonical, obj_ext = _safe_entity_id(obj_canonical)

                if subj_canonical == obj_canonical:
                    skipped += 1
                    continue

                predicate = BIOLINK_PREDICATE_MAP.get(pred_raw, "associated_with")

                if subj_canonical not in entities:
                    entities[subj_canonical] = (subj_type, subj_name, json.dumps(subj_ext) if subj_ext else "{}")
                if obj_canonical not in entities:
                    entities[obj_canonical] = (obj_type, obj_name, json.dumps(obj_ext) if obj_ext else "{}")

                total_parsed += 1
                source_id = f"monarch:{source}" if source else "monarch"
                content_id = compute_content_id(subj_canonical, predicate, obj_canonical)
                claim_id = compute_claim_id(
                    subj_canonical, predicate, obj_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    subj_canonical, obj_canonical, claim_id, content_id,
                    predicate, "relation", 0.82,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

                if len(claim_rows) >= 500_000:
                    _r = self._ingest_append_direct(entities, claim_rows, timestamp)
                    total_ingested += _r.ingested
                    logger.info("Monarch KG: %dK edges ingested so far...", total_ingested // 1_000)
                    claim_rows.clear()

        logger.info(
            "Monarch KG: %d edges parsed, %d skipped, %d ingested so far",
            total_parsed, skipped, total_ingested,
        )

        if claim_rows:
            _r = self._ingest_append_direct(entities, claim_rows, timestamp)
            total_ingested += _r.ingested

        elapsed = time.time() - t0
        logger.info(
            "Monarch KG loaded: %d entities, %d claims in %.1fs",
            len(entities), total_ingested, elapsed,
        )
        return BatchResult(ingested=total_ingested)

    # ---- PharMeBINet (composite KG, 15.9M edges, 48 sources) ----

    def download_pharmebinet(self, dest_dir: str) -> tuple[str, str]:
        """Download PharMeBINet TSV archive from Zenodo. Returns (nodes_path, edges_path)."""
        import requests
        import tarfile

        nodes_dest = os.path.join(dest_dir, "pharmebinet_nodes.tsv")
        edges_dest = os.path.join(dest_dir, "pharmebinet_edges.tsv")
        if os.path.exists(nodes_dest) and os.path.exists(edges_dest):
            n_size = os.path.getsize(nodes_dest) / (1024 * 1024)
            e_size = os.path.getsize(edges_dest) / (1024 * 1024)
            if n_size > 10 and e_size > 10:
                logger.info("PharMeBINet cached: nodes %.0f MB, edges %.0f MB", n_size, e_size)
                return nodes_dest, edges_dest

        tar_path = os.path.join(dest_dir, "pharmebinet.tar.gz")
        if not os.path.exists(tar_path):
            logger.info("Downloading PharMeBINet (~863 MB)...")
            resp = requests.get(PHARMEBINET_URL, timeout=3600, stream=True)
            resp.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            logger.info("PharMeBINet downloaded: %.1f MB", os.path.getsize(tar_path) / (1024 * 1024))

        logger.info("Extracting PharMeBINet TSVs...")
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                basename = os.path.basename(member.name)
                if basename.startswith("._"):
                    continue
                if basename == "nodes.tsv":
                    member.name = "pharmebinet_nodes.tsv"
                    tf.extract(member, dest_dir)
                elif basename == "edges.tsv":
                    member.name = "pharmebinet_edges.tsv"
                    tf.extract(member, dest_dir)

        if not os.path.exists(nodes_dest) or not os.path.exists(edges_dest):
            raise FileNotFoundError(f"Could not extract PharMeBINet TSVs from {tar_path}")
        return nodes_dest, edges_dest

    def load_pharmebinet(self, nodes_path: str, edges_path: str) -> BatchResult:
        """Load PharMeBINet from nodes + edges TSV files.

        Two-phase approach:
        1. Load nodes.tsv → register entities in LMDB, build node_id → eid lookup
        2. Load edges.tsv → resolve via lookup → insert claims (Rust resolves
           display names from LMDB)

        nodes.tsv columns: node_id, labels, properties, name, identifier, resource, license, source, url
        edges.tsv columns: relationship_id, type, start_id, end_id, properties, resource, license, source, url
        """
        t0 = time.time()
        timestamp = int(t0)

        def _pbn_entity_id(identifier: str, label: str, name: str) -> tuple[str, str, str] | None:
            """Convert PharMeBINet node to (attest_id, entity_type, display_name)."""
            etype = PHARMEBINET_LABEL_MAP.get(label, "entity")

            # Map identifier based on entity type
            if label == "Gene":
                return f"Gene_{identifier}", "gene", name or identifier
            elif label in ("Protein", "Target"):
                return f"Protein_{identifier}", "protein", name or identifier
            elif label in ("Chemical", "Compound", "Drug", "Salt", "Metabolite", "Treatment", "Product"):
                return f"Compound_{identifier}", "compound", name or identifier
            elif label in ("Disease",):
                if ":" in identifier:
                    return f"Disease_{identifier}", "disease", name or identifier
                return f"Disease_{identifier}", "disease", name or identifier
            elif label in ("Phenotype", "Symptom", "SideEffect"):
                return f"Phenotype_{identifier}", "phenotype", name or identifier
            elif label == "BiologicalProcess":
                return f"BiologicalProcess_{identifier}", "biological_process", name or identifier
            elif label == "MolecularFunction":
                return f"MolecularFunction_{identifier}", "molecular_function", name or identifier
            elif label == "CellularComponent":
                return f"CellularComponent_{identifier}", "cellular_component", name or identifier
            elif label == "Pathway":
                return f"Pathway_{identifier}", "pathway", name or identifier
            elif label in ("GeneVariant", "Variant", "Haplotype", "RNA"):
                return f"Gene_{identifier}", "gene", name or identifier
            else:
                return f"Entity_{identifier}", etype, name or identifier

        def _resolve_verb(edge_type: str) -> str:
            """Extract verb from edge type like TREATS_CHtD -> TREATS."""
            for verb, predicate in PHARMEBINET_VERB_MAP.items():
                if edge_type.startswith(verb):
                    return predicate
            return "associated_with"

        # Phase 1: Load nodes → register entities in LMDB
        logger.info("PharMeBINet Phase 1: loading nodes...")
        node_lookup: dict[str, tuple[str, str, str]] = {}  # int_id -> (identifier, label, name)
        entity_batch: dict[str, tuple[str, str, str]] = {}

        with open(nodes_path, "r", errors="replace") as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                node_id = parts[0].strip()
                labels = parts[1].strip()
                name = parts[3].strip()
                identifier = parts[4].strip()

                if not node_id or not identifier:
                    continue

                primary_label = labels.split("|")[0] if "|" in labels else labels
                node_lookup[node_id] = (identifier, primary_label, name)

                # Register entity in LMDB
                result = _pbn_entity_id(identifier, primary_label, name)
                if result:
                    eid, etype, display = result
                    canonical = normalize_entity_id(eid)
                    canonical, ext = _safe_entity_id(canonical)
                    if canonical not in entity_batch:
                        entity_batch[canonical] = (etype, display, json.dumps(ext) if ext else "{}")

        # Register all entities in one batch
        self._register_entities(entity_batch, timestamp)
        logger.info("PharMeBINet Phase 1: %d nodes loaded, %d entities registered",
                     len(node_lookup), len(entity_batch))

        # Phase 2: Process edges — pass entity_batch so Rust uses in-memory
        # HashMap (84K entities ≈ few MB) instead of LMDB B+ tree lookups
        claim_rows: list[tuple] = []
        skipped = 0
        total_ingested = 0

        with open(edges_path, "r", errors="replace") as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    skipped += 1
                    continue

                edge_type = parts[1].strip()
                start_id = parts[2].strip()
                end_id = parts[3].strip()

                # Skip negative GO annotations
                if edge_type.startswith("NOT_"):
                    skipped += 1
                    continue

                # Resolve to real identifiers
                start_node = node_lookup.get(start_id)
                end_node = node_lookup.get(end_id)
                if not start_node or not end_node:
                    skipped += 1
                    continue

                s_id, s_label, s_name = start_node
                o_id, o_label, o_name = end_node

                s_result = _pbn_entity_id(s_id, s_label, s_name)
                o_result = _pbn_entity_id(o_id, o_label, o_name)
                if not s_result or not o_result:
                    skipped += 1
                    continue

                subj_eid, subj_type, subj_name = s_result
                obj_eid, obj_type, obj_name = o_result
                subj_canonical = normalize_entity_id(subj_eid)
                subj_canonical, _ = _safe_entity_id(subj_canonical)
                obj_canonical = normalize_entity_id(obj_eid)
                obj_canonical, _ = _safe_entity_id(obj_canonical)

                if subj_canonical == obj_canonical:
                    skipped += 1
                    continue

                predicate = _resolve_verb(edge_type)

                source_id = f"pharmebinet:{edge_type}"
                content_id = compute_content_id(subj_canonical, predicate, obj_canonical)
                claim_id = compute_claim_id(
                    subj_canonical, predicate, obj_canonical,
                    source_id, "database_import", timestamp,
                )

                claim_rows.append((
                    subj_canonical, obj_canonical, claim_id, content_id,
                    predicate, "relation", 0.82,
                    "database_import", source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

                # Flush periodically to limit peak memory (~500K rows ≈ 500MB)
                if len(claim_rows) >= 500_000:
                    _r = self._ingest_append_direct(entity_batch, claim_rows, timestamp)
                    total_ingested += _r.ingested
                    logger.info("PharMeBINet Phase 2: %dK edges ingested...", total_ingested // 1_000)
                    claim_rows.clear()

        logger.info(
            "PharMeBINet: %d edges parsed (total), %d skipped",
            total_ingested + len(claim_rows), skipped,
        )

        # Flush remaining
        if claim_rows:
            _r = self._ingest_append_direct(entity_batch, claim_rows, timestamp)
            total_ingested += _r.ingested

        elapsed = time.time() - t0
        logger.info(
            "PharMeBINet loaded: %d nodes, %d claims in %.1fs",
            len(node_lookup), total_ingested, elapsed,
        )
        return BatchResult(ingested=total_ingested)

    def _register_entities(
        self,
        entities: dict[str, tuple[str, str, str]],
        timestamp: int,
    ) -> int:
        """Register entities in the store without inserting claims.

        Phase 1 of two-phase bulk loading. Entities become immediately
        queryable via the store's indexes (LMDB), so subsequent
        insert_bulk calls with entities={} can resolve display names
        from the store instead of from a passed-in dict.

        Returns the number of entities registered.
        """
        store = self._pipeline._store
        if hasattr(store, "upsert_entities_batch"):
            return store.upsert_entities_batch(entities, timestamp)
        # Fallback: use insert_bulk with no claims (registers entities only)
        if hasattr(store, "insert_bulk"):
            store.insert_bulk(entities, [], timestamp)
            return len(entities)
        return 0

    def _ingest_append_direct(
        self,
        entities: dict[str, tuple[str, str, str]],
        claim_rows: list[tuple],
        timestamp: int,
    ) -> BatchResult:
        """Direct entity/claim insertion via the Rust store.

        Prefers insert_bulk (batch Rust path, ~100x faster) when available.
        Falls back to per-claim Python→Rust loop for older wheel versions.
        """
        store = self._pipeline._store

        # Fast path: batch insert entirely in Rust (release GIL for Phase 2)
        # Chunk to limit dirty page memory in LMDB and PyO3 transfer size.
        if hasattr(store, "insert_bulk"):
            CHUNK = 500_000
            total = 0
            for i in range(0, len(claim_rows), CHUNK):
                chunk = claim_rows[i:i + CHUNK]
                total += store.insert_bulk(entities, chunk, timestamp)
                if i + CHUNK < len(claim_rows):
                    logger.info("  bulk insert progress: %d/%d", i + CHUNK, len(claim_rows))
            return BatchResult(ingested=total)

        # Slow path: per-claim Python→Rust insertion
        from attestdb.core.types import (
            Claim,
            ClaimStatus,
            EntityRef,
            PredicateRef,
            Provenance,
        )

        # Upsert all entities (idempotent — handles dedup)
        for eid, (etype, display, ext_json) in entities.items():
            ext_ids = json.loads(ext_json) if ext_json else {}
            store.upsert_entity(eid, etype, display, ext_ids, timestamp)

        # Insert claims
        ingested = 0
        for row in claim_rows:
            subj_id, obj_id = row[0], row[1]
            claim_id, content_id = row[2], row[3]
            pred_id, pred_type = row[4], row[5]
            confidence = float(row[6])
            source_type, source_id = row[7], row[8]
            ts = row[14]

            # Look up entity types from entities dict
            subj_info = entities.get(subj_id, ("entity", subj_id, "{}"))
            obj_info = entities.get(obj_id, ("entity", obj_id, "{}"))

            claim = Claim(
                claim_id=claim_id,
                content_id=content_id,
                subject=EntityRef(
                    id=subj_id, entity_type=subj_info[0],
                    display_name=subj_info[1],
                ),
                predicate=PredicateRef(id=pred_id, predicate_type=pred_type),
                object=EntityRef(
                    id=obj_id, entity_type=obj_info[0],
                    display_name=obj_info[1],
                ),
                confidence=confidence,
                provenance=Provenance(
                    source_type=source_type,
                    source_id=source_id,
                ),
                timestamp=ts,
                status=ClaimStatus.ACTIVE,
            )
            store.insert_claim(claim)
            ingested += 1

        return BatchResult(ingested=ingested)

    # --- Batch 2 loaders ---

    def load_dgidb(self, path: str) -> BatchResult:
        """Load DGIdb drug-gene interactions from NDJSON (GraphQL format).

        Each line is a JSON object with gene, drug, interactionTypes, and
        optional interactionScore. Skips records with empty gene name.

        Interaction type mapping:
            inhibitor -> inhibits, antagonist -> inhibits,
            agonist -> activates, activator -> activates,
            substrate/binder/ligand -> binds,
            default -> interacts_with
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        ITYPE_MAP = {
            "inhibitor": "inhibits", "antagonist": "inhibits",
            "inverse agonist": "inhibits", "negative modulator": "inhibits",
            "blocker": "inhibits", "suppressor": "inhibits",
            "agonist": "activates", "activator": "activates",
            "positive modulator": "activates", "inducer": "activates",
            "stimulator": "activates", "potentiator": "activates",
            "substrate": "binds", "binder": "binds", "ligand": "binds",
        }

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gene_info = rec.get("gene", {})
                drug_info = rec.get("drug", {})
                gene_name = (gene_info.get("name") or "").strip()
                drug_name = (drug_info.get("name") or "").strip()
                if not gene_name or not drug_name:
                    continue

                gene_concept = (gene_info.get("conceptId") or "").strip()
                drug_concept = (drug_info.get("conceptId") or "").strip()

                gene_entity = f"Gene_{gene_name}"
                drug_entity = f"Compound_{drug_name}"
                gene_canonical = normalize_entity_id(gene_entity)
                drug_canonical = normalize_entity_id(drug_entity)

                gene_ext_base = {}
                if gene_concept:
                    gene_ext_base["concept_id"] = gene_concept
                gene_canonical, gene_ext = _safe_entity_id(gene_canonical, ext_ids=gene_ext_base)

                drug_ext_base = {}
                if drug_concept:
                    drug_ext_base["concept_id"] = drug_concept
                drug_canonical, drug_ext = _safe_entity_id(drug_canonical, ext_ids=drug_ext_base)

                if gene_canonical not in entities:
                    entities[gene_canonical] = ("gene", gene_name, json.dumps(gene_ext) if gene_ext else "{}")
                if drug_canonical not in entities:
                    entities[drug_canonical] = ("compound", drug_name, json.dumps(drug_ext) if drug_ext else "{}")

                itypes = rec.get("interactionTypes") or [{}]
                if not itypes:
                    itypes = [{}]
                for itype_rec in itypes:
                    itype_slug = (itype_rec.get("type") or "unknown").lower()
                    pred_id = ITYPE_MAP.get(itype_slug, "interacts_with")
                    source_id = f"dgidb:{gene_name}:{drug_name}:{itype_slug}"
                    source_type = "database_import"

                    claim_id = compute_claim_id(
                        drug_canonical, pred_id, gene_canonical,
                        source_id, source_type, timestamp,
                    )
                    if claim_id in seen_claim_ids:
                        continue
                    seen_claim_ids.add(claim_id)

                    content_id = compute_content_id(drug_canonical, pred_id, gene_canonical)
                    claim_rows.append((
                        drug_canonical, gene_canonical,
                        claim_id, content_id, pred_id, "relates_to",
                        tier1_confidence(source_type),
                        source_type, source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))
                    count += 1

        logger.info("DGIdb: %d drug-gene interactions parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("DGIdb loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_omim(self, path: str) -> BatchResult:
        """Load OMIM disease-gene associations from mim2gene.txt.

        Format (TSV, # comments): MIM Number, MIM Entry Type, Entrez Gene ID,
        Approved Gene Symbol, Ensembl Gene ID.

        Keeps only 'phenotype' and 'gene/phenotype' entries with valid Entrez ID.
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0

        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                mim_number = parts[0].strip()
                entry_type = parts[1].strip()
                entrez_id = parts[2].strip()
                gene_symbol = parts[3].strip()

                if entry_type not in ("phenotype", "gene/phenotype"):
                    continue
                if not entrez_id or entrez_id == "-":
                    continue

                disease_entity = f"Disease_OMIM:{mim_number}"
                gene_entity = f"Gene_{entrez_id}"
                disease_canonical = normalize_entity_id(disease_entity)
                gene_canonical = normalize_entity_id(gene_entity)

                if disease_canonical not in entities:
                    entities[disease_canonical] = ("disease", f"OMIM:{mim_number}",
                                                    json.dumps({"omim": mim_number}))
                if gene_canonical not in entities:
                    gene_ext = {"ncbi_gene": entrez_id}
                    if gene_symbol:
                        gene_ext["symbol"] = gene_symbol
                    entities[gene_canonical] = ("gene", gene_symbol or gene_entity,
                                                json.dumps(gene_ext))

                pred_id = "associated_with"
                source_id = f"omim:{mim_number}:{entrez_id}"
                source_type = "database_import"

                claim_id = compute_claim_id(
                    disease_canonical, pred_id, gene_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        disease_canonical, pred_id, gene_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(disease_canonical, pred_id, gene_canonical)
                claim_rows.append((
                    disease_canonical, gene_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

        logger.info("OMIM: %d disease-gene associations parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("OMIM loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_gwas_catalog(self, path: str) -> BatchResult:
        """Load GWAS Catalog SNP-trait associations from TSV.

        Produces two claim types:
        - SNP -> trait: 'associated_with' (one per SNP per row)
        - SNP -> gene: 'variant_in_gene' (one per SNP per mapped gene)

        Skips rows with empty SNPS field. Multiple SNPs (semicolon-separated)
        and multiple genes (comma-separated) produce cross-product claims.
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                snps_raw = (row.get("SNPS") or "").strip()
                if not snps_raw:
                    continue
                trait = (row.get("DISEASE/TRAIT") or "").strip()
                genes_raw = (row.get("MAPPED_GENE") or "").strip()
                pubmed = (row.get("PUBMEDID") or "").strip()
                p_value = (row.get("P-VALUE") or "").strip()

                snps = [s.strip() for s in snps_raw.split(";") if s.strip()]
                genes = [g.strip() for g in genes_raw.split(",") if g.strip()] if genes_raw else []

                for snp in snps:
                    snp_entity = f"SNP_{snp}"
                    snp_canonical = normalize_entity_id(snp_entity)
                    snp_canonical, snp_ext_safe = _safe_entity_id(snp_canonical, ext_ids={"rsid": snp})
                    if snp_canonical not in entities:
                        entities[snp_canonical] = ("variant", snp, json.dumps(snp_ext_safe))

                    # SNP -> trait
                    if trait:
                        trait_entity = f"Disease_{trait}"
                        trait_canonical = normalize_entity_id(trait_entity)
                        trait_canonical, trait_ext = _safe_entity_id(trait_canonical)
                        if trait_canonical not in entities:
                            entities[trait_canonical] = ("phenotype", trait, json.dumps(trait_ext) if trait_ext else "{}")

                        pred_id = "associated_with"
                        source_id = f"gwas:{snp}:{pubmed}" if pubmed else f"gwas:{snp}"
                        source_type = "database_import"

                        claim_id = compute_claim_id(
                            snp_canonical, pred_id, trait_canonical,
                            source_id, source_type, timestamp,
                        )
                        if claim_id in seen_claim_ids:
                            timestamp += 1
                            claim_id = compute_claim_id(
                                snp_canonical, pred_id, trait_canonical,
                                source_id, source_type, timestamp,
                            )
                        seen_claim_ids.add(claim_id)

                        content_id = compute_content_id(snp_canonical, pred_id, trait_canonical)
                        claim_rows.append((
                            snp_canonical, trait_canonical,
                            claim_id, content_id, pred_id, "relates_to",
                            tier1_confidence(source_type),
                            source_type, source_id,
                            "", "[]", "", "", "", timestamp, "active",
                        ))
                        count += 1

                    # SNP -> gene (variant_in_gene)
                    for gene_name in genes:
                        gene_entity = f"Gene_{gene_name}"
                        gene_canonical = normalize_entity_id(gene_entity)
                        gene_canonical, gene_ext = _safe_entity_id(gene_canonical)
                        if gene_canonical not in entities:
                            entities[gene_canonical] = ("gene", gene_name, json.dumps(gene_ext) if gene_ext else "{}")

                        pred_id = "variant_in_gene"
                        source_id = f"gwas:{snp}:{gene_name}:{pubmed}" if pubmed else f"gwas:{snp}:{gene_name}"
                        source_type = "database_import"

                        claim_id = compute_claim_id(
                            snp_canonical, pred_id, gene_canonical,
                            source_id, source_type, timestamp,
                        )
                        if claim_id in seen_claim_ids:
                            timestamp += 1
                            claim_id = compute_claim_id(
                                snp_canonical, pred_id, gene_canonical,
                                source_id, source_type, timestamp,
                            )
                        seen_claim_ids.add(claim_id)

                        content_id = compute_content_id(snp_canonical, pred_id, gene_canonical)
                        claim_rows.append((
                            snp_canonical, gene_canonical,
                            claim_id, content_id, pred_id, "relates_to",
                            tier1_confidence(source_type),
                            source_type, source_id,
                            "", "[]", "", "", "", timestamp, "active",
                        ))
                        count += 1

        logger.info("GWAS Catalog: %d associations parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("GWAS Catalog loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_bindingdb(self, path: str) -> BatchResult:
        """Load BindingDB protein-ligand interactions from TSV.

        Columns: UniProt ID, Target Name, Ligand Name, Ki/Kd/IC50/EC50 (nM).
        Skips rows with empty UniProt ID or empty Ligand Name.
        All interactions use 'binds' predicate.
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                uniprot_id = (row.get("UniProt (SwissProt) Primary ID of Target Chain") or "").strip()
                target_name = (row.get("Target Name Assigned by Curator or DataSource") or "").strip()
                ligand_name = (row.get("Ligand Name") or "").strip()
                if not uniprot_id or not ligand_name:
                    continue

                protein_entity = f"Protein_{uniprot_id}"
                compound_entity = f"Compound_{ligand_name}"
                protein_canonical = normalize_entity_id(protein_entity)
                compound_canonical = normalize_entity_id(compound_entity)
                compound_canonical, compound_ext = _safe_entity_id(compound_canonical)

                if protein_canonical not in entities:
                    prot_ext = {"uniprot": uniprot_id}
                    entities[protein_canonical] = ("protein", target_name or protein_entity,
                                                    json.dumps(prot_ext))
                if compound_canonical not in entities:
                    entities[compound_canonical] = ("compound", ligand_name, json.dumps(compound_ext) if compound_ext else "{}")

                # Collect best affinity value
                best_affinity = None
                for col in ("Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)"):
                    val = (row.get(col) or "").strip()
                    if val:
                        try:
                            af = float(val)
                            if best_affinity is None or af < best_affinity:
                                best_affinity = af
                        except ValueError:
                            pass

                pred_id = "binds"
                source_id = f"bindingdb:{uniprot_id}:{ligand_name}"
                source_type = "database_import"

                claim_id = compute_claim_id(
                    compound_canonical, pred_id, protein_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        compound_canonical, pred_id, protein_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(compound_canonical, pred_id, protein_canonical)
                claim_rows.append((
                    compound_canonical, protein_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

        logger.info("BindingDB: %d protein-ligand interactions parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("BindingDB loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_ttd(self, path: str) -> BatchResult:
        """Load Therapeutic Target Database from flat file.

        TTD uses a record-based format with tab-separated fields:
            TargetID  FieldType  Value...

        Field types: TARGETID, TARGNAME, GENENAME, UNIPROID, DRUGINFO, INDICATI.
        Produces 'targets' claims (drug -> protein/gene) and 'treats' claims
        (drug -> disease from INDICATI). Skips targets without GENENAME or UNIPROID.
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        # Parse into target records
        targets: dict[str, dict] = {}
        current_id = None

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    current_id = None
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                tid = parts[0].strip()
                field = parts[1].strip()

                if tid not in targets:
                    targets[tid] = {"drugs": [], "indications": []}
                rec = targets[tid]

                if field == "TARGNAME":
                    rec["name"] = parts[2].strip()
                elif field == "GENENAME":
                    rec["gene"] = parts[2].strip()
                elif field == "UNIPROID":
                    rec["uniprot"] = parts[2].strip()
                elif field == "DRUGINFO" and len(parts) >= 5:
                    rec["drugs"].append({
                        "drug_id": parts[2].strip(),
                        "status": parts[3].strip(),
                        "name": parts[4].strip(),
                    })
                elif field == "INDICATI" and len(parts) >= 3:
                    # Extract disease name from "ICD-11: XX  Disease Name [CODE]"
                    raw_ind = parts[2].strip() if len(parts) == 3 else parts[2].strip()
                    # Second field after ICD code has the disease name
                    if len(parts) >= 3:
                        disease_text = parts[-1].strip()
                        # Remove trailing [CODE] if present
                        if "[" in disease_text:
                            disease_text = disease_text[:disease_text.rfind("[")].strip()
                        if disease_text:
                            rec["indications"].append(disease_text)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0

        for tid, rec in targets.items():
            # Need gene or uniprot to identify the target
            uniprot = rec.get("uniprot", "")
            gene = rec.get("gene", "")
            if not uniprot and not gene:
                continue

            if uniprot:
                target_entity = f"Protein_{uniprot}"
                target_type = "protein"
                target_display = rec.get("name", target_entity)
                target_ext = json.dumps({"uniprot": uniprot})
            else:
                target_entity = f"Gene_{gene}"
                target_type = "gene"
                target_display = gene
                target_ext = "{}"

            target_canonical = normalize_entity_id(target_entity)
            if target_canonical not in entities:
                entities[target_canonical] = (target_type, target_display, target_ext)

            for drug in rec.get("drugs", []):
                drug_id = drug.get("drug_id", "")
                drug_name = drug.get("name", "")
                if not drug_id:
                    continue

                drug_entity = f"Compound_{drug_id}"
                drug_canonical = normalize_entity_id(drug_entity)
                if drug_canonical not in entities:
                    entities[drug_canonical] = ("compound", drug_name or drug_id, "{}")

                # Drug targets protein/gene
                pred_id = "targets"
                source_id = f"ttd:{tid}:{drug_id}"
                source_type = "database_import"

                claim_id = compute_claim_id(
                    drug_canonical, pred_id, target_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        drug_canonical, pred_id, target_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(drug_canonical, pred_id, target_canonical)
                claim_rows.append((
                    drug_canonical, target_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

                # Drug treats indications
                for disease_name in rec.get("indications", []):
                    disease_entity = f"Disease_{disease_name}"
                    disease_canonical = normalize_entity_id(disease_entity)
                    disease_canonical, disease_ext = _safe_entity_id(disease_canonical)
                    if disease_canonical not in entities:
                        entities[disease_canonical] = ("disease", disease_name, json.dumps(disease_ext) if disease_ext else "{}")

                    pred_id2 = "treats"
                    source_id2 = f"ttd:{drug_id}:{disease_name}"
                    source_type2 = "database_import"

                    claim_id2 = compute_claim_id(
                        drug_canonical, pred_id2, disease_canonical,
                        source_id2, source_type2, timestamp,
                    )
                    if claim_id2 in seen_claim_ids:
                        timestamp += 1
                        claim_id2 = compute_claim_id(
                            drug_canonical, pred_id2, disease_canonical,
                            source_id2, source_type2, timestamp,
                        )
                    seen_claim_ids.add(claim_id2)

                    content_id2 = compute_content_id(drug_canonical, pred_id2, disease_canonical)
                    claim_rows.append((
                        drug_canonical, disease_canonical,
                        claim_id2, content_id2, pred_id2, "relates_to",
                        tier1_confidence(source_type2),
                        source_type2, source_id2,
                        "", "[]", "", "", "", timestamp, "active",
                    ))
                    count += 1

        logger.info("TTD: %d drug-target/drug-disease associations parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("TTD loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_uniprot(self, path: str) -> BatchResult:
        """Load UniProt protein annotations from TSV.

        Columns: Entry, Gene Names, Protein names, Gene Ontology IDs,
        Involvement in disease.

        Produces claims:
        - gene 'encodes' protein (one per primary gene name)
        - protein 'annotated_with' GO term (one per GO ID)
        - protein 'associated_with' disease (parsed from free text)
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                accession = (row.get("Entry") or "").strip()
                if not accession:
                    continue

                gene_names = (row.get("Gene Names") or "").strip()
                protein_name = (row.get("Protein names") or "").strip()
                go_ids_raw = (row.get("Gene Ontology IDs") or "").strip()
                disease_text = (row.get("Involvement in disease") or "").strip()

                protein_entity = f"Protein_{accession}"
                protein_canonical = normalize_entity_id(protein_entity)
                if protein_canonical not in entities:
                    entities[protein_canonical] = ("protein", protein_name or accession,
                                                    json.dumps({"uniprot": accession}))

                source_type = "database_import"

                # Gene -> Protein: encodes (use first gene name)
                primary_gene = gene_names.split()[0] if gene_names else ""
                if primary_gene:
                    gene_entity = f"Gene_{primary_gene}"
                    gene_canonical = normalize_entity_id(gene_entity)
                    if gene_canonical not in entities:
                        entities[gene_canonical] = ("gene", primary_gene, "{}")

                    pred_id = "encodes"
                    source_id = f"uniprot:{accession}:gene:{primary_gene}"

                    claim_id = compute_claim_id(
                        gene_canonical, pred_id, protein_canonical,
                        source_id, source_type, timestamp,
                    )
                    if claim_id in seen_claim_ids:
                        timestamp += 1
                        claim_id = compute_claim_id(
                            gene_canonical, pred_id, protein_canonical,
                            source_id, source_type, timestamp,
                        )
                    seen_claim_ids.add(claim_id)

                    content_id = compute_content_id(gene_canonical, pred_id, protein_canonical)
                    claim_rows.append((
                        gene_canonical, protein_canonical,
                        claim_id, content_id, pred_id, "relates_to",
                        tier1_confidence(source_type),
                        source_type, source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))
                    count += 1

                # Protein -> GO terms: annotated_with
                if go_ids_raw:
                    go_ids = [g.strip() for g in go_ids_raw.split(";") if g.strip()]
                    for go_id in go_ids:
                        go_entity = f"GO_{go_id}"
                        go_canonical = normalize_entity_id(go_entity)
                        if go_canonical not in entities:
                            entities[go_canonical] = ("biological_process", go_id, "{}")

                        pred_id = "annotated_with"
                        source_id = f"uniprot:{accession}:{go_id}"

                        claim_id = compute_claim_id(
                            protein_canonical, pred_id, go_canonical,
                            source_id, source_type, timestamp,
                        )
                        if claim_id in seen_claim_ids:
                            timestamp += 1
                            claim_id = compute_claim_id(
                                protein_canonical, pred_id, go_canonical,
                                source_id, source_type, timestamp,
                            )
                        seen_claim_ids.add(claim_id)

                        content_id = compute_content_id(protein_canonical, pred_id, go_canonical)
                        claim_rows.append((
                            protein_canonical, go_canonical,
                            claim_id, content_id, pred_id, "relates_to",
                            tier1_confidence(source_type),
                            source_type, source_id,
                            "", "[]", "", "", "", timestamp, "active",
                        ))
                        count += 1

                # Protein -> Disease: associated_with (parse "DISEASE: Name [MIM:xxx].")
                if disease_text:
                    import re
                    diseases = re.findall(r"DISEASE:\s*([^[]+?)(?:\s*\[MIM:\d+\])?\.", disease_text)
                    for disease_name in diseases:
                        disease_name = disease_name.strip()
                        if not disease_name:
                            continue
                        disease_entity = f"Disease_{disease_name}"
                        disease_canonical = normalize_entity_id(disease_entity)
                        disease_canonical, disease_ext = _safe_entity_id(disease_canonical)
                        if disease_canonical not in entities:
                            entities[disease_canonical] = ("disease", disease_name, json.dumps(disease_ext) if disease_ext else "{}")

                        pred_id = "associated_with"
                        source_id = f"uniprot:{accession}:disease:{disease_name}"

                        claim_id = compute_claim_id(
                            protein_canonical, pred_id, disease_canonical,
                            source_id, source_type, timestamp,
                        )
                        if claim_id in seen_claim_ids:
                            timestamp += 1
                            claim_id = compute_claim_id(
                                protein_canonical, pred_id, disease_canonical,
                                source_id, source_type, timestamp,
                            )
                        seen_claim_ids.add(claim_id)

                        content_id = compute_content_id(protein_canonical, pred_id, disease_canonical)
                        claim_rows.append((
                            protein_canonical, disease_canonical,
                            claim_id, content_id, pred_id, "relates_to",
                            tier1_confidence(source_type),
                            source_type, source_id,
                            "", "[]", "", "", "", timestamp, "active",
                        ))
                        count += 1

        logger.info("UniProt: %d protein annotations parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("UniProt loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def _parse_obo(self, path: str, id_prefix: str) -> list[dict]:
        """Parse OBO ontology format into a list of term records.

        Returns list of dicts with keys: id, name, is_a (list), xrefs (list),
        relationships (list of (rel_type, target_id)).
        Skips obsolete terms and terms not matching id_prefix.
        """
        terms = []
        current: dict | None = None

        with open(path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "[Term]":
                    if current and current.get("id", "").startswith(id_prefix) and not current.get("obsolete"):
                        terms.append(current)
                    current = {"is_a": [], "xrefs": [], "relationships": []}
                    continue
                if current is None:
                    continue
                if line.startswith("id: "):
                    current["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    current["name"] = line[6:].strip()
                elif line.startswith("is_obsolete: true"):
                    current["obsolete"] = True
                elif line.startswith("is_a: "):
                    # "is_a: DOID:14566 ! disease of cellular proliferation"
                    parent = line[6:].split("!")[0].strip()
                    if parent:
                        current["is_a"].append(parent)
                elif line.startswith("xref: "):
                    xref = line[6:].strip()
                    if xref:
                        current["xrefs"].append(xref)
                elif line.startswith("relationship: "):
                    # "relationship: has_role CHEBI:35358 ! biguanide..."
                    parts = line[14:].split("!", 1)[0].strip().split(None, 1)
                    if len(parts) == 2:
                        current["relationships"].append((parts[0], parts[1].strip()))

        # Don't forget the last term
        if current and current.get("id", "").startswith(id_prefix) and not current.get("obsolete"):
            terms.append(current)

        return terms

    def load_disease_ontology(self, path: str) -> BatchResult:
        """Load Disease Ontology from OBO format.

        Produces claims:
        - 'subclass_of' for is_a relationships
        - 'has_xref' for cross-references (MESH, OMIM, NCI, etc.)
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        terms = self._parse_obo(path, "DOID:")

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0
        source_type = "database_import"

        for term in terms:
            term_id = term["id"]
            term_name = term.get("name", term_id)
            entity = f"Disease_{term_id}"
            canonical = normalize_entity_id(entity)
            if canonical not in entities:
                entities[canonical] = ("disease", term_name, json.dumps({"doid": term_id}))

            # is_a -> subclass_of
            for parent_id in term["is_a"]:
                parent_entity = f"Disease_{parent_id}"
                parent_canonical = normalize_entity_id(parent_entity)
                if parent_canonical not in entities:
                    entities[parent_canonical] = ("disease", parent_id, json.dumps({"doid": parent_id}))

                pred_id = "subclass_of"
                source_id = f"doid:{term_id}:isa:{parent_id}"

                claim_id = compute_claim_id(
                    canonical, pred_id, parent_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        canonical, pred_id, parent_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(canonical, pred_id, parent_canonical)
                claim_rows.append((
                    canonical, parent_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

            # xrefs -> has_xref
            for xref in term["xrefs"]:
                xref_entity = f"Xref_{xref}"
                xref_canonical = normalize_entity_id(xref_entity)
                if xref_canonical not in entities:
                    entities[xref_canonical] = ("cross_reference", xref, "{}")

                pred_id = "has_xref"
                source_id = f"doid:{term_id}:xref:{xref}"

                claim_id = compute_claim_id(
                    canonical, pred_id, xref_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        canonical, pred_id, xref_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(canonical, pred_id, xref_canonical)
                claim_rows.append((
                    canonical, xref_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

        logger.info("Disease Ontology: %d claims parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("Disease Ontology loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_chebi(self, path: str) -> BatchResult:
        """Load ChEBI chemical ontology from OBO format.

        Produces claims:
        - 'subclass_of' for is_a relationships
        - relationship type (has_role, has_functional_parent, etc.) for
          relationship: lines
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        terms = self._parse_obo(path, "CHEBI:")

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0
        source_type = "database_import"

        for term in terms:
            term_id = term["id"]
            term_name = term.get("name", term_id)
            entity = f"Compound_{term_id}"
            canonical = normalize_entity_id(entity)
            if canonical not in entities:
                entities[canonical] = ("compound", term_name, json.dumps({"chebi": term_id}))

            # is_a -> subclass_of
            for parent_id in term["is_a"]:
                parent_entity = f"Compound_{parent_id}"
                parent_canonical = normalize_entity_id(parent_entity)
                if parent_canonical not in entities:
                    entities[parent_canonical] = ("compound", parent_id,
                                                   json.dumps({"chebi": parent_id}))

                pred_id = "subclass_of"
                source_id = f"chebi:{term_id}:isa:{parent_id}"

                claim_id = compute_claim_id(
                    canonical, pred_id, parent_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        canonical, pred_id, parent_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(canonical, pred_id, parent_canonical)
                claim_rows.append((
                    canonical, parent_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

            # relationship: has_role/has_functional_parent/etc.
            for rel_type, target_id in term["relationships"]:
                target_entity = f"Compound_{target_id}"
                target_canonical = normalize_entity_id(target_entity)
                if target_canonical not in entities:
                    entities[target_canonical] = ("compound", target_id,
                                                   json.dumps({"chebi": target_id}))

                pred_id = rel_type  # e.g. "has_role", "has_functional_parent"
                source_id = f"chebi:{term_id}:{rel_type}:{target_id}"

                claim_id = compute_claim_id(
                    canonical, pred_id, target_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        canonical, pred_id, target_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(canonical, pred_id, target_canonical)
                claim_rows.append((
                    canonical, target_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

        logger.info("ChEBI: %d claims parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("ChEBI loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_mesh(self, path: str) -> BatchResult:
        """Load MeSH descriptors from ASCII descriptor file (d20XX.bin).

        Record format: *NEWRECORD blocks with MH (heading), UI (unique ID),
        MN (tree number), PA (pharmacological action).

        Maps MN tree prefix to entity type:
            C -> disease, D -> compound, A -> anatomy, G -> phenomena,
            B -> organism, default -> mesh_concept.

        Produces claims:
        - 'subclass_of' for hierarchy (child.MN starts with parent.MN + ".")
        - 'pharmacological_action' for PA entries
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        # Parse records
        records: list[dict] = []
        current: dict | None = None

        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "*NEWRECORD":
                    if current and current.get("ui"):
                        records.append(current)
                    current = {"mns": [], "pas": []}
                    continue
                if current is None:
                    continue
                if line.startswith("MH = "):
                    current["name"] = line[5:].strip()
                elif line.startswith("UI = "):
                    current["ui"] = line[5:].strip()
                elif line.startswith("MN = "):
                    current["mns"].append(line[5:].strip())
                elif line.startswith("PA = "):
                    current["pas"].append(line[5:].strip())

        if current and current.get("ui"):
            records.append(current)

        # Map MN prefix to entity type
        TREE_TYPE_MAP = {
            "A": "anatomy", "B": "organism", "C": "disease",
            "D": "compound", "G": "phenomena",
        }

        # Build MN -> UI index for hierarchy resolution
        mn_to_ui: dict[str, str] = {}
        for rec in records:
            for mn in rec["mns"]:
                mn_to_ui[mn] = rec["ui"]

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0
        source_type = "database_import"

        for rec in records:
            ui = rec["ui"]
            name = rec.get("name", ui)
            mns = rec["mns"]

            # Determine entity type from first MN tree number
            etype = "mesh_concept"
            for mn in mns:
                prefix = mn[0] if mn else ""
                if prefix in TREE_TYPE_MAP:
                    etype = TREE_TYPE_MAP[prefix]
                    break

            entity = f"MeSH_{ui}"
            canonical = normalize_entity_id(entity)
            if canonical not in entities:
                entities[canonical] = (etype, name, json.dumps({"mesh": ui}))

            # Hierarchy: find parent via MN
            for mn in mns:
                if "." not in mn:
                    continue  # Top-level node, no parent
                parent_mn = mn.rsplit(".", 1)[0]
                parent_ui = mn_to_ui.get(parent_mn)
                if not parent_ui:
                    continue  # Parent not in our data

                parent_entity = f"MeSH_{parent_ui}"
                parent_canonical = normalize_entity_id(parent_entity)
                # Parent should already be in entities from its own record

                pred_id = "subclass_of"
                source_id = f"mesh:{ui}:isa:{parent_ui}"

                claim_id = compute_claim_id(
                    canonical, pred_id, parent_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        canonical, pred_id, parent_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(canonical, pred_id, parent_canonical)
                claim_rows.append((
                    canonical, parent_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

            # Pharmacological actions
            for pa_name in rec["pas"]:
                pa_entity = f"MeSH_PA_{pa_name}"
                pa_canonical = normalize_entity_id(pa_entity)
                if pa_canonical not in entities:
                    entities[pa_canonical] = ("pharmacological_action", pa_name, "{}")

                pred_id = "pharmacological_action"
                source_id = f"mesh:{ui}:pa:{pa_name}"

                claim_id = compute_claim_id(
                    canonical, pred_id, pa_canonical,
                    source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        canonical, pred_id, pa_canonical,
                        source_id, source_type, timestamp,
                    )
                seen_claim_ids.add(claim_id)

                content_id = compute_content_id(canonical, pred_id, pa_canonical)
                claim_rows.append((
                    canonical, pa_canonical,
                    claim_id, content_id, pred_id, "relates_to",
                    tier1_confidence(source_type),
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))
                count += 1

        logger.info("MeSH: %d claims parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("MeSH loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_clinicaltrials(self, path: str) -> BatchResult:
        """Load ClinicalTrials.gov study data from NDJSON.

        Each line is a JSON object with protocolSection containing:
        - identificationModule.nctId
        - conditionsModule.conditions (list)
        - armsInterventionsModule.interventions (list with type + name)

        Produces claims:
        - trial 'studied_for' condition (one per condition)
        - drug 'investigated_in' trial (DRUG and BIOLOGICAL types only)

        Skips studies with empty nctId. Non-drug intervention types
        (DEVICE, BEHAVIORAL, PROCEDURE, etc.) are ignored.
        """
        t0 = time.time()
        timestamp = int(time.time() * 1_000_000_000)

        entities: dict[str, tuple[str, str, str]] = {}
        claim_rows: list[tuple] = []
        seen_claim_ids: set[str] = set()
        count = 0
        source_type = "database_import"

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                study = json.loads(line)
                proto = study.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                nct_id = (ident.get("nctId") or "").strip()
                if not nct_id:
                    continue

                title = (ident.get("briefTitle") or "").strip()
                conditions = proto.get("conditionsModule", {}).get("conditions", [])
                interventions = proto.get("armsInterventionsModule", {}).get("interventions", [])

                trial_entity = f"ClinicalTrial_{nct_id}"
                trial_canonical = normalize_entity_id(trial_entity)
                if trial_canonical not in entities:
                    entities[trial_canonical] = ("clinical_trial", title or nct_id,
                                                  json.dumps({"nct_id": nct_id}))

                # Trial studied_for conditions
                for condition in conditions:
                    condition = condition.strip()
                    if not condition:
                        continue
                    disease_entity = f"Disease_{condition}"
                    disease_canonical = normalize_entity_id(disease_entity)
                    disease_canonical, disease_ext = _safe_entity_id(disease_canonical)
                    if disease_canonical not in entities:
                        entities[disease_canonical] = ("disease", condition, json.dumps(disease_ext) if disease_ext else "{}")

                    pred_id = "studied_for"
                    source_id = f"ct:{nct_id}:condition:{condition}"

                    claim_id = compute_claim_id(
                        trial_canonical, pred_id, disease_canonical,
                        source_id, source_type, timestamp,
                    )
                    if claim_id in seen_claim_ids:
                        timestamp += 1
                        claim_id = compute_claim_id(
                            trial_canonical, pred_id, disease_canonical,
                            source_id, source_type, timestamp,
                        )
                    seen_claim_ids.add(claim_id)

                    content_id = compute_content_id(trial_canonical, pred_id, disease_canonical)
                    claim_rows.append((
                        trial_canonical, disease_canonical,
                        claim_id, content_id, pred_id, "relates_to",
                        tier1_confidence(source_type),
                        source_type, source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))
                    count += 1

                # Drug/biological investigated_in trial
                for interv in interventions:
                    itype = (interv.get("type") or "").upper()
                    iname = (interv.get("name") or "").strip()
                    if itype not in ("DRUG", "BIOLOGICAL") or not iname:
                        continue

                    drug_entity = f"Compound_{iname}"
                    drug_canonical = normalize_entity_id(drug_entity)
                    drug_canonical, drug_ext = _safe_entity_id(drug_canonical)
                    if drug_canonical not in entities:
                        entities[drug_canonical] = ("compound", iname, json.dumps(drug_ext) if drug_ext else "{}")

                    pred_id = "investigated_in"
                    source_id = f"ct:{nct_id}:drug:{iname}"

                    claim_id = compute_claim_id(
                        drug_canonical, pred_id, trial_canonical,
                        source_id, source_type, timestamp,
                    )
                    if claim_id in seen_claim_ids:
                        timestamp += 1
                        claim_id = compute_claim_id(
                            drug_canonical, pred_id, trial_canonical,
                            source_id, source_type, timestamp,
                        )
                    seen_claim_ids.add(claim_id)

                    content_id = compute_content_id(drug_canonical, pred_id, trial_canonical)
                    claim_rows.append((
                        drug_canonical, trial_canonical,
                        claim_id, content_id, pred_id, "relates_to",
                        tier1_confidence(source_type),
                        source_type, source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))
                    count += 1

        logger.info("ClinicalTrials: %d study claims parsed", count)
        if not claim_rows:
            return BatchResult(ingested=0)
        result = self._ingest_append_direct(entities, claim_rows, timestamp)
        logger.info("ClinicalTrials loaded: %d entities, %d claims in %.1fs",
                     len(entities), len(claim_rows), time.time() - t0)
        return result

    def load_all(
        self,
        dest_dir: str,
        holdout_fraction: float = 0.1,
        include: list[str] | None = None,
        min_string_score: int = 700,
    ) -> tuple[BatchResult, list[ClaimInput]]:
        """Load all data sources in correct order.

        Order: Hetionet (canonical) -> Reactome (no mapping) ->
               DISEASES (ENSP mapping) -> STRING (ENSP mapping, largest) ->
               DisGeNET (Entrez direct) -> DrugBank (symbol mapping) ->
               ChEMBL (symbol mapping, largest) ->
               GOA (UniProt mapping) -> CTD (Entrez direct)

        Args:
            dest_dir: Directory to cache downloaded files.
            holdout_fraction: Fraction of Hetionet edges to withhold.
            include: List of sources to include. None means all 4 core sources.
                     Options: 'hetionet', 'reactome', 'diseases', 'string',
                              'disgenet', 'drugbank', 'chembl', 'goa',
                              'ctd_chem_gene', 'ctd_gene_disease', 'biogrid',
                              'clinvar', 'primekg', 'drkg', 'sider', 'hpo',
                              'intact', 'pharmgkb', 'mondo', 'stitch',
                              'ctd_chem_disease', 'tissues', 'open_targets',
                              'semmeddb', 'kg2c', 'monarch_kg', 'pharmebinet',
                              'dgidb', 'omim', 'gwas_catalog', 'bindingdb',
                              'ttd', 'uniprot', 'disease_ontology', 'chebi',
                              'mesh', 'clinicaltrials'
            min_string_score: Minimum STRING combined score (default 700).

        Returns:
            (combined BatchResult, list of withheld Hetionet ClaimInputs)
        """
        os.makedirs(dest_dir, exist_ok=True)
        # Default: original 4 sources (new sources opt-in)
        sources = set(include) if include else {"hetionet", "reactome", "diseases", "string"}
        total_ingested = 0
        withheld: list[ClaimInput] = []

        # 1. Hetionet (establishes canonical entities)
        if "hetionet" in sources:
            logger.info("=== Loading Hetionet ===")
            hetionet_path = self.download_hetionet(dest_dir)
            result, withheld = self.load_hetionet(
                hetionet_path, holdout_fraction=holdout_fraction, max_edges=None,
            )
            total_ingested += result.ingested
            logger.info("Hetionet: %d edges ingested", result.ingested)

        # 2. Build ID mapper (needed by DISEASES, STRING, DrugBank, ChEMBL, GOA)
        mapper = None
        needs_mapper = {"diseases", "string", "drugbank", "chembl", "goa", "intact",
                        "pharmgkb", "tissues", "open_targets",
                        "stitch", "ctd_chem_gene", "ctd_gene_disease",
                        "biogrid", "clinvar", "hpo"}
        if sources & needs_mapper:
            logger.info("=== Building ID mapper ===")
            mapper = GeneIDMapper(dest_dir)
            nodes_path = None
            if "hetionet" in sources:
                nodes_path = self.download_hetionet_nodes(dest_dir)
            mapper.load(hetionet_nodes_path=nodes_path)
            mapper.load_gene_info()  # +45K genes from NCBI gene_info

            # Load UniProt mapping if DrugBank, ChEMBL, GOA, or IntAct requested
            if sources & {"drugbank", "chembl", "goa", "intact"}:
                try:
                    mapper.load_uniprot_mapping()
                except Exception as e:
                    logger.warning("UniProt mapping unavailable: %s", e)

        # 3. Reactome (no mapping needed)
        if "reactome" in sources:
            logger.info("=== Loading Reactome ===")
            reactome_path = self.download_reactome(dest_dir)
            result = self.load_reactome(reactome_path)
            total_ingested += result.ingested
            logger.info("Reactome: %d edges ingested", result.ingested)

        # 4. DISEASES (ENSP mapping)
        if "diseases" in sources and mapper is not None:
            logger.info("=== Loading DISEASES ===")
            diseases_path = self.download_diseases(dest_dir)
            result = self.load_diseases(diseases_path, mapper)
            total_ingested += result.ingested
            logger.info("DISEASES: %d edges ingested", result.ingested)

        # 5. STRING (ENSP mapping, largest of original 4)
        if "string" in sources and mapper is not None:
            logger.info("=== Loading STRING PPI ===")
            string_path = self.download_string_links(dest_dir)
            result = self.load_string_ppi(string_path, mapper, min_score=min_string_score)
            total_ingested += result.ingested
            logger.info("STRING: %d edges ingested", result.ingested)

        # 6. DisGeNET (direct Entrez gene IDs, UMLS CUI diseases)
        if "disgenet" in sources:
            logger.info("=== Loading DisGeNET ===")
            disgenet_path = self.download_disgenet(dest_dir)
            if disgenet_path:
                result = self.load_disgenet(disgenet_path)
                total_ingested += result.ingested
                logger.info("DisGeNET: %d edges ingested", result.ingested)
            else:
                logger.warning("DisGeNET skipped (no credentials)")

        # 7. DrugBank (DrugBank IDs match Hetionet compounds)
        if "drugbank" in sources and mapper is not None:
            logger.info("=== Loading DrugBank ===")
            drugbank_path = self.download_drugbank_targets(dest_dir)
            if drugbank_path:
                result = self.load_drugbank(drugbank_path, mapper)
                total_ingested += result.ingested
                logger.info("DrugBank: %d edges ingested", result.ingested)
            else:
                logger.warning("DrugBank skipped (no credentials)")

        # 8. ChEMBL (fully open, but large download ~3-4GB)
        if "chembl" in sources and mapper is not None:
            logger.info("=== Loading ChEMBL ===")
            chembl_path = self.download_chembl(dest_dir)
            if chembl_path:
                result = self.load_chembl(chembl_path, mapper)
                total_ingested += result.ingested
                logger.info("ChEMBL: %d edges ingested", result.ingested)
            else:
                logger.warning("ChEMBL skipped (download failed)")

        # 9. GO Annotations (UniProt→Entrez mapping)
        if "goa" in sources and mapper is not None:
            logger.info("=== Loading GO Annotations ===")
            goa_path = self.download_goa(dest_dir)
            result = self.load_goa(goa_path, mapper)
            total_ingested += result.ingested
            logger.info("GOA: %d annotations ingested", result.ingested)

        # 10. CTD chemical-gene interactions (Entrez direct, MeSH chemicals)
        if "ctd_chem_gene" in sources:
            logger.info("=== Loading CTD chemical-gene ===")
            ctd_cg_path = self.download_ctd_chem_gene(dest_dir)
            if ctd_cg_path:
                result = self.load_ctd_chem_gene(ctd_cg_path, mapper=mapper)
                total_ingested += result.ingested
                logger.info("CTD chem-gene: %d interactions ingested", result.ingested)
            else:
                logger.warning("CTD chem-gene skipped (download failed)")

        # 11. CTD gene-disease associations (Entrez direct, MeSH/OMIM diseases)
        if "ctd_gene_disease" in sources:
            logger.info("=== Loading CTD gene-disease ===")
            ctd_gd_path = self.download_ctd_gene_disease(dest_dir)
            if ctd_gd_path:
                result = self.load_ctd_gene_disease(ctd_gd_path, mapper=mapper)
                total_ingested += result.ingested
                logger.info("CTD gene-disease: %d associations ingested", result.ingested)
            else:
                logger.warning("CTD gene-disease skipped (download failed)")

        # 12. BioGRID (experimentally validated PPIs, Entrez direct)
        if "biogrid" in sources:
            logger.info("=== Loading BioGRID ===")
            biogrid_path = self.download_biogrid(dest_dir)
            if biogrid_path:
                result = self.load_biogrid(biogrid_path, mapper=mapper)
                total_ingested += result.ingested
                logger.info("BioGRID: %d interactions ingested", result.ingested)
            else:
                logger.warning("BioGRID skipped (download failed)")

        # 13. ClinVar (gene-disease associations, Entrez direct)
        if "clinvar" in sources:
            logger.info("=== Loading ClinVar ===")
            clinvar_path = self.download_clinvar(dest_dir)
            result = self.load_clinvar(clinvar_path, mapper=mapper)
            total_ingested += result.ingested
            logger.info("ClinVar: %d associations ingested", result.ingested)

        # 14. PrimeKG (multi-relational, no mapping needed)
        if "primekg" in sources:
            logger.info("=== Loading PrimeKG ===")
            primekg_path = self.download_primekg(dest_dir)
            result = self.load_primekg(primekg_path)
            total_ingested += result.ingested
            logger.info("PrimeKG: %d edges ingested", result.ingested)

        # 15. DRKG (drug repurposing KG, no mapping needed)
        if "drkg" in sources:
            logger.info("=== Loading DRKG ===")
            drkg_path = self.download_drkg(dest_dir)
            result = self.load_drkg(drkg_path)
            total_ingested += result.ingested
            logger.info("DRKG: %d triples ingested", result.ingested)

        # 16. SIDER (drug side effects, no mapping needed)
        if "sider" in sources:
            logger.info("=== Loading SIDER ===")
            sider_path = self.download_sider(dest_dir)
            result = self.load_sider(sider_path)
            total_ingested += result.ingested
            logger.info("SIDER: %d drug-SE pairs ingested", result.ingested)

        # 17. HPO (gene-phenotype, no mapping needed)
        if "hpo" in sources:
            logger.info("=== Loading HPO ===")
            hpo_path = self.download_hpo(dest_dir)
            result = self.load_hpo(hpo_path, mapper=mapper)
            total_ingested += result.ingested
            logger.info("HPO: %d gene-phenotype pairs ingested", result.ingested)

        # 18. IntAct (molecular interactions, UniProt mapping)
        if "intact" in sources:
            logger.info("=== Loading IntAct ===")
            intact_path = self.download_intact(dest_dir)
            result = self.load_intact(intact_path, mapper)
            total_ingested += result.ingested
            logger.info("IntAct: %d interactions ingested", result.ingested)

        # 19. PharmGKB (pharmacogenomics, symbol mapping)
        if "pharmgkb" in sources:
            logger.info("=== Loading PharmGKB ===")
            pharmgkb_path = self.download_pharmgkb(dest_dir)
            result = self.load_pharmgkb(pharmgkb_path, mapper)
            total_ingested += result.ingested
            logger.info("PharmGKB: %d relationships ingested", result.ingested)

        # 20. Mondo (disease cross-references, no mapping needed)
        if "mondo" in sources:
            logger.info("=== Loading Mondo ===")
            mondo_path = self.download_mondo(dest_dir)
            result = self.load_mondo(mondo_path)
            total_ingested += result.ingested
            logger.info("Mondo: %d cross-references ingested", result.ingested)

        # 21. STITCH (chemical-protein interactions, PubChem CIDs)
        if "stitch" in sources:
            logger.info("=== Loading STITCH ===")
            stitch_path = self.download_stitch(dest_dir)
            result = self.load_stitch(stitch_path, mapper=mapper)
            total_ingested += result.ingested
            logger.info("STITCH: %d interactions ingested", result.ingested)

        # 22. CTD chemical-disease (direct evidence, MeSH IDs)
        if "ctd_chem_disease" in sources:
            logger.info("=== Loading CTD chemical-disease ===")
            ctd_cd_path = self.download_ctd_chem_disease(dest_dir)
            if ctd_cd_path:
                result = self.load_ctd_chem_disease(ctd_cd_path)
                total_ingested += result.ingested
                logger.info("CTD chem-disease: %d associations ingested", result.ingested)
            else:
                logger.warning("CTD chem-disease skipped (download failed)")

        # 23. TISSUES (Jensen Lab gene-tissue, ENSP mapping)
        if "tissues" in sources and mapper is not None:
            logger.info("=== Loading TISSUES ===")
            tissues_path = self.download_tissues(dest_dir)
            result = self.load_tissues(tissues_path, mapper)
            total_ingested += result.ingested
            logger.info("TISSUES: %d gene-tissue pairs ingested", result.ingested)

        # 24. Open Targets (target-disease associations, Parquet)
        if "open_targets" in sources:
            logger.info("=== Loading Open Targets ===")
            ot_path = self.download_open_targets(dest_dir)
            result = self.load_open_targets(ot_path, mapper)
            total_ingested += result.ingested
            logger.info("Open Targets: %d associations ingested", result.ingested)

        # 25. SemMedDB (literature-mined predications, 130M+)
        if "semmeddb" in sources:
            logger.info("=== Loading SemMedDB ===")
            semmed_path = self.download_semmeddb(dest_dir)
            if semmed_path:
                result = self.load_semmeddb(semmed_path)
                total_ingested += result.ingested
                logger.info("SemMedDB: %d predications ingested", result.ingested)
            else:
                logger.warning("SemMedDB skipped (download failed or requires manual download)")

        # 26. RTX-KG2c (composite biomedical KG, 40-55M edges)
        if "kg2c" in sources:
            logger.info("=== Loading RTX-KG2c ===")
            kg2c_path = self.download_kg2c(dest_dir)
            result = self.load_kg2c(kg2c_path)
            total_ingested += result.ingested
            logger.info("KG2c: %d edges ingested", result.ingested)

        # 27. Monarch KG (multi-source biomedical KG, 15.3M edges)
        if "monarch_kg" in sources:
            logger.info("=== Loading Monarch KG ===")
            monarch_path = self.download_monarch_kg(dest_dir)
            result = self.load_monarch_kg(monarch_path)
            total_ingested += result.ingested
            logger.info("Monarch KG: %d edges ingested", result.ingested)

        # 28. PharMeBINet (composite KG, 15.9M edges, 48 sources)
        if "pharmebinet" in sources:
            logger.info("=== Loading PharMeBINet ===")
            nodes_path, edges_path = self.download_pharmebinet(dest_dir)
            result = self.load_pharmebinet(nodes_path, edges_path)
            total_ingested += result.ingested
            logger.info("PharMeBINet: %d edges ingested", result.ingested)

        # --- Batch 2 loaders (local files, no auto-download) ---

        # 29. DGIdb (drug-gene interactions, NDJSON)
        if "dgidb" in sources:
            dgidb_path = os.path.join(dest_dir, "dgidb_interactions.ndjson")
            if os.path.exists(dgidb_path):
                logger.info("=== Loading DGIdb ===")
                result = self.load_dgidb(dgidb_path)
                total_ingested += result.ingested
                logger.info("DGIdb: %d interactions ingested", result.ingested)
            else:
                logger.warning("DGIdb skipped (file not found: %s)", dgidb_path)

        # 30. OMIM (disease-gene associations, mim2gene.txt)
        if "omim" in sources:
            omim_path = os.path.join(dest_dir, "mim2gene.txt")
            if os.path.exists(omim_path):
                logger.info("=== Loading OMIM ===")
                result = self.load_omim(omim_path)
                total_ingested += result.ingested
                logger.info("OMIM: %d associations ingested", result.ingested)
            else:
                logger.warning("OMIM skipped (file not found: %s)", omim_path)

        # 31. GWAS Catalog (SNP-trait associations, TSV)
        if "gwas_catalog" in sources:
            gwas_path = os.path.join(dest_dir, "gwas_catalog_associations.tsv")
            if not os.path.exists(gwas_path):
                gwas_path = os.path.join(dest_dir, "gwas_catalog_associations.tsv.gz")
            if os.path.exists(gwas_path):
                logger.info("=== Loading GWAS Catalog ===")
                result = self.load_gwas_catalog(gwas_path)
                total_ingested += result.ingested
                logger.info("GWAS Catalog: %d associations ingested", result.ingested)
            else:
                logger.warning("GWAS Catalog skipped (file not found in %s)", dest_dir)

        # 32. BindingDB (protein-ligand interactions, TSV)
        if "bindingdb" in sources:
            bindingdb_path = os.path.join(dest_dir, "BindingDB_All.tsv")
            if not os.path.exists(bindingdb_path):
                bindingdb_path = os.path.join(dest_dir, "BindingDB_All.tsv.gz")
            if os.path.exists(bindingdb_path):
                logger.info("=== Loading BindingDB ===")
                result = self.load_bindingdb(bindingdb_path)
                total_ingested += result.ingested
                logger.info("BindingDB: %d interactions ingested", result.ingested)
            else:
                logger.warning("BindingDB skipped (file not found in %s)", dest_dir)

        # 33. TTD (drug-target database, flat file)
        if "ttd" in sources:
            ttd_path = os.path.join(dest_dir, "ttd_targets.txt")
            if os.path.exists(ttd_path):
                logger.info("=== Loading TTD ===")
                result = self.load_ttd(ttd_path)
                total_ingested += result.ingested
                logger.info("TTD: %d associations ingested", result.ingested)
            else:
                logger.warning("TTD skipped (file not found: %s)", ttd_path)

        # 34. UniProt (protein annotations, TSV)
        if "uniprot" in sources:
            uniprot_path = os.path.join(dest_dir, "uniprot_sprot.tsv")
            if not os.path.exists(uniprot_path):
                uniprot_path = os.path.join(dest_dir, "uniprot_sprot.tsv.gz")
            if os.path.exists(uniprot_path):
                logger.info("=== Loading UniProt ===")
                result = self.load_uniprot(uniprot_path)
                total_ingested += result.ingested
                logger.info("UniProt: %d annotations ingested", result.ingested)
            else:
                logger.warning("UniProt skipped (file not found in %s)", dest_dir)

        # 35. Disease Ontology (OBO format)
        if "disease_ontology" in sources:
            doid_path = os.path.join(dest_dir, "doid.obo")
            if os.path.exists(doid_path):
                logger.info("=== Loading Disease Ontology ===")
                result = self.load_disease_ontology(doid_path)
                total_ingested += result.ingested
                logger.info("Disease Ontology: %d claims ingested", result.ingested)
            else:
                logger.warning("Disease Ontology skipped (file not found: %s)", doid_path)

        # 36. ChEBI (chemical ontology, OBO format)
        if "chebi" in sources:
            chebi_path = os.path.join(dest_dir, "chebi_lite.obo")
            if not os.path.exists(chebi_path):
                chebi_path = os.path.join(dest_dir, "chebi.obo")
            if os.path.exists(chebi_path):
                logger.info("=== Loading ChEBI ===")
                result = self.load_chebi(chebi_path)
                total_ingested += result.ingested
                logger.info("ChEBI: %d claims ingested", result.ingested)
            else:
                logger.warning("ChEBI skipped (file not found in %s)", dest_dir)

        # 37. MeSH (medical subject headings, ASCII descriptor file)
        if "mesh" in sources:
            # MeSH descriptor files are named d20XX.bin
            mesh_path = None
            for fname in sorted(os.listdir(dest_dir), reverse=True):
                if fname.startswith("d20") and fname.endswith(".bin"):
                    mesh_path = os.path.join(dest_dir, fname)
                    break
            if mesh_path:
                logger.info("=== Loading MeSH ===")
                result = self.load_mesh(mesh_path)
                total_ingested += result.ingested
                logger.info("MeSH: %d claims ingested", result.ingested)
            else:
                logger.warning("MeSH skipped (no d20XX.bin file found in %s)", dest_dir)

        # 38. ClinicalTrials.gov (study data, NDJSON)
        if "clinicaltrials" in sources:
            ct_path = os.path.join(dest_dir, "clinicaltrials_studies.ndjson")
            if os.path.exists(ct_path):
                logger.info("=== Loading ClinicalTrials.gov ===")
                result = self.load_clinicaltrials(ct_path)
                total_ingested += result.ingested
                logger.info("ClinicalTrials: %d claims ingested", result.ingested)
            else:
                logger.warning("ClinicalTrials skipped (file not found: %s)", ct_path)

        logger.info("=== All sources loaded: %d total edges ===", total_ingested)
        return BatchResult(ingested=total_ingested), withheld

    def build_entity_name_map(self, dest_dir: str) -> dict[str, str]:
        """Parse Hetionet nodes TSV to build {canonical_id: human_name} map.

        Maps Gene_9021 -> CDK4, Disease_DOID:14330 -> Parkinson's, etc.
        Falls back to display_name from the database for entities not in Hetionet.
        """
        name_map: dict[str, str] = {}

        # Parse Hetionet nodes if available
        nodes_path = os.path.join(dest_dir, "hetionet-nodes.tsv")
        if not os.path.exists(nodes_path):
            try:
                nodes_path = self.download_hetionet_nodes(dest_dir)
            except Exception:
                logger.warning("Could not download Hetionet nodes for name resolution")
                nodes_path = None

        if nodes_path and os.path.exists(nodes_path):
            with open(nodes_path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) < 2:
                        continue
                    raw_id = row[0].strip()
                    name = row[1].strip()
                    if "::" in raw_id:
                        kind, eid = raw_id.split("::", 1)
                        display = f"{kind}_{eid}"
                        canonical = normalize_entity_id(display)
                        if name:
                            name_map[canonical] = name
            logger.info("Entity name map: %d names from Hetionet nodes", len(name_map))

        return name_map

