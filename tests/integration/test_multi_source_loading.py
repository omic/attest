"""Integration tests for multi-source biomedical data loading.

All tests use synthetic data — no network downloads required.
Tests verify DISEASES, STRING, Reactome loaders and cross-source
entity deduplication.
"""

import csv
import os

import pytest

from attestdb.core.hashing import compute_claim_id, compute_content_id
from attestdb.core.normalization import normalize_entity_id
from attestdb.infrastructure.bulk_loader import BulkLoader
from attestdb.infrastructure.id_mapper import GeneIDMapper
from attestdb.infrastructure.attest_db import AttestDB


def _make_synthetic_mapper(tmp_path):
    """Create a GeneIDMapper with synthetic data (no downloads)."""
    mapper = GeneIDMapper(str(tmp_path))
    # Manually populate mappings instead of downloading
    mapper._ensp_to_entrez = {
        "ENSP00000251968": 1029,    # CDK4
        "ENSP00000269305": 7157,    # TP53
        "ENSP00000275493": 7040,    # TGFB1
        "ENSP00000344456": 675,     # BRCA2
        "ENSP00000350283": 672,     # BRCA1
    }
    mapper._symbol_to_entrez = {
        "CDK4": 1029,
        "TP53": 7157,
        "TGFB1": 7040,
        "BRCA2": 675,
        "BRCA1": 672,
    }
    mapper._entrez_to_symbol = {
        1029: "CDK4",
        7157: "TP53",
        7040: "TGFB1",
        675: "BRCA2",
        672: "BRCA1",
    }
    return mapper


def _make_synthetic_diseases_file(path):
    """Create synthetic DISEASES-format TSV."""
    with open(path, "w") as f:
        # Format: ENSP  symbol  DOID  disease_name  source_db  curation  score
        f.write("ENSP00000269305\tTP53\tDOID:1612\tbreast cancer\tTextmining\tKnowledge\t4.5\n")
        f.write("ENSP00000251968\tCDK4\tDOID:1612\tbreast cancer\tExperiments\tKnowledge\t3.8\n")
        f.write("ENSP00000275493\tTGFB1\tDOID:10283\tprostate cancer\tTextmining\tKnowledge\t3.2\n")
        # Unmappable ENSP but mappable symbol
        f.write("ENSP99999999999\tBRCA1\tDOID:1612\tbreast cancer\tExperiments\tKnowledge\t4.0\n")
        # Completely unmappable
        f.write("ENSP00000000000\tFAKEGENE\tDOID:9999\tunknown disease\tTextmining\tKnowledge\t1.0\n")


def _make_synthetic_string_file(path):
    """Create synthetic STRING-format space-separated file."""
    with open(path, "w") as f:
        f.write("protein1 protein2 combined_score\n")
        f.write("9606.ENSP00000269305 9606.ENSP00000251968 900\n")  # TP53-CDK4
        f.write("9606.ENSP00000269305 9606.ENSP00000275493 800\n")  # TP53-TGFB1
        f.write("9606.ENSP00000251968 9606.ENSP00000344456 750\n")  # CDK4-BRCA2
        f.write("9606.ENSP00000269305 9606.ENSP00000344456 500\n")  # TP53-BRCA2 (below threshold)
        f.write("9606.ENSP00000000000 9606.ENSP00000269305 900\n")  # unmappable


def _make_synthetic_reactome_file(path):
    """Create synthetic Reactome-format TSV."""
    with open(path, "w") as f:
        # Format: GeneID  ReactomeID  URL  PathwayName  EvidenceCode  Species
        f.write("7157\tR-HSA-69473\thttp://reactome.org\tG1/S DNA Damage Checkpoints\tIEA\tHomo sapiens\n")
        f.write("1029\tR-HSA-69473\thttp://reactome.org\tG1/S DNA Damage Checkpoints\tIEA\tHomo sapiens\n")
        f.write("672\tR-HSA-73894\thttp://reactome.org\tDNA Repair\tIEA\tHomo sapiens\n")
        f.write("675\tR-HSA-73894\thttp://reactome.org\tDNA Repair\tIEA\tHomo sapiens\n")
        # Non-human (should be filtered)
        f.write("12345\tR-MMU-12345\thttp://reactome.org\tMouse Pathway\tIEA\tMus musculus\n")


def test_diseases_loader_synthetic(make_db, tmp_path):
    """Load synthetic DISEASES data and verify gene-disease edges."""
    db = make_db(embedding_dim=None)
    loader = BulkLoader(db._pipeline)
    mapper = _make_synthetic_mapper(tmp_path)

    diseases_path = str(tmp_path / "diseases.tsv")
    _make_synthetic_diseases_file(diseases_path)

    result = loader.load_diseases(diseases_path, mapper)

    # 4 mappable rows (3 via ENSP, 1 via symbol fallback), 1 unmappable
    assert result.ingested == 4

    stats = db.stats()
    assert stats["total_claims"] == 4
    assert stats["entity_count"] >= 4  # genes + diseases


def test_string_ppi_loader_synthetic(make_db, tmp_path):
    """Load synthetic STRING PPI data and verify gene-gene edges."""
    db = make_db(embedding_dim=None)
    loader = BulkLoader(db._pipeline)
    mapper = _make_synthetic_mapper(tmp_path)

    string_path = str(tmp_path / "string.txt")
    _make_synthetic_string_file(string_path)

    result = loader.load_string_ppi(string_path, mapper, min_score=700)

    # 3 edges pass score threshold and both proteins map:
    # TP53-CDK4 (900), TP53-TGFB1 (800), CDK4-BRCA2 (750)
    # TP53-BRCA2 (500) below threshold, unmappable pair skipped
    assert result.ingested == 3

    stats = db.stats()
    assert stats["total_claims"] == 3
    assert stats["entity_count"] == 4  # TP53, CDK4, TGFB1, BRCA2

    # Verify adjacency
    adj = db.get_adjacency_list()
    tp53 = normalize_entity_id("Gene_7157")
    cdk4 = normalize_entity_id("Gene_1029")
    assert cdk4 in adj.get(tp53, set())


def test_reactome_loader_synthetic(make_db, tmp_path):
    """Load synthetic Reactome data and verify gene-pathway edges."""
    db = make_db(embedding_dim=None)
    loader = BulkLoader(db._pipeline)

    reactome_path = str(tmp_path / "reactome.txt")
    _make_synthetic_reactome_file(reactome_path)

    result = loader.load_reactome(reactome_path)

    # 4 human rows, 1 mouse row filtered
    assert result.ingested == 4

    stats = db.stats()
    assert stats["total_claims"] == 4
    # 4 genes + 2 pathways = 6 entities
    assert stats["entity_count"] == 6

    # Verify pathway entity exists
    pathway_id = normalize_entity_id("Pathway_R-HSA-69473")
    entity = db.get_entity(pathway_id)
    assert entity is not None
    assert entity.entity_type == "pathway"


def test_load_all_combined(make_db, tmp_path):
    """Test load_all() orchestrator with synthetic data files pre-cached."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)

    db = make_db(embedding_dim=None)
    loader = BulkLoader(db._pipeline)

    # Create synthetic Hetionet edges file
    hetionet_path = os.path.join(data_dir, "hetionet-edges.tsv")
    with open(hetionet_path, "w") as f:
        f.write("source\tmetaedge\ttarget\n")
        f.write("Gene::7157\tDaG\tDisease::DOID:1612\n")
        f.write("Gene::1029\tDaG\tDisease::DOID:1612\n")
        f.write("Gene::7157\tGiG\tGene::1029\n")

    # Create synthetic Hetionet nodes file for mapper
    nodes_path = os.path.join(data_dir, "hetionet-nodes.tsv")
    with open(nodes_path, "w") as f:
        f.write("id\tname\n")
        f.write("Gene::7157\tTP53\n")
        f.write("Gene::1029\tCDK4\n")
        f.write("Disease::DOID:1612\tbreast cancer\n")

    # Create synthetic Reactome
    reactome_path = os.path.join(data_dir, "NCBI2Reactome.txt")
    _make_synthetic_reactome_file(reactome_path)

    # We can't easily test DISEASES and STRING without the gene2ensembl download,
    # so test with just hetionet + reactome
    result, withheld = loader.load_all(
        data_dir,
        holdout_fraction=0.0,
        include=["hetionet", "reactome"],
    )

    # 3 Hetionet edges + 4 Reactome edges
    assert result.ingested == 7

    stats = db.stats()
    # Hetionet: Gene_7157, Gene_1029, Disease_DOID:1612
    # Reactome: Gene_7157 (dup), Gene_1029 (dup), Gene_672, Gene_675,
    #           Pathway_R-HSA-69473, Pathway_R-HSA-73894
    assert stats["entity_count"] >= 7
    assert stats["total_claims"] == 7


