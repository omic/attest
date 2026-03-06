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

# GitHub raw URL works for LFS files; raw.githubusercontent.com does not
HETIONET_EDGES_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
HETIONET_NODES_URL = "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv"

# External data source URLs
DISEASES_URL = "https://download.jensenlab.org/human_disease_knowledge_full.tsv"
STRING_LINKS_URL = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
REACTOME_URL = "https://reactome.org/download/current/NCBI2Reactome.txt"

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
        rng = random.Random(seed)
        raw_edges = self._parse_hetionet_edges(path)

        if max_edges and len(raw_edges) > max_edges:
            raw_edges = self._snowball_sample(raw_edges, max_edges, rng)

        claims = self._edges_to_claims(raw_edges)

        # Split holdout
        rng.shuffle(claims)
        holdout_count = int(len(claims) * holdout_fraction)
        withheld = claims[:holdout_count]
        to_ingest = claims[holdout_count:]

        logger.info(
            "Hetionet: %d total edges, %d to ingest, %d withheld",
            len(claims),
            len(to_ingest),
            len(withheld),
        )

        result = self._pipeline.ingest_batch(to_ingest)

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
                ensp, symbol, doid, disease_name = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()

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
                    entities[gene_canonical] = ("gene", gene_id, json.dumps(gene_ext) if gene_ext else "{}")
                if disease_canonical not in entities:
                    disease_ext = _extract_external_ids(disease_entity, "disease")
                    entities[disease_canonical] = ("disease", disease_name or disease_entity, json.dumps(disease_ext) if disease_ext else "{}")

                pred_id = "associates"
                source_id = "diseases_knowledge"
                source_type = "database_import"

                claim_id = compute_claim_id(
                    gene_canonical, pred_id, disease_canonical, source_id, source_type, timestamp,
                )
                if claim_id in seen_claim_ids:
                    timestamp += 1
                    claim_id = compute_claim_id(
                        gene_canonical, pred_id, disease_canonical, source_id, source_type, timestamp,
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
            header = f.readline()  # Skip header
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
                        gene_a_canonical, pred_id, gene_b_canonical, source_id, source_type, timestamp,
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
                        gene_canonical, pred_id, pathway_canonical, source_id, source_type, timestamp,
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

    def _ingest_append_direct(
        self,
        entities: dict[str, tuple[str, str, str]],
        claim_rows: list[tuple],
        timestamp: int,
    ) -> BatchResult:
        """Direct entity/claim insertion via the Rust store.

        Uses upsert_entity (handles dedup naturally) and insert_claim.
        Builds Claim objects from pre-computed row tuples.
        """
        from attestdb.core.types import (
            Claim,
            ClaimStatus,
            EntityRef,
            PredicateRef,
            Provenance,
        )

        store = self._pipeline._store

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

    def load_all(
        self,
        dest_dir: str,
        holdout_fraction: float = 0.1,
        include: list[str] | None = None,
        min_string_score: int = 700,
    ) -> tuple[BatchResult, list[ClaimInput]]:
        """Load all data sources in correct order.

        Order: Hetionet (canonical) -> Reactome (no mapping) ->
               DISEASES (ENSP mapping) -> STRING (ENSP mapping, largest)

        Args:
            dest_dir: Directory to cache downloaded files.
            holdout_fraction: Fraction of Hetionet edges to withhold.
            include: List of sources to include. None means all.
                     Options: 'hetionet', 'reactome', 'diseases', 'string'
            min_string_score: Minimum STRING combined score (default 700).

        Returns:
            (combined BatchResult, list of withheld Hetionet ClaimInputs)
        """
        os.makedirs(dest_dir, exist_ok=True)
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

        # 2. Build ID mapper (needed by DISEASES and STRING)
        mapper = None
        if "diseases" in sources or "string" in sources:
            logger.info("=== Building ID mapper ===")
            mapper = GeneIDMapper(dest_dir)
            nodes_path = None
            if "hetionet" in sources:
                nodes_path = self.download_hetionet_nodes(dest_dir)
            mapper.load(hetionet_nodes_path=nodes_path)

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

        # 5. STRING (ENSP mapping, largest)
        if "string" in sources and mapper is not None:
            logger.info("=== Loading STRING PPI ===")
            string_path = self.download_string_links(dest_dir)
            result = self.load_string_ppi(string_path, mapper, min_score=min_string_score)
            total_ingested += result.ingested
            logger.info("STRING: %d edges ingested", result.ingested)

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

