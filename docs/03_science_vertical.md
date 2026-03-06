# Attest: Science Vertical (Omic)

The computational biology use case at Omic is the first vertical that validates the core Attest engine. It's the most demanding test: provenance matters (experimental vs. computational evidence), contamination is dangerous (LLM hallucinations entering as ground truth), and cross-domain discovery is the highest-value output.

## What the Science Vertical Adds

Everything below is built **on top of** the core engine, not inside it.

### Biology Vocabulary

```python
db.register_vocabulary("bio", {
    "entity_types": [
        "protein", "gene", "compound", "disease", "pathway",
        "cell_line", "organism", "phenotype", "assay",
        "molecular_function", "biological_process", "cellular_component"
    ],
    "predicate_types": [
        "binds", "inhibits", "activates", "upregulates", "downregulates",
        "expressed_in", "associated_with", "substrate_of", "phosphorylates",
        "transported_by", "metabolized_by", "variant_of", "homolog_of"
    ],
    "source_types": [
        "experimental_crystallography", "experimental_spr", "experimental_assay",
        "experimental_mass_spec", "experimental_cryo_em",
        "computational_docking", "computational_md", "computational_alphafold",
        "computational_ml_prediction",
        "literature_extraction", "database_import", "expert_annotation"
    ]
})
```

### Predicate Constraints

```python
db.register_predicate("binds", {
    "subject_types": ["protein", "compound"],
    "object_types": ["protein", "compound", "gene"],
    "expected_payload": "binding_affinity"
})

db.register_predicate("inhibits", {
    "subject_types": ["compound"],
    "object_types": ["protein", "pathway", "biological_process"],
    "expected_payload": "inhibition_data"
})

db.register_predicate("expressed_in", {
    "subject_types": ["gene", "protein"],
    "object_types": ["cell_line", "organism", "cellular_component"],
    "expected_payload": "expression_data"
})
```

### Payload Schemas

```python
db.register_payload_schema("binding_affinity", {
    "properties": {
        "metric": {"enum": ["Kd", "Ki", "IC50", "EC50", "Km"]},
        "value": {"type": "number"},
        "unit": {"enum": ["nM", "uM", "mM", "M", "pM"]},
        "conditions": {
            "type": "object",
            "properties": {
                "pH": {"type": "number"},
                "temperature_C": {"type": "number"},
                "buffer": {"type": "string"}
            }
        }
    },
    "required": ["metric", "value", "unit"]
})

db.register_payload_schema("expression_data", {
    "properties": {
        "metric": {"enum": ["TPM", "FPKM", "fold_change", "log2FC"]},
        "value": {"type": "number"},
        "p_value": {"type": "number"},
        "method": {"enum": ["RNA-seq", "microarray", "qPCR", "proteomics"]}
    },
    "required": ["metric", "value"]
})
```

### Confidence Function (Biology-Calibrated)

The bio vertical registers custom source_type weights for Tier 1 confidence, and domain-specific parameters for Tier 2:

```python
# Bio source type weights (override defaults for bio-specific source types)
BIO_SOURCE_WEIGHTS = {
    "experimental_crystallography": 1.0,
    "experimental_cryo_em": 0.95,
    "experimental_spr": 0.9,
    "experimental_assay": 0.85,
    "experimental_mass_spec": 0.85,
    "computational_alphafold": 0.6,
    "computational_docking": 0.4,
    "computational_md": 0.5,
    "computational_ml_prediction": 0.35,
    "literature_extraction": 0.5,
    "database_import": 0.7,
    "expert_annotation": 0.8,
    "llm_inference": 0.15
}

db.register_source_weights("bio", BIO_SOURCE_WEIGHTS)

# Bio-specific Tier 2 parameters:
# - Biochemistry claims decay slowly (half-life: 5 years vs. default 1 year)
# - Corroboration from different experimental methods is worth more
db.register_confidence_config("bio", {
    "half_life_days": 1825,                    # 5 years
    "cross_method_corroboration_boost": 1.5     # extra boost when sources use different methods
})
```

### Ontology Plugins

External ID namespaces anchored to authoritative databases:

| Namespace | Database | Entity Types |
|-----------|----------|-------------|
| `uniprot` | UniProt | protein |
| `ensembl` | Ensembl | gene |
| `pubchem` | PubChem | compound |
| `chembl` | ChEMBL | compound |
| `mesh` | MeSH | disease, phenotype |
| `go` | Gene Ontology | molecular_function, biological_process, cellular_component |
| `reactome` | Reactome | pathway |
| `doi` | DOI | literature source |

Entities with external IDs have a ground-truth check. Entities that don't map to any known database are flagged as unresolved/suspicious — catching hallucinated entities automatically.

### Seeding Pipeline

Phased, provenance-tagged:

1. **Structured databases** (highest quality): UniProt, ChEMBL, Reactome, KEGG. Clean entities, strong provenance. Entity resolution in conservative mode (exact ID matches only).
2. **Literature extraction** (medium quality): Claims extracted from papers with `source_type: literature_extraction`. Lower confidence. Fuzzy entity resolution activated after base graph is established.
3. **Agent-generated** (variable quality): Claims from Omic research agents with full provenance chains. Curator handles triage.

## Curator (Biology-Specific)

An LLM-powered editorial agent that processes claims before ingestion. **Not part of the core engine** — it's an application that calls `attestdb.ingest()`.

Responsibilities:
- **Significance triage**: Is this claim worth storing? Routine expected results are filtered. Unexpected findings, contradictions, novel connections are kept. Curator has read access to the graph for context on what's already known.
- **Entity resolution**: Maps entities to canonical IDs using ontology plugins. Exact matches auto-resolved. Fuzzy matches flagged. Hallucinated entities (no external ID match) flagged as suspicious.
- **Contradiction detection**: Checks new claims against existing beliefs. Flags contradictions rather than silently overwriting.
- **Provenance tagging**: Enriches source metadata. Classifies source type from agent output.

Design: Haiku-class model, fast triage decisions. Async — claims accepted to staging buffer immediately, curator processes in background. Fast-path for structured sources (database imports skip LLM triage). Batch processing for efficiency.

Curator's own decisions are ingested as claims with `source_type: llm_inference, method: curator_v1`. Auditable, retractable.

**Cost**: ~$75-150/month at 10K-20K claims/day.

## Insight Engine (Biology-Specific)

An active learning system that scans the graph and generates hypotheses. **Not part of the core engine** — it's an application that queries Attest.

### Bridge Predictions

"Subgraph A (oncology) and subgraph B (neurodegeneration) are both connected to entity X through different pathways. This cross-domain connection hasn't been explored."

Process:
1. Embedding similarity as cheap first pass (zero LLM cost) — find semantically similar but structurally disconnected subgraph pairs
2. Filter out domain-specific hub entities (ATP, water, common cofactors — these connect to everything and produce noise)
3. LLM evaluation of top candidates only (budget-constrained)
4. Each prediction includes a suggested validation step

### Gap Identification

"Entity X has 47 relationships but zero data on its interaction with pathway Y, which based on structural similarity to related entities is likely relevant."

Mostly structural analysis (computation, minimal LLM). Compare entity relationship profiles against expected patterns for their entity type.

### Confidence Alerts

"Belief Z has one supporting claim from `llm_inference`, is 8 months old, and is upstream of 12 other beliefs. Recommend experimental corroboration."

Pure computation over provenance metadata. Zero LLM cost.

### Multi-Resolution Summaries

Domain-specific hierarchy:

```
L0: Individual claims (protein X binds compound Y with Ki=5nM)
L1: Entity-level summaries (~10-100 claims per entity)
L2: Pathway/process-level summaries (~10-100 entities)
L3: Disease/system-level summaries
L4: Domain-level summaries (oncology, neuroscience, infectious disease)
```

The core engine provides the grouping framework. The science vertical defines the levels and grouping logic.

### Cost

Daily token budget (configurable). ~$15-50/month at steady state. Embedding generation ~$30-60/month. Total insight engine: ~$45-110/month.

## Example Usage

```python
import attestdb
from attestdb_bio import BioCurator, BioInsightEngine, bio_vocabulary

# Open database with biology vocabulary
db = attestdb.open("./omic_evidence.db", embedding_dim=768)
bio_vocabulary.register(db)

# Curator processes agent output
curator = BioCurator(db, model="haiku")
curator.process_agent_run(agent_id="docking_pipeline_v3", run_id="4523", results=raw_output)

# Query with ContextFrame
frame = db.query(
    focal_entity="TREM2",
    depth=2,
    min_confidence=0.5,
    exclude_source_types=["llm_inference"],
    max_tokens=4000
)

# Time travel
jan_frame = db.at("2026-01-15").query(focal_entity="TREM2")

# Insight engine generates hypotheses
engine = BioInsightEngine(db, daily_token_budget=1_000_000)
bridges = engine.find_bridges(min_confidence=0.6)
gaps = engine.find_gaps(entity_type="protein", min_relationships=20)
alerts = engine.confidence_alerts(max_age_months=6, min_downstream=5)
```

## Why This Vertical Validates the Core

| Core Capability | How Biology Tests It |
|----------------|---------------------|
| Provenance enforcement | Distinguishing crystal structures from docking predictions from LLM inferences |
| Type system | Biology has thousands of entity types and relationship types — stress-tests vocabulary registration |
| Circular evidence detection | Agent A reads graph, produces "corroborating" claim that's actually derived from the original |
| Confidence recomputation | "Show me knowledge excluding all computational predictions" — trivial with claim-native storage |
| Time travel | "What did we know about this target before the retracted paper?" |
| Content-addressed dedup | Multiple agents independently confirming the same binding interaction |
| Compaction | Millions of claims/month from research pipelines — compaction tiers critical |
| Co-located indexes | Insight engine needs graph + vector + temporal in single pass for bridge detection |
| ContextFrame queries | Research agents need structured context, not text chunks |

If the core engine handles biology well — the most demanding provenance requirements, the richest type vocabulary, the highest claim volume — it handles DevOps, legal, finance, and ML experiment tracking trivially.
