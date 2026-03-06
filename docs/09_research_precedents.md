# Attest: Research Precedents & Lessons

Three items on the watchlist. Each has direct precedents we can learn from.

---

## 1. Knowledge Topology at Scale

**The watchlist item:** Leiden community detection on 100M+ claim graphs. Our Phase 2 exit criteria targets <60 seconds on 10M claims. What happens at 100M? 1B?

### The Precedent: Microsoft GraphRAG

GraphRAG is the closest production system to what we're building for topology. It uses **hierarchical Leiden community detection** on LLM-extracted knowledge graphs, producing multi-level topic hierarchies with community summaries at each level. Released open-source July 2024, now in production use.

**What they proved:**
- Leiden on KG-scale graphs (10K-100K entities) works and produces meaningful hierarchical communities
- Multi-level community summaries (LLM-generated) enable "global queries" — questions about the shape of knowledge, not specific entities
- Community-level retrieval outperforms naive RAG on comprehensiveness and diversity (70-80% win rate in their benchmarks)
- Dynamic community selection (Nov 2024 update) traverses the hierarchy top-down, pruning irrelevant branches — directly analogous to our TopicContextFrame assembly

**What they got wrong (our opportunity):**
- GraphRAG communities are **static snapshots** — rebuild the entire index to update. Attest's topics should be incrementally updated as claims arrive
- No provenance. GraphRAG communities have LLM-generated summaries but no chain back to source evidence. An Attest topic knows exactly which claims, from which sources, with what confidence, constitute its knowledge
- No confidence or contradiction awareness. GraphRAG treats all extracted triples equally. Attest's topics inherit confidence distributions from their constituent claims
- No cross-domain bridge detection. GraphRAG identifies communities but doesn't analyze the boundaries between them or identify structural gaps
- Mutually exclusive communities — each entity belongs to exactly one community. Real knowledge has overlapping domain membership. We need overlapping communities, which means **Leiden with the Constant Potts Model (CPM)** quality function instead of modularity, or a separate overlapping community detection step

**Key engineering lesson:** The parallel Leiden implementation (GVE-Leiden, 2024) runs on CSR graph format — exactly what we're building for the Rust engine. On a 64-core CPU, it processes graphs with billions of edges in seconds. The original Traag et al. paper (2019, Nature Scientific Reports) guarantees connected communities (unlike Louvain which produces up to 25% badly connected communities). Our CSR graph index should be designed to feed directly into parallel Leiden from day 1.

**Research to read:**
- Traag, Waltman, van Eck. "From Louvain to Leiden: guaranteeing well-connected communities." Scientific Reports, 2019
- Sahu. "GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting." ACM, 2024
- Microsoft GraphRAG paper: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." 2024
- Microsoft blog on dynamic community selection (Nov 2024) — their approach to navigating hierarchical communities at query time is directly relevant to our TopicContextFrame

**Design decision for Attest:** Use CPM (Constant Potts Model) quality function instead of modularity for Leiden. CPM doesn't suffer from the resolution limit that prevents detection of small communities in large graphs. This matters because in a biology graph, a small cluster of 10 proteins in a specific signaling pathway should be detectable even when the graph has millions of entities. GraphRAG uses modularity and misses these.

### Scaling Concern Resolution

The NVIDIA nx-cugraph team demonstrated GPU-accelerated Leiden on a genomics graph of 14.7K nodes and 83.8M edges in under 4 seconds. Our 10M claim graph (roughly 1-5M entities, 10M+ edges) is well within this range. At 100M claims, we'd likely need GPU acceleration or the parallel CPU implementation from GVE-Leiden. The Phase 2 Rust engine's CSR format is ideal for both paths.

**Bottom line:** Topology at our scale is solved engineering, not research. The open question is incremental updates (recomputing only affected communities when new claims arrive), which GraphRAG punts on. This is worth a research spike in Phase 2 week 5.

---

## 2. Text-to-Claims Extraction Quality

**The watchlist item:** The text ingestion pipeline depends on LLM extraction quality, which is inherently noisy. How noisy? Can we bound it?

### The Precedent: DARPA SCORE Program (2019-2024)

DARPA spent $7.6M+ on exactly this problem: can automated tools reliably extract and score scientific claims from text? The Center for Open Science manually extracted 7,066 claims from social/behavioral science papers, had human forecasters and algorithms assign confidence scores, and then ran actual replications to test accuracy.

**What they found:**
- Manual claim extraction from scientific text is expensive but achievable ($7.6M for 7,066 claims ≈ $1,075/claim at scale)
- Automated extraction was viable but required structured decomposition: identify the claim, identify the evidence, identify the methodology, score confidence
- Prediction markets (groups of human experts) were better than individual experts at scoring claim reproducibility
- The key insight: confidence should be a **spectrum**, not binary. Their "confidence score" ranged continuously, which is exactly our Tier 1-3 system

**Lesson for Attest:** Our curator already has the right architecture (LLM triage with structured output). But DARPA SCORE showed that the extraction step and the scoring step should be **independent** — don't let the same system that extracts claims also judge their confidence. Our separation of text extraction → curator triage → confidence computation follows this principle correctly.

### The Precedent: LLM-Driven Knowledge Graph Construction (2024-2025)

The research literature on LLM triple extraction has exploded. Key findings:

**Accuracy ranges:**
- Zero-shot LLM extraction: 40-65% F1 on standard benchmarks (REBEL, WebNLG)
- Few-shot with ontology guidance: 65-80% F1
- Fine-tuned smaller models: 70-85% F1
- With vocabulary constraints (our approach): likely 75-90% F1, because we constrain entity types and predicates to a registered vocabulary

**The ATLAS system** (HKUST, 2025): Built 900M+ nodes and 5.9B edges from 50M documents in the Dolma corpus, achieving 95% semantic alignment with human-crafted schemas. This is the scale existence proof for automated KG construction.

**Key finding from TextMineX (2025):** Ontology-aligned prompts (analogous to our vocabulary-guided extraction) improved triple extraction accuracy by 44.2% and reduced hallucinations by 22.5% compared to unconstrained LLM extraction. This directly validates our design choice of vocabulary-constrained extraction.

**The hallucination problem is real.** Multiple papers document LLMs generating "hallucinated triples" — relationships not grounded in source text. Our architecture handles this correctly through three mechanisms:
1. Vocabulary constraints reject entities and predicates not in the registered vocabulary
2. The curator acts as a second-pass filter
3. All text-extracted claims carry `source_type: "document_extraction"` and can be filtered out for high-confidence queries

**KGGEN (Mo et al., 2025):** Decomposed extraction into two sequential LLM calls — first entity detection, then relation generation — to reduce error propagation. Our text pipeline should follow this two-step pattern rather than asking for complete triples in a single pass.

**Design decision for Attest:** Text extraction should use a two-step pipeline (entities first, then relations), with vocabulary-constrained prompts. Expected accuracy: 75-85% F1 on in-domain text. The 15-25% error rate is acceptable because (a) the curator provides a second filter, (b) text-extracted claims carry explicit provenance marking, and (c) confidence scores for `document_extraction` source type start at 0.60, not 0.85.

---

## 3. Federated Evidence Sharing

**The watchlist item:** Cross-organization evidence sharing is Phase 3+ with only an `organization` field as an architectural hook. Is that sufficient?

### The Precedent: Federated Knowledge Graph Embedding (FedE and successors, 2021-2025)

A rich body of research on federated KG learning, starting with FedE (Chen et al., 2021) and evolving through FedEC, FedLU, FedR, and OFKGE.

**Core problem they solve:** Multiple organizations have local knowledge graphs. They want to share information to improve completion (filling in missing edges) without exposing raw data. Sound familiar? It's exactly Attest's future federated scenario.

**What works:**
- **Entity alignment tables** — a shared mapping of which entities correspond across organizations. Our `external_ids` and `same_as` mechanism are exactly this. The research validates that alignment must be maintained separately from the data itself
- **One-shot federation** (OFKGE, 2025) — instead of iterative communication, each organization trains locally and shares model parameters once. For Attest, this translates to: share embeddings and topic summaries rather than raw claims. An organization can share "our neurodegeneration topic has 142 entities and connects to lipid metabolism through these 3 connector entities" without exposing proprietary claims
- **Privacy via embedding sharing** — share entity embeddings rather than raw triples. FedR (2022) showed that sharing relation embeddings instead of entity embeddings reduces privacy leakage. For Attest, this means the commercial intelligence layer could offer: "share your topic-level embeddings and community summaries, not your claims"

**What failed:**
- Multi-round federated learning is expensive and leaks privacy (FedE sends ~1.35B parameters over multiple rounds)
- Aggregation strategies that ignore semantic diversity produce poor results
- Entity embedding sharing enables inference attacks (FedR, 2022 — "known entity embedding can be used to infer whether a specific relation between two entities exists")

**The Federated Virtual Knowledge Graph approach (FVKG, 2025):** Each organization maintains its own graph with its own vocabulary. A semantic layer translates between them using ontology mappings. Queries are distributed to relevant organizations, results are integrated. This maps well to Attest's vocabulary registration system — each organization registers its own vocabulary, and a federation layer maps between them.

**Lesson for Attest:** The `organization` field on ProvenanceChain is necessary but not sufficient. Phase 3 also needs:
1. **Topic-level sharing** — share TopicNode summaries and cross-domain connectors without exposing claims
2. **Embedding-level sharing** — share named vector space embeddings for entity alignment across organizations
3. **Vocabulary mapping** — a registry of vocabulary equivalences across organizations (our vocabulary registration system already supports namespacing)
4. **Provenance firewall** — claims from external organizations carry provenance but internal claims referenced by those external claims are not exposed

**However**, the `organization` field IS sufficient as an architectural hook for Phase 1-2. The research shows the hard federation problems are in alignment and privacy, not in data modeling. Our claim schema already has the right primitives (external_ids, same_as, vocabulary namespacing, provenance chains). We won't need to change the core data model — federation is an intelligence layer feature built on top of existing primitives.

---

## Summary: What To Build Differently Based On Research

| Watchlist Item | Precedent | Key Lesson | Design Change |
|---|---|---|---|
| Topology at scale | GraphRAG, GVE-Leiden | Leiden on CSR is fast; use CPM not modularity; overlapping communities needed | Switch from modularity to CPM quality function. Design CSR for parallel Leiden. Plan incremental update research spike. |
| Text extraction quality | DARPA SCORE, KGGEN, TextMineX, ATLAS | Two-step extraction (entities then relations). Ontology-constrained prompts add ~44% accuracy. Separate extraction from scoring. | Two-step extraction pipeline. Vocabulary-constrained prompts. Keep extraction and confidence as independent systems. |
| Federated sharing | FedE/OFKGE, FVKG | Share embeddings and topic summaries, not raw claims. Entity alignment is the hard problem. One-shot > iterative. | `organization` field is sufficient for Phase 1-2. Phase 3 adds topic-level sharing, embedding exchange, vocabulary mapping, provenance firewalls. |

### Additional Architectural Insight from the Research

One pattern appears across all three areas: **the thing that makes Attest different (structural provenance on every claim) is also what makes each of these problems more tractable.** 

- GraphRAG's community summaries are opaque LLM outputs. Attest's topic summaries can trace to specific claims with known confidence — making them auditable and correctable.
- LLM triple extraction is notoriously hallucinatory. Attest's vocabulary constraints + curator + explicit `document_extraction` provenance type creates a triple-layered defense that no existing system has.
- Federated KG sharing leaks privacy through embeddings. Attest's provenance chain means you can share topic-level summaries with provenance metadata (source types, confidence distributions) without sharing the actual claims — a more information-rich yet privacy-preserving approach than raw embedding sharing.

The claim-native architecture isn't just a storage optimization. It's the primitive that makes topology, text ingestion, and federation work better than in any existing system. The research validates this wasn't just a good idea — it's the structurally correct design choice.
