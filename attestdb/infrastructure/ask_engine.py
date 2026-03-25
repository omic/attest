"""AskEngine — question-answering subsystem extracted from AttestDB."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.core.types import EntitySummary

from attestdb.core.types import AskResult, Citation, claim_from_dict

DEFAULT_MAX_RELATIONSHIPS = 400
ENTITY_BUDGET_FLOOR = 60


class AskEngine:
    """Encapsulates the ask() pipeline and its helpers.

    Holds a back-reference to the owning ``AttestDB`` instance so it can
    call ``db.search_entities``, ``db.claims_for``, ``db._store``, etc.
    """

    def __init__(self, db):
        self.db = db

    # --- LLM access ---

    def _get_llm_client(self):
        """Return (client, model) for LLM calls.

        Prefers direct OpenAI client initialization from env keys (avoids httpx
        conflicts with uvicorn when text extractor creates its own httpx client).
        Falls back to text extractor if no env keys are available.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Prefer direct initialization from env keys — creates a fresh httpx client
        # that doesn't conflict with uvicorn's event loop (unlike text extractor's)
        try:
            from attestdb.core.providers import PROVIDERS, EXTRACTION_FALLBACK_CHAIN
            import os
            from openai import OpenAI

            for provider_name in EXTRACTION_FALLBACK_CHAIN:
                provider = PROVIDERS.get(provider_name)
                if not provider:
                    continue
                key = os.environ.get(provider["env_key"], "")
                if not key:
                    continue
                client = OpenAI(api_key=key, base_url=provider["base_url"])
                model = provider["default_model"]
                logger.info("_get_llm_client: initialized %s/%s directly from env", provider_name, model)
                return client, model
        except ImportError:
            logger.debug("_get_llm_client: openai package not available")

        # Last resort: try text extractor (may cause httpx conflicts in uvicorn)
        try:
            ext = self.db._get_text_extractor()
            if ext._client and ext._llm_model:
                logger.info("_get_llm_client: using text extractor client (last resort)")
                return ext._client, ext._llm_model
        except Exception as exc:
            logger.debug("_get_llm_client: text extractor unavailable: %s", exc)

        logger.warning("_get_llm_client: no LLM provider available")
        return None, None

    def _llm_call(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str | None:
        """Single LLM call with provider fallback and proper error handling.

        Tries max_completion_tokens first (for reasoning models), falls back
        to max_tokens. If the primary provider fails, walks the fallback chain.
        All failures are logged.
        """
        import logging
        logger = logging.getLogger(__name__)

        client, model = self._get_llm_client()
        if not client:
            logger.warning("_llm_call: no LLM client available (configure_curator not called or no API key)")
            return None

        messages = [{"role": "user", "content": prompt}]

        # Try with max_completion_tokens first (reasoning models need this)
        for param_name, param_kwargs in [
            ("max_completion_tokens", {"max_completion_tokens": max_tokens}),
            ("max_tokens", {"max_tokens": max_tokens}),
        ]:
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=60,
                    **param_kwargs,
                )
                content = r.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                # Empty content — try next param style
                logger.debug("_llm_call: %s/%s returned empty content with %s", model, param_name, param_name)
                continue
            except Exception as exc:
                logger.debug("_llm_call: %s/%s failed with %s: %s", model, param_name, type(exc).__name__, exc)
                continue

        # Primary provider failed or returned empty — try fallback providers
        try:
            from attestdb.core.providers import PROVIDERS, EXTRACTION_FALLBACK_CHAIN
            import os
            from openai import OpenAI

            current_model = model
            for provider_name in EXTRACTION_FALLBACK_CHAIN:
                provider = PROVIDERS.get(provider_name)
                if not provider:
                    continue
                key = os.environ.get(provider["env_key"], "")
                if not key:
                    continue
                fallback_model = provider["default_model"]
                if fallback_model == current_model:
                    continue  # already tried this one
                try:
                    fallback_client = OpenAI(api_key=key, base_url=provider["base_url"])
                    r = fallback_client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=30,
                    )
                    content = r.choices[0].message.content
                    if content and content.strip():
                        logger.info("_llm_call: fallback to %s/%s succeeded", provider_name, fallback_model)
                        return content.strip()
                except Exception as exc:
                    logger.debug("_llm_call: fallback %s/%s failed: %s", provider_name, fallback_model, exc)
                    continue
        except ImportError:
            pass

        logger.warning("_llm_call: all providers failed for prompt (%d chars)", len(prompt))
        return None

    # --- Graph-neighborhood clustering for ask() ---

    def _cluster_entities(self, entity_ids: list[str]) -> list[list[str]]:
        """Cluster candidate entities by 2-hop graph connectivity.

        Two candidates are "topic-connected" if directly adjacent OR share
        at least one graph neighbor.  Returns connected components sorted
        by size descending.
        """
        if len(entity_ids) <= 1:
            return [list(entity_ids)] if entity_ids else []

        # Build neighbor sets only for the candidate entities (not the full graph)
        neighbors: dict[str, set[str]] = {}
        for eid in entity_ids:
            raw = self.db._store.claims_for(eid, None, None, 0.0)
            adj: set[str] = set()
            for c in raw:
                claim = claim_from_dict(c)
                other = claim.object.id if claim.subject.id == eid else claim.subject.id
                adj.add(other)
            neighbors[eid] = adj

        # Build candidate-level adjacency: two candidates are connected if
        # they share a direct edge OR share >= 1 common neighbor (2-hop)
        cand_adj: dict[str, set[str]] = {eid: set() for eid in entity_ids}
        ids = list(entity_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                # Direct edge?
                if b in neighbors[a]:
                    cand_adj[a].add(b)
                    cand_adj[b].add(a)
                # Shared neighbor (2-hop)?
                elif neighbors[a] & neighbors[b]:
                    cand_adj[a].add(b)
                    cand_adj[b].add(a)

        # BFS connected components
        visited: set[str] = set()
        clusters: list[list[str]] = []
        for eid in entity_ids:
            if eid in visited:
                continue
            component: list[str] = []
            queue = deque([eid])
            visited.add(eid)
            while queue:
                node = queue.popleft()
                component.append(node)
                for nb in cand_adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            clusters.append(component)

        clusters.sort(key=lambda c: -len(c))
        return clusters

    def _label_cluster(self, cluster: list[str], entity_map: dict[str, "EntitySummary"]) -> str:
        """Generate a human-readable label for a cluster of entities.

        Uses topology community labels if computed, otherwise falls back
        to dominant entity type + top 2 entity names.
        """
        # Try topology labels (opportunistic — only if already computed)
        if hasattr(self.db, "_topology") and self.db._topology is not None:
            # Find the community that contains the most cluster members
            best_label = ""
            best_overlap = 0
            cluster_set = set(cluster)
            for _res, communities in self.db._topology.communities.items():
                for comm in communities:
                    overlap = len(cluster_set & set(comm.members))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_label = comm.label
            if best_label:
                return best_label

        # Fallback: dominant entity type + top 2 names
        type_counts: dict[str, int] = {}
        for eid in cluster:
            e = entity_map.get(eid)
            if e:
                etype = e.entity_type or "unknown"
                type_counts[etype] = type_counts.get(etype, 0) + 1

        dominant = max(type_counts, key=type_counts.get) if type_counts else "unknown"

        # Top 2 entities by claim count
        ranked = sorted(
            (entity_map[eid] for eid in cluster if eid in entity_map),
            key=lambda e: -e.claim_count,
        )
        names = [e.name or e.id for e in ranked[:2]]
        if names:
            return f"{dominant} ({', '.join(names)})"
        return dominant

    def _gather_clustered_evidence(
        self,
        clusters: list[list[str]],
        entity_map: dict[str, "EntitySummary"],
        max_rels: int = DEFAULT_MAX_RELATIONSHIPS,
        collect_citations: bool = False,
    ) -> str | tuple:
        """Build evidence organized by topic cluster.

        Each cluster gets a ``# Topic: {label}`` section header.
        Budget is allocated proportionally (at least 60 rels per cluster).
        Capped at 5 topic sections.
        """
        clusters = clusters[:5]
        total_entities = sum(len(c) for c in clusters)

        sections: list[str] = []
        all_citations: list[Citation] = []
        all_contradictions: list[dict] = []
        all_gaps: list[str] = []
        for cluster in clusters:
            label = self._label_cluster(cluster, entity_map)
            budget = max(ENTITY_BUDGET_FLOOR, int(max_rels * len(cluster) / max(total_entities, 1)))
            if collect_citations:
                evidence, cit, contra, gaps = self._gather_evidence(
                    cluster, max_rels=budget, collect_citations=True,
                )
                all_citations.extend(cit)
                all_contradictions.extend(contra)
                for g in gaps:
                    if g not in all_gaps:
                        all_gaps.append(g)
            else:
                evidence = self._gather_evidence(cluster, max_rels=budget)
            if evidence.strip():
                sections.append(f"# Topic: {label}\n{evidence}")

        text = "\n\n".join(sections)
        if collect_citations:
            return text, all_citations, all_contradictions, all_gaps
        return text

    def ask(self, question: str, top_k: int = 10) -> AskResult:
        """Answer a natural-language question using the knowledge graph.

        Fast path (search hits >= 3): search -> evidence -> single LLM call.
        Slow path (few search hits): adds type catalog + LLM type selection.

        Returns AskResult with structured citations, contradictions, and gaps.
        Dict-compatible for backward compat (r["answer"] works).
        """
        # Step 1: Search for entities whose names match the question
        _stop = frozenset({
            "what", "who", "how", "why", "when", "where", "which", "does",
            "the", "are", "for", "and", "that", "this", "with", "from",
            "about", "has", "have", "been", "is", "was", "were", "tell",
            "me", "show", "find", "get", "list", "of", "in", "on", "at",
            "by", "an", "a", "do", "did", "our", "we", "us", "made",
            "being", "last", "recent", "done", "any", "all", "some",
            "much", "many", "can", "could", "would", "should", "will",
            "to", "it", "its", "be", "not", "no", "or", "but", "if",
            "than", "them", "they", "their", "there", "then", "top",
            "best", "most", "also", "just", "only", "very", "more",
            "good", "bad", "new", "old", "like", "use", "used", "using",
            "target", "role", "effect", "type", "cause", "work", "works",
            "related", "involved", "between", "affect", "impact",
        })
        words = [w.strip("?.,!\"'()[]{}:;") for w in question.lower().split()]
        query_terms = " ".join(w for w in words if w and len(w) >= 2 and w not in _stop)

        search_hits: list = []
        if query_terms:
            search_hits = self.db.search_entities(query_terms, top_k=30)

            # Also search individual content words (catches entities whose
            # names match only one query term, e.g. "Pain" for "gpr55 pain")
            # and progressively shorter prefixes for stem variations.
            hit_ids = {e.id for e in search_hits}
            content_words = [w for w in query_terms.split() if len(w) >= 3]
            for word in content_words:
                for hit in self.db.search_entities(word, top_k=10):
                    if hit.id not in hit_ids:
                        search_hits.append(hit)
                        hit_ids.add(hit.id)
                if len(word) >= 7:
                    for plen in range(len(word) - 2, 4, -1):
                        prefix = word[:plen]
                        before = len(hit_ids)
                        for hit in self.db.search_entities(prefix, top_k=10):
                            if hit.id not in hit_ids:
                                search_hits.append(hit)
                                hit_ids.add(hit.id)
                        if len(hit_ids) > before:
                            break

            search_hits.sort(key=lambda e: -e.claim_count)
            search_hits = search_hits[:top_k]

        # Step 2: Build entity set
        # Fast path: search hits are sufficient — skip type catalog + LLM type selection
        # Filter to entities whose names actually contain a query term (BM25 can
        # return high-claim-count entities with marginal text overlap in small DBs)
        selected_types: set[str] = set()
        if search_hits and query_terms:
            terms_lower = set(query_terms.lower().split())
            relevant_hits: list = []
            for e in search_hits:
                ename = (e.name or e.id).lower()
                if any(t in ename for t in terms_lower):
                    relevant_hits.append(e)
            if relevant_hits:
                search_hits = relevant_hits
            # Drop entities with 0 claims — they produce no evidence
            search_hits = [e for e in search_hits if e.claim_count > 0]
            # Deduplicate by display name — keep highest-claim-count per name
            seen_names: dict[str, int] = {}  # name -> best claim_count
            deduped: list = []
            for e in search_hits:
                key = (e.name or e.id).lower()
                if key not in seen_names or e.claim_count > seen_names[key]:
                    if key in seen_names:
                        deduped = [x for x in deduped if (x.name or x.id).lower() != key]
                    seen_names[key] = e.claim_count
                    deduped.append(e)
            search_hits = deduped

        clusters: list[list[str]] = []
        cluster_labels: list[str] = []
        entity_map: dict = {}

        if len(search_hits) >= 1:
            # Fast path: search hits found — go straight to evidence
            selected = list(search_hits)
        else:
            # Slow path: no search hits — type catalog + LLM + clustering
            selected = list(search_hits)
            selected, selected_types = self._ask_type_expand(
                question, selected, top_k,
            )
            # Cluster and filter/organize by topic
            entity_map = {e.id: e for e in selected[:20]}
            entity_ids_for_cluster = [e.id for e in selected[:20]]
            clusters = self._cluster_entities(entity_ids_for_cluster)
            search_hit_ids = {e.id for e in search_hits}
            if len(clusters) > 1 and search_hit_ids:
                relevant = [c for c in clusters if search_hit_ids & set(c)]
                if relevant:
                    clusters = relevant
                    kept = set()
                    for c in clusters:
                        kept.update(c)
                    selected = [e for e in selected if e.id in kept]

        if not selected:
            return AskResult(meta={
                "n_searched": 0, "n_search_hits": len(search_hits),
                "selected_types": sorted(selected_types),
            })

        # Step 3: Gather evidence AND citations in a single pass (no re-querying)
        # On large databases, cap entities and hop-2 depth to keep latency bounded
        _large_db = False
        try:
            _large_db = self.db._store.count_claims() > 1_000_000
        except Exception:
            pass

        multi_topic = len(clusters) > 1
        if multi_topic:
            evidence, citations, all_contradictions, gaps = self._gather_clustered_evidence(
                clusters, entity_map, max_rels=DEFAULT_MAX_RELATIONSHIPS,
                collect_citations=True,
            )
            cluster_labels = [
                self._label_cluster(c, entity_map) for c in clusters
            ]
        else:
            # Cap entities for evidence gathering — each triggers query() calls
            evidence_cap = 2 if _large_db else top_k
            entity_ids = [e.id for e in selected[:evidence_cap]]
            evidence, citations, all_contradictions, gaps = self._gather_evidence(
                entity_ids, max_rels=DEFAULT_MAX_RELATIONSHIPS, collect_citations=True,
            )

        # Step 4: Single LLM synthesis call
        base_instructions = (
            "- Trace multi-hop reasoning chains explicitly (A \u2192 B \u2192 C) when they "
            "answer the question.\n"
            "- Distinguish well-supported facts (high confidence, multiple sources) "
            "from weaker inferences.\n"
            "- Note contradictions when present \u2014 don't hide conflicting evidence.\n"
            "- Items tagged with \u26a0 WARNING, BUG, VULNERABILITY, or PATTERN are "
            "knowledge annotations \u2014 surface these prominently as they represent "
            "critical operational learnings.\n"
            "- Quote evidence text when available and relevant.\n"
            "- Answer in 3-8 sentences using specific names, facts, and "
            "relationships from the evidence. Synthesize directly from the "
            "evidence \u2014 do not hedge or say information is unavailable if the "
            "evidence contains relevant facts."
        )
        if multi_topic:
            base_instructions += (
                "\n- The evidence is organized by topic area. Address each topic "
                "area separately.\n"
                "- Do not mix evidence from different topic areas in the same "
                "paragraph.\n"
                "- Use the topic labels as section headers in your answer."
            )
        prompt = (
            "You are answering questions about an organization's knowledge "
            "graph. Below is detailed evidence \u2014 entities, their relationships, "
            "multi-hop chains (marked [2-hop]), confidence scores, source counts, "
            "and evidence quotes from real conversations and documents.\n\n"
            f"{evidence}\n\n"
            "---\n\n"
            f"Question: {question}\n\n"
            f"Instructions:\n{base_instructions}"
        )
        answer = self._llm_call(prompt, max_tokens=1024)

        selected.sort(key=lambda e: -e.claim_count)

        return AskResult(
            answer=answer,
            citations=citations,
            contradictions=all_contradictions,
            gaps=gaps,
            entities=selected[:top_k],
            evidence=evidence,
            meta={
                "n_searched": len(selected),
                "n_search_hits": len(search_hits),
                "selected_types": sorted(selected_types),
                "n_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
                "cluster_labels": cluster_labels,
            },
        )

    def _ask_type_expand(
        self,
        question: str,
        search_hits: list,
        top_k: int,
    ) -> tuple:
        """Slow path: use type catalog + LLM to find relevant entity types.

        Called when search_entities() returns fewer than 3 hits, so we need
        LLM guidance to figure out which entity types are relevant.

        Returns (selected_entities, selected_types).
        """
        db_stats = self.db.stats()
        type_counts: dict[str, int] = db_stats.get("entity_types", {})
        if not type_counts:
            return list(search_hits), set()

        min_type_size = 3
        eligible_types = sorted(
            ((t, c) for t, c in type_counts.items() if c >= min_type_size),
            key=lambda x: -x[1],
        )
        catalog_parts = []
        for t, count in eligible_types[:20]:
            examples = self.db.list_entities(entity_type=t, limit=2)
            name_str = ", ".join((e.name or e.id) for e in examples)
            catalog_parts.append(f"{t} ({count}: {name_str})")
        type_catalog = ", ".join(catalog_parts)
        type_prompt = (
            f"Entity types in a knowledge graph: {type_catalog}\n\n"
            f"Question: {question}\n\n"
            "Which 5-8 entity types contain the most interesting "
            "domain-specific knowledge for answering this question? "
            "Avoid generic types like process, concept, information, "
            "activity, task, or change. "
            "Return ONLY type names, comma-separated."
        )
        type_response = self._llm_call(type_prompt, max_tokens=100, temperature=0.0)

        selected_types: set[str] = set()
        if type_response:
            for part in type_response.split(","):
                t = part.strip().lower()
                if t in type_counts and type_counts[t] >= min_type_size:
                    selected_types.add(t)

        seen_ids: set[str] = set()
        selected: list = []
        for e in search_hits:
            if e.id not in seen_ids:
                seen_ids.add(e.id)
                selected.append(e)
        for t in selected_types:
            for e in self.db.list_entities(entity_type=t, limit=3):
                if e.id not in seen_ids and len(selected) < top_k:
                    seen_ids.add(e.id)
                    selected.append(e)

        return selected, selected_types

    def _gather_evidence(
        self,
        entity_ids: list[str],
        max_rels: int = 60,
        collect_citations: bool = False,
    ) -> str | tuple:
        """Build rich evidence from claims for a set of entities.

        Uses claims_for() directly (O(1) LMDB index lookup) instead of
        query() (which does full BFS + contradiction detection per call).
        This keeps latency sub-second on databases of any size.

        If *collect_citations* is True, returns
        ``(evidence_text, citations, contradictions, gaps)`` so ask() can
        reuse the data (no redundant re-queries).
        """
        from attestdb.core.vocabulary import KNOWLEDGE_PRIORITY, knowledge_label, OPPOSITE_PREDICATES

        lines: list[str] = []
        citations: list[Citation] = []
        all_contradictions: list[dict] = []
        all_gaps: list[str] = []
        seen: set[str] = set()
        seen_claim_ids: set[str] = set()

        # Budget: distribute max_rels across entities
        per_entity = max(max_rels // max(len(entity_ids), 1), 5)

        for eid in entity_ids:
            claims = self.db.claims_for(eid)
            if not claims:
                continue

            # Get entity display name
            entity = self.db._store.get_entity(eid)
            ename = (entity or {}).get("display_name", eid)
            etype = (entity or {}).get("entity_type", "entity")
            lines.append(f"\n## {ename} ({etype})")

            # Sort by confidence, take budget
            claims.sort(key=lambda c: -c.confidence)
            budgeted = claims[:per_entity]

            # Group by predicate for contradiction detection
            pred_targets: dict[str, list] = {}  # (subj, obj) -> [predicate_ids]
            for c in budgeted:
                key = (c.subject.id, c.object.id)
                pred_targets.setdefault(key, []).append(c.predicate.id)

            for c in budgeted:
                subj = c.subject.display_name or c.subject.id
                obj = c.object.display_name or c.object.id
                triple = f"{subj} \u2192 {c.predicate.id} \u2192 {obj}"
                if triple in seen:
                    continue
                seen.add(triple)

                # Tag knowledge claims
                tag = ""
                if c.predicate.id in KNOWLEDGE_PRIORITY:
                    tag = f" \u26a0 {knowledge_label(c.predicate.id).upper()}"

                src = c.provenance.source_type if c.provenance else ""
                ann = f"[conf={c.confidence:.2f}, source: {src}]{tag}"
                lines.append(f"- {triple} {ann}")

                # Evidence text from payload
                if c.payload and hasattr(c.payload, 'data') and isinstance(c.payload.data, dict):
                    ev = c.payload.data.get("evidence_text", "")
                    if ev:
                        lines.append(f'    Evidence: "{ev}"')

                # Collect citation
                if collect_citations and len(citations) < 50:
                    if c.claim_id not in seen_claim_ids:
                        seen_claim_ids.add(c.claim_id)
                        citations.append(Citation(
                            claim_id=c.claim_id,
                            subject=c.subject.id,
                            predicate=c.predicate.id,
                            object=c.object.id,
                            confidence=c.confidence,
                            source_id=c.provenance.source_id if c.provenance else "",
                            source_type=c.provenance.source_type if c.provenance else "",
                        ))

            # Detect contradictions from opposing predicates in this entity's claims
            for (subj, obj), preds in pred_targets.items():
                pred_set = set(preds)
                for p1, p2 in OPPOSITE_PREDICATES.items():
                    if p1 in pred_set and p2 in pred_set:
                        desc = f"{subj} has both '{p1}' and '{p2}' relationship with {obj}"
                        lines.append(f"  \u26a0 Contradiction: {desc}")
                        if collect_citations:
                            all_contradictions.append({
                                "claim_a": "", "claim_b": "",
                                "description": desc, "status": "unresolved",
                            })

        evidence_text = "\n".join(lines)
        if collect_citations:
            return evidence_text, citations, all_contradictions, all_gaps
        return evidence_text
