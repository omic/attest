"""IntelligenceGateway — lazy-init and configuration for LLM-backed components.

Also hosts curation, text ingestion, connector, and sync methods extracted
from AttestDB so that attest_db.py stays focused on core CRUD.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.connectors.base import Connector
    from attestdb.infrastructure.attest_db import AttestDB

from attestdb.core.types import ClaimInput
from attestdb.infrastructure.event_bus import EventType

logger = logging.getLogger(__name__)


class IntelligenceGateway:
    """Manages curator, text extractor, insight engine, temporal engine, and researcher.

    All intelligence imports are lazy (inside method bodies) so the core engine
    works without intelligence dependencies installed.
    """

    def __init__(self, store, db=None):
        self._store = store
        self._db = db  # back-reference for components that need AttestDB
        self._curator_model: str = "heuristic"
        self._curator_api_key: str | None = None
        self._curator_env_path: str | None = None
        self._curator = None
        self._text_extractor = None
        self._insight_engine = None
        self._temporal_engine = None
        self._researcher = None
        self._domain_context: str | None = None

    # --- Configuration ---

    def configure_curator(
        self,
        model: str = "heuristic",
        api_key: str | None = None,
        env_path: str | None = None,
    ) -> None:
        """Configure the curator model. Resets cached intelligence instances."""
        self._curator_model = model
        self._curator_api_key = api_key
        self._curator_env_path = env_path
        self._curator = None
        self._text_extractor = None
        self._researcher = None

    def set_domain_context(self, context: str) -> None:
        """Set domain context for LLM extraction prioritization."""
        self._domain_context = context
        self._text_extractor = None  # force rebuild

    # --- Lazy-init accessors ---

    def _get_curator(self):
        if self._curator is None:
            from attestdb.intelligence.curator import CuratorV1
            ops_cb = getattr(self._db, "_ops_log", None)
            self._curator = CuratorV1(
                self._db, model=self._curator_model,
                api_key=self._curator_api_key, env_path=self._curator_env_path,
                ops_callback=ops_cb.write if ops_cb else None,
            )
        return self._curator

    def _get_text_extractor(self):
        if self._text_extractor is None:
            from attestdb.core.vocabulary import BUILT_IN_ENTITY_TYPES, BUILT_IN_PREDICATE_TYPES
            from attestdb.intelligence.text_extractor import TextExtractor

            # Merge built-in + registered entity types and predicates
            entity_types = set(BUILT_IN_ENTITY_TYPES)
            predicates = set(BUILT_IN_PREDICATE_TYPES)
            try:
                for vocab in self._store.get_registered_vocabularies().values():
                    entity_types.update(vocab.get("entity_types", []))
                    predicates.update(vocab.get("predicate_types", []))
            except Exception:
                pass

            # Get predicate constraints
            predicate_constraints = {}
            try:
                predicate_constraints = self._store.get_predicate_constraints()
            except Exception:
                pass

            # Use "auto" fallback chain unless a specific LLM provider was configured.
            extractor_model = self._curator_model if self._curator_model != "heuristic" else "auto"
            self._text_extractor = TextExtractor(
                model=extractor_model,
                api_key=self._curator_api_key,
                env_path=self._curator_env_path,
                entity_types=list(entity_types),
                predicates=list(predicates),
                predicate_constraints=predicate_constraints,
                discovery_mode=True,
                domain_context=self._domain_context,
            )
        return self._text_extractor

    def _get_insight_engine(self):
        if self._insight_engine is None:
            from attestdb.intelligence.insight_engine import InsightEngineV1
            self._insight_engine = InsightEngineV1(self._db)
        return self._insight_engine

    def _get_temporal_engine(self):
        if self._temporal_engine is None:
            from attestdb.intelligence.temporal import TemporalEngine
            self._temporal_engine = TemporalEngine(self._db)
        return self._temporal_engine

    def _get_researcher(self):
        if self._researcher is None:
            try:
                from attestdb_enterprise.researcher import Researcher
            except ImportError:
                raise ImportError(
                    "Researcher requires attestdb-enterprise: "
                    "pip install attestdb-enterprise"
                )
            self._researcher = Researcher(
                self._db, model=self._curator_model,
                api_key=self._curator_api_key, env_path=self._curator_env_path,
            )
        return self._researcher

    def invalidate_text_extractor(self) -> None:
        """Force rebuild of text extractor (e.g. after vocabulary registration)."""
        self._text_extractor = None

    # --- Curation & text ingestion ---

    def curate(self, claims: list[ClaimInput], agent_id: str = "default"):
        """Triage and ingest claims through the curator."""
        return self._get_curator().process_agent_output(agent_id, claims)

    def ingest_text(self, text: str, source_id: str = "", use_curator: bool = True):
        """Extract claims from text and ingest. Optional curator triage."""
        curator = self._get_curator() if use_curator else None
        return self._get_text_extractor().extract_and_ingest(
            text, self._db, source_id=source_id, curator=curator,
        )

    def ingest_texts(
        self,
        texts: list[dict],
        use_curator: bool = True,
    ) -> dict:
        """Batch wrapper around ingest_text().

        Args:
            texts: List of dicts with keys: text, source_id, and optionally source_type.
            use_curator: Whether to triage claims through the curator.

        Returns:
            Summary dict with total_extracted, total_stored, total_skipped, and per-text results.
        """
        total_extracted = 0
        total_stored = 0
        total_skipped = 0
        results = []

        for item in texts:
            text = item["text"]
            source_id = item.get("source_id", "")
            result = self._db.ingest_text(text, source_id=source_id, use_curator=use_curator)
            n_extracted = result.raw_count
            n_stored = result.n_valid
            n_skipped = n_extracted - n_stored
            total_extracted += n_extracted
            total_stored += n_stored
            total_skipped += n_skipped
            results.append({
                "source_id": source_id,
                "extracted": n_extracted,
                "stored": n_stored,
                "skipped": n_skipped,
            })

        return {
            "total_extracted": total_extracted,
            "total_stored": total_stored,
            "total_skipped": total_skipped,
            "results": results,
        }

    def _get_extractor(self, mode: str = "llm"):
        """Get the appropriate extractor for the given mode.

        Modes:
            "llm" — Full LLM extraction (requires API key).
            "heuristic" — Pattern-based, no API key needed.
            "smart" — Heuristic pre-scan → novelty check → LLM only for novel.
        """
        if mode in ("heuristic", "smart"):
            from attestdb.intelligence.heuristic_extractor import HeuristicExtractor
            constraints = {}
            try:
                constraints = self._store.get_predicate_constraints()
            except Exception:
                pass
            # Build known entity dictionary from existing entities
            known = {}
            try:
                for es in self._db.list_entities():
                    known[es.id] = es.entity_type
            except Exception:
                pass
            heuristic = HeuristicExtractor(
                predicate_constraints=constraints,
                known_entities=known if known else None,
            )
            if mode == "smart":
                from attestdb.intelligence.smart_extractor import SmartExtractor
                return SmartExtractor(self._get_text_extractor(), heuristic, self._db)
            return heuristic
        return self._get_text_extractor()

    # --- Chat ingestion ---

    def ingest_chat(
        self,
        messages: list[dict],
        conversation_id: str = "",
        platform: str = "generic",
        use_curator: bool = True,
        extraction: str = "llm",
    ):
        """Extract and ingest claims from a chat conversation.

        Args:
            messages: OpenAI-format messages [{role, content}, ...].
            conversation_id: Optional ID for the conversation.
            platform: Platform hint ("generic", "chatgpt", "claude").
            use_curator: Whether to triage claims through the curator.
            extraction: "llm", "heuristic", or "smart" (heuristic + LLM for novel).

        Returns:
            ChatIngestionResult with per-turn breakdown.
        """
        from attestdb.intelligence.chat_ingestor import ChatIngestor
        curator = self._get_curator() if use_curator else None
        extractor = self._get_extractor(extraction)
        ingestor = ChatIngestor(extractor, self._db, curator)
        return ingestor.ingest_messages(messages, conversation_id, platform)

    def ingest_chat_file(
        self,
        path: str,
        platform: str = "auto",
        use_curator: bool = True,
        extraction: str = "llm",
    ):
        """Extract and ingest claims from a chat log file (.zip, .json, .txt, .md).

        Args:
            path: Path to the chat log file.
            platform: Format hint ("auto", "chatgpt", "openai", "generic").
            use_curator: Whether to triage claims through the curator.
            extraction: "llm", "heuristic", or "smart" (heuristic + LLM for novel).

        Returns:
            List of ChatIngestionResult (one per conversation in file).
        """
        from attestdb.intelligence.chat_ingestor import ChatIngestor
        curator = self._get_curator() if use_curator else None
        extractor = self._get_extractor(extraction)
        ingestor = ChatIngestor(extractor, self._db, curator)
        return ingestor.ingest_file(path, platform)

    def ingest_slack(
        self,
        path: str,
        bot_ids: set[str] | None = None,
        channels: list[str] | None = None,
        use_curator: bool = True,
        extraction: str = "llm",
    ):
        """Extract and ingest claims from a Slack workspace export ZIP.

        Args:
            path: Path to the Slack export ZIP file.
            bot_ids: Only treat these bot_ids as "assistant". None = all bots.
            channels: Only process these channel names. None = all.
            use_curator: Whether to triage claims through the curator.
            extraction: "llm", "heuristic", or "smart" (heuristic + LLM for novel).

        Returns:
            List of ChatIngestionResult (one per channel/thread with bot interaction).
        """
        from attestdb.intelligence.chat_ingestor import ChatIngestor
        curator = self._get_curator() if use_curator else None
        extractor = self._get_extractor(extraction)
        ingestor = ChatIngestor(extractor, self._db, curator)
        return ingestor.ingest_slack_export(path, bot_ids=bot_ids, channels=channels)

    # --- Connectors ---

    def connect(self, name: str, *, save: bool = False, **kwargs) -> "Connector":
        """Create a connector for an external data source.

        Args:
            name: Connector name (e.g. "slack", "postgres", "notion").
            save: Persist credentials to the encrypted token store.
            **kwargs: Passed to the connector constructor.

        Returns:
            A :class:`~attestdb.connectors.base.Connector` instance.
            Call ``.run(db)`` to execute.

        Example::

            conn = db.connect("slack", token="xoxb-...")
            result = conn.run(db)
        """
        from attestdb.connectors import connect as _connect

        conn = _connect(name, db_path=self._db._db_path, **kwargs)
        if save and kwargs:
            from attestdb.connectors.token_store import TokenStore

            ts = TokenStore(self._db._db_path)
            ts.save_token(name, dict(kwargs))
        return conn

    # --- Continuous sync ---

    def _get_scheduler(self):
        """Lazy-init the connector scheduler."""
        if self._db._scheduler is None:
            from attestdb.connectors.scheduler import ConnectorScheduler
            self._db._scheduler = ConnectorScheduler()
        return self._db._scheduler

    def _post_sync_check(self, connector_name: str, claims_ingested: int, **kwargs):
        """Lightweight insight check after each sync cycle. Fires insight events."""
        if claims_ingested == 0:
            return
        try:
            engine = self._get_insight_engine()
            alerts = engine.find_confidence_alerts(min_claims=3)
            if alerts:
                self._db._fire(
                    EventType.INSIGHT_ALERTS, alerts=alerts,
                    trigger="sync", connector=connector_name,
                )
        except Exception as e:
            logger.warning("Post-sync insight check failed: %s", e)

    def sync(
        self,
        name: str,
        interval: float = 300.0,
        *,
        save: bool = False,
        run_immediately: bool = True,
        jitter: float = 0.1,
        **connector_kwargs,
    ):
        """Start continuous sync for a connector.

        Creates the connector, schedules it for periodic execution in a
        background thread, and returns a SyncHandle for monitoring.

        Args:
            name: Connector name (e.g. "slack", "github", "jira").
            interval: Seconds between runs (default: 300 = 5 min).
            save: Persist connector credentials to encrypted token store.
            run_immediately: Run once before waiting for the first interval.
            jitter: Random jitter fraction to prevent thundering herd.
            **connector_kwargs: Passed to the connector constructor.

        Returns:
            SyncHandle with status, last_run, total_claims, etc.

        Example::

            db.sync("slack", interval=300, token="xoxb-...")
            db.sync("github", interval=600, token="ghp_...")
            print(db.sync_status())
        """
        # Register post-sync insight hook once
        if not getattr(self._db, '_sync_hook_registered', False):
            self._db.on(EventType.SYNC_COMPLETED, self._post_sync_check)
            self._db._sync_hook_registered = True

        conn = self._db.connect(name, save=save, **connector_kwargs)
        scheduler = self._get_scheduler()
        return scheduler.schedule(
            conn, self._db, interval=interval, jitter=jitter,
            run_immediately=run_immediately,
        )

    def sync_status(self) -> list[dict]:
        """Status of all scheduled connectors.

        Returns:
            List of dicts with name, interval, status, last_run, next_run,
            total_runs, total_claims, error_count, last_error.
        """
        if self._db._scheduler is None:
            return []
        return self._db._scheduler.status()

    def sync_stop(self, name: str) -> None:
        """Stop a scheduled connector."""
        if self._db._scheduler:
            self._db._scheduler.stop(name)

    def sync_stop_all(self) -> None:
        """Stop all scheduled connectors."""
        if self._db._scheduler:
            self._db._scheduler.stop_all()

    def sync_pause(self, name: str) -> None:
        """Pause a connector (thread stays alive but skips execution)."""
        if self._db._scheduler:
            self._db._scheduler.pause(name)

    def sync_resume(self, name: str) -> None:
        """Resume a paused connector."""
        if self._db._scheduler:
            self._db._scheduler.resume(name)

    def sync_run_now(self, name: str) -> None:
        """Trigger an immediate run of a connector."""
        if self._db._scheduler:
            self._db._scheduler.run_now(name)

    # --- Insight engine ---

    def find_bridges(self, **kwargs) -> list:
        cache_key = f"find_bridges:{sorted(kwargs.items())}"
        cached = self._db._cache.get(cache_key)
        if cached is not None and (time.time() - self._db._cache_ts.get(cache_key, 0)) < 300:
            return cached
        result = self._get_insight_engine().find_bridges(**kwargs)
        self._db._cache[cache_key] = result
        self._db._cache_ts[cache_key] = time.time()
        return result

    def find_gaps(self, expected_patterns, **kwargs) -> list:
        return self._get_insight_engine().find_gaps(expected_patterns, **kwargs)

    def find_confidence_alerts(self, **kwargs) -> list:
        cache_key = f"find_confidence_alerts:{sorted(kwargs.items())}"
        cached = self._db._cache.get(cache_key)
        if cached is not None and (time.time() - self._db._cache_ts.get(cache_key, 0)) < 120:
            return cached
        result = self._get_insight_engine().find_confidence_alerts(**kwargs)
        self._db._cache[cache_key] = result
        self._db._cache_ts[cache_key] = time.time()
        return result

    # --- Temporal analysis ---

    def temporal_analyze(
        self,
        entity_id: str,
        analysis_type: str = "regime_shifts",
        metric: str = "claim_count",
        bucket: str | None = None,
        **kwargs,
    ) -> "TemporalResult":
        """Analyze temporal patterns for an entity's claims.

        Args:
            entity_id: Entity to analyze.
            analysis_type: ``"regime_shifts"``, ``"velocity"``, or ``"cycles"``.
            metric: ``"claim_count"`` (default) or ``"avg_confidence"``.
            bucket: ``"day"``, ``"week"``, ``"month"``, or ``None`` (auto).
            **kwargs: Analysis-specific params (window_size, threshold, etc.).

        Returns:
            :class:`TemporalResult` with detected patterns.
        """
        return self._get_temporal_engine().analyze(
            entity_id, analysis_type=analysis_type,
            metric=metric, bucket=bucket, **kwargs,
        )

    def temporal_summary(
        self,
        entity_id: str,
        metric: str = "claim_count",
        bucket: str | None = None,
    ) -> "TemporalResult":
        """Run all temporal analyses (shifts, velocity, cycles) at once.

        Args:
            entity_id: Entity to analyze.
            metric: ``"claim_count"`` or ``"avg_confidence"``.
            bucket: ``"day"``, ``"week"``, ``"month"``, or ``None`` (auto).

        Returns:
            :class:`TemporalResult` with all three analyses populated.
        """
        return self._get_temporal_engine().summary(
            entity_id, metric=metric, bucket=bucket,
        )

    # --- Autonomous research ---

    def investigate(
        self,
        max_questions: int = 20,
        use_curator: bool = True,
        search_fn=None,
    ) -> "InvestigationReport":
        """Autonomous gap-closing loop: detect blindspots, research, ingest.

        Args:
            max_questions: Max questions to generate and research.
            use_curator: Whether to triage discovered claims through the curator.
            search_fn: Optional callback(question_text) -> text.

        Returns:
            InvestigationReport with before/after blindspot counts.
        """
        return self._get_researcher().investigate(
            max_questions=max_questions,
            use_curator=use_curator,
            search_fn=search_fn,
        )

    def research_question(
        self,
        question: str,
        entity_id: str | None = None,
        entity_type: str = "",
        predicate_hint: str = "",
    ) -> "ResearchResult":
        """Research a single question and ingest discovered claims.

        Args:
            question: Natural-language research question.
            entity_id: Optional focal entity.
            entity_type: Optional entity type.
            predicate_hint: Optional predicate to hint at.

        Returns:
            ResearchResult with ingestion counts.
        """
        from attestdb.core.types import ResearchQuestion

        rq = ResearchQuestion(
            entity_id=entity_id or "",
            entity_type=entity_type,
            gap_type="manual",
            question=question,
            predicate_hint=predicate_hint,
        )
        # Register as inquiry if entity_id provided
        if entity_id:
            try:
                claim_id = self._db.ingest_inquiry(
                    question=question,
                    subject=(entity_id, entity_type),
                    object=(entity_id, entity_type),
                    predicate_hint=predicate_hint,
                )
                rq.inquiry_claim_id = claim_id
            except Exception:
                pass

        return self._get_researcher().research(rq)
