"""Attest: a brain for AI agents that actually learns."""

__version__ = "0.1.36"

# Lazy-import table: name -> (module_path, attribute_name)
# Open-source engine only — intelligence & connectors are in attestdb-enterprise.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Infrastructure
    "AttestDB": ("attestdb.infrastructure.attest_db", "AttestDB"),
    "open": ("attestdb.infrastructure.attest_db", "open"),
    # Core types
    "AskResult": ("attestdb.core.types", "AskResult"),
    "ClaimInput": ("attestdb.core.types", "ClaimInput"),
    "Claim": ("attestdb.core.types", "Claim"),
    "ClaimStatus": ("attestdb.core.types", "ClaimStatus"),
    "ContextFrame": ("attestdb.core.types", "ContextFrame"),
    "EntitySummary": ("attestdb.core.types", "EntitySummary"),
    "BatchResult": ("attestdb.core.types", "BatchResult"),
    "RetractResult": ("attestdb.core.types", "RetractResult"),
    "QualityReport": ("attestdb.core.types", "QualityReport"),
    "CascadeResult": ("attestdb.core.types", "CascadeResult"),
    "DownstreamNode": ("attestdb.core.types", "DownstreamNode"),
    "KnowledgeHealth": ("attestdb.core.types", "KnowledgeHealth"),
    "QueryProfile": ("attestdb.core.types", "QueryProfile"),
    "PathStep": ("attestdb.core.types", "PathStep"),
    "PathResult": ("attestdb.core.types", "PathResult"),
    "SchemaDescriptor": ("attestdb.core.types", "SchemaDescriptor"),
    # New API types (Stage 7)
    "ImpactReport": ("attestdb.core.types", "ImpactReport"),
    "BlindspotMap": ("attestdb.core.types", "BlindspotMap"),
    "ConsensusReport": ("attestdb.core.types", "ConsensusReport"),
    "AuditTrail": ("attestdb.core.types", "AuditTrail"),
    "DriftReport": ("attestdb.core.types", "DriftReport"),
    "HypotheticalReport": ("attestdb.core.types", "HypotheticalReport"),
    # Knowledge-Intelligence API types
    "EvidenceChain": ("attestdb.core.types", "EvidenceChain"),
    "ConfidenceGap": ("attestdb.core.types", "ConfidenceGap"),
    "HypothesisVerdict": ("attestdb.core.types", "HypothesisVerdict"),
    "ConfidenceChange": ("attestdb.core.types", "ConfidenceChange"),
    "EvolutionReport": ("attestdb.core.types", "EvolutionReport"),
    "Investigation": ("attestdb.core.types", "Investigation"),
    "ReasoningHop": ("attestdb.core.types", "ReasoningHop"),
    "SourceOverlap": ("attestdb.core.types", "SourceOverlap"),
    "ReasoningChain": ("attestdb.core.types", "ReasoningChain"),
    # Research types
    "ResearchQuestion": ("attestdb.core.types", "ResearchQuestion"),
    "ResearchResult": ("attestdb.core.types", "ResearchResult"),
    "InvestigationReport": ("attestdb.core.types", "InvestigationReport"),
    "CloseGapsReport": ("attestdb.core.types", "CloseGapsReport"),
    # Discovery engine types
    "Discovery": ("attestdb.core.types", "Discovery"),
    "Analogy": ("attestdb.core.types", "Analogy"),
    # Crown jewels types
    "BeliefChange": ("attestdb.core.types", "BeliefChange"),
    "KnowledgeDiff": ("attestdb.core.types", "KnowledgeDiff"),
    "ContradictionSide": ("attestdb.core.types", "ContradictionSide"),
    "ContradictionAnalysis": ("attestdb.core.types", "ContradictionAnalysis"),
    "ContradictionReport": ("attestdb.core.types", "ContradictionReport"),
    "ConnectionLoss": ("attestdb.core.types", "ConnectionLoss"),
    "ConfidenceShift": ("attestdb.core.types", "ConfidenceShift"),
    "SimulationReport": ("attestdb.core.types", "SimulationReport"),
    "Citation": ("attestdb.core.types", "Citation"),
    "BriefSection": ("attestdb.core.types", "BriefSection"),
    "KnowledgeBrief": ("attestdb.core.types", "KnowledgeBrief"),
    "ExplanationStep": ("attestdb.core.types", "ExplanationStep"),
    "Explanation": ("attestdb.core.types", "Explanation"),
    "ForecastConnection": ("attestdb.core.types", "ForecastConnection"),
    "Forecast": ("attestdb.core.types", "Forecast"),
    "MergeConflict": ("attestdb.core.types", "MergeConflict"),
    "MergeReport": ("attestdb.core.types", "MergeReport"),
    # Consensus types
    "JudgeVote": ("attestdb.core.types", "JudgeVote"),
    "ProviderResponse": ("attestdb.core.types", "ProviderResponse"),
    "AgentConsensusResult": ("attestdb.core.types", "AgentConsensusResult"),
    # Consensus engine + chat
    "ConsensusEngine": ("attestdb.core.consensus", "ConsensusEngine"),
    "MultiChat": ("attestdb.core.chat", "MultiChat"),
    "BrowserChat": ("attestdb.core.browser_chat", "BrowserChat"),
}

# Backward-compat alias
SubstrateDB = None  # resolved lazily below

# Enterprise symbols that proxy to attestdb-enterprise when installed
_ENTERPRISE_IMPORTS: dict[str, tuple[str, str]] = {
    # Intelligence (closed source)
    "CuratorV1": ("attestdb.intelligence.curator", "CuratorV1"),
    "TextExtractor": ("attestdb.intelligence.text_extractor", "TextExtractor"),
    "ChatIngestor": ("attestdb.intelligence.chat_ingestor", "ChatIngestor"),
    "HeuristicExtractor": ("attestdb.intelligence.heuristic_extractor", "HeuristicExtractor"),
    "SlackExportReader": ("attestdb.intelligence.slack_connector", "SlackExportReader"),
    "NoveltyChecker": ("attestdb.intelligence.novelty", "NoveltyChecker"),
    "ClaimMerger": ("attestdb.intelligence.novelty", "ClaimMerger"),
    "SmartExtractor": ("attestdb.intelligence.smart_extractor", "SmartExtractor"),
    "InsightEngineV1": ("attestdb.intelligence.insight_engine", "InsightEngineV1"),
    "OmicPipeline": ("attestdb.intelligence.omic_pipeline", "OmicPipeline"),
    # Connectors (closed source)
    "Connector": ("attestdb.connectors.base", "Connector"),
    "ConnectorResult": ("attestdb.connectors.base", "ConnectorResult"),
    # Vocabularies (closed source)
    "register_bio_vocabulary": ("attestdb.intelligence.bio_vocabulary", "register_bio_vocabulary"),
    "register_devops_vocabulary": (
        "attestdb.intelligence.devops_vocabulary",
        "register_devops_vocabulary",
    ),
    "register_ml_vocabulary": (
        "attestdb.intelligence.ml_vocabulary",
        "register_ml_vocabulary",
    ),
    "register_ai_tools_vocabulary": (
        "attestdb.intelligence.ai_tools_vocabulary",
        "register_ai_tools_vocabulary",
    ),
    # Enterprise service layer
    "Researcher": ("attestdb_enterprise.researcher", "Researcher"),
    "Client": ("attestdb_enterprise.client", "Client"),
    "AdminClient": ("attestdb_enterprise.client", "AdminClient"),
}


def __getattr__(name: str):
    if name == "SubstrateDB":
        from attestdb.infrastructure.attest_db import AttestDB
        globals()["SubstrateDB"] = AttestDB
        return AttestDB
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _ENTERPRISE_IMPORTS:
        module_path, attr_name = _ENTERPRISE_IMPORTS[name]
        try:
            import importlib
            module = importlib.import_module(module_path)
            value = getattr(module, attr_name)
            globals()[name] = value
            return value
        except ImportError:
            raise ImportError(
                f"{name} requires closed-source components. "
                f"Install attestdb-enterprise: pip install attestdb-enterprise"
            ) from None
    raise AttributeError(f"module 'attestdb' has no attribute {name!r}")


__all__ = ["__version__", "SubstrateDB"] + list(_LAZY_IMPORTS.keys())
