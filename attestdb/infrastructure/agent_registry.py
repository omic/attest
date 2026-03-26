"""AgentRegistry — register agents and track their domain expertise and eval scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

from attestdb.core.types import claim_from_dict


class AgentRegistry:
    """Manage agent registrations and eval score tracking.

    Stores registrations as claims in the knowledge graph:
    - (agent_id, has_domain, domain)
    - (agent_id, has_capability, capability)
    - (agent_id, uses_model, model)
    """

    def __init__(self, db: "AttestDB"):
        self.db = db

    def register(
        self,
        agent_id: str,
        domains: list[str] | None = None,
        capabilities: list[str] | None = None,
        model: str | None = None,
    ) -> None:
        """Register an agent and its domain expertise.

        Stores registration as claims in the graph.
        """
        for domain in (domains or []):
            self.db.ingest(
                subject=(agent_id, "agent"),
                predicate=("has_domain", "agent_meta"),
                object=(domain, "domain"),
                provenance={"source_type": "agent_registry", "source_id": f"register:{agent_id}"},
                confidence=1.0,
            )

        for capability in (capabilities or []):
            self.db.ingest(
                subject=(agent_id, "agent"),
                predicate=("has_capability", "agent_meta"),
                object=(capability, "capability"),
                provenance={"source_type": "agent_registry", "source_id": f"register:{agent_id}"},
                confidence=1.0,
            )

        if model:
            self.db.ingest(
                subject=(agent_id, "agent"),
                predicate=("uses_model", "agent_meta"),
                object=(model, "model"),
                provenance={"source_type": "agent_registry", "source_id": f"register:{agent_id}"},
                confidence=1.0,
            )

    def list_agents(self) -> list[dict]:
        """List all registered agents with their domains and latest eval scores."""
        agents: dict[str, dict] = {}

        # Collect agent metadata from domain/capability/model claims
        for pred_id in ("has_domain", "has_capability", "uses_model"):
            raw = self.db._store.claims_by_predicate_id(pred_id)
            for d in raw:
                claim = claim_from_dict(d)
                if claim.subject.entity_type != "agent":
                    continue
                aid = claim.subject.id
                if aid not in agents:
                    agents[aid] = {
                        "agent_id": aid,
                        "domains": [],
                        "capabilities": [],
                        "model": None,
                        "latest_score": None,
                    }
                if pred_id == "has_domain":
                    if claim.object.id not in agents[aid]["domains"]:
                        agents[aid]["domains"].append(claim.object.id)
                elif pred_id == "has_capability":
                    if claim.object.id not in agents[aid]["capabilities"]:
                        agents[aid]["capabilities"].append(claim.object.id)
                elif pred_id == "uses_model":
                    agents[aid]["model"] = claim.object.id

        # Attach latest eval scores
        eval_claims = self.db._store.claims_by_predicate_id("eval_scored")
        for d in eval_claims:
            claim = claim_from_dict(d)
            aid = claim.subject.id
            if aid in agents:
                current = agents[aid].get("latest_score")
                if current is None or claim.timestamp > (agents[aid].get("_latest_ts") or 0):
                    agents[aid]["latest_score"] = claim.confidence
                    agents[aid]["_latest_ts"] = claim.timestamp

        # Clean up internal fields
        for info in agents.values():
            info.pop("_latest_ts", None)

        return list(agents.values())

    def agent_profile(self, agent_id: str) -> dict:
        """Get full profile: domains, capabilities, eval history, score trends."""
        profile: dict = {
            "agent_id": agent_id,
            "domains": [],
            "capabilities": [],
            "model": None,
            "eval_history": [],
            "score_trend": [],
        }

        # Gather metadata
        for pred_id in ("has_domain", "has_capability", "uses_model"):
            raw = self.db._store.claims_by_predicate_id(pred_id)
            for d in raw:
                claim = claim_from_dict(d)
                if claim.subject.id != agent_id:
                    continue
                if pred_id == "has_domain":
                    if claim.object.id not in profile["domains"]:
                        profile["domains"].append(claim.object.id)
                elif pred_id == "has_capability":
                    if claim.object.id not in profile["capabilities"]:
                        profile["capabilities"].append(claim.object.id)
                elif pred_id == "uses_model":
                    profile["model"] = claim.object.id

        # Gather eval history
        eval_claims = self.db._store.claims_by_predicate_id("eval_scored")
        for d in eval_claims:
            claim = claim_from_dict(d)
            if claim.subject.id != agent_id:
                continue
            entry = {
                "eval_id": claim.object.id,
                "score": claim.confidence,
                "timestamp": claim.timestamp,
            }
            if claim.payload and claim.payload.data:
                entry.update(claim.payload.data)
            profile["eval_history"].append(entry)

        # Sort by timestamp and extract score trend
        profile["eval_history"].sort(key=lambda e: e.get("timestamp", 0))
        profile["score_trend"] = [e.get("score", 0.0) for e in profile["eval_history"]]

        return profile

    def leaderboard(self, domain: str | None = None) -> list[dict]:
        """Rank agents by eval score. Optionally filter by domain.

        Returns a list of dicts sorted by latest score descending.
        """
        agents = self.list_agents()

        if domain:
            agents = [a for a in agents if domain in a.get("domains", [])]

        # Only include agents that have been scored
        scored = [a for a in agents if a.get("latest_score") is not None]

        # Sort by score descending
        scored.sort(key=lambda a: a.get("latest_score", 0.0), reverse=True)

        # Add rank
        for i, agent in enumerate(scored, 1):
            agent["rank"] = i

        return scored
