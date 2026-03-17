"""ConsensusEngine — Multi-LLM consensus through parallel querying and cross-pollination.

Round-robin tournament: every provider is both participant and judge. Each round,
providers revise their answers after seeing all others, then each votes on whether
the group has converged. Consensus is declared only when ALL providers agree the
best answer has been reached.

Reuses the PROVIDERS dict and client initialization pattern from curator.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import shared provider configs
from attestdb.core.providers import PROVIDERS, load_env_file as _load_env_file

# Default to open/free models for the OSS consensus engine.
# These providers offer free tiers or very cheap rates.
OPEN_PROVIDERS = ["gemini", "groq", "deepseek", "together", "glm"]

# Per-1K-token pricing (USD). Input/output rates for default models.
# Updated March 2026. Free-tier providers show 0.0.
PROVIDER_PRICING: dict[str, dict[str, float]] = {
    "gemini": {"input": 0.0, "output": 0.0},             # free tier
    "together": {"input": 0.00018, "output": 0.00018},    # Qwen3-Next-80B-A3B
    "openai": {"input": 0.0004, "output": 0.0016},        # gpt-4.1-mini
    "deepseek": {"input": 0.00014, "output": 0.00028},    # deepseek-chat (V3.2)
    "grok": {"input": 0.003, "output": 0.015},            # grok-4-1-fast
    "openrouter": {"input": 0.00014, "output": 0.00028},  # deepseek-v3.2 via OR
    "groq": {"input": 0.0, "output": 0.0},                # free tier
    "anthropic": {"input": 0.0008, "output": 0.004},      # claude-haiku-4-5
    "glm": {"input": 0.0001, "output": 0.0001},           # glm-4-flash
}


@dataclass
class JudgeVote:
    """A single provider's judgment on whether the group has converged."""
    provider: str
    converged: bool                  # does this provider think we've converged?
    best_provider: str = ""          # who does this provider think gave the best answer?
    rating: int = 0                  # 1-10 agreement rating
    critique: str = ""               # what's still wrong / missing


@dataclass
class ProviderResponse:
    provider: str
    model: str
    response: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    error: str = ""
    round_number: int = 1
    cost_usd: float = 0.0


@dataclass
class ConsensusResult:
    question: str
    consensus: str                          # final synthesized answer
    confidence: float                       # 0-1 agreement level
    rounds: int                             # how many rounds it took
    providers_used: list[str] = field(default_factory=list)
    responses: list[ProviderResponse] = field(default_factory=list)
    dissents: list[str] = field(default_factory=list)
    votes: list[JudgeVote] = field(default_factory=list)  # final round votes
    total_tokens: int = 0
    total_cost: float = 0.0
    converged: bool = False


class ConsensusEngine:
    """Multi-LLM consensus through parallel querying and round-robin judging.

    Every provider participates and judges. Consensus requires unanimous (or
    supermajority) agreement that the best answer has been reached.
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        model_overrides: dict[str, str] | None = None,
        env_path: str | None = None,
    ):
        """Auto-detect available providers from API keys in environment.

        Defaults to open/free providers (Gemini, Groq, DeepSeek, Together, GLM).
        Pass providers=list(PROVIDERS.keys()) to use all available providers
        including proprietary ones.

        Args:
            providers: Filter to specific providers. None = open providers only.
            model_overrides: Override default models, e.g. {"openai": "gpt-4o"}.
            env_path: Path to .env file for API key resolution.
        """
        self._clients: dict[str, tuple] = {}  # name -> (client, model)
        self._env_path = env_path
        self._model_overrides = model_overrides or {}

        env_vars: dict[str, str] = {}
        if env_path:
            env_vars = _load_env_file(env_path)

        candidate_providers = providers or OPEN_PROVIDERS

        for name in candidate_providers:
            provider = PROVIDERS.get(name)
            if not provider:
                logger.warning("Unknown provider '%s', skipping", name)
                continue

            api_key = os.environ.get(provider["env_key"])
            if not api_key and env_vars:
                api_key = env_vars.get(provider["env_key"])
            if not api_key:
                continue

            model = self._model_overrides.get(name, provider["default_model"])

            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=provider["base_url"])
                self._clients[name] = (client, model)
                logger.info("Consensus provider ready: %s (%s)", name, model)
            except Exception as exc:
                logger.warning("Failed to init %s: %s", name, exc)

    @property
    def available_providers(self) -> list[str]:
        return list(self._clients.keys())

    def run(
        self,
        question: str,
        context: str = "",
        max_rounds: int = 3,
        convergence_threshold: float = 0.8,
        providers: list[str] | None = None,
    ) -> ConsensusResult:
        """Execute round-robin consensus. Returns when all agree or max_rounds hit.

        Each round:
        1. Providers see all responses and revise their answer
        2. Every provider votes: converged? best response? rating? critique?
        3. If all vote converged → done. Otherwise, critiques feed next round.

        Args:
            question: The question to get consensus on.
            context: Optional document/chat context to include.
            max_rounds: Maximum number of rounds (1 = no cross-pollination).
            convergence_threshold: Fraction of providers that must agree to converge
                                   (1.0 = unanimous, 0.67 = supermajority).
            providers: Filter to specific providers for this run.
        """
        active_clients = self._clients
        if providers:
            active_clients = {
                k: v for k, v in self._clients.items() if k in providers
            }

        if len(active_clients) < 2:
            raise ValueError(
                f"Need at least 2 providers for consensus, got {len(active_clients)}: "
                f"{list(active_clients.keys())}. Check API keys."
            )

        all_responses: list[ProviderResponse] = []
        all_votes: list[JudgeVote] = []
        confidence = 0.0
        converged = False
        round_num = 0

        # Round 1: Initial fan-out
        round_num = 1
        messages = self._initial_prompt(question, context)
        round_responses = self._fan_out(active_clients, messages, round_num)
        all_responses.extend(round_responses)

        # After round 1, run the tournament
        for round_num in range(2, max_rounds + 1):
            # Every provider judges: has the group converged?
            valid_responses = [r for r in round_responses if not r.error]
            if len(valid_responses) >= 2:
                votes = self._round_robin_judge(
                    active_clients, question, valid_responses,
                )
                all_votes = votes

                # Check if enough providers say "converged"
                n_converged = sum(1 for v in votes if v.converged)
                n_total = len(votes)
                confidence = n_converged / n_total if n_total else 0.0

                if confidence >= convergence_threshold:
                    converged = True
                    break

                # Feed critiques into next round's cross-pollination
                critiques = {v.provider: v.critique for v in votes if v.critique}
            else:
                critiques = {}

            # Cross-pollinate with critiques
            round_responses = self._cross_pollinate(
                active_clients, question, context,
                round_responses, round_num, critiques,
            )
            all_responses.extend(round_responses)

        # Final judgment if we exhausted rounds without converging
        if not converged:
            valid_last = [r for r in round_responses if not r.error]
            if len(valid_last) >= 2:
                votes = self._round_robin_judge(
                    active_clients, question, valid_last,
                )
                all_votes = votes
                n_converged = sum(1 for v in votes if v.converged)
                n_total = len(votes)
                confidence = n_converged / n_total if n_total else 0.0
                converged = confidence >= convergence_threshold

        # Synthesize: use the provider voted "best" most often, or synthesize
        valid_final = [r for r in round_responses if not r.error]
        consensus_text = self._synthesize_from_votes(
            active_clients, question, valid_final, all_votes,
        )

        # Providers that errored on every round
        responding_providers = {r.provider for r in all_responses if not r.error}
        dissents = [
            name for name in active_clients
            if name not in responding_providers
        ]
        # Also add providers that voted "not converged" in final round
        for v in all_votes:
            if not v.converged and v.provider not in dissents:
                dissents.append(v.provider)

        total_tokens = sum(r.tokens_in + r.tokens_out for r in all_responses)
        total_cost = sum(r.cost_usd for r in all_responses)

        return ConsensusResult(
            question=question,
            consensus=consensus_text,
            confidence=confidence,
            rounds=round_num,
            providers_used=list(active_clients.keys()),
            responses=all_responses,
            dissents=dissents,
            votes=all_votes,
            total_tokens=total_tokens,
            total_cost=total_cost,
            converged=converged,
        )

    def _initial_prompt(self, question: str, context: str) -> list[dict]:
        """Build the initial prompt messages.

        If context contains evidence from a knowledge database, providers
        evaluate that evidence rather than answering from training data alone.
        """
        if context:
            system = (
                "You are an expert evaluating claims from a knowledge database. "
                "Below you will see existing claims with their sources, confidence "
                "levels, and corroboration status. Your job is to:\n"
                "1. Assess whether the evidence supports the question being asked\n"
                "2. Identify which claims are well-supported and which are weak\n"
                "3. Synthesize the evidence into a thorough answer\n"
                "4. Note any gaps, contradictions, or areas needing more evidence\n\n"
                "Ground your answer in the provided evidence. Cite specific sources "
                "and confidence levels. If the evidence is insufficient, say so."
            )
            user_content = f"Evidence from knowledge database:\n{context}\n\nQuestion: {question}"
        else:
            system = (
                "You are a knowledgeable expert. Give a thorough, accurate answer. "
                "Be specific and cite mechanisms, evidence, or reasoning where possible."
            )
            user_content = question
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _fan_out(
        self,
        clients: dict[str, tuple],
        messages: list[dict],
        round_num: int,
    ) -> list[ProviderResponse]:
        """Query all providers in parallel via ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            futures = {
                pool.submit(
                    self._call_provider, name, client, model, messages, round_num,
                ): name
                for name, (client, model) in clients.items()
            }
            results = []
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(ProviderResponse(
                        provider=name, model="", response="",
                        error=str(exc), round_number=round_num,
                    ))
            return results

    def _call_provider(
        self,
        name: str,
        client,
        model: str,
        messages: list[dict],
        round_num: int,
    ) -> ProviderResponse:
        """Call a single provider and return the response."""
        start = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                timeout=60,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            choice = resp.choices[0]
            content = choice.message.content or ""
            usage = resp.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            pricing = PROVIDER_PRICING.get(name, {"input": 0.0, "output": 0.0})
            cost = (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1000.0
            return ProviderResponse(
                provider=name,
                model=model,
                response=content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=elapsed_ms,
                round_number=round_num,
                cost_usd=cost,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning("Provider %s failed: %s", name, exc)
            return ProviderResponse(
                provider=name, model=model, response="",
                error=str(exc), latency_ms=elapsed_ms,
                round_number=round_num,
            )

    def _round_robin_judge(
        self,
        clients: dict[str, tuple],
        question: str,
        responses: list[ProviderResponse],
    ) -> list[JudgeVote]:
        """Every provider judges all responses in parallel.

        Each provider answers:
        - Has the group converged on the best answer? (yes/no)
        - Which response is best? (provider name)
        - Agreement rating (1-10)
        - What's still wrong or missing? (critique — empty if converged)

        Returns a JudgeVote from each provider.
        """
        provider_names = [r.provider for r in responses if r.response]
        responses_text = "\n\n".join(
            f"=== Response from {r.provider} ===\n{r.response}"
            for r in responses if r.response
        )

        messages = [
            {"role": "system", "content": (
                "You are an expert judge evaluating multiple AI responses to a question. "
                "You must decide: has the group converged on the best possible answer?\n\n"
                "Respond with EXACTLY this JSON (no other text):\n"
                '{"converged": true/false, "best": "provider_name", '
                '"rating": 1-10, "critique": "what is still wrong or missing (empty if converged)"}'
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Responses to evaluate:\n{responses_text}\n\n"
                f"Available provider names: {provider_names}\n\n"
                "Has the group reached the best answer? Judge now:"
            )},
        ]

        votes: list[JudgeVote] = []

        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            futures = {
                pool.submit(
                    self._call_provider, name, client, model, messages, 0,
                ): name
                for name, (client, model) in clients.items()
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    resp = future.result()
                    vote = self._parse_judge_vote(name, resp.response, provider_names)
                    votes.append(vote)
                except Exception as exc:
                    logger.warning("Judge vote failed for %s: %s", name, exc)
                    votes.append(JudgeVote(
                        provider=name, converged=False,
                        critique=f"Judge failed: {exc}",
                    ))

        return votes

    def _parse_judge_vote(
        self,
        judge_name: str,
        raw_response: str,
        valid_providers: list[str],
    ) -> JudgeVote:
        """Parse a judge's JSON response into a JudgeVote."""
        # Try to extract JSON from the response (models sometimes wrap in markdown)
        text = raw_response.strip()
        if "```" in text:
            # Extract from code block
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    text = block
                    break

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: look for key signals in the text
            text_lower = text.lower()
            is_converged = any(
                phrase in text_lower
                for phrase in ['"converged": true', '"converged":true', "yes, converged"]
            )
            return JudgeVote(
                provider=judge_name,
                converged=is_converged,
                rating=8 if is_converged else 4,
                critique=text[:200] if not is_converged else "",
            )

        converged = bool(data.get("converged", False))
        best = str(data.get("best", ""))
        # Validate best is an actual provider
        if best not in valid_providers:
            best = ""
        rating = int(data.get("rating", 5))
        rating = max(1, min(10, rating))
        critique = str(data.get("critique", ""))

        return JudgeVote(
            provider=judge_name,
            converged=converged,
            best_provider=best,
            rating=rating,
            critique=critique,
        )

    def _cross_pollinate(
        self,
        clients: dict[str, tuple],
        question: str,
        context: str,
        prior_responses: list[ProviderResponse],
        round_num: int,
        critiques: dict[str, str] | None = None,
    ) -> list[ProviderResponse]:
        """Build cross-pollination prompts and fan out round N+1.

        If critiques are provided (from the judge round), they're included
        so providers know what to fix.
        """
        valid_prior = [r for r in prior_responses if not r.error and r.response]
        results = []

        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            futures = {}
            for name, (client, model) in clients.items():
                messages = self._cross_pollinate_prompt(
                    question, context, name, valid_prior, round_num, critiques,
                )
                futures[pool.submit(
                    self._call_provider, name, client, model, messages, round_num,
                )] = name

            for future in as_completed(futures):
                fname = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(ProviderResponse(
                        provider=fname, model="", response="",
                        error=str(exc), round_number=round_num,
                    ))
        return results

    def _cross_pollinate_prompt(
        self,
        question: str,
        context: str,
        current_provider: str,
        prior_responses: list[ProviderResponse],
        round_num: int,
        critiques: dict[str, str] | None = None,
    ) -> list[dict]:
        """Build messages for round N+1: include all prior responses + critiques."""
        if context:
            system = (
                "You are an expert evaluating evidence from a knowledge database, "
                "participating in a multi-expert review. Other experts have reviewed "
                "all responses and provided feedback. Revise your answer to address "
                "their critiques. Stay grounded in the evidence provided."
            )
        else:
            system = (
                "You are a knowledgeable expert participating in a multi-expert review. "
                "Other experts have reviewed all responses and provided feedback. "
                "Revise your answer to address their critiques and produce the best "
                "possible answer."
            )

        # Find this provider's own prior response
        own_response = ""
        for r in prior_responses:
            if r.provider == current_provider:
                own_response = r.response
                break

        other_responses = "\n\n".join(
            f"=== {r.provider} ===\n{r.response}"
            for r in prior_responses
            if r.provider != current_provider and r.response
        )

        context_section = f"\nContext:\n{context}\n" if context else ""

        critique_section = ""
        if critiques:
            critique_lines = []
            for provider, critique in critiques.items():
                if critique:
                    critique_lines.append(f"- {provider}: {critique}")
            if critique_lines:
                critique_section = (
                    "\n\nCritiques from the review panel:\n"
                    + "\n".join(critique_lines)
                )

        user_content = (
            f"Question: {question}{context_section}\n\n"
            f"Your previous answer (round {round_num - 1}):\n{own_response}\n\n"
            f"Other experts' answers:\n{other_responses}"
            f"{critique_section}\n\n"
            "Give your revised, best answer. Address the critiques."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _synthesize_from_votes(
        self,
        clients: dict[str, tuple],
        question: str,
        responses: list[ProviderResponse],
        votes: list[JudgeVote],
    ) -> str:
        """Synthesize final answer, weighted by judge votes.

        If judges agreed on a "best" provider, use that response as the basis.
        Otherwise, ask the most-voted-for provider to synthesize.
        """
        valid = [r for r in responses if not r.error and r.response]
        if not valid:
            return ""

        # Count "best" votes
        best_counts: dict[str, int] = {}
        for v in votes:
            if v.best_provider:
                best_counts[v.best_provider] = best_counts.get(v.best_provider, 0) + 1

        # If there's a clear winner (majority), use their response directly
        if best_counts:
            winner = max(best_counts, key=best_counts.get)
            winner_count = best_counts[winner]
            total_votes = len(votes)
            if winner_count > total_votes / 2:
                for r in valid:
                    if r.provider == winner:
                        return r.response

        # No clear winner — ask the highest-rated provider to synthesize
        avg_ratings: dict[str, float] = {}
        for v in votes:
            if v.best_provider:
                avg_ratings.setdefault(v.best_provider, []).append(v.rating)
        if avg_ratings:
            synthesizer = max(
                avg_ratings,
                key=lambda p: sum(avg_ratings[p]) / len(avg_ratings[p]),
            )
        else:
            # Fallback: fastest responder
            valid.sort(key=lambda r: r.latency_ms)
            synthesizer = valid[0].provider

        if synthesizer not in clients:
            synthesizer = next(iter(clients))

        client, model = clients[synthesizer]

        responses_text = "\n\n".join(
            f"=== {r.provider} ===\n{r.response}"
            for r in valid
        )

        # Include critiques if any
        critique_summary = ""
        unresolved = [v.critique for v in votes if v.critique and not v.converged]
        if unresolved:
            critique_summary = (
                "\n\nRemaining disagreements:\n"
                + "\n".join(f"- {c}" for c in unresolved)
            )

        messages = [
            {"role": "system", "content": (
                "You are synthesizing multiple expert responses into a single "
                "authoritative answer. Include all points of agreement. "
                "Address any remaining disagreements noted below. "
                "Be thorough but concise."
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Expert responses:\n{responses_text}"
                f"{critique_summary}\n\n"
                "Write the final synthesized answer:"
            )},
        ]

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
                timeout=60,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Synthesis failed: %s; using best individual response", exc)
            return valid[0].response

    # Keep old method name as alias for backward compat with tests
    def _check_convergence(
        self,
        clients: dict[str, tuple],
        question: str,
        responses: list[ProviderResponse],
    ) -> float:
        """Convenience: run round-robin judge and return confidence as float."""
        votes = self._round_robin_judge(clients, question, responses)
        n_converged = sum(1 for v in votes if v.converged)
        return n_converged / len(votes) if votes else 0.0
