"""Interactive multi-LLM chat with round-robin consensus.

Launch: attest chat [--providers openai,gemini] [--primary openai] [--db attest.db]

The user chats with multiple LLM providers simultaneously. Each provider
maintains its own conversation history. The user controls when to share
responses cross-provider (/share), when to request consensus (/consensus),
and can reference local files with @filename.

All responses and consensus results are stored in AttestDB with provenance.
Provider performance (latency, user preference, task-type affinity) is
tracked so AttestDB learns which providers excel at which tasks.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

from attestdb.core.providers import PROVIDERS, load_env_file as _load_env_file

# Terminal colors
_COLORS = {
    "gemini": "\033[34m",      # blue
    "together": "\033[36m",    # cyan
    "openai": "\033[32m",      # green
    "deepseek": "\033[35m",    # magenta
    "grok": "\033[33m",        # yellow
    "openrouter": "\033[36m",  # cyan
    "groq": "\033[91m",        # light red
    "anthropic": "\033[95m",   # light magenta
    "glm": "\033[94m",         # light blue
    "consensus": "\033[97;42m",  # white on green
    "system": "\033[90m",      # gray
}
_RESET = "\033[0m"
_BOLD = "\033[1m"

MAX_FILE_CHARS = 10_000
FILE_REF_PATTERN = re.compile(r"@([\w./\-*\[\]{}]+)")


@dataclass
class ProviderSession:
    """Per-provider conversation state."""
    name: str
    model: str
    client: object  # OpenAI client
    messages: list[dict] = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    call_count: int = 0
    last_response: str = ""


def _color(provider: str, text: str) -> str:
    c = _COLORS.get(provider, "")
    return f"{c}{text}{_RESET}" if c else text


def _resolve_file_refs(text: str, cwd: str) -> str:
    """Replace @filename references with file contents."""
    def _replace(match):
        pattern = match.group(1)
        # Try exact path first, then glob
        candidates = []
        exact = os.path.join(cwd, pattern)
        if os.path.isfile(exact):
            candidates = [exact]
        else:
            candidates = sorted(glob.glob(os.path.join(cwd, pattern)))
            candidates = [c for c in candidates if os.path.isfile(c)]

        if not candidates:
            return match.group(0)  # leave as-is if not found

        parts = []
        for path in candidates[:5]:  # max 5 files per glob
            rel = os.path.relpath(path, cwd)
            try:
                content = open(path, "r", errors="replace").read()
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + f"\n... (truncated, {len(content)} chars total)"
                parts.append(f"--- @{rel} ---\n{content}\n--- end @{rel} ---")
            except Exception as exc:
                parts.append(f"--- @{rel} --- (error reading: {exc})")

        return "\n".join(parts)

    return FILE_REF_PATTERN.sub(_replace, text)


def _detect_task_type(text: str) -> str:
    """Simple heuristic to classify the task type for provider learning."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["bug", "error", "fix", "debug", "crash", "exception", "traceback"]):
        return "debugging"
    if any(w in text_lower for w in ["review", "improve", "refactor", "optimize"]):
        return "code_review"
    if any(w in text_lower for w in ["summarize", "summary", "brief", "tldr"]):
        return "summarization"
    if any(w in text_lower for w in ["write", "create", "implement", "build", "generate", "code"]):
        return "code_generation"
    if any(w in text_lower for w in ["explain", "what is", "how does", "why", "describe"]):
        return "explanation"
    return "general"


class MultiChat:
    """Interactive multi-LLM chat session."""

    def __init__(
        self,
        db=None,
        providers: list[str] | None = None,
        primary: str | None = None,
        model_overrides: dict[str, str] | None = None,
        env_path: str | None = None,
        cwd: str | None = None,
    ):
        self.db = db
        self.cwd = cwd or os.getcwd()
        self.sessions: dict[str, ProviderSession] = {}
        self.primary: str | None = primary
        self._model_overrides = model_overrides or {}

        env_vars: dict[str, str] = {}
        if env_path:
            env_vars = _load_env_file(env_path)

        candidate_providers = providers or list(PROVIDERS.keys())

        for name in candidate_providers:
            provider = PROVIDERS.get(name)
            if not provider:
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
                self.sessions[name] = ProviderSession(
                    name=name, model=model, client=client,
                    messages=[{"role": "system", "content": (
                        "You are a knowledgeable expert assistant. Be thorough, "
                        "accurate, and specific. When shown code or files, analyze "
                        "them carefully."
                    )}],
                )
            except Exception as exc:
                logger.warning("Failed to init %s: %s", name, exc)

        # If no primary set, pick the first available
        if not self.primary and self.sessions:
            self.primary = next(iter(self.sessions))

        # If primary was specified but not available, warn
        if primary and primary not in self.sessions:
            logger.warning("Primary provider '%s' not available", primary)

    def _print_header(self):
        providers_str = ", ".join(
            f"{_BOLD}{name}{_RESET}" if name == self.primary
            else name
            for name in self.sessions
        )
        print(f"\n{_BOLD}attest chat{_RESET} ({len(self.sessions)} providers: {providers_str})")
        print(f"Primary: {_BOLD}{self.primary}{_RESET} (used for summarization)")
        print(f"Working directory: {self.cwd}")
        print(f"Use @filename to include files. Commands: /share /consensus /primary /providers /quit")
        print()

    def run(self):
        """Main interactive loop."""
        if len(self.sessions) < 1:
            print("Error: No providers available. Set API keys for at least one provider.")
            print("  e.g. export OPENAI_API_KEY=sk-...")
            return

        self._print_header()

        try:
            while True:
                try:
                    user_input = input(f"{_BOLD}>{_RESET} ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        continue
                    else:
                        break  # /quit

                # Resolve file references
                resolved = _resolve_file_refs(user_input, self.cwd)

                # Send to all providers
                self._send_to_all(resolved)

        except KeyboardInterrupt:
            print(f"\n{_color('system', 'Interrupted.')}")
        finally:
            self._save_session()
            print(f"{_color('system', 'Session saved.')}")

    def _handle_command(self, cmd: str) -> bool:
        """Handle a / command. Returns True to continue, False to quit."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("/quit", "/exit", "/q"):
            return False

        elif command == "/share":
            self._share_responses()
            return True

        elif command == "/consensus":
            self._run_consensus()
            return True

        elif command == "/tournament":
            max_rounds = int(arg) if arg.strip().isdigit() else 5
            self._run_tournament(max_rounds=max_rounds)
            return True

        elif command == "/primary":
            if arg and arg in self.sessions:
                self.primary = arg
                print(f"{_color('system', f'Primary provider set to {arg}')}")
            elif arg:
                print(f"{_color('system', f'Unknown provider: {arg}. Available: {list(self.sessions.keys())}')}")
            else:
                print(f"{_color('system', f'Current primary: {self.primary}')}")
                print(f"{_color('system', f'Usage: /primary <provider_name>')}")
            return True

        elif command == "/providers":
            for name, session in self.sessions.items():
                primary_tag = " (primary)" if name == self.primary else ""
                print(
                    f"  {_color(name, name)}{primary_tag}: {session.model} "
                    f"({session.call_count} calls, "
                    f"{session.total_tokens_in + session.total_tokens_out} tokens, "
                    f"{session.total_latency_ms:.0f}ms total)"
                )
            return True

        elif command == "/save":
            self._save_session()
            print(f"{_color('system', 'Session saved to AttestDB.')}")
            return True

        elif command == "/summarize":
            self._summarize()
            return True

        elif command == "/help":
            print("Commands:")
            print("  /share       — One round: share responses cross-provider for revision")
            print("  /tournament  — Loop: providers improve + score each other until they agree")
            print("  /tournament N — Same, with max N rounds (default: 5)")
            print("  /consensus   — Round-robin judge + synthesize best answer")
            print("  /summarize   — Ask primary provider to summarize the conversation")
            print("  /primary <p> — Set primary provider (used for summarization)")
            print("  /providers   — List active providers with stats")
            print("  /save        — Save session to AttestDB")
            print("  /quit        — Exit (auto-saves)")
            return True

        else:
            print(f"{_color('system', f'Unknown command: {command}. Type /help for commands.')}")
            return True

    def _send_to_all(self, message: str):
        """Send a message to all providers in parallel, print responses as they arrive."""
        task_type = _detect_task_type(message)

        # Add user message to all conversation histories
        for session in self.sessions.values():
            session.messages.append({"role": "user", "content": message})

        print()

        with ThreadPoolExecutor(max_workers=len(self.sessions)) as pool:
            futures = {
                pool.submit(self._call_session, name): name
                for name in self.sessions
            }

            for future in as_completed(futures):
                name = futures[future]
                session = self.sessions[name]
                try:
                    response, tokens_in, tokens_out, latency_ms = future.result()
                    session.last_response = response
                    session.messages.append({"role": "assistant", "content": response})
                    session.total_tokens_in += tokens_in
                    session.total_tokens_out += tokens_out
                    session.total_latency_ms += latency_ms
                    session.call_count += 1

                    # Print response
                    header = f"─── {name} ({session.model}) ── {latency_ms:.0f}ms ───"
                    print(_color(name, header))
                    print(response)
                    print()

                    # Track performance for learning
                    self._record_performance(name, task_type, latency_ms, tokens_out)

                except Exception as exc:
                    print(_color(name, f"─── {name} ── ERROR ───"))
                    print(f"  {exc}")
                    print()

    def _call_session(self, name: str) -> tuple[str, int, int, float]:
        """Call a single provider. Returns (response, tokens_in, tokens_out, latency_ms)."""
        session = self.sessions[name]
        start = time.monotonic()
        resp = session.client.chat.completions.create(
            model=session.model,
            messages=session.messages,
            max_tokens=2048,
            temperature=0.7,
            timeout=60,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        return content, tokens_in, tokens_out, elapsed_ms

    def _share_responses(self):
        """Cross-pollinate: share all responses with each provider for revision."""
        responses = {
            name: session.last_response
            for name, session in self.sessions.items()
            if session.last_response
        }
        if len(responses) < 2:
            print(f"{_color('system', 'Need at least 2 responses to share. Send a message first.')}")
            return

        print(f"{_color('system', 'Sharing responses cross-provider...')}")

        # Build cross-pollination messages per provider
        for name, session in self.sessions.items():
            others = "\n\n".join(
                f"=== {p} ({self.sessions[p].model}) ===\n{r}"
                for p, r in responses.items()
                if p != name
            )
            cross_msg = (
                f"Here's what other AI assistants said about the same question:\n\n"
                f"{others}\n\n"
                "Considering these perspectives, revise your answer. "
                "Where do you agree? Where do you disagree and why? "
                "Give your revised, best answer."
            )
            session.messages.append({"role": "user", "content": cross_msg})

        # Fan out (messages already appended above, so call sessions directly)
        print()
        with ThreadPoolExecutor(max_workers=len(self.sessions)) as pool:
            futures = {
                pool.submit(self._call_session, name): name
                for name in self.sessions
            }
            for future in as_completed(futures):
                name = futures[future]
                session = self.sessions[name]
                try:
                    response, tokens_in, tokens_out, latency_ms = future.result()
                    session.last_response = response
                    session.messages.append({"role": "assistant", "content": response})
                    session.total_tokens_in += tokens_in
                    session.total_tokens_out += tokens_out
                    session.total_latency_ms += latency_ms
                    session.call_count += 1

                    header = f"─── {name} (revised) ── {latency_ms:.0f}ms ───"
                    print(_color(name, header))
                    print(response)
                    print()
                except Exception as exc:
                    print(_color(name, f"─── {name} ── ERROR ───"))
                    print(f"  {exc}")
                    print()

    def _run_tournament(self, max_rounds: int = 5):
        """Tournament: providers iteratively improve and score each other until convergence."""
        import re as _re

        responses = {
            name: session.last_response
            for name, session in self.sessions.items()
            if session.last_response
        }
        if len(responses) < 2:
            print(f"{_color('system', 'Need at least 2 responses. Send a message first.')}")
            return

        provider_names = list(responses.keys())
        provider_list_str = ", ".join(provider_names)

        for round_num in range(1, max_rounds + 1):
            print(f"{_color('system', f'─── Tournament Round {round_num}/{max_rounds} ───')}")
            print()

            # Build per-provider prompts
            for name, session in self.sessions.items():
                if name not in responses:
                    continue

                others_text = "\n\n".join(
                    f"=== {p} ===\n{r}"
                    for p, r in responses.items()
                    if p != name
                )

                prompt = (
                    f"We put your question into other LLMs and they said the following:\n\n"
                    f"{others_text}\n\n"
                    f"Now do two things:\n"
                    f"1. Give your REVISED best answer, considering what the other models said. "
                    f"Where do you agree? Where do you still disagree and why?\n"
                    f"2. After your answer, on a new line write SCORES: and rate each response "
                    f"(including your own previous one) from 1-10 for accuracy, completeness, "
                    f"and clarity. Format: SCORES: {provider_list_str} = X, Y, Z\n\n"
                    f"Start with your revised answer:"
                )
                session.messages.append({"role": "user", "content": prompt})

            # Fan out
            scores_by_judge: dict[str, dict[str, int]] = {}

            with ThreadPoolExecutor(max_workers=len(self.sessions)) as pool:
                futures = {
                    pool.submit(self._call_session, name): name
                    for name in self.sessions if name in responses
                }
                for future in as_completed(futures):
                    name = futures[future]
                    session = self.sessions[name]
                    try:
                        response, tokens_in, tokens_out, latency_ms = future.result()
                        session.messages.append({"role": "assistant", "content": response})
                        session.total_tokens_in += tokens_in
                        session.total_tokens_out += tokens_out
                        session.total_latency_ms += latency_ms
                        session.call_count += 1

                        # Parse revised answer + scores
                        revised, provider_scores = self._parse_tournament_response(
                            response, provider_names,
                        )
                        responses[name] = revised
                        session.last_response = revised
                        if provider_scores:
                            scores_by_judge[name] = provider_scores

                        header = f"─── {name} (round {round_num}) ── {latency_ms:.0f}ms ───"
                        print(_color(name, header))
                        print(revised[:3000] if len(revised) > 3000 else revised)
                        if provider_scores:
                            scores_str = "  ".join(f"{p}: {s}/10" for p, s in provider_scores.items())
                            print(_color("system", f"  Scores: {scores_str}"))
                        print()

                    except Exception as exc:
                        print(_color(name, f"─── {name} ── ERROR ───"))
                        print(f"  {exc}")
                        print()

            # Check convergence
            if len(scores_by_judge) >= 2:
                winners = {}
                for judge, scores in scores_by_judge.items():
                    if scores:
                        winners[judge] = max(scores, key=scores.get)

                winner_counts: dict[str, int] = {}
                for w in winners.values():
                    winner_counts[w] = winner_counts.get(w, 0) + 1

                n_judges = len(winners)
                top_winner = max(winner_counts, key=winner_counts.get) if winner_counts else None
                top_count = winner_counts.get(top_winner, 0) if top_winner else 0

                # Average scores
                avg_scores: dict[str, float] = {}
                for p in provider_names:
                    vals = [s.get(p, 0) for s in scores_by_judge.values() if p in s]
                    avg_scores[p] = sum(vals) / len(vals) if vals else 0

                avg_str = "  ".join(f"{p}: {s:.1f}" for p, s in sorted(avg_scores.items(), key=lambda x: -x[1]))
                print(_color("system", f"  Average scores: {avg_str}"))

                if top_count == n_judges:
                    print()
                    print(_color("consensus", f"─── CONSENSUS: {top_winner} wins (unanimous, round {round_num}) ───"))
                    print(responses[top_winner])
                    print()
                    self._record_consensus_winner(top_winner, top_count / n_judges)
                    return

                if n_judges >= 3 and top_count >= n_judges - 1:
                    dissenters = [j for j, w in winners.items() if w != top_winner]
                    print()
                    print(_color("consensus", f"─── CONSENSUS: {top_winner} wins ({top_count}/{n_judges} agree, round {round_num}) ───"))
                    if dissenters:
                        print(_color("system", f"  Dissent from: {', '.join(dissenters)}"))
                    print(responses[top_winner])
                    print()
                    self._record_consensus_winner(top_winner, top_count / n_judges)
                    return

                vote_str = ", ".join(f"{j} → {w}" for j, w in winners.items())
                print(_color("system", f"  Votes: {vote_str} (no consensus yet)"))
                print()

        # Max rounds
        print(_color("system", f"─── Tournament ended after {max_rounds} rounds (no full consensus) ───"))
        if scores_by_judge:
            avg_scores = {}
            for p in provider_names:
                vals = [s.get(p, 0) for s in scores_by_judge.values() if p in s]
                avg_scores[p] = sum(vals) / len(vals) if vals else 0
            best_avg = max(avg_scores, key=avg_scores.get)
            print(_color("system", f"  Best by average score: {best_avg} ({avg_scores[best_avg]:.1f}/10)"))
            print()
            print(_color("consensus", f"─── BEST ANSWER: {best_avg} ───"))
            print(responses[best_avg])
            print()
            self._record_consensus_winner(best_avg, avg_scores[best_avg] / 10)

    @staticmethod
    def _parse_tournament_response(
        response: str, provider_names: list[str],
    ) -> tuple[str, dict[str, int]]:
        """Extract the revised answer and SCORES: line from a tournament response."""
        import re as _re

        scores: dict[str, int] = {}
        lines = response.split("\n")

        scores_line_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if "SCORES:" in lines[i].upper():
                scores_line_idx = i
                break

        if scores_line_idx >= 0:
            scores_part = lines[scores_line_idx].split(":", 1)[-1].strip()
            for provider in provider_names:
                pattern = _re.compile(
                    rf"{_re.escape(provider)}\s*[=:]\s*(\d+)", _re.IGNORECASE,
                )
                match = pattern.search(scores_part)
                if match:
                    scores[provider] = min(int(match.group(1)), 10)
            revised = "\n".join(lines[:scores_line_idx]).strip()
        else:
            revised = response.strip()

        return revised, scores

    def _run_consensus(self):
        """Round-robin judge: each provider evaluates all responses."""
        responses = {
            name: session.last_response
            for name, session in self.sessions.items()
            if session.last_response
        }
        if len(responses) < 2:
            print(f"{_color('system', 'Need at least 2 responses for consensus.')}")
            return

        print(f"{_color('system', 'Running round-robin consensus...')}")
        print()

        # Build judge prompt
        responses_text = "\n\n".join(
            f"=== {name} ({self.sessions[name].model}) ===\n{resp}"
            for name, resp in responses.items()
        )

        judge_prompt = (
            "You are judging multiple AI responses to the same question. "
            "Respond with EXACTLY this JSON (no other text):\n"
            '{"converged": true/false, "best": "provider_name", '
            '"rating": 1-10, "critique": "what is still wrong or missing"}\n\n'
            f"Responses:\n{responses_text}\n\n"
            f"Available provider names: {list(responses.keys())}\n"
            "Judge now:"
        )

        # Each provider judges
        import json
        votes = {}
        for name, session in self.sessions.items():
            try:
                resp = session.client.chat.completions.create(
                    model=session.model,
                    messages=[
                        {"role": "system", "content": "You are an expert judge evaluating AI responses."},
                        {"role": "user", "content": judge_prompt},
                    ],
                    max_tokens=256,
                    temperature=0.0,
                    timeout=30,
                )
                text = (resp.choices[0].message.content or "").strip()
                # Try to parse JSON
                if "```" in text:
                    for block in text.split("```"):
                        block = block.strip()
                        if block.startswith("json"):
                            block = block[4:].strip()
                        if block.startswith("{"):
                            text = block
                            break
                try:
                    vote = json.loads(text)
                except json.JSONDecodeError:
                    vote = {"converged": False, "rating": 5, "critique": text[:200]}

                votes[name] = vote
                converged_str = "YES" if vote.get("converged") else "NO"
                print(
                    f"  {_color(name, name)}: {converged_str} "
                    f"(rating: {vote.get('rating', '?')}/10, "
                    f"best: {vote.get('best', '?')})"
                )
                if vote.get("critique"):
                    print(f"    critique: {vote['critique'][:150]}")
            except Exception as exc:
                print(f"  {_color(name, name)}: ERROR — {exc}")
                votes[name] = {"converged": False, "rating": 0}

        # Count convergence
        n_converged = sum(1 for v in votes.values() if v.get("converged"))
        n_total = len(votes)
        confidence = n_converged / n_total if n_total else 0

        print()
        if confidence >= 0.5:
            # Find the most-voted "best" provider
            best_counts: dict[str, int] = {}
            for v in votes.values():
                b = v.get("best", "")
                if b in responses:
                    best_counts[b] = best_counts.get(b, 0) + 1

            if best_counts:
                winner = max(best_counts, key=best_counts.get)
                winner_votes = best_counts[winner]
                if winner_votes > n_total / 2:
                    # Clear majority — use winner's response
                    print(_color("consensus", f"─── CONSENSUS (confidence: {confidence:.0%}, winner: {winner}) ───"))
                    print(responses[winner])
                    print()
                    self._record_consensus_winner(winner, confidence)
                    return

            # No clear winner — synthesize via primary provider
            self._synthesize_consensus(responses, votes, confidence)
        else:
            print(_color("system", f"No consensus ({n_converged}/{n_total} agreed). Use /share to cross-pollinate, then try again."))
            # Show critiques
            for name, vote in votes.items():
                if not vote.get("converged") and vote.get("critique"):
                    print(f"  {_color(name, name)}: {vote['critique'][:200]}")
            print()

    def _synthesize_consensus(self, responses: dict, votes: dict, confidence: float):
        """Use the primary provider to synthesize a consensus answer."""
        if not self.primary or self.primary not in self.sessions:
            return

        session = self.sessions[self.primary]
        responses_text = "\n\n".join(
            f"=== {name} ===\n{resp}" for name, resp in responses.items()
        )
        critiques = "\n".join(
            f"- {name}: {v.get('critique', '')}"
            for name, v in votes.items()
            if v.get("critique")
        )

        synth_prompt = (
            f"Synthesize these expert responses into one authoritative answer. "
            f"Address remaining critiques.\n\n"
            f"Responses:\n{responses_text}\n\n"
            f"Critiques:\n{critiques}\n\n"
            f"Write the final synthesized answer:"
        )

        try:
            resp = session.client.chat.completions.create(
                model=session.model,
                messages=[
                    {"role": "system", "content": "You are synthesizing multiple expert responses."},
                    {"role": "user", "content": synth_prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
                timeout=60,
            )
            synthesis = (resp.choices[0].message.content or "").strip()
            print(_color("consensus", f"─── CONSENSUS (confidence: {confidence:.0%}, synthesized by {self.primary}) ───"))
            print(synthesis)
            print()
        except Exception as exc:
            print(f"{_color('system', f'Synthesis failed: {exc}')}")

    def _summarize(self):
        """Ask the primary provider to summarize the conversation so far."""
        if not self.primary or self.primary not in self.sessions:
            print(f"{_color('system', 'No primary provider set.')}")
            return

        session = self.sessions[self.primary]

        # Build a summary of all providers' contributions
        all_exchanges = []
        for name, s in self.sessions.items():
            for msg in s.messages:
                if msg["role"] == "user":
                    all_exchanges.append(f"User: {msg['content'][:200]}")
                elif msg["role"] == "assistant":
                    all_exchanges.append(f"{name}: {msg['content'][:300]}")

        summary_prompt = (
            "Summarize this multi-AI conversation concisely. "
            "Highlight key agreements, disagreements, and conclusions.\n\n"
            + "\n".join(all_exchanges[-30:])  # last 30 exchanges
        )

        try:
            resp = session.client.chat.completions.create(
                model=session.model,
                messages=[
                    {"role": "system", "content": "You are summarizing a multi-AI discussion."},
                    {"role": "user", "content": summary_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
                timeout=60,
            )
            summary = (resp.choices[0].message.content or "").strip()
            print(_color("consensus", f"─── SUMMARY (by {self.primary}) ───"))
            print(summary)
            print()
        except Exception as exc:
            print(f"{_color('system', f'Summary failed: {exc}')}")

    def _record_performance(self, provider: str, task_type: str, latency_ms: float, tokens_out: int):
        """Record provider performance in AttestDB for learning."""
        if not self.db:
            return
        try:
            from attestdb.core.types import ClaimInput
            self.db.ingest(ClaimInput(
                subject=(provider, "llm_provider"),
                predicate=("performed_task", "provider_performance"),
                object=(task_type, "task_type"),
                provenance={
                    "source_id": "attest_chat",
                    "source_type": "provider_telemetry",
                    "method": "chat_session",
                },
                confidence=0.5,
                payload={
                    "schema_ref": "provider_performance/v1",
                    "data": {
                        "latency_ms": latency_ms,
                        "tokens_out": tokens_out,
                        "task_type": task_type,
                    },
                },
            ))
        except Exception:
            pass  # Don't let telemetry failures break the chat

    def _record_consensus_winner(self, winner: str, confidence: float):
        """Record which provider won consensus — feeds future primary selection."""
        if not self.db:
            return
        try:
            from attestdb.core.types import ClaimInput
            self.db.ingest(ClaimInput(
                subject=(winner, "llm_provider"),
                predicate=("won_consensus", "provider_performance"),
                object=("consensus_round", "event"),
                provenance={
                    "source_id": "attest_chat",
                    "source_type": "provider_telemetry",
                },
                confidence=confidence,
                payload={
                    "schema_ref": "consensus_winner/v1",
                    "data": {"confidence": confidence},
                },
            ))
        except Exception:
            pass

    def _save_session(self):
        """Save conversation to AttestDB."""
        if not self.db:
            return

        from attestdb.core.types import ClaimInput

        for name, session in self.sessions.items():
            # Save final response as a claim
            if session.last_response:
                try:
                    self.db.ingest(ClaimInput(
                        subject=("chat_session", "session"),
                        predicate=("has_response", "chat_response"),
                        object=(name, "llm_provider"),
                        provenance={
                            "source_id": name,
                            "source_type": "llm_api",
                            "method": "attest_chat",
                            "model_version": session.model,
                        },
                        confidence=0.5,
                        payload={
                            "schema_ref": "chat_session/response/v1",
                            "data": {
                                "response": session.last_response[:4000],
                                "total_calls": session.call_count,
                                "total_tokens": session.total_tokens_in + session.total_tokens_out,
                                "total_latency_ms": session.total_latency_ms,
                            },
                        },
                    ))
                except Exception as exc:
                    logger.warning("Failed to save session for %s: %s", name, exc)
