"""Browser-based multi-LLM chat — uses your existing ChatGPT, Claude, and Gemini subscriptions.

Launch: attest chat [--mode browser] [--providers chatgpt,claude,gemini]

Opens a Chromium browser with persistent login sessions. On first run, you'll
log into each chat app once — after that, sessions are remembered. Your question
is sent to all providers simultaneously via their web UIs.

Judging/scoring/synthesis uses a free API tier (Gemini or Groq) so you don't
need any API keys for the primary chat providers.

Requires: pip install playwright && playwright install chromium
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Terminal colors
_COLORS = {
    "chatgpt": "\033[32m",     # green
    "claude": "\033[95m",      # light magenta
    "gemini": "\033[34m",      # blue
    "consensus": "\033[97;42m",  # white on green
    "system": "\033[90m",      # gray
    "judge": "\033[33m",       # yellow
}
_RESET = "\033[0m"
_BOLD = "\033[1m"

PROFILE_DIR = Path.home() / ".attestdb" / "browser-profile"

MAX_FILE_CHARS = 10_000
FILE_REF_PATTERN = re.compile(r"@([\w./\-*\[\]{}]+)")

# CSS selectors for chat UIs.
# These WILL break when UIs change — keep them here for easy updates.
PROVIDER_SELECTORS = {
    "chatgpt": {
        "url": "https://chatgpt.com",
        "input": 'div[id="prompt-textarea"]',
        "send": 'button[data-testid="send-button"]',
        "response": 'div[data-message-author-role="assistant"]',
        "streaming_indicator": 'button[aria-label="Stop generating"]',
        "new_chat": 'a[data-testid="create-new-chat-button"]',
    },
    "gemini": {
        "url": "https://gemini.google.com",
        "input": 'div.ql-editor',
        "send": 'button[aria-label="Send message"]',
        "response": 'message-content',
        "streaming_indicator": 'button[aria-label="Stop response"]',
    },
    "claude": {
        "url": "https://claude.ai",
        "input": 'div[contenteditable="true"]',
        "send": 'button[aria-label="Send Message"]',
        "response": 'div[data-is-streaming]',
        "streaming_indicator": 'button[aria-label="Stop Response"]',
    },
}


def _color(provider: str, text: str) -> str:
    c = _COLORS.get(provider, "")
    return f"{c}{text}{_RESET}" if c else text


def _resolve_file_refs(text: str, cwd: str) -> str:
    """Replace @filename references with file contents."""
    import glob as globmod

    def _replace(match):
        pattern = match.group(1)
        exact = os.path.join(cwd, pattern)
        if os.path.isfile(exact):
            candidates = [exact]
        else:
            candidates = sorted(globmod.glob(os.path.join(cwd, pattern)))
            candidates = [c for c in candidates if os.path.isfile(c)]

        if not candidates:
            return match.group(0)

        parts = []
        for path in candidates[:5]:
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


@dataclass
class BrowserSession:
    """A single browser tab connected to a chat provider."""
    provider: str
    page: object  # playwright Page
    selectors: dict
    messages: list[dict] = field(default_factory=list)
    last_response: str = ""
    call_count: int = 0
    logged_in: bool = False


def _init_judge_client():
    """Create a free-tier API client for judging/scoring.

    Tries Gemini (free), then Groq (free), then any available provider.
    Returns (client, model, provider_name) or (None, None, None).
    """
    from attestdb.core.providers import PROVIDERS, load_env_file

    # Load .env if present
    env_vars = load_env_file(os.path.join(os.getcwd(), ".env"))

    # Order: free-tier first
    judge_preference = ["gemini", "groq", "deepseek", "openai", "grok", "together"]

    for name in judge_preference:
        provider = PROVIDERS.get(name)
        if not provider:
            continue
        api_key = os.environ.get(provider["env_key"]) or env_vars.get(provider["env_key"])
        if not api_key:
            continue
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=provider["base_url"])
            return client, provider["default_model"], name
        except Exception:
            continue

    return None, None, None


class BrowserChat:
    """Interactive multi-LLM chat using browser automation.

    Opens ChatGPT, Claude, and Gemini in Chromium tabs using the user's
    saved login sessions. No API keys needed for the primary providers.
    A free-tier API (Gemini/Groq) handles judging and synthesis.
    """

    def __init__(
        self,
        db=None,
        providers: list[str] | None = None,
        cwd: str | None = None,
        profile_dir: str | None = None,
        headless: bool = False,
    ):
        self.db = db
        self.cwd = cwd or os.getcwd()
        self.profile_dir = Path(profile_dir) if profile_dir else PROFILE_DIR
        self.headless = headless
        self.requested_providers = providers or ["chatgpt", "claude", "gemini"]
        self.sessions: dict[str, BrowserSession] = {}
        self._context = None
        self._playwright = None

        # Judge client (free API for scoring/synthesis)
        self._judge_client = None
        self._judge_model = None
        self._judge_name = None

    async def _start_browser(self):
        """Launch persistent Chromium context with saved logins."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("Playwright required. Install with:")
            print("  pip install playwright && playwright install chromium")
            sys.exit(1)

        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            str(self.profile_dir),
            headless=self.headless,
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )

        # Open tabs for each provider
        for provider in self.requested_providers:
            if provider not in PROVIDER_SELECTORS:
                print(_color("system", f"Unknown provider '{provider}', skipping"))
                continue

            selectors = PROVIDER_SELECTORS[provider]
            page = await self._context.new_page()

            print(_color("system", f"Opening {provider}..."))
            try:
                await page.goto(selectors["url"], wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(2)
            except Exception as exc:
                print(_color("system", f"  Failed to load {provider}: {exc}"))
                continue

            session = BrowserSession(
                provider=provider,
                page=page,
                selectors=selectors,
            )

            # Check if we need to log in
            logged_in = await self._check_login(session)
            if not logged_in:
                print(f"\n  {_BOLD}Log into {provider} in the browser window, then press Enter here.{_RESET}")
                await asyncio.get_event_loop().run_in_executor(None, input, "  Press Enter when logged in... ")
                # Wait a moment for the page to settle after login
                await asyncio.sleep(2)

            session.logged_in = True
            self.sessions[provider] = session
            print(_color("system", f"  {provider} ready"))

        # Close the default blank tab
        pages = self._context.pages
        if len(pages) > len(self.sessions):
            for p in pages:
                if p.url in ("about:blank", "chrome://newtab/"):
                    await p.close()
                    break

    async def _check_login(self, session: BrowserSession) -> bool:
        """Heuristic check if the user is logged into this provider."""
        page = session.page
        provider = session.provider

        try:
            if provider == "chatgpt":
                # If we see the prompt input, we're logged in
                el = await page.query_selector(session.selectors["input"])
                return el is not None
            elif provider == "claude":
                el = await page.query_selector(session.selectors["input"])
                return el is not None
            elif provider == "gemini":
                el = await page.query_selector(session.selectors["input"])
                return el is not None
        except Exception:
            pass
        return False

    async def _send_message(self, session: BrowserSession, text: str) -> str | None:
        """Type and send a message, wait for response. Returns response text."""
        page = session.page
        sel = session.selectors

        try:
            # Wait for input to be ready
            input_el = await page.wait_for_selector(sel["input"], timeout=10000)

            # Clear and type (use fill for regular inputs, special handling for contenteditable)
            if session.provider == "chatgpt":
                # ChatGPT uses a div with id="prompt-textarea"
                await input_el.click()
                await page.keyboard.press("Control+a")
                await page.keyboard.press("Meta+a")
                await input_el.fill(text)
            elif session.provider == "claude":
                # Claude uses contenteditable div
                await input_el.click()
                await page.keyboard.press("Control+a")
                await page.keyboard.press("Meta+a")
                await page.keyboard.type(text, delay=5)
            elif session.provider == "gemini":
                # Gemini uses Quill editor
                await input_el.click()
                await page.keyboard.press("Control+a")
                await page.keyboard.press("Meta+a")
                await input_el.fill(text)

            await asyncio.sleep(0.5)

            # Click send (or press Enter)
            send_btn = await page.query_selector(sel["send"])
            if send_btn:
                await send_btn.click()
            else:
                await page.keyboard.press("Enter")

            # Wait for streaming to start
            await asyncio.sleep(2)

            # Wait for streaming to finish
            response = await self._wait_for_response(session)
            if response:
                session.last_response = response
                session.call_count += 1
                session.messages.append({"role": "user", "content": text})
                session.messages.append({"role": "assistant", "content": response})
            return response

        except Exception as exc:
            logger.error("Failed to send to %s: %s", session.provider, exc)
            return None

    async def _wait_for_response(self, session: BrowserSession, timeout: float = 120.0) -> str | None:
        """Poll until the assistant finishes responding."""
        page = session.page
        sel = session.selectors
        start = time.monotonic()
        interval = 2.0

        while (time.monotonic() - start) < timeout:
            # Check if still streaming
            streaming = await page.query_selector(sel.get("streaming_indicator", "nonexistent"))
            if streaming:
                await asyncio.sleep(interval)
                interval = min(interval * 1.3, 10.0)
                continue

            # Small delay to let final content render
            await asyncio.sleep(1.0)

            # Extract the last response
            response_els = await page.query_selector_all(sel["response"])
            if response_els:
                last = response_els[-1]
                text = await last.inner_text()
                if text and text.strip():
                    return text.strip()

            await asyncio.sleep(interval)
            interval = min(interval * 1.3, 10.0)

        logger.warning("Timeout waiting for %s", session.provider)
        return None

    async def _send_to_all(self, message: str):
        """Send message to all providers in parallel, print responses."""
        print()

        # Send to all providers concurrently
        tasks = {
            provider: asyncio.create_task(self._send_message(session, message))
            for provider, session in self.sessions.items()
        }

        for provider, task in tasks.items():
            response = await task
            if response:
                header = f"─── {provider} ───"
                print(_color(provider, header))
                print(response)
                print()
            else:
                print(_color(provider, f"─── {provider} ── NO RESPONSE ───"))
                print()

    async def _share_responses(self):
        """Cross-pollinate: show each provider what the others said."""
        responses = {
            name: session.last_response
            for name, session in self.sessions.items()
            if session.last_response
        }
        if len(responses) < 2:
            print(_color("system", "Need at least 2 responses to share."))
            return

        print(_color("system", "Sharing responses cross-provider..."))
        print()

        tasks = {}
        for name, session in self.sessions.items():
            others = "\n\n".join(
                f"=== {p} ===\n{r}"
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
            tasks[name] = asyncio.create_task(self._send_message(session, cross_msg))

        for provider, task in tasks.items():
            response = await task
            if response:
                header = f"─── {provider} (revised) ───"
                print(_color(provider, header))
                print(response)
                print()

    async def _run_consensus(self):
        """Use free-tier API to judge all browser responses."""
        responses = {
            name: session.last_response
            for name, session in self.sessions.items()
            if session.last_response
        }
        if len(responses) < 2:
            print(_color("system", "Need at least 2 responses for consensus."))
            return

        if not self._judge_client:
            self._judge_client, self._judge_model, self._judge_name = _init_judge_client()

        if not self._judge_client:
            print(_color("system", "No free API key found for judging. Set GOOGLE_API_KEY (free) in .env"))
            # Fallback: ask each browser provider to judge
            await self._browser_consensus(responses)
            return

        print(_color("judge", f"Judging via {self._judge_name} API (free tier)..."))
        print()

        responses_text = "\n\n".join(
            f"=== {name} ===\n{resp[:2000]}"
            for name, resp in responses.items()
        )

        judge_prompt = (
            "You are an expert judge comparing AI responses to the same question. "
            "Rate each response for accuracy, completeness, and clarity.\n\n"
            f"Responses:\n{responses_text}\n\n"
            f"Provider names: {list(responses.keys())}\n\n"
            "Respond with EXACTLY this JSON:\n"
            '{"ranking": ["best_provider", "second", ...], '
            '"scores": {"provider": 1-10, ...}, '
            '"synthesis": "Combined best answer from all responses", '
            '"agreements": "What all providers agree on", '
            '"disagreements": "Where they differ and who is more likely correct"}'
        )

        try:
            resp = self._judge_client.chat.completions.create(
                model=self._judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator comparing AI model responses."},
                    {"role": "user", "content": judge_prompt},
                ],
                max_tokens=2048,
                temperature=0.2,
                timeout=60,
            )
            text = (resp.choices[0].message.content or "").strip()

            # Parse JSON
            if "```" in text:
                for block in text.split("```"):
                    block = block.strip()
                    if block.startswith("json"):
                        block = block[4:].strip()
                    if block.startswith("{"):
                        text = block
                        break

            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                # Show raw judge response
                print(_color("judge", f"─── JUDGE ({self._judge_name}) ───"))
                print(text)
                print()
                return

            # Display results
            print(_color("judge", f"─── SCORES (judged by {self._judge_name}) ───"))
            scores = result.get("scores", {})
            ranking = result.get("ranking", [])
            for i, name in enumerate(ranking):
                score = scores.get(name, "?")
                medal = ["  ", "  ", "  "][i] if i < 3 else "   "
                print(f"  {medal} {_color(name, name)}: {score}/10")
            print()

            if result.get("agreements"):
                print(f"  {_BOLD}Agreements:{_RESET} {result['agreements']}")
            if result.get("disagreements"):
                print(f"  {_BOLD}Disagreements:{_RESET} {result['disagreements']}")
            print()

            if result.get("synthesis"):
                print(_color("consensus", f"─── CONSENSUS (synthesized by {self._judge_name}) ───"))
                print(result["synthesis"])
                print()

            # Record in AttestDB
            if self.db and ranking:
                self._record_consensus_winner(ranking[0], scores.get(ranking[0], 5) / 10)

        except Exception as exc:
            print(_color("system", f"Judging failed: {exc}"))
            print(_color("system", "Falling back to browser-based consensus..."))
            await self._browser_consensus(responses)

    async def _browser_consensus(self, responses: dict[str, str]):
        """Fallback: ask the first browser provider to judge (no API needed)."""
        first_provider = next(iter(self.sessions))
        session = self.sessions[first_provider]

        responses_text = "\n\n".join(
            f"=== {name} ===\n{resp[:2000]}"
            for name, resp in responses.items()
        )

        judge_prompt = (
            "Multiple AI assistants answered the same question. "
            "Compare their responses. Which is best and why? "
            "Synthesize the best answer combining their strengths.\n\n"
            f"Responses:\n{responses_text}"
        )

        print(_color("system", f"Asking {first_provider} to judge (browser fallback)..."))
        response = await self._send_message(session, judge_prompt)
        if response:
            print(_color("consensus", f"─── CONSENSUS (judged by {first_provider}) ───"))
            print(response)
            print()

    def _record_consensus_winner(self, winner: str, confidence: float):
        """Record which provider won consensus."""
        if not self.db:
            return
        try:
            from attestdb.core.types import ClaimInput
            self.db.ingest(ClaimInput(
                subject=(winner, "llm_provider"),
                predicate=("won_consensus", "provider_performance"),
                object=("consensus_round", "event"),
                provenance={
                    "source_id": "attest_chat_browser",
                    "source_type": "provider_telemetry",
                },
                confidence=confidence,
                payload={
                    "schema_ref": "consensus_winner/v1",
                    "data": {"confidence": confidence, "mode": "browser"},
                },
            ))
        except Exception:
            pass

    def _handle_command(self, cmd: str) -> str | None:
        """Parse a / command. Returns command name or None for regular message."""
        parts = cmd.split(maxsplit=1)
        return parts[0].lower() if parts else None

    async def run(self):
        """Main interactive loop."""
        await self._start_browser()

        if not self.sessions:
            print("No providers available. Check your browser logins.")
            return

        providers_str = ", ".join(f"{_BOLD}{name}{_RESET}" for name in self.sessions)
        print(f"\n{_BOLD}attest chat{_RESET} (browser mode, {len(self.sessions)} providers: {providers_str})")
        judge_info = f" | Judge: {self._judge_name} API" if self._judge_name else ""
        if not self._judge_name:
            self._judge_client, self._judge_model, self._judge_name = _init_judge_client()
            judge_info = f" | Judge: {self._judge_name} API (free)" if self._judge_name else " | Judge: browser fallback"
        print(f"Working directory: {self.cwd}{judge_info}")
        print(f"Use @filename to include files. Commands: /share /consensus /quit /help")
        print()

        try:
            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input(f"{_BOLD}>{_RESET} ").strip()
                    )
                except EOFError:
                    break

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd = self._handle_command(user_input)

                    if cmd in ("/quit", "/exit", "/q"):
                        break
                    elif cmd == "/share":
                        await self._share_responses()
                    elif cmd == "/consensus":
                        await self._run_consensus()
                    elif cmd == "/providers":
                        for name, session in self.sessions.items():
                            status = "logged in" if session.logged_in else "not logged in"
                            print(f"  {_color(name, name)}: {status}, {session.call_count} messages")
                    elif cmd == "/help":
                        print("Commands:")
                        print("  /share       — Share all responses cross-provider for revision")
                        print("  /consensus   — Judge all responses + synthesize best answer")
                        print("  /providers   — List active providers")
                        print("  /quit        — Exit")
                    else:
                        print(_color("system", f"Unknown command: {cmd}. Type /help"))
                    continue

                # Resolve file references
                resolved = _resolve_file_refs(user_input, self.cwd)

                # Send to all providers
                await self._send_to_all(resolved)

        except KeyboardInterrupt:
            print(f"\n{_color('system', 'Interrupted.')}")
        finally:
            await self._stop()
            print(_color("system", "Browser closed."))

    async def _stop(self):
        """Close browser and clean up."""
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()


def run_browser_chat(
    db=None,
    providers: list[str] | None = None,
    cwd: str | None = None,
    headless: bool = False,
):
    """Entry point for browser-based chat. Called from __main__.py."""
    chat = BrowserChat(db=db, providers=providers, cwd=cwd, headless=headless)
    asyncio.run(chat.run())
