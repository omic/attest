"""Sync Slack Web API connector for the Attest library.

Direct :class:`Connector` subclass with custom ``run()`` that produces
both **structural claims** (person posted_in channel, person mentioned
person — with external_ids for cross-connector resolution) and
**text-extracted claims** via ``db.ingest_text()`` on cleaned message
batches and file attachment summaries.

Usage::

    from attestdb.connectors.slack import SlackConnector

    conn = SlackConnector(token=os.environ["SLACK_BOT_TOKEN"], channels=["general"])
    result = conn.run(db)
"""

from __future__ import annotations

import io
import logging
import re
import time
from typing import TYPE_CHECKING, Iterator

from attestdb.connectors.base import Connector, ConnectorResult

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    import openpyxl
except ImportError:
    openpyxl = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SLACK_API = "https://slack.com/api"
RATE_LIMIT_PAUSE = 1.2
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB safety valve
_TABULAR_TYPES = frozenset({"csv", "tsv", "xlsx", "xls"})

# Subtypes that are system/meta noise — skip these.
# Content subtypes (file_share, me_message, thread_broadcast) are kept.
_SKIP_SUBTYPES = frozenset({
    "channel_join", "channel_leave", "channel_topic",
    "channel_purpose", "channel_name", "channel_archive",
    "channel_unarchive", "group_join", "group_leave",
    "group_topic", "group_purpose", "group_name",
    "group_archive", "group_unarchive",
    "pinned_item", "unpinned_item",
})

# Slack markup patterns
_SLACK_USER_RE = re.compile(r"<@([A-Z0-9]+)(?:\|([^>]+))?>")
_SLACK_CHANNEL_RE = re.compile(r"<#([A-Z0-9]+)\|([^>]+)>")
_SLACK_URL_RE = re.compile(r"<(https?://[^|>]+)(?:\|([^>]+))?>")
_SLACK_EMOJI_RE = re.compile(r":([a-z0-9_+-]+):")
_TRIVIAL_MSG_RE = re.compile(
    r"^(?:ok|okay|yes|no|yep|nope|sure|thanks|ty|thx|lgtm|done|\+1|"
    r"\ud83d[\ude00-\udeff]|:\w+:|got it|nice|cool|great|sounds good|"
    r"will do|on it|ack|k|y|n)\.?!?\s*$",
    re.IGNORECASE,
)


class SlackConnector(Connector):
    """Sync Slack connector that ingests channel history via ``db.ingest_chat()``."""

    name = "slack"

    def __init__(
        self,
        token: str,
        channels: list[str] | None = None,
        extraction: str = "heuristic",
        oldest: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError(
                "pip install requests for the Slack connector"
            )
        self._token = token
        self._channels = set(channels) if channels else None
        self._extraction = extraction
        # Explicit oldest takes priority; otherwise use checkpoint
        if oldest is not None:
            self._oldest = oldest
        else:
            self._oldest = self._checkpoint.get("oldest", 0)
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # Search (for autodidact evidence)
    # ------------------------------------------------------------------

    @property
    def supports_search(self) -> bool:
        return True

    def search(self, query: str) -> str:
        """Search Slack messages matching *query* via search.messages API."""
        try:
            resp = self._request_with_retry(
                "GET",
                "https://slack.com/api/search.messages",
                params={"query": query, "count": 10, "sort": "score"},
            )
            data = resp.json()
            if not data.get("ok"):
                return ""
            matches = data.get("messages", {}).get("matches", [])
            parts = []
            for m in matches[:10]:
                text = m.get("text", "")
                channel = m.get("channel", {}).get("name", "")
                if text:
                    cleaned, _ = self._clean_message(text)
                    parts.append(f"[#{channel}] {cleaned}")
            return "\n\n".join(parts)[:4000]
        except Exception as exc:
            logger.warning("Slack search failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Message preprocessing
    # ------------------------------------------------------------------

    def _clean_message(
        self,
        text: str,
        users: dict[str, dict[str, str]] | None = None,
    ) -> tuple[str, list[tuple[str, str]]]:
        """Strip Slack markup and resolve user mentions.

        Returns ``(cleaned_text, mentions)`` where *mentions* is a list
        of ``(uid, display_name)`` pairs for each resolved ``<@U...>``
        reference.
        """
        resolved_mentions: list[tuple[str, str]] = []

        def _resolve_user(m):
            uid = m.group(1)
            label = m.group(2)
            if label:
                resolved_mentions.append((uid, label))
                return label
            if users and uid in users:
                name = users[uid]["name"]
                resolved_mentions.append((uid, name))
                return name
            return ""  # drop unresolvable mentions
        text = _SLACK_USER_RE.sub(_resolve_user, text)
        # Resolve <#C123|channel-name> to #channel-name
        text = _SLACK_CHANNEL_RE.sub(lambda m: f"#{m.group(2)}", text)
        # Resolve <url|label> to label, or just url
        text = _SLACK_URL_RE.sub(lambda m: m.group(2) or m.group(1), text)
        # Strip emoji shortcodes
        text = _SLACK_EMOJI_RE.sub("", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text, resolved_mentions

    @staticmethod
    def _is_trivial(text: str) -> bool:
        """Return True if the message is too short or trivial for extraction."""
        if len(text) < 8:
            return True
        return bool(_TRIVIAL_MSG_RE.match(text))

    def _fetch_users(self) -> dict[str, dict[str, str]]:
        """Build a uid → ``{"name": ..., "email": ...}`` map.

        Email may be absent if the token lacks ``users:read.email``
        scope or the user has no email set.
        """
        users: dict[str, dict[str, str]] = {}
        try:
            cursor: str | None = None
            for _ in range(self._MAX_PAGES):
                params: dict = {"limit": 200}
                if cursor:
                    params["cursor"] = cursor
                data = self._get("users.list", params)
                for u in data.get("members", []):
                    profile = u.get("profile", {})
                    name = (
                        profile.get("display_name")
                        or profile.get("real_name")
                        or u.get("name", "")
                    )
                    if name:
                        info: dict[str, str] = {"name": name}
                        email = profile.get("email", "")
                        if email:
                            info["email"] = email
                        users[u["id"]] = info
                cursor = (
                    data.get("response_metadata", {})
                    .get("next_cursor")
                )
                if not cursor:
                    break
                time.sleep(RATE_LIMIT_PAUSE)
        except Exception as exc:
            logger.warning("Failed to fetch user list: %s", exc)
        return users

    # ------------------------------------------------------------------
    # fetch() is not used — chat connectors go through run() directly.
    # ------------------------------------------------------------------

    def fetch(self) -> Iterator[dict]:
        raise NotImplementedError("Use run() for chat connectors")

    # ------------------------------------------------------------------
    # Slack API helpers (private, paginated)
    # ------------------------------------------------------------------

    def _get(self, method: str, params: dict | None = None) -> dict:
        """Call a Slack Web API method (GET) and return the JSON response."""
        resp = self._request_with_retry(
            "GET",
            f"{SLACK_API}/{method}",
            params=params or {},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise ValueError(f"Slack API error ({method}): {data.get('error')}")
        return data

    def _post(self, method: str, json_body: dict | None = None) -> dict:
        """Call a Slack Web API method (POST) and return the JSON response."""
        resp = self._request_with_retry(
            "POST",
            f"{SLACK_API}/{method}",
            json=json_body or {},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise ValueError(f"Slack API error ({method}): {data.get('error')}")
        return data

    def _join_channel(self, channel_id: str) -> None:
        """Join a public channel so the bot can read its history."""
        self._post("conversations.join", {"channel": channel_id})

    def _list_channels(self) -> list[dict]:
        """Return list of channel dicts the bot can see (paginated).

        Tries public + private channels first; falls back to public-only
        if the token lacks ``groups:read``.
        """
        for types in ("public_channel,private_channel", "public_channel"):
            try:
                return self._list_channels_with_types(types)
            except ValueError as exc:
                if "missing_scope" in str(exc) and types != "public_channel":
                    logger.info("No private-channel scope; falling back to public only")
                    continue
                raise
        return []  # unreachable, keeps mypy happy

    def _list_channels_with_types(self, types: str) -> list[dict]:
        """Paginated channel listing for the given *types* string."""
        channels: list[dict] = []
        cursor: str | None = None
        for _ in range(self._MAX_PAGES):
            params: dict = {"types": types, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._get("conversations.list", params)
            for ch in data.get("channels", []):
                channels.append({
                    "id": ch["id"],
                    "name": ch["name"],
                    "is_archived": ch.get("is_archived", False),
                })
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(RATE_LIMIT_PAUSE)
        return channels

    def _fetch_history(
        self,
        channel_id: str,
        limit: int = 1000,
    ) -> tuple[list[dict], float]:
        """Return (messages, latest_ts) for *channel_id* in chronological order.

        Fetches top-level messages and thread replies. ``latest_ts`` is the
        maximum message timestamp seen, useful for checkpoint-based incremental
        fetching.
        """
        messages: list[dict] = []
        thread_parents: list[str] = []  # ts values of messages with replies
        latest_ts: float = 0.0
        cursor: str | None = None
        count = 0
        while count < limit:
            params: dict = {
                "channel": channel_id,
                "limit": min(200, limit - count),
            }
            if self._oldest:
                params["oldest"] = str(self._oldest)
            if cursor:
                params["cursor"] = cursor
            data = self._get("conversations.history", params)
            for msg in data.get("messages", []):
                if msg.get("subtype") in _SKIP_SUBTYPES:
                    continue  # skip join/leave/topic-change etc.
                if msg.get("bot_id"):
                    continue  # skip bot messages (GitLab, Jira, etc.)
                ts = float(msg.get("ts", 0))
                if ts > latest_ts:
                    latest_ts = ts
                messages.append({
                    "role": "user",
                    "content": msg.get("text", ""),
                    "user": msg.get("user", ""),
                    "ts": msg.get("ts", ""),
                    "files": msg.get("files", []),
                })
                count += 1
                # Track threads that have replies
                if msg.get("reply_count", 0) > 0:
                    thread_parents.append(msg["ts"])
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(RATE_LIMIT_PAUSE)

        # Fetch thread replies for messages that have them
        for thread_ts in thread_parents:
            try:
                thread_msgs = self._fetch_thread_replies(channel_id, thread_ts)
                messages.extend(thread_msgs)
            except Exception as exc:
                logger.debug("Failed to fetch thread %s: %s", thread_ts, exc)

        messages.reverse()  # Slack returns newest-first; we want chronological
        return messages, latest_ts

    def _fetch_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
        limit: int = 200,
    ) -> list[dict]:
        """Fetch replies to a thread (excludes parent message)."""
        replies: list[dict] = []
        cursor: str | None = None
        for _ in range(self._MAX_PAGES):
            params: dict = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": min(200, limit),
            }
            if cursor:
                params["cursor"] = cursor
            data = self._get("conversations.replies", params)
            for msg in data.get("messages", []):
                # Skip the parent message (same ts as thread_ts)
                if msg.get("ts") == thread_ts:
                    continue
                if msg.get("subtype") in _SKIP_SUBTYPES or msg.get("bot_id"):
                    continue
                replies.append({
                    "role": "user",
                    "content": msg.get("text", ""),
                    "user": msg.get("user", ""),
                    "ts": msg.get("ts", ""),
                    "files": msg.get("files", []),
                })
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(RATE_LIMIT_PAUSE)
        return replies

    # ------------------------------------------------------------------
    # File attachment helpers
    # ------------------------------------------------------------------

    def _download_file(self, file_info: dict) -> bytes | None:
        """Download a Slack-hosted file. Returns None on error.

        Uses a 50 MB safety ceiling to avoid unbounded downloads, but the
        real data limit is the per-format row cap applied during parsing.

        Detects auth failures (Slack returns HTML login page when the bot
        token lacks ``files:read`` scope) and logs a clear diagnostic.
        """
        size = file_info.get("size", 0)
        if size > _MAX_DOWNLOAD_BYTES:
            logger.info(
                "Skipping file %s (%.1f MB exceeds download ceiling)",
                file_info.get("name", "?"), size / 1024 / 1024,
            )
            return None
        url = file_info.get("url_private_download") or file_info.get("url_private")
        if not url:
            return None
        try:
            resp = self._request_with_retry("GET", url, timeout=60)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type:
                logger.error(
                    "File download for %s returned HTML instead of file data. "
                    "The Slack bot token likely needs the 'files:read' OAuth scope.",
                    file_info.get("name", "?"),
                )
                return None
            return resp.content
        except Exception as exc:
            logger.warning("Failed to download %s: %s", file_info.get("name", "?"), exc)
            return None

    def _parse_file_to_text(self, file_info: dict, content: bytes) -> str | None:
        """Analyze file with pandas and produce a compact summary for extraction.

        Falls back to stdlib csv parsing if pandas is unavailable.
        """
        filetype = file_info.get("filetype", "").lower()
        name = file_info.get("name", "file")

        # Load into DataFrame
        df = self._load_dataframe(content, filetype, name)
        if df is None or df.empty:
            return None

        return self._summarize_dataframe(df, name)

    def _load_dataframe(self, content: bytes, filetype: str, name: str):
        """Load file bytes into a pandas DataFrame."""
        if pd is None:
            # Fallback: minimal stdlib parsing
            return self._load_dataframe_stdlib(content, filetype, name)
        try:
            buf = io.BytesIO(content)
            if filetype in ("csv", "tsv"):
                sep = "\t" if filetype == "tsv" else ","
                return pd.read_csv(buf, sep=sep, encoding="utf-8-sig")
            if filetype in ("xlsx", "xls"):
                if openpyxl is None:
                    logger.warning("openpyxl not installed — skipping Excel file %s", name)
                    return None
                return pd.read_excel(buf, engine="openpyxl")
        except Exception as exc:
            logger.warning("Failed to load %s: %s", name, exc)
        return None

    def _load_dataframe_stdlib(self, content: bytes, filetype: str, name: str):
        """Minimal fallback when pandas is unavailable — returns None (skip file)."""
        logger.warning("pandas not installed — skipping tabular file %s", name)
        return None

    def _summarize_dataframe(self, df, name: str) -> str:
        """Produce a data-rich summary of a DataFrame for LLM claim extraction.

        Prioritizes actual row-level data over metadata so the LLM extracts
        claims about entities (drugs, genes, organisms) rather than about
        the file structure (column names, data types).

        Strategy:
        - Brief header (shape, columns)
        - Auto-detect score/rank columns → show top-ranked rows
        - Show up to 30 representative rows with key columns
        - Keep it compact by selecting the most informative columns
        """
        nrows, ncols = df.shape
        lines: list[str] = [f"Dataset: {name} ({nrows} rows, {ncols} columns)"]

        # Identify key columns: prefer name/id/label columns + score/rank columns
        # Drop long blob columns (sequences, SMILES > 50 chars, etc.)
        key_cols = self._select_key_columns(df)
        if not key_cols:
            key_cols = df.columns.tolist()[:8]

        lines.append(f"Key columns: {', '.join(key_cols)}")

        # Auto-detect best sort column (score, rank, confidence, priority)
        sort_col = self._detect_sort_column(df, key_cols)

        # Build the data view: top-ranked rows if we found a sort column,
        # otherwise a representative sample
        max_rows = min(30, nrows)
        if sort_col:
            ascending = "rank" in sort_col.lower()  # rank=1 is best; score=1.0 is best
            view = df.sort_values(sort_col, ascending=ascending).head(max_rows)
            lines.append(f"Top {max_rows} rows by {sort_col}:")
        else:
            view = df.head(max_rows)
            lines.append(f"First {max_rows} rows:")

        for _, row in view.iterrows():
            parts = []
            for col in key_cols:
                v = row.get(col)
                if pd.notna(v):
                    s = str(v).strip()
                    if s and len(s) < 120:  # skip very long values
                        parts.append(f"{col}: {s}")
            if parts:
                lines.append("  " + "; ".join(parts))

        return "\n".join(lines)

    def _select_key_columns(self, df) -> list[str]:
        """Pick the most informative columns for the summary.

        Prefers: name/id/label columns, categorical columns with moderate
        cardinality, and numeric score columns. Drops: long text blobs,
        binary flags, and columns with all-unique values (like hashes).
        """
        cols = df.columns.tolist()
        scored: list[tuple[float, str]] = []

        for col in cols:
            s = df[col]
            nuniq = s.nunique()
            score = 0.0

            # Name/ID/label columns are high-value
            name_lower = col.lower()
            if any(kw in name_lower for kw in ("name", "id", "label", "title", "type", "class")):
                score += 3.0
            if any(kw in name_lower for kw in ("organism", "gene", "drug", "target", "species")):
                score += 4.0
            if any(kw in name_lower for kw in ("score", "rank", "confidence", "priority")):
                score += 3.0
            if any(kw in name_lower for kw in ("outcome", "result", "status", "category")):
                score += 2.0

            # Moderate cardinality is informative
            if pd.api.types.is_numeric_dtype(s):
                score += 1.0
            elif 2 <= nuniq <= 100:
                score += 2.0

            # Penalize columns where most values are very long (blobs)
            if s.dtype == object:
                mean_len = s.dropna().astype(str).str.len().mean()
                if mean_len > 80:
                    score -= 3.0

            # Penalize all-unique text columns (likely hashes/IDs with no semantic value)
            if s.dtype == object and nuniq == len(df) and nuniq > 20:
                score -= 2.0

            scored.append((score, col))

        scored.sort(key=lambda x: -x[0])
        return [col for _, col in scored[:10]]

    def _detect_sort_column(self, df, key_cols: list[str]) -> str | None:
        """Find the best column to sort by for showing top entries."""
        candidates = []
        for col in key_cols:
            name_lower = col.lower()
            if pd.api.types.is_numeric_dtype(df[col]):
                if any(kw in name_lower for kw in ("score", "rank", "confidence", "priority")):
                    candidates.append(col)
        # Prefer "score" over "rank"
        for c in candidates:
            if "score" in c.lower():
                return c
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # run() — main entry point
    # ------------------------------------------------------------------

    def run(self, db: AttestDB, *, batch_size: int = 500) -> ConnectorResult:
        """Fetch Slack history and ingest via ``db.ingest_chat()``.

        Iterates over visible channels, fetches message history for each,
        cleans Slack markup, filters trivial messages, and feeds them
        through the chat extraction pipeline with the configured extraction
        mode (heuristic/llm/smart).

        Tracks the latest message timestamp and saves it as a checkpoint
        so that subsequent runs only fetch new messages.
        """
        start = time.monotonic()
        result = ConnectorResult(connector_name=self.name)
        max_ts: float = 0.0

        try:
            channels = self._list_channels()
        except Exception as exc:
            result.errors.append(f"Failed to list channels: {exc}")
            self._finalize_result(result, start)
            return result

        # Fetch user map once for resolving <@U...> mentions
        users = self._fetch_users()

        # Pre-filter to active, wanted channels
        active_channels = [
            ch for ch in channels
            if not ch["is_archived"]
            and (not self._channels or ch["name"] in self._channels)
        ]
        total_channels = len(active_channels)
        logger.info("Found %d active channel(s) to process", total_channels)

        for ch_idx, ch in enumerate(active_channels, 1):
            logger.info("[%d/%d] Fetching #%s (%s)", ch_idx, total_channels, ch["name"], ch["id"])
            try:
                messages, ch_latest_ts = self._fetch_history(ch["id"])
            except ValueError as exc:
                if "not_in_channel" in str(exc):
                    try:
                        logger.info("Auto-joining #%s", ch["name"])
                        self._join_channel(ch["id"])
                        time.sleep(RATE_LIMIT_PAUSE)
                        messages, ch_latest_ts = self._fetch_history(ch["id"])
                    except Exception as join_exc:
                        result.errors.append(f"#{ch['name']}: {join_exc}")
                        continue
                else:
                    result.errors.append(f"#{ch['name']}: {exc}")
                    continue
            except Exception as exc:
                result.errors.append(f"#{ch['name']}: {exc}")
                continue

            if ch_latest_ts > max_ts:
                max_ts = ch_latest_ts

            if not messages:
                logger.info("[%d/%d] #%s: 0 messages", ch_idx, total_channels, ch["name"])
                continue

            # Clean and filter messages — preserve user identity.
            # Annotate message text with attached file names so the LLM
            # has context about what data was shared alongside the message.
            # Generate structural claims (posted_in, mentioned).
            cleaned: list[dict] = []
            structural_claims: list[dict] = []
            for msg in messages:
                text, mentions = self._clean_message(
                    msg["content"], users,
                )
                for f in msg.get("files", []):
                    fname = f.get("name", "")
                    if fname:
                        text = (
                            (text + "\n" if text else "")
                            + f"[Attached: {fname}]"
                        )
                if text and not self._is_trivial(text):
                    uid = msg.get("user", "")
                    user_info = users.get(uid, {})
                    user_name = user_info.get("name", "") if user_info else ""
                    role = "user"
                    content = (
                        f"{user_name}: {text}" if user_name else text
                    )
                    cleaned.append({"role": role, "content": content})

                    # --- Structural claims ---
                    msg_ts = msg.get("ts", "")
                    if user_name and uid:
                        # posted_in: person → channel
                        subj_ext: dict[str, str] = {"slack_id": uid}
                        email = user_info.get("email", "")
                        if email:
                            subj_ext["email"] = email
                        structural_claims.append(
                            self._make_claim(
                                user_name, "person",
                                "posted_in",
                                f"#{ch['name']}", "channel",
                                f"slack:{ch['name']}:{msg_ts}",
                                subj_ext=subj_ext,
                                obj_ext={"slack_id": ch["id"]},
                            )
                        )

                    # mentioned: person → person
                    for m_uid, m_name in mentions:
                        if m_uid != uid:  # skip self-mentions
                            m_info = users.get(m_uid, {})
                            m_ext: dict[str, str] = {
                                "slack_id": m_uid,
                            }
                            m_email = (
                                m_info.get("email", "")
                                if m_info else ""
                            )
                            if m_email:
                                m_ext["email"] = m_email
                            structural_claims.append(
                                self._make_claim(
                                    user_name, "person",
                                    "mentioned",
                                    m_name, "person",
                                    f"slack:{ch['name']}:{msg_ts}",
                                    subj_ext=subj_ext,
                                    obj_ext=m_ext,
                                )
                            )

            # Flush structural claims (posted_in, mentioned)
            if structural_claims:
                self._flush(db, structural_claims, result)

            if not cleaned:
                logger.info(
                    "[%d/%d] #%s: %d messages -> 0 after cleanup",
                    ch_idx, total_channels, ch["name"], len(messages),
                )
                continue

            # Batch messages for text extraction (group into chunks)
            batches: list[list[dict]] = []
            current: list[dict] = []
            current_len = 0
            for msg in cleaned:
                msg_len = len(msg["content"])
                if current and current_len + msg_len > batch_size * 8:
                    batches.append(current)
                    current = []
                    current_len = 0
                current.append(msg)
                current_len += msg_len
            if current:
                batches.append(current)

            ch_claims = 0
            for batch_idx, batch_msgs in enumerate(batches):
                text = "\n".join(m["content"] for m in batch_msgs)
                try:
                    ext_result = db.ingest_text(
                        text,
                        source_id=f"slack:{ch['name']}",
                    )
                    ch_claims += ext_result.n_valid
                    result.claims_ingested += ext_result.n_valid
                    result.claims_skipped += ext_result.n_rejected
                    result.prompt_tokens += getattr(ext_result, "prompt_tokens", 0)
                    result.completion_tokens += getattr(ext_result, "completion_tokens", 0)
                except Exception as exc:
                    result.errors.append(f"#{ch['name']} batch {batch_idx}: {exc}")

            # ----- File attachments: analyze with pandas, extract claims -----
            # Instead of dumping raw rows to the LLM, we use pandas to produce
            # a compact analytical summary (stats, distributions, samples) and
            # send that as a single LLM call per file.
            file_claims = 0
            for msg in messages:
                for file_info in msg.get("files", []):
                    filetype = file_info.get("filetype", "").lower()
                    if filetype not in _TABULAR_TYPES:
                        continue
                    raw = self._download_file(file_info)
                    if raw is None:
                        continue
                    parsed = self._parse_file_to_text(file_info, raw)
                    if not parsed:
                        continue
                    fname = file_info.get("name", "file")
                    source_id = f"slack:{ch['name']}:file:{fname}"
                    logger.info("Analyzing file %s for claims", fname)
                    try:
                        ext_result = db.ingest_text(parsed, source_id=source_id)
                        file_claims += ext_result.n_valid
                        result.claims_ingested += ext_result.n_valid
                        result.claims_skipped += ext_result.n_rejected
                        result.prompt_tokens += getattr(ext_result, "prompt_tokens", 0)
                        result.completion_tokens += getattr(ext_result, "completion_tokens", 0)
                    except Exception as exc:
                        result.errors.append(f"#{ch['name']} file {fname}: {exc}")

            logger.info(
                "[%d/%d] #%s: %d messages -> %d after cleanup -> %d claims (%d from files)",
                ch_idx, total_channels, ch["name"],
                len(messages), len(cleaned), ch_claims + file_claims, file_claims,
            )
            time.sleep(RATE_LIMIT_PAUSE)

        # Save checkpoint so next run fetches only newer messages
        if max_ts > 0:
            self._checkpoint["oldest"] = max_ts

        self._finalize_result(result, start)
        self._save_checkpoint()
        return result
