"""Launch CLI commands for AttestDB.

Implements: quickstart, mcp-config, doctor, stats (new), trial, upgrade, share, telemetry.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# telemetry prompt (first run only)
# ---------------------------------------------------------------------------

def _maybe_ask_telemetry():
    """Ask the user about telemetry on first run, then never again."""
    try:
        from attestdb import config

        if config.telemetry_asked():
            return
        answer = input("Help improve AttestDB by sharing anonymous usage stats? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            config.set_key("telemetry_enabled", True)
        else:
            config.set_key("telemetry_enabled", False)
        config.set_key("telemetry_asked", True)
    except (EOFError, KeyboardInterrupt):
        # Non-interactive or cancelled — default to off, mark as asked
        try:
            from attestdb import config
            config.set_key("telemetry_enabled", False)
            config.set_key("telemetry_asked", True)
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# quickstart
# ---------------------------------------------------------------------------

def cmd_quickstart(args):
    """Create a local DB, ingest sample data, run example queries."""
    _maybe_ask_telemetry()

    from attestdb.infrastructure.attest_db import AttestDB
    from attestdb.core.types import ClaimInput

    db_path = getattr(args, "db", None) or os.path.join(
        tempfile.gettempdir(), "attestdb_quickstart.attest"
    )

    # Load bundled sample data
    sample_path = Path(__file__).parent / "data" / "sample_claims.json"
    if not sample_path.exists():
        print("Error: sample data not found. Reinstall attestdb.")
        sys.exit(1)

    raw_claims = json.loads(sample_path.read_text())

    print("AttestDB Quickstart")
    print("=" * 40)
    print()

    # 1. Create DB and ingest
    print("1. Creating local database...")
    t0 = time.monotonic()
    db = AttestDB(db_path, embedding_dim=None)

    claims_built = []
    for row in raw_claims:
        ci = ClaimInput(
            subject=(row["subject_id"], row["subject_type"]),
            predicate=(row["predicate_id"], row["predicate_type"]),
            object=(row["object_id"], row["object_type"]),
            confidence=row["confidence"],
            provenance=row["provenance"],
        )
        claims_built.append(ci)

    result = db.ingest_batch(claims_built)
    elapsed = time.monotonic() - t0
    print(f"   Ingested {result.ingested} claims from {len(set(r['provenance']['source_id'] for r in raw_claims))} sources ({elapsed:.1f}s)")
    print()

    # 2. Query: What do we know about TP53?
    print("2. Query: What do we know about TP53?")
    print("-" * 40)
    tp53_claims = db.claims_for(entity_id="tp53")
    for c in tp53_claims[:6]:
        src = getattr(c.provenance, "source_id", "") if c.provenance else ""
        source = f" [{src}]" if src else ""
        print(f"   tp53 --({c.predicate.id})--> {c.object.id}  (conf: {c.confidence:.0%}){source}")
    print()

    # 3. Query: Which drugs target which genes?
    print("3. Query: Which drugs treat diseases?")
    print("-" * 40)
    drug_claims = db.claims_for_predicate("treats")
    for c in drug_claims[:5]:
        src = getattr(c.provenance, "source_id", "") if c.provenance else ""
        source = f" [{src}]" if src else ""
        print(f"   {c.subject.id} --treats--> {c.object.id}  (conf: {c.confidence:.0%}){source}")
    print()

    # 4. Query: What genes are associated with breast cancer?
    print("4. Query: What's linked to breast cancer?")
    print("-" * 40)
    bc_claims = db.claims_for(entity_id="breast cancer")
    for c in bc_claims[:5]:
        src = getattr(c.provenance, "source_id", "") if c.provenance else ""
        source = f" [{src}]" if src else ""
        other = c.subject.id if c.object.id == "breast cancer" else c.object.id
        print(f"   {other} --({c.predicate.id})--> breast cancer  (conf: {c.confidence:.0%}){source}")
    print()

    stats = db.stats()
    n_claims = stats.get("total_claims", 0)
    n_entities = stats.get("total_entities", 0) or stats.get("entity_count", 0)
    n_sources = len(set(r["provenance"]["source_id"] for r in raw_claims))
    db.close()

    # Clean up temp DB (LMDB creates a directory, not a file)
    temp_prefix = os.path.join(tempfile.gettempdir(), "attestdb_quickstart")
    if db_path.startswith(temp_prefix):
        shutil.rmtree(db_path, ignore_errors=True)

    print("=" * 40)
    print(f"Quickstart complete! You queried {n_claims} claims from {n_sources} sources.")
    print()
    print("  Next steps:")
    print("  * Connect your own data:  attestdb ingest --help")
    print("  * Try cloud API:          attestdb trial start   (7-day Pro trial, no credit card)")
    print("  * Connect to Claude Code: attestdb mcp-config")
    print("  * Full docs:              https://attestdb.com/quickstart.html")

    try:
        from attestdb.telemetry import track
        track("quickstart_completed", {"claims": n_claims})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# mcp-config
# ---------------------------------------------------------------------------

def cmd_mcp_config(args):
    """Generate and write MCP server config for Claude Code / Cursor."""
    from attestdb.mcp_install import _find_attest_mcp

    attest_mcp = _find_attest_mcp()
    if not attest_mcp:
        print("Error: attest-mcp not found. Reinstall: pip install attestdb")
        sys.exit(1)

    config_entry = {
        "command": attest_mcp,
        "args": [],
    }

    if args.print_only:
        print(json.dumps({"mcpServers": {"brain": config_entry}}, indent=2))
        return

    # Determine target config file
    if args.cursor:
        config_path = Path.cwd() / ".cursor" / "mcp.json"
    else:
        # Claude Code: try project first, fall back to user
        project_path = Path.cwd() / ".mcp.json"
        user_path = Path.home() / ".claude" / "mcp.json"
        config_path = project_path if project_path.exists() else user_path

    # Merge with existing config
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    servers = existing.setdefault("mcpServers", {})
    servers["brain"] = config_entry

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2) + "\n")

    tool_name = "Cursor" if args.cursor else "Claude Code"
    print(f"MCP config written to {config_path}")
    print()
    print(f"  Restart {tool_name} to activate AttestDB memory.")
    print("  Your agent now has tools for querying, ingesting, and verifying claims.")

    try:
        from attestdb.telemetry import track
        track("mcp_configured")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args):
    """Run health checks and diagnose common issues."""
    ok_count = 0
    fail_count = 0

    def check(label, passed, detail="", fix=""):
        nonlocal ok_count, fail_count
        if passed:
            ok_count += 1
            status = "ok"
            print(f"  {status}  {label}: {detail}")
        else:
            fail_count += 1
            status = "FAIL"
            print(f"  {status}  {label}: {detail}")
            if fix:
                print(f"        Fix: {fix}")

    print("AttestDB Doctor")
    print("=" * 40)

    # Python version
    v = sys.version_info
    check(
        "Python version",
        v >= (3, 11),
        f"{v.major}.{v.minor}.{v.micro}",
        "AttestDB requires Python 3.11+. Install from python.org.",
    )

    # Rust engine
    try:
        import attest_py
        rust_version = getattr(attest_py, "__version__", "loaded")
        check("Rust engine", True, f"loaded (v{rust_version})")
    except ImportError:
        check(
            "Rust engine", False, "not installed",
            "pip install attest-py",
        )

    # Local database
    from attestdb import config
    db_path = Path(config.default_db_path())
    if db_path.exists():
        # LMDB creates a directory — sum file sizes inside
        if db_path.is_dir():
            size_mb = sum(f.stat().st_size for f in db_path.iterdir() if f.is_file()) / (1024 * 1024)
        else:
            size_mb = db_path.stat().st_size / (1024 * 1024)
        try:
            from attestdb.infrastructure.attest_db import AttestDB
            db = AttestDB(str(db_path), embedding_dim=None)
            stats = db.stats()
            n = stats.get("total_claims", 0)
            db.close()
            check("Local database", True, f"healthy ({n:,} claims, {size_mb:.1f} MB at {db_path})")
        except Exception as e:
            check("Local database", False, f"error: {e}", "Try removing and recreating the database.")
    else:
        check("Local database", True, "not yet created (run `attest quickstart` to try)")

    # MCP config (check Claude Code and Cursor paths)
    mcp_paths = [
        Path.cwd() / ".mcp.json",
        Path.home() / ".claude" / "mcp.json",
        Path.cwd() / ".cursor" / "mcp.json",
    ]
    mcp_found = False
    for p in mcp_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if "brain" in data.get("mcpServers", {}):
                    mcp_found = True
                    check("MCP config", True, f"found at {p}")
                    break
            except Exception:
                pass
    if not mcp_found:
        check("MCP config", False, "not configured", "Run `attest mcp-config` to set up.")

    # Cloud API
    api_key = config.get_api_key()
    if api_key:
        check("Cloud API", True, "configured")
    else:
        check("Cloud API", False, "not configured", "Run `attest trial start` to try cloud.")

    # Disk space
    usage = shutil.disk_usage(Path.home())
    free_gb = usage.free / (1024 ** 3)
    check(
        "Disk space",
        free_gb > 1.0,
        f"{free_gb:.1f} GB available",
        "Free up disk space. AttestDB needs at least 1 GB.",
    )

    print()
    if fail_count == 0:
        print(f"All {ok_count} checks passed.")
    else:
        print(f"{ok_count} passed, {fail_count} failed.")

    try:
        from attestdb.telemetry import track
        track("doctor_run")
    except Exception:
        pass

    if fail_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# stats (new version — screenshot-friendly)
# ---------------------------------------------------------------------------

def cmd_stats_new(args):
    """Show usage summary — designed to be screenshot-friendly."""
    from attestdb.infrastructure.attest_db import AttestDB
    from attestdb import config

    db_path = getattr(args, "db", None) or config.default_db_path()
    if not Path(db_path).exists():
        print("No local database found. Run `attest quickstart` to get started.")
        return

    db = AttestDB(db_path, embedding_dim=None)
    stats = db.stats()
    n_claims = stats.get("total_claims", 0)
    n_entities = stats.get("total_entities", 0) or stats.get("entity_count", 0)

    # Count source types from a sample of claims
    sources = set()
    try:
        sample = db.claims_for_predicate("causes") or db.claims_for_predicate("associated_with") or []
        for c in sample[:200]:
            if c.provenance:
                st = getattr(c.provenance, "source_type", "")
                if st:
                    sources.add(st)
    except Exception:
        pass

    # Count predicates as proxy for "verified" (claims with provenance)
    verified = n_claims  # all claims have provenance (required by validation)

    db.close()

    # Plan info
    trial_info = config.get_trial_info()
    api_key = config.get_api_key()
    if trial_info:
        import datetime
        expires = trial_info.get("expires_at", "")
        if expires:
            try:
                exp_dt = datetime.datetime.fromisoformat(expires)
                remaining = (exp_dt - datetime.datetime.now(datetime.timezone.utc)).days
                plan_line = f"Pro trial ({remaining} days remaining)"
            except Exception:
                plan_line = "Pro trial"
        else:
            plan_line = "Pro trial"
    elif api_key:
        plan_line = config.get("plan", "Pro")
    else:
        plan_line = "Free (local only)"

    print("AttestDB Stats")
    print("-" * 35)
    print(f"  Claims:      {n_claims:,} from {len(sources) or '?'} source types")
    print(f"  Entities:    {n_entities:,}")
    if sources:
        print(f"  Sources:     {', '.join(sorted(sources))}")
    print(f"  Verified:    {verified:,} claims (100% — provenance required)")
    print(f"  Plan:        {plan_line}")
    print("-" * 35)


# ---------------------------------------------------------------------------
# trial
# ---------------------------------------------------------------------------

def cmd_trial_start(args):
    """Start a 7-day Pro trial (no credit card required)."""
    from attestdb import config

    # Check if already on trial
    existing = config.get_trial_info()
    if existing:
        print("You already have an active trial.")
        print("Run `attest trial status` to check your trial.")
        return

    email = args.email
    if not email:
        try:
            email = input("Email address (for trial): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

    if not email or "@" not in email:
        print("Error: valid email required.")
        sys.exit(1)

    endpoint = config.get_api_endpoint()
    print(f"Starting 7-day Pro trial for {email}...")

    try:
        import requests
        resp = requests.post(
            f"{endpoint}/api/v1/trial/start",
            json={"email": email},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # Store trial info locally
        config.set_key("api_key", data["api_key"])
        config.set_trial_info({
            "email": email,
            "tenant_id": data.get("tenant_id", ""),
            "started_at": data.get("started_at", ""),
            "expires_at": data.get("expires_at", ""),
        })

        print()
        print(f"Pro trial activated! Expires: {data.get('expires_at', '7 days')}")
        print()
        print("  Your cloud API is ready. Try:")
        print("  * attest stats              — see your usage")
        print("  * attest mcp-config         — connect to Claude Code")
        print("  * attest trial status       — check trial status")

        try:
            from attestdb.telemetry import track
            track("trial_started")
        except Exception:
            pass

    except ImportError:
        print("Error: `requests` package required. pip install requests")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting trial: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)


def cmd_trial_status(args):
    """Show trial status."""
    from attestdb import config
    import datetime

    trial = config.get_trial_info()
    if not trial:
        print("No active trial. Run `attest trial start` to begin.")
        return

    print("AttestDB Trial Status")
    print("-" * 35)
    print(f"  Email:       {trial.get('email', '?')}")
    print(f"  Tenant:      {trial.get('tenant_id', '?')}")
    print(f"  Started:     {trial.get('started_at', '?')}")
    expires = trial.get("expires_at", "")
    if expires:
        try:
            exp_dt = datetime.datetime.fromisoformat(expires)
            remaining = (exp_dt - datetime.datetime.now(datetime.timezone.utc)).days
            print(f"  Expires:     {expires} ({remaining} days remaining)")
        except Exception:
            print(f"  Expires:     {expires}")
    print("-" * 35)


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------

def cmd_upgrade(args):
    """Open Stripe checkout to upgrade to a paid plan."""
    import webbrowser
    from attestdb import config

    tier = getattr(args, "tier", "pro")
    interval = getattr(args, "interval", "month")
    endpoint = config.get_api_endpoint()
    email = ""
    trial = config.get_trial_info()
    if trial:
        email = trial.get("email", "")

    url = f"{endpoint}/signup?plan={tier}&interval={interval}"
    if email:
        url += f"&email={email}"

    prices = {"pro": 99, "growth": 249, "team": 499}
    print(f"Opening Stripe checkout for {tier.title()} plan (${prices[tier]}/mo)...")
    webbrowser.open(url)
    print()
    print("  Complete the checkout in your browser.")
    print("  After payment, your cloud API will be upgraded automatically.")
    print("  Run `attest stats` to verify.")


# ---------------------------------------------------------------------------
# share
# ---------------------------------------------------------------------------

def cmd_share(args):
    """Generate a shareable link to a visual claim graph."""
    import webbrowser
    from attestdb.infrastructure.attest_db import AttestDB
    from attestdb import config

    db_path = getattr(args, "db", None) or config.default_db_path()
    if not Path(db_path).exists():
        print("No local database found. Run `attest quickstart` first.")
        return

    db = AttestDB(db_path, embedding_dim=None)
    stats = db.stats()
    n_claims = stats.get("total_claims", 0)

    if args.private:
        print(f"Your database: {n_claims:,} claims")
        print("Opening AttestDB demo viewer...")
        print("(Note: the demo shows the public reference database, not your local data)")
        webbrowser.open("https://attestdb.com/demo/")
        db.close()
        return

    # Upload to cloud for sharing
    api_key = config.get_api_key()
    if not api_key:
        print("Cloud access required for sharing. Run `attest trial start` first.")
        print("Or use `attest share --private` to view locally.")
        db.close()
        return

    endpoint = config.get_api_endpoint()
    try:
        import requests
        resp = requests.post(
            f"{endpoint}/api/v1/share",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"claim_count": n_claims},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        share_url = data.get("url", f"{endpoint}/share/{data.get('id', 'unknown')}")

        print(f"Your claim graph: {share_url}")
        print()
        print(f"  {n_claims:,} claims")
        print("  Share this link -- anyone can explore your knowledge graph (read-only).")
    except ImportError:
        print("Error: `requests` package required. pip install requests")
    except Exception as e:
        print(f"Error creating share link: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------

def cmd_telemetry(args):
    """Manage telemetry preferences."""
    from attestdb import config

    action = getattr(args, "telemetry_action", None)

    if action == "on":
        config.set_key("telemetry_enabled", True)
        config.set_key("telemetry_asked", True)
        print("Telemetry enabled. Only anonymous usage stats are collected.")
        print("Run `attest telemetry off` to disable at any time.")
    elif action == "off":
        config.set_key("telemetry_enabled", False)
        config.set_key("telemetry_asked", True)
        print("Telemetry disabled. No usage data will be collected.")
    elif action == "status":
        enabled = config.is_telemetry_enabled()
        asked = config.telemetry_asked()
        if not asked:
            print("Telemetry: not configured yet")
        elif enabled:
            print("Telemetry: enabled (anonymous usage stats only)")
        else:
            print("Telemetry: disabled")
    else:
        print("Usage: attest telemetry {on|off|status}")
