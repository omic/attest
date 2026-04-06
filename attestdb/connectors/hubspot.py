"""HubSpot CRM connector — imports contacts, companies, deals, and notes.

Fetches structured objects from HubSpot's CRM API and produces claims
about relationships, deal status, and contact information.  Notes are
optionally run through text extraction to pull claims from sales notes.

Usage::

    from attestdb.connectors.hubspot import HubSpotConnector

    conn = HubSpotConnector(
        token=os.environ["HUBSPOT_TOKEN"],
        objects=["contacts", "companies", "deals", "notes"],
        max_items=500,
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from attestdb.connectors.base import HybridConnector

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

HUBSPOT_API = "https://api.hubapi.com"

# Properties to request per object type
_CONTACT_PROPS = "firstname,lastname,email,company,jobtitle,lifecyclestage"
_COMPANY_PROPS = "name,domain,industry,numberofemployees,annualrevenue"
_DEAL_PROPS = "dealname,dealstage,pipeline,amount,closedate,hubspot_owner_id"
_NOTE_PROPS = "hs_note_body,hs_timestamp"

_DEFAULT_OBJECTS = ("contacts", "companies", "deals", "notes")


class HubSpotConnector(HybridConnector):
    """Sync connector that imports HubSpot CRM objects as claims."""

    name = "hubspot"

    def __init__(
        self,
        token: str,
        objects: list[str] | None = None,
        max_items: int = 500,
        pipeline: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError("pip install requests for the HubSpot connector")
        self._token = token
        self._objects = tuple(objects) if objects else _DEFAULT_OBJECTS
        self._max_items = max_items
        self._pipeline = pipeline
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        # Pipeline stage cache: internal ID → human-readable name
        self._stage_names: dict[str, str] = {}
        # Collected note bodies for text extraction
        self._note_bodies: list[tuple[str, str]] = []  # (source_id, body)

    # ------------------------------------------------------------------
    # Pipeline stage resolution
    # ------------------------------------------------------------------

    def _load_pipeline_stages(self) -> None:
        """Fetch deal pipelines and cache stage ID → name mapping."""
        if self._stage_names:
            return
        try:
            resp = self._request_with_retry(
                "GET", f"{HUBSPOT_API}/crm/v3/pipelines/deals",
            )
            resp.raise_for_status()
            for pipe in resp.json().get("results", []):
                for stage in pipe.get("stages", []):
                    stage_id = stage.get("id", "")
                    label = stage.get("label", stage_id)
                    if stage_id:
                        self._stage_names[stage_id] = label
        except Exception as exc:
            logger.warning("hubspot: failed to load pipeline stages: %s", exc)

    def _resolve_stage(self, stage_id: str) -> str:
        """Return human-readable stage name, falling back to the raw ID."""
        if not self._stage_names:
            self._load_pipeline_stages()
        return self._stage_names.get(stage_id, stage_id)

    # ------------------------------------------------------------------
    # Pagination helper
    # ------------------------------------------------------------------

    def _paginate(
        self,
        endpoint: str,
        properties: str,
        associations: str | None = None,
    ) -> Iterator[dict]:
        """Yield CRM objects from a paginated HubSpot endpoint."""
        after: str | None = None
        count = 0
        pages = 0

        while count < self._max_items and pages < self._MAX_PAGES:
            params: dict = {
                "limit": min(100, self._max_items - count),
                "properties": properties,
            }
            if associations:
                params["associations"] = associations
            if after:
                params["after"] = after

            resp = self._request_with_retry("GET", endpoint, params=params)

            # Rate limit: 429 with Retry-After
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "10"))
                logger.warning("hubspot: rate limited, sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for obj in results:
                yield obj
                count += 1

            paging = data.get("paging", {}).get("next", {})
            after = paging.get("after")
            if not after:
                break
            pages += 1
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Object → claim converters
    # ------------------------------------------------------------------

    def _contact_to_claims(self, obj: dict) -> Iterator[dict]:
        props = obj.get("properties", {})
        email = (props.get("email") or "").strip()
        if not email:
            return

        hs_id = obj.get("id", "")
        sid = f"hubspot:contacts/{hs_id}"
        first = (props.get("firstname") or "").strip()
        last = (props.get("lastname") or "").strip()
        full_name = f"{first} {last}".strip()
        contact_ext: dict[str, str] = {"email": email}
        if full_name:
            contact_ext["person_name"] = full_name

        company = (props.get("company") or "").strip()
        if company:
            yield self._make_claim(
                email, "contact", "works_at",
                company, "company", sid,
                subj_ext=contact_ext,
                obj_ext={"company_name": company},
            )

        jobtitle = (props.get("jobtitle") or "").strip()
        if jobtitle:
            yield self._make_claim(
                email, "contact", "has_role",
                jobtitle, "role", sid,
                subj_ext=contact_ext,
            )

        stage = (props.get("lifecyclestage") or "").strip()
        if stage:
            yield self._make_claim(
                email, "contact", "has_stage",
                stage, "stage", sid,
            )

    def _company_to_claims(self, obj: dict) -> Iterator[dict]:
        props = obj.get("properties", {})
        name = (props.get("name") or "").strip()
        if not name:
            return

        sid = f"hubspot:companies/{obj.get('id', '')}"

        industry = (props.get("industry") or "").strip()
        if industry:
            yield self._make_claim(
                name, "company", "in_industry",
                industry, "industry", sid,
            )

        domain = (props.get("domain") or "").strip()
        if domain:
            yield self._make_claim(
                name, "company", "has_domain",
                domain, "domain", sid,
            )

        size = (props.get("numberofemployees") or "").strip()
        if size:
            yield self._make_claim(
                name, "company", "has_size",
                size, "size", sid,
            )

    def _deal_to_claims(self, obj: dict) -> Iterator[dict]:
        props = obj.get("properties", {})
        dealname = (props.get("dealname") or "").strip()
        if not dealname:
            return

        sid = f"hubspot:deals/{obj.get('id', '')}"

        stage_raw = (props.get("dealstage") or "").strip()
        if stage_raw:
            yield self._make_claim(
                dealname, "deal", "has_stage",
                self._resolve_stage(stage_raw), "stage", sid,
            )

        amount = (props.get("amount") or "").strip()
        if amount:
            yield self._make_claim(
                dealname, "deal", "has_amount",
                f"${amount}", "currency", sid,
            )

        pipeline = (props.get("pipeline") or "").strip()
        if pipeline:
            yield self._make_claim(
                dealname, "deal", "in_pipeline",
                pipeline, "pipeline", sid,
            )

        closedate = (props.get("closedate") or "").strip()
        if closedate:
            yield self._make_claim(
                dealname, "deal", "closes_on",
                closedate, "date", sid,
            )

    def _note_to_claims(self, obj: dict) -> Iterator[dict]:
        props = obj.get("properties", {})
        body = (props.get("hs_note_body") or "").strip()
        hs_id = obj.get("id", "")
        sid = f"hubspot:notes/{hs_id}"

        associations = obj.get("associations", {})
        for assoc_type in ("contacts", "companies", "deals"):
            results = (
                (associations.get(assoc_type) or {})
                .get("results", [])
            )
            for assoc in results:
                assoc_id = assoc.get("id", "")
                if assoc_id:
                    yield self._make_claim(
                        f"note:{hs_id}", "note",
                        "associated_with",
                        f"{assoc_type}/{assoc_id}",
                        assoc_type.rstrip("s"), sid,
                    )

        if body:
            self._note_bodies.append((sid, body))

    # ------------------------------------------------------------------
    # Main fetch / run
    # ------------------------------------------------------------------

    def fetch(self) -> Iterator[dict]:
        """Yield structural claim dicts from HubSpot CRM objects."""
        self._note_bodies = []

        if "contacts" in self._objects:
            for obj in self._paginate(
                f"{HUBSPOT_API}/crm/v3/objects/contacts", _CONTACT_PROPS,
            ):
                yield from self._contact_to_claims(obj)

        if "companies" in self._objects:
            for obj in self._paginate(
                f"{HUBSPOT_API}/crm/v3/objects/companies", _COMPANY_PROPS,
            ):
                yield from self._company_to_claims(obj)

        if "deals" in self._objects:
            # Filter by pipeline if requested
            for obj in self._paginate(
                f"{HUBSPOT_API}/crm/v3/objects/deals", _DEAL_PROPS,
            ):
                if self._pipeline:
                    pipe = (obj.get("properties", {}).get("pipeline") or "").strip()
                    if pipe != self._pipeline:
                        continue
                yield from self._deal_to_claims(obj)

        if "notes" in self._objects:
            for obj in self._paginate(
                f"{HUBSPOT_API}/crm/v3/objects/notes",
                _NOTE_PROPS,
                associations="contacts,companies,deals",
            ):
                yield from self._note_to_claims(obj)

    def fetch_bodies(self) -> Iterator[tuple[str, str]]:
        """Yield collected note bodies for text extraction."""
        return iter(self._note_bodies)
