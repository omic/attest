"""Meeting notes extraction template — v1.

Special handling:
- Action items: "ACTION: person to do X by date" patterns
- Decisions: "DECIDED:", "agreed to", "we will" patterns
- Attendee context: who said what matters for attribution
"""

VERSION = "1.0"

PROMPT = """\
Extract claims from these meeting notes. Look for:
- Decisions made ("decided to", "agreed", "approved")
- Action items with owners and deadlines
- Blockers raised during the meeting
- Status updates on projects or accounts
- Commitments and promises ("will deliver", "committed to")
- Risk mentions and escalation needs
- Stakeholder sentiment expressed during the meeting
- Deadlines and timeline updates

Attribute claims to specific people when the notes indicate who said what.

Few-shot examples:

Notes: "Sync with Acme team. Decision: migrate to new API by March. \
ACTION: @john to draft migration plan by Friday. Blocker: SSO integration \
not ready — need eng support."
Output:
[
  {"subject": "Acme", "predicate": "action.decision", \
"object": "migrate to new API by March", "confidence": 0.9, \
"source_snippet": "Decision: migrate to new API by March"},
  {"subject": "john", "predicate": "commitment.deadline", \
"object": "draft migration plan by Friday", "confidence": 0.85, \
"source_snippet": "ACTION: @john to draft migration plan by Friday"},
  {"subject": "Acme", "predicate": "action.blocker", \
"object": "SSO integration not ready", "confidence": 0.9, \
"source_snippet": "Blocker: SSO integration not ready"}
]

Now extract claims from these meeting notes:\
"""
