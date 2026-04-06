"""Slack message extraction template — v1.

Special handling:
- Channel context: #support vs #sales implies different claim types
- Emoji reactions: 🔥 = urgency, 👍 = acknowledgment (metadata, not extracted)
- Thread vs channel: thread replies are scoped to parent topic
- @mentions: indicate assignment or escalation targets
"""

VERSION = "1.0"

PROMPT = """\
Extract claims from this Slack message. Look for:
- Escalations and urgency signals ("urgent", "ASAP", "blocked")
- Customer feedback and sentiment mentions
- Timeline changes ("pushed back", "delayed", "moved up")
- Blocker reports and incident mentions
- Status updates on deals, projects, or accounts
- Action items and assignments (@person patterns)
- Metric mentions (NPS, CSAT, revenue, usage)

Slack messages are informal — extract the business signal, not filler. \
Treat @mentions as person entities.

Few-shot examples:

Message: "Heads up — BigCo just pushed back their go-live by 3 months 😬"
Output:
[
  {"subject": "BigCo", "predicate": "timeline.delay", \
"object": "go-live delayed 3 months", "confidence": 0.9, \
"source_snippet": "BigCo just pushed back their go-live by 3 months"}
]

Message: "@channel Globex integration has been blocked for 3 days. Customer is furious."
Output:
[
  {"subject": "Globex", "predicate": "action.blocker", \
"object": "integration blocked 3 days", "confidence": 0.9, \
"source_snippet": "Globex integration has been blocked for 3 days"},
  {"subject": "Globex", "predicate": "satisfaction.sentiment", \
"object": "furious", "confidence": 0.85, \
"source_snippet": "Customer is furious"}
]

Now extract claims from this message:\
"""
