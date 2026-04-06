"""Email extraction template — v1.

Special handling:
- Thread context: reply chains indicate conversation history
- Sender role inference: exec emails carry higher implicit authority
- Commitment detection: "I'll", "we will", "by Friday" patterns
"""

VERSION = "1.0"

PROMPT = """\
Extract business claims from the following email. Look for:
- Customer sentiment and satisfaction signals
- Risk signals (churn risk, escalation, competitor mentions, champion departure)
- Revenue mentions (ARR, deal size, expansion/contraction)
- Relationship updates (champion changes, new stakeholders, departures)
- Commitments and deadlines ("will deliver by", "expected to close")
- Escalation indicators (urgency language, executive involvement)
- Metric mentions (NPS, CSAT, health scores)

If this is a reply thread, note that earlier messages provide context but \
focus extraction on the most recent message.

Few-shot examples:

Email: "I'm concerned about Acme's renewal. Their NPS dropped to 15 last \
quarter and the champion left."
Output:
[
  {"subject": "Acme", "predicate": "risk.escalation", "object": "renewal concern", \
"confidence": 0.8, "source_snippet": "I'm concerned about Acme's renewal"},
  {"subject": "Acme", "predicate": "satisfaction.nps", "object": "15", \
"confidence": 0.9, "source_snippet": "NPS dropped to 15 last quarter"},
  {"subject": "Acme", "predicate": "relationship.champion_change", \
"object": "champion departed", "confidence": 0.85, \
"source_snippet": "the champion left"}
]

Email: "Great news — BigCo signed the expansion. New ARR is $2.4M, up from $1.8M."
Output:
[
  {"subject": "BigCo", "predicate": "revenue.arr", "object": "$2.4M", \
"confidence": 0.95, "source_snippet": "New ARR is $2.4M"},
  {"subject": "BigCo", "predicate": "revenue.expansion", "object": "expanded from $1.8M to $2.4M", \
"confidence": 0.9, "source_snippet": "signed the expansion. New ARR is $2.4M, up from $1.8M"}
]

Now extract claims from this email:\
"""
