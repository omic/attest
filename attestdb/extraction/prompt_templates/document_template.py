"""Document / QBR extraction template — v1.

Special handling:
- Table extraction: satisfaction scores, renewal dates, revenue in tabular form
- Section awareness: executive summary claims are higher confidence than appendix
- Metric density: QBRs often have many metrics per paragraph
"""

VERSION = "1.0"

PROMPT = """\
Extract key business claims from this document (may be a QBR, report, or \
presentation). Look for:
- Health scores, NPS, CSAT, and other satisfaction metrics
- Revenue figures (ARR, MRR, TCV, contract values)
- Risk assessments and mitigation plans
- Expansion or contraction signals
- Renewal dates and contract terms
- Strategic goals and OKR progress
- Competitive intelligence mentions
- Personnel changes (new champion, departed stakeholder)

If the document contains tables, extract each metric row as a separate claim. \
Executive summary sections carry higher implicit confidence than appendices.

Few-shot examples:

Document: "Q2 Business Review — TechStart Inc\\nHealth score: 85. NPS: 72. \
ARR: $1.2M. Renewal: Sept 2026. Key risk: champion departing in Q3."
Output:
[
  {"subject": "TechStart Inc", "predicate": "satisfaction.health", "object": "85", \
"confidence": 0.95, "source_snippet": "Health score: 85"},
  {"subject": "TechStart Inc", "predicate": "satisfaction.nps", "object": "72", \
"confidence": 0.95, "source_snippet": "NPS: 72"},
  {"subject": "TechStart Inc", "predicate": "revenue.arr", "object": "$1.2M", \
"confidence": 0.95, "source_snippet": "ARR: $1.2M"},
  {"subject": "TechStart Inc", "predicate": "renewal.date", "object": "September 2026", \
"confidence": 0.9, "source_snippet": "Renewal: Sept 2026"},
  {"subject": "TechStart Inc", "predicate": "risk.escalation", \
"object": "champion departing Q3", "confidence": 0.8, \
"source_snippet": "Key risk: champion departing in Q3"}
]

Now extract claims from this document:\
"""
