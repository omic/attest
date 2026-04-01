# Using the Text Extractor

## What It Does

TextExtractor takes unstructured text (Slack messages, docs, emails) and produces structured claims. It uses LLM extraction with a fallback chain of 5+ providers.

## Basic Usage

```python
from attestdb.intelligence.text_extractor import TextExtractor

extractor = TextExtractor(model="auto")  # walks fallback chain
result = extractor.extract(
    text="TP53 mutations cause resistance to cisplatin in NSCLC",
    source_type="paper",
    source_id="pmid:12345",
)
print(f"Extracted {result.n_valid} claims")
```

`model="auto"` walks `EXTRACTION_FALLBACK_CHAIN` (gemini → together → openai → deepseek → grok), using the first provider with an API key set.

## Batch Extraction

For multiple messages (e.g., Slack ingestion), batch 10 messages per LLM call:

```python
results = extractor.extract_batch(messages, batch_size=10)
```

## Entity Validation

TextExtractor must reject claims where subject, predicate, or object is `None`:

```python
# The extractor rejects these BEFORE str() coercion:
# str(None) → "None" entity — this is a real bug that existed
if any(v is None for v in [subj, pred, obj]):
    continue  # skip this claim
```

Also reject string `"none"` / `"null"` entities at parse time in `_parse_llm_response()`.

## Return Type

`ExtractionResult` has:
- `.n_valid` — number of valid claims extracted (NOT `.claims_ingested`)
- `.claims` — list of extracted claim dicts
- `.prompt_tokens` / `.completion_tokens` — LLM usage

## Heuristic Extractor

`HeuristicExtractor` is a non-LLM fallback that uses regex patterns. Important: it must use the actual parsed predicate, not hard-coded `"relates_to"`.

## Testing

- Test extraction returns valid claims with proper structure
- Test `None` entity rejection
- Test batch extraction with multiple messages
- Test fallback chain activates when primary provider fails
- Mock the LLM API, not the extractor itself
