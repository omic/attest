# Building a Data Connector

## Quick Start

1. Create `attestdb/connectors/myservice.py`
2. Extend `Connector` base class
3. Implement `fetch()` — yields claim dicts
4. Register in `attestdb/connectors/__init__.py`

## Skeleton

```python
from attestdb.connectors.base import Connector, ConnectorResult

class MyServiceConnector(Connector):
    name = "myservice"

    def __init__(self, token: str, **kwargs):
        super().__init__(**kwargs)
        self.token = token

    def fetch(self):
        for item in self._paginate():
            yield {
                "subject": (item["id"], "ticket"),
                "predicate": ("assigned_to", "relates_to"),
                "object": (item["assignee"], "person"),
                "provenance": {"source_type": "myservice", "source_id": f"myservice:{item['id']}"},
                "confidence": 1.0,
            }
```

## Claim Format

Every `fetch()` yield must include:
- `subject`: tuple of (id, entity_type)
- `predicate`: tuple of (predicate_id, predicate_type)
- `object`: tuple of (id, entity_type)
- `provenance`: dict with `source_type` and `source_id`
- `confidence`: float 0.0-1.0

## Pagination

Use bounded loops — never `while True`:
```python
def _paginate(self):
    cursor = None
    for _ in range(10000):  # bounded
        data = self._request(cursor=cursor)
        yield from data["items"]
        cursor = data.get("next_cursor")
        if not cursor:
            break
        time.sleep(0.3)  # rate limit
```

## Error Handling

Wrap row-level conversion in try/except — one bad row shouldn't crash the whole connector:
```python
for row in rows:
    try:
        yield self._row_to_claim(row)
    except (KeyError, ValueError) as e:
        logger.warning("Skipping row: %s", e)
```

## Registration

Add to `attestdb/connectors/__init__.py`:
```python
CONNECTOR_REGISTRY["myservice"] = ("attestdb.connectors.myservice", "MyServiceConnector")
```

Then users can: `db.connect("myservice", token="...")`

## Testing

- Test `fetch()` returns valid claim dicts
- Test pagination terminates
- Test error handling doesn't crash on bad data
- Mock the external API, don't call it in tests
