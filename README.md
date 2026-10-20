# AttestDB

An embedded database for claims with sources, confidence scores and timestamps.
The Python API uses a Rust storage engine backed by LMDB.

```bash
pip install attestdb
```

```python
from attestdb import AttestDB

db = AttestDB("example.attest", embedding_dim=None)
try:
    db.ingest(
        subject=("service-a", "service"),
        predicate=("depends_on", "relates_to"),
        object=("cache", "service"),
        provenance={"source_id": "deployment-1", "source_type": "manual"},
        confidence=0.9,
    )
    assert len(db.claims_for("service-a")) == 1

    result = db.retract("deployment-1", reason="Superseded deployment record")
    assert result.retracted_count == 1
finally:
    db.close()
```

The open source package contains the claim engine and selected basic connectors.
Proprietary models, model training and evaluation assets, enterprise connectors,
plugins and hosted services are outside this release.

For a source checkout, run the Python tests with `pytest tests/` and the storage
engine tests with `cargo test --manifest-path rust/Cargo.toml --all`.

Python 3.11 and 3.12 are the release test targets. If a matching native wheel is
unavailable, building `attest-py` from source requires a Rust toolchain.

CI and release validation run on private runners. See [release validation](RELEASE_POLICY.md).

Licensed under [Apache-2.0](LICENSE).
