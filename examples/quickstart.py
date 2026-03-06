"""Attest quickstart — core engine features (no enterprise dependencies)."""

from attestdb.infrastructure.attest_db import AttestDB
from attestdb.core.types import ClaimInput


def main():
    # 1. Create a database
    db = AttestDB(":memory:", embedding_dim=None)

    # 2. Ingest claims with provenance
    claims = [
        ClaimInput(
            subject=("BRCA1", "gene"),
            predicate=("associated_with", "relates_to"),
            object=("Breast Cancer", "disease"),
            provenance={"source_type": "literature", "source_id": "PMID:12345"},
            confidence=0.92,
        ),
        ClaimInput(
            subject=("BRCA1", "gene"),
            predicate=("interacts_with", "relates_to"),
            object=("RAD51", "protein"),
            provenance={"source_type": "literature", "source_id": "PMID:12346"},
            confidence=0.87,
        ),
        ClaimInput(
            subject=("TP53", "gene"),
            predicate=("associated_with", "relates_to"),
            object=("Breast Cancer", "disease"),
            provenance={"source_type": "literature", "source_id": "PMID:12347"},
            confidence=0.95,
        ),
        ClaimInput(
            subject=("TP53", "gene"),
            predicate=("regulates", "relates_to"),
            object=("BRCA1", "gene"),
            provenance={"source_type": "experimental", "source_id": "EXP:001"},
            confidence=0.78,
        ),
    ]

    result = db.ingest_batch(claims)
    print(f"Ingested {result.ingested} claims ({result.duplicates} duplicates)")

    # 3. Query the knowledge graph
    frame = db.query("BRCA1", depth=2)
    print(f"\n{frame.focal_entity.name} ({frame.focal_entity.entity_type})")
    print(f"  {frame.claim_count} claims")
    for rel in frame.direct_relationships:
        print(f"  --[{rel.predicate}]--> {rel.target.name} (conf={rel.confidence:.2f})")

    # 4. Explain with profiling
    frame, profile = db.explain("TP53", depth=1)
    print(f"\nQuery for TP53 took {profile.elapsed_ms:.1f}ms")
    for step in profile.steps:
        print(f"  {step}")

    # 5. Retraction
    result = db.retract("PMID:12345", reason="Retracted by journal")
    print(f"\nRetracted: {result.retracted_count} claims")

    # 6. Time travel — view before retraction
    import time
    snapshot = db.at(timestamp=int(time.time() * 1e9))
    snap_frame = snapshot.query("BRCA1", depth=1)
    print(f"\nSnapshot query: {snap_frame.claim_count} claims for BRCA1")

    # 7. Stats
    stats = db.stats()
    print(f"\nDatabase: {stats['total_claims']} claims, {stats['total_entities']} entities")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
