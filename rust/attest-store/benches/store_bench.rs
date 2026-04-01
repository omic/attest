//! Criterion benchmarks for RustStore.
//!
//! Run with: cargo bench -p attest-store

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use attest_core::types::{Claim, ClaimStatus, EntityRef, PredicateRef, Provenance};
use attest_store::RustStore;

/// Helper — build a synthetic claim with the given identifiers.
fn make_claim(claim_id: &str, subj: &str, pred: &str, obj: &str) -> Claim {
    Claim {
        claim_id: claim_id.to_string(),
        content_id: attest_core::compute_content_id(subj, pred, obj),
        subject: EntityRef {
            id: subj.to_string(),
            entity_type: "entity".to_string(),
            display_name: subj.to_string(),
            external_ids: Default::default(),
        },
        predicate: PredicateRef {
            id: pred.to_string(),
            predicate_type: "relates_to".to_string(),
        },
        object: EntityRef {
            id: obj.to_string(),
            entity_type: "entity".to_string(),
            display_name: obj.to_string(),
            external_ids: Default::default(),
        },
        confidence: 0.7,
        provenance: Provenance {
            source_type: "observation".to_string(),
            source_id: "test".to_string(),
            method: None,
            chain: vec![],
            model_version: None,
            organization: None,
        },
        embedding: None,
        payload: None,
        timestamp: 1000,
        status: ClaimStatus::Active,
        namespace: String::new(),
        expires_at: 0,
    }
}

/// Insert 1,000 claims into a fresh in-memory store.
fn bench_insert_claim(c: &mut Criterion) {
    // Pre-build the claims so allocation is not included in the measurement.
    let claims: Vec<Claim> = (0..1_000)
        .map(|i| {
            let subj = format!("entity_{}", i % 100);
            let obj = format!("entity_{}", (i + 1) % 100);
            make_claim(&format!("claim_{i}"), &subj, "relates_to", &obj)
        })
        .collect();

    c.bench_function("insert_claim_1000", |b| {
        b.iter(|| {
            let mut store = RustStore::in_memory();
            for claim in &claims {
                store.insert_claim(black_box(claim.clone()));
            }
            store
        });
    });
}

/// Query claims_for a single entity after populating the store with 1,000 claims.
fn bench_claims_for(c: &mut Criterion) {
    let mut store = RustStore::in_memory();
    for i in 0..1_000 {
        let subj = format!("entity_{}", i % 100);
        let obj = format!("entity_{}", (i + 1) % 100);
        store.insert_claim(make_claim(&format!("claim_{i}"), &subj, "relates_to", &obj));
    }

    c.bench_function("claims_for_entity", |b| {
        b.iter(|| {
            store.claims_for(black_box("entity_0"), None, None, 0.0, 0)
        });
    });
}

/// BFS traversal at depth 2 on a graph with 100 entities and 500 claims.
fn bench_bfs_depth_2(c: &mut Criterion) {
    let mut store = RustStore::in_memory();

    // Build a graph: 100 entities, 500 claims connecting them in various patterns
    for i in 0..500 {
        let subj = format!("entity_{}", i % 100);
        let obj = format!("entity_{}", (i * 7 + 3) % 100); // pseudo-random wiring
        store.insert_claim(make_claim(&format!("claim_{i}"), &subj, "relates_to", &obj));
    }

    c.bench_function("bfs_depth_2", |b| {
        b.iter(|| {
            store.bfs_claims(black_box("entity_0"), black_box(2))
        });
    });
}

/// Build the full adjacency list from 1,000 claims.
fn bench_get_adjacency_list(c: &mut Criterion) {
    let mut store = RustStore::in_memory();
    for i in 0..1_000 {
        let subj = format!("entity_{}", i % 100);
        let obj = format!("entity_{}", (i + 1) % 100);
        store.insert_claim(make_claim(&format!("claim_{i}"), &subj, "relates_to", &obj));
    }

    c.bench_function("get_adjacency_list_1000", |b| {
        b.iter(|| store.get_adjacency_list());
    });
}

// ---------------------------------------------------------------------------
// LMDB benchmarks — more realistic conditions (disk-backed, larger datasets)
// ---------------------------------------------------------------------------

/// Query claims_for on LMDB with 100K claims (warm cache).
fn bench_lmdb_claims_for_100k(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.attest");
    let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
    for i in 0..100_000 {
        let subj = format!("entity_{}", i % 10_000);
        let obj = format!("entity_{}", (i + 1) % 10_000);
        store.insert_claim(make_claim(&format!("claim_{i}"), &subj, "relates_to", &obj));
    }

    c.bench_function("lmdb_claims_for_100k", |b| {
        b.iter(|| {
            store.claims_for(black_box("entity_0"), None, None, 0.0, 0)
        });
    });
    let _ = store.close();
}

/// Query claims_for on LMDB with 100K claims, limit=10 (paginated).
fn bench_lmdb_claims_for_100k_limit10(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.attest");
    let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
    for i in 0..100_000 {
        let subj = format!("entity_{}", i % 10_000);
        let obj = format!("entity_{}", (i + 1) % 10_000);
        store.insert_claim(make_claim(&format!("claim_{i}"), &subj, "relates_to", &obj));
    }

    c.bench_function("lmdb_claims_for_100k_limit10", |b| {
        b.iter(|| {
            store.claims_for(black_box("entity_0"), None, None, 0.0, 10)
        });
    });
    let _ = store.close();
}

/// BFS depth-2 on LMDB with 100K claims.
fn bench_lmdb_bfs_100k(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.attest");
    let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
    for i in 0..100_000 {
        let subj = format!("entity_{}", i % 10_000);
        let obj = format!("entity_{}", (i * 7 + 3) % 10_000);
        store.insert_claim(make_claim(&format!("claim_{i}"), &subj, "relates_to", &obj));
    }

    c.bench_function("lmdb_bfs_depth2_100k", |b| {
        b.iter(|| {
            store.bfs_claims(black_box("entity_0"), black_box(2))
        });
    });
    let _ = store.close();
}

criterion_group!(
    benches,
    bench_insert_claim,
    bench_claims_for,
    bench_bfs_depth_2,
    bench_get_adjacency_list,
    bench_lmdb_claims_for_100k,
    bench_lmdb_claims_for_100k_limit10,
    bench_lmdb_bfs_100k,
);
criterion_main!(benches);
