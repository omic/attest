//! PyO3 bindings for Attest Rust storage engine.
//!
//! Exposes `RustStore` as a Python class matching the KuzuStore interface.
//! Complex types (Claim, EntitySummary) are marshalled as Python dicts;
//! a Python-side adapter reconstructs the proper dataclass instances.

use std::collections::HashMap;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use pyo3::IntoPyObjectExt;

use attest_core::types::{
    Claim, ClaimStatus, EntityRef, Payload, PredicateRef, Provenance,
};
use attest_store::store::RustStore;
use attest_store::metadata::Vocabulary;

// ── Helpers: Rust → Python ─────────────────────────────────────────────

/// Convert an EntitySummary-like struct to a Python dict.
fn entity_summary_to_dict(py: Python<'_>, es: &attest_core::types::EntitySummary) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("id", &es.id)?;
    dict.set_item("name", &es.name)?;
    dict.set_item("entity_type", &es.entity_type)?;
    dict.set_item("external_ids", es.external_ids.clone().into_py_any(py)?)?;
    dict.set_item("claim_count", es.claim_count)?;
    Ok(dict.into())
}

/// Convert a Claim to a Python dict.
fn claim_to_dict(py: Python<'_>, c: &Claim) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("claim_id", &c.claim_id)?;
    dict.set_item("content_id", &c.content_id)?;

    // Subject
    let subj = PyDict::new(py);
    subj.set_item("id", &c.subject.id)?;
    subj.set_item("entity_type", &c.subject.entity_type)?;
    subj.set_item("display_name", &c.subject.display_name)?;
    subj.set_item("external_ids", c.subject.external_ids.clone().into_py_any(py)?)?;
    dict.set_item("subject", subj)?;

    // Predicate
    let pred = PyDict::new(py);
    pred.set_item("id", &c.predicate.id)?;
    pred.set_item("predicate_type", &c.predicate.predicate_type)?;
    dict.set_item("predicate", pred)?;

    // Object
    let obj = PyDict::new(py);
    obj.set_item("id", &c.object.id)?;
    obj.set_item("entity_type", &c.object.entity_type)?;
    obj.set_item("display_name", &c.object.display_name)?;
    obj.set_item("external_ids", c.object.external_ids.clone().into_py_any(py)?)?;
    dict.set_item("object", obj)?;

    dict.set_item("confidence", c.confidence)?;

    // Provenance
    let prov = PyDict::new(py);
    prov.set_item("source_type", &c.provenance.source_type)?;
    prov.set_item("source_id", &c.provenance.source_id)?;
    prov.set_item("method", c.provenance.method.as_deref())?;
    prov.set_item("chain", c.provenance.chain.clone().into_py_any(py)?)?;
    prov.set_item("model_version", c.provenance.model_version.as_deref())?;
    prov.set_item("organization", c.provenance.organization.as_deref())?;
    dict.set_item("provenance", prov)?;

    // Payload
    if let Some(ref payload) = c.payload {
        let pl = PyDict::new(py);
        pl.set_item("schema_ref", &payload.schema_ref)?;
        let data_str = serde_json::to_string(&payload.data)
            .map_err(|e| PyValueError::new_err(format!("payload serialization error: {e}")))?;
        pl.set_item("data_json", &data_str)?;
        dict.set_item("payload", pl)?;
    } else {
        dict.set_item("payload", py.None())?;
    }

    dict.set_item("timestamp", c.timestamp)?;
    dict.set_item("status", c.status.as_str())?;
    dict.set_item("namespace", &c.namespace)?;
    dict.set_item("expires_at", c.expires_at)?;

    Ok(dict.into())
}

// ── Helpers: Python → Rust ─────────────────────────────────────────────

/// Extract a Claim from a Python Claim dataclass instance.
fn claim_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Claim> {
    let claim_id: String = obj.getattr("claim_id")?.extract()?;
    let content_id: String = obj.getattr("content_id")?.extract()?;
    let confidence: f64 = obj.getattr("confidence")?.extract()?;
    let timestamp: i64 = obj.getattr("timestamp")?.extract()?;

    // Subject
    let subj_obj = obj.getattr("subject")?;
    let subject = EntityRef {
        id: subj_obj.getattr("id")?.extract()?,
        entity_type: subj_obj.getattr("entity_type")?.extract()?,
        display_name: subj_obj.getattr("display_name")?.extract().unwrap_or_default(),
        external_ids: extract_string_dict(&subj_obj.getattr("external_ids")?)?,
    };

    // Predicate
    let pred_obj = obj.getattr("predicate")?;
    let predicate = PredicateRef {
        id: pred_obj.getattr("id")?.extract()?,
        predicate_type: pred_obj.getattr("predicate_type")?.extract()?,
    };

    // Object
    let obj_ref = obj.getattr("object")?;
    let object = EntityRef {
        id: obj_ref.getattr("id")?.extract()?,
        entity_type: obj_ref.getattr("entity_type")?.extract()?,
        display_name: obj_ref.getattr("display_name")?.extract().unwrap_or_default(),
        external_ids: extract_string_dict(&obj_ref.getattr("external_ids")?)?,
    };

    // Provenance
    let prov_obj = obj.getattr("provenance")?;
    let provenance = Provenance {
        source_type: prov_obj.getattr("source_type")?.extract()?,
        source_id: prov_obj.getattr("source_id")?.extract()?,
        method: prov_obj.getattr("method")?.extract().ok(),
        chain: prov_obj.getattr("chain")?.extract().unwrap_or_default(),
        model_version: prov_obj.getattr("model_version")?.extract().ok(),
        organization: prov_obj.getattr("organization")?.extract().ok(),
    };

    // Payload
    let payload_py = obj.getattr("payload")?;
    let payload = if payload_py.is_none() {
        None
    } else {
        let schema_ref: String = payload_py.getattr("schema_ref")?.extract()?;
        let data_obj = payload_py.getattr("data")?;
        // Convert Python dict to serde_json::Value via JSON roundtrip
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (&data_obj,))?.extract()?;
        let data: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(format!("invalid payload JSON: {e}")))?;
        Some(Payload { schema_ref, data })
    };

    // Status
    let status_obj = obj.getattr("status")?;
    let status_str: String = status_obj.getattr("value")?.extract()?;
    let status = status_str.parse::<ClaimStatus>()
        .map_err(|_| PyValueError::new_err(format!("invalid claim status: {status_str}")))?;

    // Namespace (optional, default "")
    let namespace: String = obj.getattr("namespace")
        .and_then(|v| v.extract())
        .unwrap_or_default();

    // expires_at (optional, default 0)
    let expires_at: i64 = obj.getattr("expires_at")
        .and_then(|v| v.extract())
        .unwrap_or(0);

    Ok(Claim {
        claim_id,
        content_id,
        subject,
        predicate,
        object,
        confidence,
        provenance,
        embedding: None,
        payload,
        timestamp,
        status,
        namespace,
        expires_at,
    })
}

fn extract_string_dict(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, String>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    obj.extract::<HashMap<String, String>>().or_else(|_| Ok(HashMap::new()))
}

// ── Python vocab dict → Rust Vocabulary ────────────────────────────────

fn vocab_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vocabulary> {
    // Try structured extraction first ({"entity_types": [...], ...})
    if let Ok(dict) = obj.extract::<HashMap<String, Vec<String>>>() {
        return Ok(Vocabulary {
            entity_types: dict.get("entity_types").cloned().unwrap_or_default(),
            predicate_types: dict.get("predicate_types").cloned().unwrap_or_default(),
            source_types: dict.get("source_types").cloned().unwrap_or_default(),
        });
    }
    // Fallback: accept arbitrary dicts (e.g. {"protein": True} from bio_vocabulary).
    // These don't contribute types to the vocabulary — stored for compatibility.
    Ok(Vocabulary::default())
}

fn vocab_to_py(py: Python<'_>, v: &Vocabulary) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("entity_types", v.entity_types.clone().into_py_any(py)?)?;
    dict.set_item("predicate_types", v.predicate_types.clone().into_py_any(py)?)?;
    dict.set_item("source_types", v.source_types.clone().into_py_any(py)?)?;
    Ok(dict.into())
}

// ── PyO3 RustStore class ───────────────────────────────────────────────

#[pyclass(name = "RustStore")]
struct PyRustStore {
    inner: RustStore,
}

#[pymethods]
impl PyRustStore {
    #[new]
    fn new(db_path: &str) -> PyResult<Self> {
        let inner = RustStore::new(db_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create an in-memory store (no persistence).
    #[staticmethod]
    fn in_memory() -> Self {
        Self {
            inner: RustStore::in_memory(),
        }
    }

    /// Open a database in native read-only mode (no WAL recovery, no write lock).
    /// Multiple readers can coexist with a single writer.
    #[staticmethod]
    fn open_read_only(db_path: &str) -> PyResult<Self> {
        let inner = RustStore::open_read_only(db_path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Compact the database file, reclaiming free pages.
    /// Returns True if compaction freed any space.
    fn compact(&mut self) -> PyResult<bool> {
        self.inner
            .compact()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the schema version of this database.
    fn schema_version(&self) -> u32 {
        self.inner.schema_version()
    }

    /// Returns True if this store was opened in read-only mode.
    fn is_read_only(&self) -> bool {
        self.inner.is_read_only()
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        // Release the GIL during persistence — serialization + fsync can be slow
        py.allow_threads(|| self.inner.close())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Write a full checkpoint without closing. Flushes WAL to .attest file.
    fn checkpoint(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.checkpoint())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
        py: Python<'_>,
    ) -> PyResult<bool> {
        py.allow_threads(|| self.inner.close())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(false) // don't suppress exceptions
    }

    // ── Metadata ───────────────────────────────────────────────────

    fn register_vocabulary(&mut self, namespace: &str, vocab: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = vocab_from_py(vocab)?;
        self.inner.register_vocabulary(namespace, v);
        Ok(())
    }

    fn register_predicate(
        &mut self,
        predicate_id: &str,
        constraints: &Bound<'_, PyAny>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let json_str = dict_to_json_string(py, constraints)?;
        let val: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.register_predicate(predicate_id, val);
        Ok(())
    }

    fn register_payload_schema(
        &mut self,
        schema_id: &str,
        schema: &Bound<'_, PyAny>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let json_str = dict_to_json_string(py, schema)?;
        let val: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.register_payload_schema(schema_id, val);
        Ok(())
    }

    fn get_registered_vocabularies<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let vocabs = self.inner.get_registered_vocabularies();
        let dict = PyDict::new(py);
        for (ns, v) in vocabs {
            dict.set_item(ns, vocab_to_py(py, v)?)?;
        }
        Ok(dict.into())
    }

    fn get_predicate_constraints<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let constraints = self.inner.get_predicate_constraints();
        let dict = PyDict::new(py);
        for (k, v) in &constraints {
            let json_str = serde_json::to_string(v)
                .map_err(|e| PyValueError::new_err(format!("JSON serialization error: {e}")))?;
            let json_mod = py.import("json")?;
            let py_obj = json_mod.call_method1("loads", (&json_str,))?;
            dict.set_item(k, py_obj)?;
        }
        Ok(dict.into())
    }

    fn get_payload_schemas<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let schemas = self.inner.get_payload_schemas();
        let dict = PyDict::new(py);
        for (k, v) in &schemas {
            let json_str = serde_json::to_string(v)
                .map_err(|e| PyValueError::new_err(format!("JSON serialization error: {e}")))?;
            let json_mod = py.import("json")?;
            let py_obj = json_mod.call_method1("loads", (&json_str,))?;
            dict.set_item(k, py_obj)?;
        }
        Ok(dict.into())
    }

    // ── Alias resolution ───────────────────────────────────────────

    fn resolve(&mut self, entity_id: &str) -> String {
        self.inner.resolve(entity_id)
    }

    fn get_alias_group<'py>(&mut self, entity_id: &str, py: Python<'py>) -> PyResult<PyObject> {
        let group = self.inner.get_alias_group(entity_id);
        let set = PySet::new(py, group.into_iter().collect::<Vec<_>>())?;
        Ok(set.into())
    }

    // ── Cache management ───────────────────────────────────────────

    fn warm_caches(&self) {
        self.inner.warm_caches();
    }

    // ── Entity CRUD ────────────────────────────────────────────────

    #[pyo3(signature = (entity_id, entity_type, display_name="", external_ids=None, timestamp=0))]
    fn upsert_entity(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        display_name: &str,
        external_ids: Option<HashMap<String, String>>,
        timestamp: i64,
    ) {
        self.inner
            .upsert_entity(entity_id, entity_type, display_name, external_ids.as_ref(), timestamp);
    }

    fn get_entity<'py>(&self, entity_id: &str, py: Python<'py>) -> PyResult<PyObject> {
        match self.inner.get_entity(entity_id) {
            Some(es) => entity_summary_to_dict(py, &es),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (entity_type=None, min_claims=0, offset=0, limit=0))]
    fn list_entities<'py>(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
        offset: usize,
        limit: usize,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let entities = self.inner.list_entities(entity_type, min_claims, offset, limit);
        let list = PyList::empty(py);
        for es in &entities {
            list.append(entity_summary_to_dict(py, es)?)?;
        }
        Ok(list.into())
    }

    #[pyo3(signature = (entity_type=None, min_claims=0))]
    fn count_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
    ) -> usize {
        self.inner.count_entities(entity_type, min_claims)
    }

    // ── Claim operations ───────────────────────────────────────────

    fn insert_claim(&mut self, claim: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<()> {
        let c = claim_from_py(py, claim)?;
        self.inner.insert_claim(c);
        Ok(())
    }

    fn claim_exists(&self, claim_id: &str) -> bool {
        self.inner.claim_exists(claim_id)
    }

    /// Get a single claim by ID. Returns None if not found.
    fn get_claim<'py>(&self, claim_id: &str, py: Python<'py>) -> PyResult<Option<PyObject>> {
        match self.inner.get_claim(claim_id) {
            Some(c) => Ok(Some(claim_to_dict(py, &c)?)),
            None => Ok(None),
        }
    }

    fn claims_by_content_id<'py>(
        &self,
        content_id: &str,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let claims = self.inner.claims_by_content_id(content_id);
        let list = PyList::empty(py);
        for c in &claims {
            list.append(claim_to_dict(py, c)?)?;
        }
        Ok(list.into())
    }

    #[pyo3(signature = (entity_id, predicate_type=None, source_type=None, min_confidence=0.0))]
    fn claims_for<'py>(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let claims =
            self.inner
                .claims_for(entity_id, predicate_type, source_type, min_confidence);
        let list = PyList::empty(py);
        for c in &claims {
            list.append(claim_to_dict(py, c)?)?;
        }
        Ok(list.into())
    }

    fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        self.inner.get_claim_provenance_chain(claim_id)
    }

    #[pyo3(signature = (offset=0, limit=0))]
    fn all_claims<'py>(&self, offset: usize, limit: usize, py: Python<'py>) -> PyResult<PyObject> {
        let claims = self.inner.all_claims(offset, limit);
        let list = PyList::empty(py);
        for c in &claims {
            list.append(claim_to_dict(py, c)?)?;
        }
        Ok(list.into())
    }

    fn count_claims(&self) -> usize {
        self.inner.count_claims()
    }

    fn claims_by_source_id<'py>(
        &self,
        source_id: &str,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let claims = self.inner.claims_by_source_id(source_id);
        let list = PyList::empty(py);
        for c in &claims {
            list.append(claim_to_dict(py, c)?)?;
        }
        Ok(list.into())
    }

    fn claims_by_predicate_id<'py>(
        &self,
        predicate_id: &str,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let claims = self.inner.claims_by_predicate_id(predicate_id);
        let list = PyList::empty(py);
        for c in &claims {
            list.append(claim_to_dict(py, c)?)?;
        }
        Ok(list.into())
    }

    // ── Graph traversal ────────────────────────────────────────────

    #[pyo3(signature = (entity_id, max_depth=2))]
    fn bfs_claims<'py>(
        &mut self,
        entity_id: &str,
        max_depth: usize,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let results = self.inner.bfs_claims(entity_id, max_depth);
        let list = PyList::empty(py);
        for (claim, depth) in &results {
            let tuple = (claim_to_dict(py, claim)?, *depth);
            list.append(tuple)?;
        }
        Ok(list.into())
    }

    #[pyo3(signature = (entity_a, entity_b, max_depth=3))]
    fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        self.inner.path_exists(entity_a, entity_b, max_depth)
    }

    fn get_adjacency_list<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let adj = self.inner.get_adjacency_list();
        let dict = PyDict::new(py);
        for (k, v) in &adj {
            let set = PySet::new(py, v.iter().collect::<Vec<_>>())?;
            dict.set_item(k, set)?;
        }
        Ok(dict.into())
    }

    // ── Temporal queries ────────────────────────────────────────────

    fn claims_in_range<'py>(
        &mut self,
        min_ts: i64,
        max_ts: i64,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let claims = self.inner.claims_in_range(min_ts, max_ts);
        let list = PyList::empty(py);
        for c in &claims {
            list.append(claim_to_dict(py, c)?)?;
        }
        Ok(list.into())
    }

    fn most_recent_claims<'py>(&mut self, n: usize, py: Python<'py>) -> PyResult<PyObject> {
        let claims = self.inner.most_recent_claims(n);
        let list = PyList::empty(py);
        for c in &claims {
            list.append(claim_to_dict(py, c)?)?;
        }
        Ok(list.into())
    }

    // ── Text search ───────────────────────────────────────────────

    #[pyo3(signature = (query, top_k=10))]
    fn search_entities<'py>(
        &self,
        query: &str,
        top_k: usize,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let entities = self.inner.search_entities(query, top_k);
        let list = PyList::empty(py);
        for es in &entities {
            list.append(entity_summary_to_dict(py, es)?)?;
        }
        Ok(list.into())
    }

    // ── Stats ──────────────────────────────────────────────────────

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let s = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("total_claims", s.total_claims)?;
        dict.set_item("entity_count", s.entity_count)?;
        dict.set_item("entity_types", s.entity_types.clone().into_py_any(py)?)?;
        dict.set_item("predicate_types", s.predicate_types.clone().into_py_any(py)?)?;
        dict.set_item("source_types", s.source_types.clone().into_py_any(py)?)?;
        Ok(dict.into())
    }

    // ── Batch insert (bulk loader fast path) ─────────────────────

    /// Batch-insert entities and claims from pre-computed Python data.
    ///
    /// Accepts the same tuple formats used by `_ingest_append_direct`:
    /// - entities: {entity_id: (entity_type, display_name, ext_ids_json)}
    /// - claim_rows: list of 16-element tuples matching bulk loader format
    ///
    /// This avoids constructing Python Claim dataclass objects per row,
    /// extracting all data in one pass then inserting without GIL.
    ///
    /// Returns the number of claims inserted.
    fn insert_bulk(
        &mut self,
        entities: &Bound<'_, PyDict>,
        claim_rows: &Bound<'_, PyList>,
        timestamp: i64,
        _py: Python<'_>,
    ) -> PyResult<usize> {
        // Phase 1: Extract all data from Python into Rust-native types (with GIL)

        // Extract entities: {id: (type, display, ext_ids_json)}
        let mut entity_vec: Vec<(String, String, String, HashMap<String, String>)> =
            Vec::with_capacity(entities.len());
        for (key, value) in entities.iter() {
            let entity_id: String = key.extract()?;
            let tuple: &Bound<'_, PyTuple> = value.downcast()?;
            let entity_type: String = tuple.get_item(0)?.extract()?;
            let display_name: String = tuple.get_item(1)?.extract()?;
            let ext_ids_json: String = tuple.get_item(2)?.extract()?;
            let ext_ids: HashMap<String, String> = if ext_ids_json.is_empty() || ext_ids_json == "{}" {
                HashMap::new()
            } else {
                serde_json::from_str(&ext_ids_json).unwrap_or_default()
            };
            entity_vec.push((entity_id, entity_type, display_name, ext_ids));
        }

        // Build entity lookup for Claim construction
        let mut entity_map: HashMap<String, (String, String)> = entity_vec
            .iter()
            .map(|(id, etype, display, _)| (id.clone(), (etype.clone(), display.clone())))
            .collect();

        // Extract claim rows: (subj, obj, claim_id, content_id, pred_id, pred_type,
        //                      confidence, source_type, source_id, ..., timestamp, status)
        struct BulkRow {
            subj_id: String,
            obj_id: String,
            claim_id: String,
            content_id: String,
            pred_id: String,
            pred_type: String,
            confidence: f64,
            source_type: String,
            source_id: String,
            ts: i64,
        }

        let n_rows = claim_rows.len();
        let mut rows: Vec<BulkRow> = Vec::with_capacity(n_rows);
        for item in claim_rows.iter() {
            rows.push(BulkRow {
                subj_id: item.get_item(0)?.extract()?,
                obj_id: item.get_item(1)?.extract()?,
                claim_id: item.get_item(2)?.extract()?,
                content_id: item.get_item(3)?.extract()?,
                pred_id: item.get_item(4)?.extract()?,
                pred_type: item.get_item(5)?.extract()?,
                confidence: item.get_item(6)?.extract()?,
                source_type: item.get_item(7)?.extract()?,
                source_id: item.get_item(8)?.extract()?,
                ts: item.get_item(14)?.extract()?,
            });
        }

        // Phase 2: Insert in Rust (GIL held — mutation must be single-threaded)
        // Upsert all entities in a single transaction (batched)
        if !entity_vec.is_empty() {
            self.inner.upsert_entities_batch(&entity_vec, timestamp)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Resolve entity IDs missing from the passed dict by looking them up
        // in the store (LMDB). This enables two-phase loading: register entities
        // first via upsert_entities_batch, then insert claims with entities={}.
        if !rows.is_empty() {
            let mut missing: Vec<String> = Vec::new();
            for row in &rows {
                if !entity_map.contains_key(&row.subj_id) {
                    missing.push(row.subj_id.clone());
                }
                if !entity_map.contains_key(&row.obj_id) {
                    missing.push(row.obj_id.clone());
                }
            }
            if !missing.is_empty() {
                missing.sort_unstable();
                missing.dedup();
                let missing_refs: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                let resolved = self.inner.get_entities_batch(&missing_refs);
                entity_map.extend(resolved);
            }
        }

        // Build all Claim structs
        let claims: Vec<Claim> = rows
            .into_iter()
            .map(|row| {
                let (subj_type, subj_display) = entity_map
                    .get(&row.subj_id)
                    .cloned()
                    .unwrap_or(("entity".into(), row.subj_id.clone()));
                let (obj_type, obj_display) = entity_map
                    .get(&row.obj_id)
                    .cloned()
                    .unwrap_or(("entity".into(), row.obj_id.clone()));

                Claim {
                    claim_id: row.claim_id,
                    content_id: row.content_id,
                    subject: EntityRef {
                        id: row.subj_id,
                        entity_type: subj_type,
                        display_name: subj_display,
                        external_ids: HashMap::new(),
                    },
                    predicate: PredicateRef {
                        id: row.pred_id,
                        predicate_type: row.pred_type,
                    },
                    object: EntityRef {
                        id: row.obj_id,
                        entity_type: obj_type,
                        display_name: obj_display,
                        external_ids: HashMap::new(),
                    },
                    confidence: row.confidence,
                    provenance: Provenance {
                        source_type: row.source_type,
                        source_id: row.source_id,
                        method: None,
                        chain: vec![],
                        model_version: None,
                        organization: None,
                    },
                    embedding: None,
                    payload: None,
                    timestamp: row.ts,
                    status: ClaimStatus::Active,
                    namespace: String::new(),
            expires_at: 0,
                }
            })
            .collect();

        // Batch insert with single WAL sync
        let count = self.inner.insert_claims_batch(claims);

        Ok(count)
    }

    // ── Entity registration (two-phase bulk loading) ──────────────

    /// Register entities in the store without inserting claims.
    ///
    /// Used as Phase 1 of two-phase bulk loading: register entities first,
    /// then call insert_bulk with entities={} — Rust resolves display names
    /// from the store's own indexes.
    ///
    /// Accepts the same dict format as insert_bulk:
    ///   {entity_id: (entity_type, display_name, ext_ids_json)}
    fn upsert_entities_batch(
        &mut self,
        entities: &Bound<'_, PyDict>,
        timestamp: i64,
    ) -> PyResult<usize> {
        let mut entity_vec: Vec<(String, String, String, HashMap<String, String>)> =
            Vec::with_capacity(entities.len());
        for (key, value) in entities.iter() {
            let entity_id: String = key.extract()?;
            let tuple: &Bound<'_, PyTuple> = value.downcast()?;
            let entity_type: String = tuple.get_item(0)?.extract()?;
            let display_name: String = tuple.get_item(1)?.extract()?;
            let ext_ids_json: String = tuple.get_item(2)?.extract()?;
            let ext_ids: HashMap<String, String> = if ext_ids_json.is_empty() || ext_ids_json == "{}" {
                HashMap::new()
            } else {
                serde_json::from_str(&ext_ids_json).unwrap_or_default()
            };
            entity_vec.push((entity_id, entity_type, display_name, ext_ids));
        }
        let count = entity_vec.len();
        self.inner.upsert_entities_batch(&entity_vec, timestamp)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(count)
    }

    // ── Retraction / status overlay ─────────────────────────────────

    /// Retract a claim by ID (set status to Tombstoned).
    /// Returns True if the claim existed, False otherwise.
    fn retract_claim(&mut self, claim_id: &str) -> PyResult<bool> {
        self.inner.retract_claim(claim_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Retract all claims from a source. Returns list of retracted claim IDs.
    fn retract_source(&mut self, source_id: &str) -> PyResult<Vec<String>> {
        self.inner.retract_source(source_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Update the status of a claim by string name.
    /// Accepted values: "active", "archived", "tombstoned"/"retracted", "provenance_degraded"/"degraded".
    fn update_claim_status(&mut self, claim_id: &str, status: &str) -> PyResult<bool> {
        self.inner.update_claim_status(claim_id, status)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Batch-update claim statuses. Each element is (claim_id, status_string).
    /// Returns the number of claims actually updated.
    fn update_claim_status_batch(&mut self, updates: Vec<(String, String)>) -> PyResult<usize> {
        let mut parsed: Vec<(String, attest_core::types::ClaimStatus)> = Vec::with_capacity(updates.len());
        for (id, s) in updates {
            let status = match s.to_lowercase().as_str() {
                "active" => attest_core::types::ClaimStatus::Active,
                "archived" => attest_core::types::ClaimStatus::Archived,
                "tombstoned" | "retracted" => attest_core::types::ClaimStatus::Tombstoned,
                "provenance_degraded" | "degraded" => attest_core::types::ClaimStatus::ProvenanceDegraded,
                other => return Err(PyValueError::new_err(format!("Unknown claim status: {other}"))),
            };
            parsed.push((id, status));
        }
        let updates = parsed;
        self.inner.update_claim_status_batch(&updates)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Set whether query methods include retracted (Tombstoned) claims.
    fn set_include_retracted(&mut self, include: bool) -> PyResult<()> {
        self.inner.set_include_retracted(include);
        Ok(())
    }

    /// Set the namespace filter. Empty list = no filter (all namespaces visible).
    fn set_namespace_filter(&mut self, namespaces: Vec<String>) -> PyResult<()> {
        self.inner.set_namespace_filter(namespaces);
        Ok(())
    }

    /// Get the current namespace filter.
    fn get_namespace_filter(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.get_namespace_filter().to_vec())
    }

    // ── Raw query (not applicable for Rust backend) ────────────────

    #[pyo3(signature = (_query, _params=None))]
    fn raw_query<'py>(
        &self,
        _query: &str,
        _params: Option<&Bound<'_, PyDict>>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        // Rust store has no raw query support — return empty list
        Ok(PyList::empty(py).into())
    }

    // ── Analytics (Rust-native aggregation) ───────────────────────────

    /// Backfill pred_id_counts counter table for existing databases.
    /// Scans PREDICATE_IDX once and persists counts. Only needed once per DB.
    fn backfill_pred_id_counts(&mut self) -> PyResult<usize> {
        self.inner.backfill_pred_id_counts()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Backfill claim_summaries table for existing databases.
    /// Builds lightweight summaries (~200 bytes each) for fast analytics queries.
    /// Auto-triggered on schema upgrade, but can be called manually.
    fn backfill_claim_summaries(&mut self) -> PyResult<u64> {
        self.inner.backfill_claim_summaries()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Backfill analytics counter tables for sub-second analytics on large databases.
    /// Populates entity_pred_counts, entity_src_counts, and pred_pair_counts from claim_summaries.
    /// Must be called after backfill_claim_summaries. Returns number of claims processed.
    fn backfill_analytics_counters(&mut self) -> PyResult<u64> {
        self.inner.backfill_analytics_counters()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Enable or disable bulk load mode. Skips analytics counters during insert
    /// for ~2.5× throughput. Call rebuild_counters() after loading is complete.
    fn set_bulk_load_mode(&mut self, enabled: bool) {
        self.inner.set_bulk_load_mode(enabled);
    }

    /// Rebuild all analytics counters and claim_summaries from the claims table.
    /// Call after bulk loading/merging with bulk_load_mode enabled.
    fn rebuild_counters(&mut self) -> PyResult<()> {
        self.inner.rebuild_counters()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Merge all claims and entities from another LMDB database into this store.
    /// Stays entirely in Rust for maximum throughput. Returns number of merged claims.
    fn merge_from(&mut self, source_path: &str) -> PyResult<usize> {
        self.inner.merge_from(source_path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Count claims per predicate_id. Returns dict {predicate_id: count}.
    fn predicate_counts<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let counts = self.inner.predicate_counts();
        let dict = PyDict::new(py);
        for (k, v) in counts {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    /// Count claims per source_id. Returns dict {source_id: count}.
    fn source_id_counts<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let counts = self.inner.source_id_counts();
        let dict = PyDict::new(py);
        for (k, v) in counts {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    /// Find contradictions: (subject, object) pairs under both pred_a and pred_b.
    /// Returns list of (subject_id, object_id, count_a, count_b, avg_conf_a, avg_conf_b).
    fn find_contradictions<'py>(
        &self,
        pred_a: &str,
        pred_b: &str,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let results = self.inner.find_contradictions(pred_a, pred_b);
        let list = PyList::empty(py);
        for (subj, obj, ca, cb, conf_a, conf_b) in results {
            let tup = PyTuple::new(py, &[
                subj.into_py_any(py)?,
                obj.into_py_any(py)?,
                ca.into_py_any(py)?,
                cb.into_py_any(py)?,
                conf_a.into_py_any(py)?,
                conf_b.into_py_any(py)?,
            ])?;
            list.append(tup)?;
        }
        Ok(list.into())
    }

    /// Gap analysis: entities of a type missing expected predicates.
    /// Returns list of (entity_id, [missing_pred_ids]).
    #[pyo3(signature = (entity_type, expected_predicates, limit=0))]
    fn gap_analysis<'py>(
        &self,
        entity_type: &str,
        expected_predicates: Vec<String>,
        limit: usize,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let pred_refs: Vec<&str> = expected_predicates.iter().map(|s| s.as_str()).collect();
        let results = self.inner.gap_analysis(entity_type, &pred_refs, limit);
        let list = PyList::empty(py);
        for (entity_id, missing) in results {
            let missing_list = PyList::new(py, &missing)?;
            let tup = PyTuple::new(py, &[
                entity_id.into_py_any(py)?,
                missing_list.into_py_any(py)?,
            ])?;
            list.append(tup)?;
        }
        Ok(list.into())
    }

    /// Get predicate_id → count for a single entity.
    /// Returns list of (predicate_id, count) sorted by count desc.
    fn entity_predicate_counts<'py>(
        &self,
        entity_id: &str,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let results = self.inner.entity_predicate_counts(entity_id);
        let list = PyList::empty(py);
        for (pred_id, count) in results {
            let tup = PyTuple::new(py, &[
                pred_id.into_py_any(py)?,
                count.into_py_any(py)?,
            ])?;
            list.append(tup)?;
        }
        Ok(list.into())
    }

    /// Get source_id → (count, avg_confidence) for a single entity.
    /// Returns list of (source_id, count, avg_confidence) sorted by count desc.
    fn entity_source_counts<'py>(
        &self,
        entity_id: &str,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let results = self.inner.entity_source_counts(entity_id);
        let list = PyList::empty(py);
        for (src, count, avg_conf) in results {
            let tup = PyTuple::new(py, &[
                src.into_py_any(py)?,
                count.into_py_any(py)?,
                avg_conf.into_py_any(py)?,
            ])?;
            list.append(tup)?;
        }
        Ok(list.into())
    }
}

/// Helper: convert a Python dict to a JSON string via Python's json module.
fn dict_to_json_string(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    Ok(json_str)
}

/// Python module definition.
#[pymodule]
fn attest_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRustStore>()?;
    Ok(())
}
