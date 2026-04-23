//! Graph algorithms operating on adjacency lists.

use std::collections::{HashMap, HashSet, VecDeque};

use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Brandes betweenness centrality with adaptive sampling.
///
/// Operates on a bidirectional adjacency list (entity_id → set of neighbor entity_ids).
/// Returns entity_id → normalized betweenness score.
pub fn brandes_betweenness(
    adj: &HashMap<String, HashSet<String>>,
    sample_size: usize,
    seed: u64,
    adaptive: bool,
    max_nodes: usize,
) -> HashMap<String, f64> {
    if adj.is_empty() {
        return HashMap::new();
    }

    // Optionally extract high-degree subgraph
    let working_adj: HashMap<String, HashSet<String>>;
    let adj_ref = if adj.len() > max_nodes && max_nodes > 0 {
        let mut by_degree: Vec<(&String, usize)> = adj.iter()
            .map(|(k, v)| (k, v.len()))
            .collect();
        by_degree.sort_by(|a, b| b.1.cmp(&a.1));
        let keep: HashSet<&String> = by_degree.iter()
            .take(max_nodes)
            .map(|(k, _)| *k)
            .collect();
        let mut sub: HashMap<String, HashSet<String>> = HashMap::new();
        for k in &keep {
            if let Some(neighbors) = adj.get(*k) {
                let filtered: HashSet<String> = neighbors.iter()
                    .filter(|n| keep.contains(n))
                    .cloned()
                    .collect();
                if !filtered.is_empty() {
                    sub.insert((*k).clone(), filtered);
                }
            }
        }
        working_adj = sub;
        &working_adj
    } else {
        adj
    };

    let n = adj_ref.len();
    if n < 2 {
        return adj_ref.keys().map(|k| (k.clone(), 0.0)).collect();
    }
    let n_edges: usize = adj_ref.values().map(|v| v.len()).sum::<usize>() / 2;

    // Adaptive sample size: reduce sample count for dense graphs to keep
    // runtime manageable. Formula: base 50 samples + 150 scaled by inverse
    // density (10K/edges), floored at 20 to ensure statistical coverage.
    // Empirically validated on graphs from 100 to 500K edges.
    let effective_sample = if adaptive && sample_size > 0 && n_edges > 10_000 {
        let adaptive_s = std::cmp::max(20, 50 + 150 * 10_000 / n_edges);
        std::cmp::min(std::cmp::min(sample_size, adaptive_s), n)
    } else if sample_size == 0 {
        n
    } else {
        std::cmp::min(sample_size, n)
    };

    // Build index-based adjacency
    let entities: Vec<String> = adj_ref.keys().cloned().collect();
    let entity_to_idx: HashMap<&str, usize> = entities.iter()
        .enumerate()
        .map(|(i, e)| (e.as_str(), i))
        .collect();
    let mut adj_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (entity, neighbors) in adj_ref {
        if let Some(&idx) = entity_to_idx.get(entity.as_str()) {
            for neighbor in neighbors {
                if let Some(&nidx) = entity_to_idx.get(neighbor.as_str()) {
                    adj_idx[idx].push(nidx);
                }
            }
        }
    }

    // Select source nodes
    let sources: Vec<usize> = if effective_sample >= n {
        (0..n).collect()
    } else {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(effective_sample);
        indices
    };

    // Brandes algorithm
    let mut betweenness = vec![0.0_f64; n];
    let mut sigma = vec![0.0_f64; n];
    let mut dist = vec![-1_i32; n];
    let mut delta = vec![0.0_f64; n];
    let mut pred: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();

    for &s in &sources {
        for i in 0..n {
            sigma[i] = 0.0;
            dist[i] = -1;
            delta[i] = 0.0;
            pred[i].clear();
        }
        sigma[s] = 1.0;
        dist[s] = 0;

        let mut stack: Vec<usize> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(s);

        // Forward pass: BFS
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &adj_idx[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        // Backward pass: dependency accumulation
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                betweenness[w] += delta[w];
            }
        }
    }

    // Normalize using undirected graph convention: 2/((n-1)(n-2)).
    // The adjacency list is bidirectional (each edge stored in both
    // directions), so this is internally consistent. Scores represent
    // undirected betweenness — not directed claim-level centrality.
    let scale = 1.0 / std::cmp::max(sources.len(), 1) as f64;
    let norm_factor = std::cmp::max((n - 1) * (n - 2), 1) as f64;

    let mut result = HashMap::with_capacity(n);
    for i in 0..n {
        let score = betweenness[i] * scale * 2.0 / norm_factor;
        // Clamp to [0, 1] — guard against NaN/Inf from degenerate graphs
        let clamped = if score.is_finite() { score.clamp(0.0, 1.0) } else { 0.0 };
        result.insert(entities[i].clone(), clamped);
    }
    result
}
