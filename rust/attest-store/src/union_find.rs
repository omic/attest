//! Union-find (disjoint set) for alias resolution.
//!
//! Supports same_as (union) and not_same_as (split) operations,
//! with path compression and union by rank.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Union-find with path compression and union by rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnionFind {
    parent: HashMap<String, String>,
    rank: HashMap<String, usize>,
    /// Reverse index: root → set of all members (including the root itself).
    /// Maintained on union/split for O(group_size) `get_group()` lookups.
    /// Lazily rebuilt from `parent` if empty after deserialization.
    #[serde(default)]
    groups: HashMap<String, HashSet<String>>,
}

impl UnionFind {
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
            groups: HashMap::new(),
        }
    }

    /// Rebuild the `groups` reverse index from `parent` map.
    /// Called after deserialization when `#[serde(default)]` groups may be empty.
    pub fn rebuild_groups(&mut self) {
        if !self.groups.is_empty() || self.parent.is_empty() {
            return;
        }
        // Find the root for every node and group them
        let keys: Vec<String> = self.parent.keys().cloned().collect();
        for key in keys {
            let root = self.find(&key);
            self.groups
                .entry(root)
                .or_default()
                .insert(key);
        }
        // Ensure each root includes itself
        let roots: Vec<String> = self.groups.keys().cloned().collect();
        for root in roots {
            self.groups.entry(root.clone()).or_default().insert(root);
        }
    }

    /// Find the root representative for `x`, with path compression.
    pub fn find(&mut self, x: &str) -> String {
        if !self.parent.contains_key(x) {
            return x.to_string();
        }

        // Find root
        let mut root = x.to_string();
        while let Some(p) = self.parent.get(&root) {
            if p == &root {
                break;
            }
            root = p.clone();
        }

        // Path compression
        let mut current = x.to_string();
        while current != root {
            if let Some(next) = self.parent.get(&current).cloned() {
                self.parent.insert(current.clone(), root.clone());
                current = next;
            } else {
                break;
            }
        }

        root
    }

    /// Merge the sets containing `a` and `b` (same_as).
    pub fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }

        let rank_a = *self.rank.get(&ra).unwrap_or(&0);
        let rank_b = *self.rank.get(&rb).unwrap_or(&0);

        // Determine winner (new root) and loser
        let (winner, loser) = if rank_a < rank_b {
            (rb.clone(), ra.clone())
        } else if rank_a > rank_b {
            (ra.clone(), rb.clone())
        } else {
            (rb.clone(), ra.clone())
        };

        // Union by rank: attach smaller tree under larger
        self.parent.insert(loser.clone(), winner.clone());
        if rank_a == rank_b {
            self.rank.insert(winner.clone(), rank_b + 1);
        }

        // Maintain groups reverse index: merge loser's group into winner's
        let loser_group = self.groups.remove(&loser).unwrap_or_else(|| {
            let mut s = HashSet::new();
            s.insert(loser);
            s
        });
        let winner_group = self.groups.entry(winner.clone()).or_insert_with(|| {
            let mut s = HashSet::new();
            s.insert(winner);
            s
        });
        winner_group.extend(loser_group);
    }

    /// Detach `a` from its group (not_same_as).
    ///
    /// Re-parents any nodes that point directly to `a` so they point
    /// to `a`'s root instead, preventing silent group fragmentation.
    pub fn split(&mut self, a: &str) {
        let root = self.find(a);
        // Re-parent direct children of `a` to the root
        if root != a {
            // `a` is not the root — children pointing to `a` should go to root
            let children: Vec<String> = self
                .parent
                .iter()
                .filter(|(k, v)| v.as_str() == a && k.as_str() != a)
                .map(|(k, _)| k.clone())
                .collect();
            for child in children {
                self.parent.insert(child, root.clone());
            }
            // Update groups: remove `a` from root's group
            if let Some(group) = self.groups.get_mut(&root) {
                group.remove(a);
            }
        } else {
            // `a` IS the root — pick a new root for remaining members
            let children: Vec<String> = self
                .parent
                .iter()
                .filter(|(k, v)| v.as_str() == a && k.as_str() != a)
                .map(|(k, _)| k.clone())
                .collect();
            if let Some(new_root) = children.first().cloned() {
                // Remove new_root's parent link (it becomes a root)
                self.parent.remove(&new_root);
                // Point remaining children to new_root
                for child in &children[1..] {
                    self.parent.insert(child.clone(), new_root.clone());
                }
                // Transfer rank
                let old_rank = self.rank.remove(a).unwrap_or(0);
                self.rank.insert(new_root.clone(), old_rank);

                // Update groups: move group from old root to new root, minus `a`
                if let Some(mut group) = self.groups.remove(a) {
                    group.remove(a);
                    self.groups.insert(new_root, group);
                }
            } else {
                // No children — just remove the singleton group entry
                self.groups.remove(a);
            }
        }
        self.parent.remove(a);
        self.rank.remove(a);
    }

    /// Get all entity IDs that resolve to the same root as `entity_id`.
    /// Uses the reverse index for O(group_size) instead of O(N).
    pub fn get_group(&mut self, entity_id: &str) -> HashSet<String> {
        let root = self.find(entity_id);
        if let Some(group) = self.groups.get(&root) {
            group.clone()
        } else {
            // Singleton — not part of any union
            let mut group = HashSet::new();
            group.insert(entity_id.to_string());
            group
        }
    }
}

impl Default for UnionFind {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_self() {
        let mut uf = UnionFind::new();
        assert_eq!(uf.find("a"), "a");
    }

    #[test]
    fn test_union_and_find() {
        let mut uf = UnionFind::new();
        uf.union("a", "b");
        assert_eq!(uf.find("a"), uf.find("b"));
    }

    #[test]
    fn test_transitive_union() {
        let mut uf = UnionFind::new();
        uf.union("a", "b");
        uf.union("b", "c");
        assert_eq!(uf.find("a"), uf.find("c"));
    }

    #[test]
    fn test_split_detaches() {
        let mut uf = UnionFind::new();
        uf.union("a", "b");
        assert_eq!(uf.find("a"), uf.find("b"));
        uf.split("a");
        // After split, "a" is its own root again
        assert_eq!(uf.find("a"), "a");
    }

    #[test]
    fn test_split_intermediate_preserves_group() {
        let mut uf = UnionFind::new();
        // Build chain: a -> b -> d, c -> d
        uf.union("a", "b");
        uf.union("c", "d");
        uf.union("a", "c"); // merges groups, root is d (or b depending on rank)
        let root = uf.find("a");
        assert_eq!(uf.find("b"), root);
        assert_eq!(uf.find("c"), root);
        assert_eq!(uf.find("d"), root);

        // Split "b" — only "b" should leave; a, c, d stay connected
        uf.split("b");
        assert_eq!(uf.find("b"), "b"); // b is detached

        // The remaining group {a, c, d} should still be connected
        let remaining_root = uf.find("a");
        assert_eq!(uf.find("c"), remaining_root);
        assert_eq!(uf.find("d"), remaining_root);
        assert_ne!(remaining_root, "b"); // not connected to b
    }

    #[test]
    fn test_split_root_preserves_children() {
        let mut uf = UnionFind::new();
        uf.union("a", "b");
        uf.union("c", "b");
        // b is likely the root
        let root = uf.find("a");

        // Split the root
        uf.split(&root);
        assert_eq!(uf.find(&root), root); // root is detached

        // The other two should still be connected
        let remaining: Vec<&str> = ["a", "b", "c"]
            .iter()
            .copied()
            .filter(|x| *x != root.as_str())
            .collect();
        assert_eq!(uf.find(remaining[0]), uf.find(remaining[1]));
    }

    #[test]
    fn test_get_group() {
        let mut uf = UnionFind::new();
        uf.union("a", "b");
        uf.union("b", "c");
        let group = uf.get_group("a");
        assert!(group.contains("a"));
        assert!(group.contains("b"));
        assert!(group.contains("c"));
        assert!(!group.contains("d"));
    }

    #[test]
    fn test_separate_groups() {
        let mut uf = UnionFind::new();
        uf.union("a", "b");
        uf.union("c", "d");
        assert_ne!(uf.find("a"), uf.find("c"));
    }

    #[test]
    fn test_union_by_rank_balances() {
        let mut uf = UnionFind::new();
        // Create a chain to test rank
        uf.union("a", "b");
        uf.union("c", "d");
        uf.union("a", "c");
        // All should resolve to same root
        let root = uf.find("a");
        assert_eq!(uf.find("b"), root);
        assert_eq!(uf.find("c"), root);
        assert_eq!(uf.find("d"), root);
    }
}
