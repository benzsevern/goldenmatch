"""Union-Find clustering for GoldenMatch."""

from __future__ import annotations

from collections import defaultdict


class UnionFind:
    """Union-Find (disjoint set) with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def add(self, x: int) -> None:
        """Add an element as its own root."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        """Find the root of x with path compression."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: int, b: int) -> None:
        """Union the sets containing a and b using union by rank."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def get_clusters(self) -> list[set[int]]:
        """Return all clusters as a list of sets."""
        groups: dict[int, set[int]] = defaultdict(set)
        for x in self._parent:
            groups[self.find(x)].add(x)
        return list(groups.values())


def build_clusters(
    pairs: list[tuple[int, int, float]],
    all_ids: list[int],
    max_cluster_size: int = 100,
) -> dict[int, dict]:
    """Build clusters from scored pairs using Union-Find.

    Args:
        pairs: List of (id_a, id_b, score) tuples.
        all_ids: All record IDs (ensures singletons are included).
        max_cluster_size: Clusters exceeding this size are flagged as oversized.

    Returns:
        Dict mapping monotonic cluster ID (starting at 1) to cluster info dict
        with keys: members, size, oversized, pair_scores.
    """
    uf = UnionFind()
    for rid in all_ids:
        uf.add(rid)
    for id_a, id_b, _score in pairs:
        uf.union(id_a, id_b)

    # Build pair_scores lookup keyed by root
    root_pairs: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
    for id_a, id_b, score in pairs:
        root = uf.find(id_a)
        root_pairs[root][(id_a, id_b)] = score

    clusters = uf.get_clusters()
    result: dict[int, dict] = {}
    for cluster_id, members in enumerate(sorted(clusters, key=lambda s: min(s)), start=1):
        root = uf.find(min(members))
        size = len(members)
        result[cluster_id] = {
            "members": sorted(members),
            "size": size,
            "oversized": size > max_cluster_size,
            "pair_scores": root_pairs.get(root, {}),
        }

    return result
