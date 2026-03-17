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

    def add_many(self, ids: list[int]) -> None:
        """Add multiple elements at once, more efficient than individual add() calls."""
        parent = self._parent
        rank = self._rank
        for x in ids:
            if x not in parent:
                parent[x] = x
                rank[x] = 0

    def find(self, x: int) -> int:
        """Find the root of x with iterative path compression."""
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: point all nodes on the path directly to root
        while self._parent[x] != root:
            next_x = self._parent[x]
            self._parent[x] = root
            x = next_x
        return root

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
    uf.add_many(all_ids)
    for id_a, id_b, _score in pairs:
        uf.union(id_a, id_b)

    clusters = uf.get_clusters()

    # Build member-to-cluster-id mapping for fast pair assignment
    member_to_cid: dict[int, int] = {}
    sorted_clusters = sorted(clusters, key=lambda s: min(s))
    for cluster_id, members in enumerate(sorted_clusters, start=1):
        for m in members:
            member_to_cid[m] = cluster_id

    # Build result with empty pair_scores first
    result: dict[int, dict] = {}
    for cluster_id, members in enumerate(sorted_clusters, start=1):
        size = len(members)
        result[cluster_id] = {
            "members": sorted(members),
            "size": size,
            "oversized": size > max_cluster_size,
            "pair_scores": {},
        }

    # Populate pair_scores in a single pass over pairs
    for id_a, id_b, score in pairs:
        cid = member_to_cid[id_a]
        result[cid]["pair_scores"][(id_a, id_b)] = score

    return result


def get_cluster_pair_scores(
    cluster_members: list[int],
    all_pairs: list[tuple[int, int, float]],
) -> dict[tuple[int, int], float]:
    """Get pair scores for a specific cluster. Call on-demand, not in hot path."""
    member_set = set(cluster_members)
    return {
        (a, b): s
        for a, b, s in all_pairs
        if a in member_set and b in member_set
    }
