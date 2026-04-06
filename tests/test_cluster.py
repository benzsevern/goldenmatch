"""Tests for goldenmatch clustering."""


from goldenmatch.core.cluster import UnionFind, build_clusters


class TestUnionFind:
    """Tests for UnionFind."""

    def test_basic_union(self):
        """Unioning two elements puts them in the same set."""
        uf = UnionFind()
        uf.add(1)
        uf.add(2)
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)

    def test_singletons(self):
        """Elements not unioned remain in separate sets."""
        uf = UnionFind()
        uf.add(1)
        uf.add(2)
        uf.add(3)
        assert uf.find(1) != uf.find(2)
        assert uf.find(2) != uf.find(3)

    def test_transitive_union(self):
        """Union is transitive: union(1,2) + union(2,3) => 1,2,3 in same set."""
        uf = UnionFind()
        for i in range(1, 4):
            uf.add(i)
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.find(1) == uf.find(3)

    def test_get_clusters(self):
        """get_clusters returns correct groupings."""
        uf = UnionFind()
        for i in range(1, 6):
            uf.add(i)
        uf.union(1, 2)
        uf.union(3, 4)
        clusters = uf.get_clusters()
        # Should have 3 clusters: {1,2}, {3,4}, {5}
        cluster_sets = [frozenset(c) for c in clusters]
        assert frozenset({1, 2}) in cluster_sets
        assert frozenset({3, 4}) in cluster_sets
        assert frozenset({5}) in cluster_sets
        assert len(clusters) == 3


class TestBuildClusters:
    """Tests for build_clusters."""

    def test_builds_from_pairs(self):
        """Builds clusters from scored pairs."""
        pairs = [(1, 2, 0.95), (2, 3, 0.90)]
        all_ids = [1, 2, 3, 4]
        result = build_clusters(pairs, all_ids)
        # IDs 1,2,3 should be in one cluster; 4 is a singleton
        assert len(result) == 2
        # Find the cluster containing member 1
        cluster_with_1 = [c for c in result.values() if 1 in c["members"]][0]
        assert cluster_with_1["members"] == [1, 2, 3]
        assert cluster_with_1["size"] == 3
        assert cluster_with_1["oversized"] is False
        # pair_scores should contain the pairs
        assert (1, 2) in cluster_with_1["pair_scores"] or (2, 1) in cluster_with_1["pair_scores"]

    def test_oversized_auto_split(self):
        """Clusters exceeding max_cluster_size are auto-split."""
        pairs = [(1, 2, 0.9), (2, 3, 0.9)]
        all_ids = [1, 2, 3]
        result = build_clusters(pairs, all_ids, max_cluster_size=2)
        # Cluster of 3 was split — no cluster should exceed size 2
        for cluster in result.values():
            assert cluster["size"] <= 2

    def test_no_pairs_all_singletons(self):
        """No pairs means all IDs become singleton clusters."""
        pairs = []
        all_ids = [1, 2, 3]
        result = build_clusters(pairs, all_ids)
        assert len(result) == 3
        for cluster in result.values():
            assert cluster["size"] == 1
            assert cluster["oversized"] is False

    def test_monotonic_cluster_ids(self):
        """Cluster IDs start at 1 and are monotonically increasing."""
        pairs = [(1, 2, 0.9)]
        all_ids = [1, 2, 3]
        result = build_clusters(pairs, all_ids)
        assert sorted(result.keys()) == [1, 2]

    def test_pair_scores_stored(self):
        """Pair scores are stored in the cluster dict."""
        pairs = [(10, 20, 0.85), (20, 30, 0.77)]
        all_ids = [10, 20, 30]
        result = build_clusters(pairs, all_ids)
        cluster = list(result.values())[0]
        assert cluster["pair_scores"][(10, 20)] == 0.85
        assert cluster["pair_scores"][(20, 30)] == 0.77

    def test_members_sorted(self):
        """Members list is sorted."""
        pairs = [(5, 3, 0.9), (3, 1, 0.9)]
        all_ids = [5, 3, 1]
        result = build_clusters(pairs, all_ids)
        cluster_with_all = [c for c in result.values() if c["size"] == 3][0]
        assert cluster_with_all["members"] == [1, 3, 5]


def test_auto_split_oversized():
    """build_clusters auto-splits oversized clusters."""
    # Chain: 0-1(0.9) - 1-2(0.5) - 2-3(0.8), max_cluster_size=2
    pairs = [(0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.8)]
    clusters = build_clusters(pairs, [0, 1, 2, 3], max_cluster_size=2)
    for cinfo in clusters.values():
        assert cinfo["size"] <= 2


def test_split_recursive_oversized():
    """build_clusters recursively splits clusters exceeding max_cluster_size."""
    pairs = [(0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.8), (3, 4, 0.7), (4, 5, 0.6)]
    all_ids = list(range(6))
    clusters = build_clusters(pairs, all_ids, max_cluster_size=3)
    for cinfo in clusters.values():
        assert cinfo["size"] <= 3


def test_cluster_quality_strong():
    """Clusters with tight edges get quality='strong'."""
    pairs = [(0, 1, 0.95), (1, 2, 0.90), (0, 2, 0.92)]
    clusters = build_clusters(pairs, [0, 1, 2])
    cinfo = list(clusters.values())[0]
    assert cinfo["cluster_quality"] == "strong"


def test_cluster_quality_weak():
    """Clusters with large edge gap get quality='weak'."""
    pairs = [(0, 1, 0.95), (1, 2, 0.85), (0, 2, 0.40)]
    clusters = build_clusters(pairs, [0, 1, 2])
    cinfo = list(clusters.values())[0]
    assert cinfo["cluster_quality"] == "weak"


def test_cluster_quality_split_precedence():
    """Split clusters get quality='split' even if also weak."""
    pairs = [(0, 1, 0.9), (1, 2, 0.3), (2, 3, 0.9)]
    clusters = build_clusters(pairs, [0, 1, 2, 3], max_cluster_size=2)
    for cinfo in clusters.values():
        if cinfo["size"] > 1:
            assert cinfo["cluster_quality"] in ("strong", "split")
