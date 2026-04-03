"""CCMS — Case Count Metric System for comparing ER clustering outcomes.

Based on: Talburt et al., "Case Count Metric for Comparative Analysis of
Entity Resolution Results" (arXiv:2601.02824v1, Jan 2026).

Classifies each cluster from ER1 into one of four transformation cases
relative to ER2: unchanged, merged, partitioned, or overlapping.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ClusterCase:
    """Classification of a single ER1 cluster's transformation to ER2."""
    cluster_id: int
    case: str  # "unchanged" | "merged" | "partitioned" | "overlapping"
    members: list[int]
    er2_clusters: dict[int, list[int]]


@dataclass
class CompareResult:
    """Result of comparing two sets of ER clusters."""
    unchanged: int = 0
    merged: int = 0
    partitioned: int = 0
    overlapping: int = 0
    rc: int = 0
    cc1: int = 0
    cc2: int = 0
    sc1: int = 0
    sc2: int = 0
    twi: float = 0.0
    cases: list[ClusterCase] = field(default_factory=list)

    def summary(self) -> dict:
        """Return a summary dict of all metrics."""
        total = self.cc1 or 1
        return {
            "unchanged": self.unchanged,
            "merged": self.merged,
            "partitioned": self.partitioned,
            "overlapping": self.overlapping,
            "rc": self.rc,
            "cc1": self.cc1,
            "cc2": self.cc2,
            "sc1": self.sc1,
            "sc2": self.sc2,
            "twi": round(self.twi, 4),
            "unchanged_pct": round(self.unchanged / total, 4),
            "merged_pct": round(self.merged / total, 4),
            "partitioned_pct": round(self.partitioned / total, 4),
            "overlapping_pct": round(self.overlapping / total, 4),
        }


def compare_clusters(
    clusters_a: dict[int, dict],
    clusters_b: dict[int, dict],
) -> CompareResult:
    """Compare two ER clustering outcomes on the same dataset.

    Classifies each cluster in clusters_a (ER1) into one of four
    transformation cases relative to clusters_b (ER2).

    Args:
        clusters_a: Baseline clusters (ER1). Standard GoldenMatch format.
        clusters_b: Comparison clusters (ER2). Same format.

    Returns:
        CompareResult with counts, TWI, and per-cluster case details.

    Raises:
        ValueError: If the two cluster dicts cover different row IDs.
    """
    # Extract member sets
    ids_a: set[int] = set()
    sets_a: dict[int, set[int]] = {}
    for cid, info in clusters_a.items():
        members = set(info["members"])
        sets_a[cid] = members
        ids_a |= members

    ids_b: set[int] = set()
    sets_b: dict[int, set[int]] = {}
    for cid, info in clusters_b.items():
        members = set(info["members"])
        sets_b[cid] = members
        ids_b |= members

    if ids_a != ids_b:
        diff = ids_a.symmetric_difference(ids_b)
        only_a = ids_a - ids_b
        only_b = ids_b - ids_a
        raise ValueError(
            f"Cluster dicts cover different row IDs. "
            f"{len(only_a)} IDs only in A, {len(only_b)} IDs only in B. "
            f"First few: {sorted(diff)[:10]}"
        )

    rc = len(ids_a)

    # Build reverse lookup: row_id -> ER2 cluster ID
    row_to_b: dict[int, int] = {}
    for cid, members in sets_b.items():
        for m in members:
            row_to_b[m] = cid

    # Classify each ER1 cluster
    cases: list[ClusterCase] = []
    uc = mc = pc = oc = 0
    non_empty_intersections = 0

    for cid_a, members_a in sets_a.items():
        # Find which ER2 clusters this ER1 cluster's members map to
        er2_mapping: dict[int, list[int]] = defaultdict(list)
        for m in members_a:
            cid_b = row_to_b[m]
            er2_mapping[cid_b].append(m)

        non_empty_intersections += len(er2_mapping)

        # Classify
        if len(er2_mapping) == 1:
            cid_b = next(iter(er2_mapping))
            if sets_b[cid_b] == members_a:
                case = "unchanged"
                uc += 1
            else:
                case = "merged"
                mc += 1
        else:
            # Multiple ER2 clusters intersect this ER1 cluster
            all_subsets = all(
                sets_b[cid_b] <= members_a for cid_b in er2_mapping
            )
            if all_subsets:
                case = "partitioned"
                pc += 1
            else:
                case = "overlapping"
                oc += 1

        cases.append(ClusterCase(
            cluster_id=cid_a,
            case=case,
            members=sorted(members_a),
            er2_clusters={k: sorted(v) for k, v in er2_mapping.items()},
        ))

    cc1 = len(clusters_a)
    cc2 = len(clusters_b)
    sc1 = sum(1 for s in sets_a.values() if len(s) == 1)
    sc2 = sum(1 for s in sets_b.values() if len(s) == 1)

    # TWI = sqrt(CC1 * CC2) / V
    if non_empty_intersections > 0:
        twi = math.sqrt(cc1 * cc2) / non_empty_intersections
    else:
        twi = 1.0 if cc1 == 0 and cc2 == 0 else 0.0

    return CompareResult(
        unchanged=uc,
        merged=mc,
        partitioned=pc,
        overlapping=oc,
        rc=rc,
        cc1=cc1,
        cc2=cc2,
        sc1=sc1,
        sc2=sc2,
        twi=twi,
        cases=cases,
    )
