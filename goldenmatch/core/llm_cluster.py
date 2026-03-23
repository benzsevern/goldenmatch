"""In-context LLM clustering -- send blocks of borderline records to an LLM
for direct cluster assignment instead of pairwise yes/no scoring.

Flow:
1. Collect borderline pairs (candidate_lo <= score < auto_threshold)
2. Build a graph of borderline records, find connected components
3. Send each component as a block to the LLM: "Group these records into clusters"
4. LLM returns cluster assignments with confidence scores
5. Synthesize pair_scores from LLM results for cluster contract compatibility
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import polars as pl

from goldenmatch.config.schemas import LLMScorerConfig
from goldenmatch.core.llm_budget import BudgetTracker

logger = logging.getLogger(__name__)


def llm_cluster_pairs(
    pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    config: LLMScorerConfig,
    api_key: str | None = None,
    return_budget: bool = False,
) -> "list[tuple[int, int, float]] | tuple[list[tuple[int, int, float]], dict | None]":
    """Cluster borderline records via in-context LLM prompting.

    Replaces pairwise LLM scoring with block-level clustering. Records in the
    borderline range are grouped into connected components and sent to the LLM
    as a single prompt. The LLM returns cluster assignments with confidence.

    Returns pairs in the same format as llm_score_pairs for pipeline compatibility.
    """
    auto_threshold = config.auto_threshold
    candidate_lo = config.candidate_lo
    candidate_hi = config.candidate_hi
    cluster_max = config.cluster_max_size
    cluster_min = config.cluster_min_size

    # Classify pairs into tiers
    auto_accept = []
    candidates = []
    below = []
    for i, (a, b, s) in enumerate(pairs):
        if s >= auto_threshold:
            auto_accept.append(i)
        elif candidate_lo <= s < candidate_hi:
            candidates.append(i)
        else:
            below.append(i)

    # Build result with auto-accepted pairs
    result_pairs = list(pairs)
    for i in auto_accept:
        a, b, s = pairs[i]
        result_pairs[i] = (a, b, 1.0)

    if not candidates:
        budget_summary = None
        if return_budget:
            return result_pairs, budget_summary
        return result_pairs

    # Initialize budget tracker
    budget = BudgetTracker(config.budget) if config.budget else None

    # Build graph of borderline records
    components = _build_components(pairs, candidates)

    # Prepare row lookup
    row_lookup = {}
    for row in df.to_dicts():
        row_lookup[row["__row_id__"]] = row

    # Determine display columns
    display_cols = [c for c in df.columns if not c.startswith("__")][:6]

    # Process each component
    provider = config.provider
    model = config.model
    if not api_key:
        import os
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not provider:
        import os
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = "openai"
    if not model:
        model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5-20251001"

    # Map candidate pair indices to the pairs they reference
    candidate_pair_map = {}  # pair_index -> (a, b, score)
    for i in candidates:
        candidate_pair_map[i] = pairs[i]

    for component_records, component_pair_indices in components:
        if len(component_records) < cluster_min:
            # Fall back to pairwise for small components
            _pairwise_fallback(
                component_pair_indices, pairs, result_pairs, row_lookup,
                display_cols, provider, model, api_key, budget,
            )
            continue

        if budget and budget.budget_exhausted:
            logger.warning("Budget exhausted, skipping remaining LLM cluster blocks")
            break

        # Split oversized components
        blocks = _split_component(component_records, component_pair_indices, pairs, cluster_max)

        for block_records, block_pair_indices in blocks:
            if budget and budget.budget_exhausted:
                break

            try:
                cluster_result = _call_llm_cluster(
                    block_records, row_lookup, display_cols,
                    provider, model, api_key, budget,
                )
            except Exception as e:
                logger.warning("LLM cluster call failed (%s), falling back to pairwise", e)
                _pairwise_fallback(
                    block_pair_indices, pairs, result_pairs, row_lookup,
                    display_cols, provider, model, api_key, budget,
                )
                continue

            # Apply cluster results to pairs
            _apply_cluster_results(
                cluster_result, block_pair_indices, pairs, result_pairs,
            )

    budget_summary = budget.summary() if budget else None
    if return_budget:
        return result_pairs, budget_summary
    return result_pairs


def _build_components(
    pairs: list[tuple[int, int, float]],
    candidate_indices: list[int],
) -> list[tuple[list[int], list[int]]]:
    """Build connected components from candidate pairs.

    Returns list of (record_ids, pair_indices) tuples.
    """
    # Build adjacency from candidate pairs
    adj: dict[int, set[int]] = defaultdict(set)
    record_to_pairs: dict[int, list[int]] = defaultdict(list)

    for i in candidate_indices:
        a, b, _ = pairs[i]
        adj[a].add(b)
        adj[b].add(a)
        record_to_pairs[a].append(i)
        record_to_pairs[b].append(i)

    # Find connected components via BFS
    visited = set()
    components = []

    for start in adj:
        if start in visited:
            continue
        component_records = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component_records.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        # Collect pair indices for this component
        component_pair_set = set()
        for rec_id in component_records:
            for pi in record_to_pairs[rec_id]:
                a, b, _ = pairs[pi]
                if a in visited and b in visited:
                    component_pair_set.add(pi)

        components.append((sorted(component_records), sorted(component_pair_set)))

    return components


def _split_component(
    records: list[int],
    pair_indices: list[int],
    pairs: list[tuple[int, int, float]],
    max_size: int,
) -> list[tuple[list[int], list[int]]]:
    """Split an oversized component by removing edges below median score."""
    if len(records) <= max_size:
        return [(records, pair_indices)]

    # Sort edges by score ascending
    edges = [(i, pairs[i][2]) for i in pair_indices]
    edges.sort(key=lambda x: x[1])

    # Remove weakest edges until all components are under max_size
    removed = set()
    adj: dict[int, set[int]] = defaultdict(set)
    for i in pair_indices:
        a, b, _ = pairs[i]
        adj[a].add(b)
        adj[b].add(a)

    for edge_idx, score in edges:
        a, b, _ = pairs[edge_idx]
        adj[a].discard(b)
        adj[b].discard(a)
        removed.add(edge_idx)

        # Check if all components are now small enough
        max_comp = _largest_component_size(adj, records)
        if max_comp <= max_size:
            break

    # Rebuild components with remaining edges
    remaining_pairs = [i for i in pair_indices if i not in removed]
    remaining_adj: dict[int, set[int]] = defaultdict(set)
    for i in remaining_pairs:
        a, b, _ = pairs[i]
        remaining_adj[a].add(b)
        remaining_adj[b].add(a)

    visited = set()
    blocks = []
    for start in records:
        if start in visited:
            continue
        comp = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nb in remaining_adj.get(node, set()):
                if nb not in visited:
                    queue.append(nb)

        comp_set = set(comp)
        comp_pairs = [i for i in remaining_pairs if pairs[i][0] in comp_set and pairs[i][1] in comp_set]
        blocks.append((sorted(comp), comp_pairs))

    return blocks


def _largest_component_size(adj: dict[int, set[int]], records: list[int]) -> int:
    """Find the largest connected component size."""
    visited = set()
    max_size = 0
    for start in records:
        if start in visited:
            continue
        size = 0
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            size += 1
            for nb in adj.get(node, set()):
                if nb not in visited:
                    queue.append(nb)
        max_size = max(max_size, size)
    return max_size


def _call_llm_cluster(
    record_ids: list[int],
    row_lookup: dict[int, dict],
    display_cols: list[str],
    provider: str,
    model: str,
    api_key: str,
    budget: "BudgetTracker | None",
) -> dict:
    """Call the LLM to cluster a block of records.

    Returns dict with 'clusters' (list of {members, confidence}) and 'singletons' (list of ids).
    """
    # Build prompt
    lines = ["Group these records into clusters of duplicates. Return JSON only.\n"]
    lines.append("Records:")
    for rid in record_ids:
        row = row_lookup.get(rid, {})
        fields = " | ".join(str(row.get(c, "")) for c in display_cols)
        lines.append(f"  [{rid}] {fields}")

    lines.append("")
    lines.append('Return JSON: {"clusters": [{"members": [id1, id2, ...], "confidence": 0.0-1.0}, ...], "singletons": [id1, ...]}')
    lines.append("Rules:")
    lines.append("- Each record appears in exactly one cluster or as a singleton")
    lines.append("- confidence = how certain you are that all members are the same entity")
    lines.append("- Only group records that are clearly the same real-world entity")

    prompt = "\n".join(lines)

    if budget and not budget.can_send(len(prompt) // 4):
        raise RuntimeError("Budget insufficient for this block")

    # Call LLM
    if provider == "openai":
        from goldenmatch.core.llm_scorer import _call_openai
        response, in_tok, out_tok = _call_openai(api_key, model, prompt)
    else:
        from goldenmatch.core.llm_scorer import _call_anthropic
        response, in_tok, out_tok = _call_anthropic(api_key, model, prompt)

    if budget:
        budget.record_usage(in_tok, out_tok, model)

    # Parse JSON response
    return _parse_cluster_response(response, record_ids)


def _parse_cluster_response(response: str, valid_ids: list[int]) -> dict:
    """Parse LLM cluster response, handling various JSON formats."""
    # Try to extract JSON from response
    text = response.strip()

    # Handle markdown code blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM cluster response")
                return {"clusters": [], "singletons": valid_ids}
        else:
            return {"clusters": [], "singletons": valid_ids}

    # Validate structure
    valid_set = set(valid_ids)
    clusters = []
    assigned = set()

    for cluster in result.get("clusters", []):
        members = cluster.get("members", [])
        confidence = float(cluster.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Filter to valid IDs only
        valid_members = [m for m in members if m in valid_set and m not in assigned]
        if len(valid_members) >= 2:
            clusters.append({"members": valid_members, "confidence": confidence})
            assigned.update(valid_members)

    singletons = [rid for rid in valid_ids if rid not in assigned]

    return {"clusters": clusters, "singletons": singletons}


def _apply_cluster_results(
    cluster_result: dict,
    pair_indices: list[int],
    pairs: list[tuple[int, int, float]],
    result_pairs: list[tuple[int, int, float]],
) -> None:
    """Apply LLM cluster results to pair scores.

    For pairs where both records are in the same LLM cluster, set score to the
    LLM confidence. For pairs where records are in different clusters or singletons,
    set score to 0.0.
    """
    # Build record -> cluster mapping
    record_cluster: dict[int, float] = {}  # record_id -> confidence
    for cluster in cluster_result.get("clusters", []):
        conf = cluster["confidence"]
        for member in cluster["members"]:
            record_cluster[member] = conf

    for i in pair_indices:
        a, b, orig_score = pairs[i]
        conf_a = record_cluster.get(a)
        conf_b = record_cluster.get(b)

        if conf_a is not None and conf_b is not None and conf_a == conf_b:
            # Both in same cluster
            result_pairs[i] = (a, b, conf_a)
        else:
            # Different clusters or singleton
            result_pairs[i] = (a, b, 0.0)


def _pairwise_fallback(
    pair_indices: list[int],
    pairs: list[tuple[int, int, float]],
    result_pairs: list[tuple[int, int, float]],
    row_lookup: dict[int, dict],
    display_cols: list[str],
    provider: str,
    model: str,
    api_key: str,
    budget: "BudgetTracker | None",
) -> None:
    """Fall back to pairwise LLM scoring for a set of pairs."""
    from goldenmatch.core.llm_scorer import _batch_score

    decisions = _batch_score(
        pair_indices, pairs, row_lookup, display_cols,
        provider, model, api_key, 20, budget,
    )
    for i, is_match in decisions.items():
        a, b, s = pairs[i]
        result_pairs[i] = (a, b, 1.0 if is_match else 0.0)
