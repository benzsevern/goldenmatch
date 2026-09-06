"""Shared decision core for the #1207 strong-identifier blocking union.

This is the Python side of the cross-surface ``autoconfig-core`` blocking-union
kernel (Rust ``select_blocking.rs`` → the ``goldenmatch-native`` wheel + the
TS-wasm surface). The DECISION (assembly + the coverage/survivor/re-gate)
lives once in Rust; this module either delegates to that native kernel (when the
wheel carries the symbol) or runs a **byte-identical pure-Python mirror** as the
fallback — the reference-mode posture (`docs/design/2026-07-01-rust-is-the-
reference-roadmap.md`).

Two phases, over the same JSON-boundary shapes the native/wasm shims use:

  * ``assemble_union(cols)`` — phase 1, pure assembly from column profiles
    (``{name, col_type, null_rate, cardinality_ratio}``). Returns candidate
    passes (``{fields, transforms, is_strong_id}``) or ``None`` (needs ≥1
    strong-id AND ≥2 passes). Name-column detection uses ``_classify_by_name``
    (the name-*pattern* classifier), exactly as the Rust core does.
  * ``finalize_union(passes, coverage, pass_survives, max_safe_block)`` — phase
    2, pure gates: coverage ≥ target, survivor filter (host-measured), re-gate
    (≥1 surviving strong-id AND ≥2 survivors). Returns the emitted config
    (``{strategy, keys, passes, max_block_size, skip_oversized}``) or ``None``.

The HOST (``build_blocking`` in ``autoconfig.py``) measures the two row-level
signals between the phases — OR-coverage and per-pass scale-safety — because
those need polars, not a profile aggregate (the smart-core / dumb-measurement
split the planner already uses).

Pinned byte-for-byte to the Rust core by ``select_blocking_vectors.json`` (the
same golden fixture the Rust ``tests/golden.rs`` and the TS parity test read),
so native == pure-Python == Rust == TS.
"""
from __future__ import annotations

from typing import Any

# Mirror of the union constants in ``autoconfig.py`` (kept in lockstep with the
# Rust ``select_blocking.rs`` constants). ``exact_derived`` (#2876/#2881) is a
# domain-extracted exact-scored column (e.g. ``__title_key__``) -- the same
# strong-identity signal as identifier/email/phone for blocking purposes.
_STRONG_EXACT_TYPES = ("identifier", "email", "phone", "exact_derived")
_UNION_PASS_MIN_NONNULL = 0.02
_BLOCKING_UNION_COVERAGE_TARGET = 0.95


def _transforms_for(field: str, cols: list[dict[str, Any]]) -> list[str]:
    """``email``/``exact_derived`` → ``[lowercase, strip]``, else ``[strip]``
    (keyed on the FIRST field's col_type, matching the Rust core +
    ``_build_strong_identifier_union``)."""
    for c in cols:
        if c["name"] == field:
            if c.get("col_type") in ("email", "exact_derived"):
                return ["lowercase", "strip"]
            break
    return ["strip"]


def assemble_strong_id_union_pure(cols: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Pure-Python mirror of ``assemble_strong_id_union`` (phase 1).

    ``None`` unless ≥1 strong-id pass AND ≥2 distinct passes. NO coverage check
    (that is phase 2, once the host has measured it).
    """
    from goldenmatch.core.autoconfig import _classify_by_name  # noqa: PLC0415

    passes: list[dict[str, Any]] = []
    strong_id_count = 0
    for c in cols:
        if c.get("col_type") not in _STRONG_EXACT_TYPES:
            continue
        nonnull = 1.0 - float(c.get("null_rate", 0.0))
        if nonnull < _UNION_PASS_MIN_NONNULL:
            continue
        # #876 surrogate guard: a perfect-surrogate id (card_ratio >= 1.0) makes
        # singleton blocks. blocking_max_ratio is deliberately NOT applied here.
        if float(c.get("cardinality_ratio") or 0.0) >= 1.0:
            continue
        passes.append({
            "fields": [c["name"]],
            "transforms": _transforms_for(c["name"], cols),
            "is_strong_id": True,
        })
        strong_id_count += 1

    if strong_id_count < 1:
        return None

    # name+geo passes for rows missing every strong id.
    name_cols = [c for c in cols if _classify_by_name(c["name"]) == "name"]
    first = next((c["name"] for c in name_cols if "first" in c["name"].lower()), None)
    last = next(
        (c["name"] for c in name_cols
         if "last" in c["name"].lower() or "surname" in c["name"].lower()),
        None,
    )
    geo = next((c["name"] for c in cols if c.get("col_type") in ("zip", "geo")), None)

    if first is not None and last is not None:
        passes.append({
            "fields": [first, last],
            "transforms": _transforms_for(first, cols),
            "is_strong_id": False,
        })
    if last is not None and geo is not None:
        passes.append({
            "fields": [last, geo],
            "transforms": _transforms_for(last, cols),
            "is_strong_id": False,
        })

    if len(passes) < 2:
        return None
    return passes


def finalize_strong_id_union_pure(
    passes: list[dict[str, Any]],
    coverage: float,
    pass_survives: list[bool],
    max_safe_block: int,
) -> dict[str, Any] | None:
    """Pure-Python mirror of ``finalize_strong_id_union`` (phase 2).

    ``None`` (fall through) when the coverage target is not cleared, or < 2
    passes survive, or no strong-id survives. ``pass_survives[i]`` is the host's
    scale-safety verdict for ``passes[i]``.
    """
    if coverage < _BLOCKING_UNION_COVERAGE_TARGET:
        return None
    if len(pass_survives) != len(passes):
        return None
    survivors = [p for p, ok in zip(passes, pass_survives) if ok]
    any_strong_id = any(p["is_strong_id"] for p in survivors)
    if not any_strong_id or len(survivors) < 2:
        return None
    return {
        "strategy": "multi_pass",
        "keys": [survivors[0]],
        "passes": survivors,
        "max_block_size": max_safe_block,
        "skip_oversized": True,
    }


# ── Native-routing wrappers ────────────────────────────────────────────────
#
# When the ``autoconfig`` component is native-enabled AND the wheel carries the
# symbol, delegate to the Rust kernel (byte-identical by construction); else run
# the pure-Python mirror. The ``hasattr`` guard keeps a wheel that predates these
# symbols (but has the Phase-1 ``autoconfig_decide_plan``) on the pure path.


def _native():
    """Return the native module iff the autoconfig union kernel is usable, else None."""
    from goldenmatch.core._native_loader import native_enabled, native_module  # noqa: PLC0415

    if not native_enabled("autoconfig"):
        return None
    nm = native_module()
    if nm is None or not hasattr(nm, "autoconfig_assemble_strong_id_union"):
        return None
    if not hasattr(nm, "autoconfig_finalize_strong_id_union"):
        return None
    return nm


def assemble_union(cols: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Phase 1 — native when available, else the pure-Python mirror."""
    import json  # noqa: PLC0415

    nm = _native()
    if nm is not None:
        out = nm.autoconfig_assemble_strong_id_union(json.dumps(cols))
        return json.loads(out)
    return assemble_strong_id_union_pure(cols)


def finalize_union(
    passes: list[dict[str, Any]],
    coverage: float,
    pass_survives: list[bool],
    max_safe_block: int,
) -> dict[str, Any] | None:
    """Phase 2 — native when available, else the pure-Python mirror."""
    import json  # noqa: PLC0415

    nm = _native()
    if nm is not None:
        payload = json.dumps({
            "passes": passes,
            "coverage": coverage,
            "pass_survives": pass_survives,
            "max_safe_block": int(max_safe_block),
        })
        out = nm.autoconfig_finalize_strong_id_union(payload)
        return json.loads(out)
    return finalize_strong_id_union_pure(passes, coverage, pass_survives, max_safe_block)


def union_via_core_enabled() -> bool:
    """True when the native union kernel should drive ``build_blocking`` (the
    wheel carries the symbol AND autoconfig is native-enabled)."""
    return _native() is not None
