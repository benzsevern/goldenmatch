#!/usr/bin/env python
"""Where does the identity CONTROL PLANE actually cost wall + RSS at scale?

Motivation
----------
Open question: is Postgres the right long-horizon store for the identity
control plane, or is the row-oriented/no-zero-copy shape of it the thing
costing us? Before answering that with architecture, measure it.

What this ISOLATES (and why)
----------------------------
The control plane is the *write* half of `resolve_clusters` -> `apply_batch`
(`identity/resolve.py:444` -> `:527`). Getting real clusters would normally
mean running the full dedupe compute path first -- which at 5M costs far more
than the thing under test and would swamp it, and compute is explicitly NOT
what is in question here. So the cluster partition is synthesized DIRECTLY
from QIS ground truth (`quality_invariant_scale.generate_with_gt`, the
established labeled generator): `gt_cids` already IS the correct partition.
No scoring, no blocking, no clustering -- just the control-plane write.

The two phases matter more than the totals
------------------------------------------
`apply_batch`'s bulk fast path only engages for brand-new clusters with no
existing-entity overlap and no weak-conflict edge (`resolve.py:915-919`).
So:

  * ``cold``        -- empty store, every cluster brand-new. The FAVORABLE
                       case: set-based `bulk_upsert_identities/records`,
                       Postgres COPY / psycopg pipeline, SQLite TEMP-staging
                       + executemany.
  * ``incremental`` -- re-resolve a slice against the now-populated store.
                       Every cluster overlaps an existing entity, so this
                       falls to the PER-ROW path: `store.upsert_record` once
                       per member (`resolve.py:1165`), `store.get_record` per
                       record in the loser-reassign loop (`:1154`),
                       `upsert_identity` per cluster (`:1033/1071/1109`).

Reporting both is the point. If cold is fine and incremental is 100x worse
per record, the finding is "the per-row write path is the cost", and swapping
the storage engine underneath a 5M-round-trip loop would not fix it. If both
are heavy in the store methods themselves, that is a real engine signal.

Instrumentation
---------------
Deliberately NO production diff: there are no `stage()` markers anywhere in
`identity/` today, and adding permanent ones before knowing which boundaries
matter is guessing. Instead every public method on the live `IdentityStore`
instance is wrapped in a counting + timing proxy, which yields the exact
per-method call counts (the N+1 evidence) and wall attribution. Peak RSS uses
the `bench_fs_peak_probe.py` convention: a VmRSS sampler thread plus
`ru_maxrss` as a cross-check.

ONE rung per process (same reason as `bench_fs_peak_probe.py`): `ru_maxrss` is
a process high-water mark, so the caller loops over N in separate subprocesses
rather than this script looping internally.

Usage
-----
    uv run python scripts/bench_identity_control_plane.py \
        --n 5000000 --backend sqlite --phase both --out-json ident_5m.json

    # Postgres (CI `services: postgres:16`, so ~zero network -- deliberately
    # flattering to PG, which is the honest way to isolate engine cost):
    uv run python scripts/bench_identity_control_plane.py \
        --n 5000000 --backend postgres --dsn "$GOLDENMATCH_TEST_DATABASE_URL" ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# --- env BEFORE importing goldenmatch (native loader + planner read these) ---
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("_RJEM_MALLOC_CONF", "dirty_decay_ms:1000,muzzy_decay_ms:0")
os.environ.setdefault("GOLDENMATCH_AUTOCONFIG_MEMORY", "0")

REPO = Path(__file__).resolve().parent.parent

try:
    import resource  # POSIX only; absent on Windows (local smoke runs)
except ImportError:  # pragma: no cover -- Windows smoke path
    resource = None  # type: ignore[assignment]


def _vmrss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass  # not Linux -- sampler degrades to 0.0, ru_maxrss still reported
    return 0.0


class Sampler(threading.Thread):
    """Peak-RSS sampler (bench_fs_peak_probe.py convention)."""

    def __init__(self, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0.0
        self._stopev = threading.Event()

    def run(self) -> None:
        while not self._stopev.is_set():
            self.peak = max(self.peak, _vmrss_mb())
            time.sleep(self.interval)

    def halt(self) -> None:
        self._stopev.set()
        self.join(timeout=1)


class StoreProbe:
    """Counting + timing proxy over a live IdentityStore.

    Wraps every public callable attribute in place. The N+1 patterns in
    `apply_batch` are per-METHOD-call, so call COUNT is the primary evidence
    (5M `upsert_record` calls is a finding regardless of what each costs), and
    per-method wall tells you which of them to care about.
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self.counts: dict[str, int] = {}
        self.wall: dict[str, float] = {}
        self._install()

    def _install(self) -> None:
        for name in dir(self.store):
            if name.startswith("_"):
                continue
            attr = getattr(self.store, name, None)
            if not callable(attr):
                continue
            setattr(self.store, name, self._wrap(name, attr))

    def _wrap(self, name: str, fn: Any) -> Any:
        def probed(*a: Any, **kw: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.perf_counter() - t0
                self.counts[name] = self.counts.get(name, 0) + 1
                self.wall[name] = self.wall.get(name, 0.0) + dt

        return probed

    def report(self, top: int = 25) -> list[dict]:
        rows = [
            {
                "method": k,
                "calls": self.counts[k],
                "wall_s": round(self.wall.get(k, 0.0), 4),
                "mean_us": round(1e6 * self.wall.get(k, 0.0) / max(self.counts[k], 1), 1),
            }
            for k in self.counts
        ]
        rows.sort(key=lambda r: -float(r["wall_s"]))
        return rows[:top]

    def reset(self) -> None:
        self.counts.clear()
        self.wall.clear()


def build_clusters(gt_cids: Any, pair_scores: str) -> dict[int, dict]:
    """The cluster partition `apply_batch` consumes, straight from ground truth.

    Shape per `resolve.py:884-958`: {cid: {"members": [row_id...],
    "pair_scores": {(a,b): score}}}.

    `pair_scores` density is a knob because it is the harness's OWN memory
    cost, not the control plane's: QIS clusters are 5 members, so "full" is
    C(5,2)=10 tuples per cluster = 10M dict entries at 5M rows, which would
    confound the RSS number this script exists to measure. "spanning" (k-1
    per cluster) keeps evidence edges present without the O(k^2) tail.
    """
    import numpy as np

    order = np.argsort(gt_cids, kind="stable")
    sorted_cids = gt_cids[order]
    boundaries = np.flatnonzero(np.diff(sorted_cids)) + 1
    groups = np.split(order, boundaries)

    clusters: dict[int, dict] = {}
    for grp in groups:
        if grp.size == 0:
            continue
        cid = int(gt_cids[grp[0]])
        members = [int(x) for x in grp]
        info: dict[str, Any] = {"members": members}
        if pair_scores != "none" and len(members) > 1:
            ps: dict[tuple[int, int], float] = {}
            if pair_scores == "full":
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = sorted((members[i], members[j]))
                        ps[(a, b)] = 0.95
            else:  # spanning
                for i in range(len(members) - 1):
                    a, b = sorted((members[i], members[i + 1]))
                    ps[(a, b)] = 0.95
            info["pair_scores"] = ps
        clusters[cid] = info
    return clusters


@contextmanager
def open_store(backend: str, dsn: str | None, path: str) -> Generator[Any, None, None]:
    """A store guaranteed EMPTY at entry, torn down at exit.

    Postgres gets a freshly CREATEd database per invocation, mirroring
    `tests/_pg_helpers.py`. That is not incidental hygiene -- reusing one
    database across rungs silently corrupted the first 5M profile: SQLite got
    a fresh temp file per rung while Postgres kept the shared DSN, and because
    the QIS generator is PREFIX-STABLE (the first k rows of an n-row draw
    equal a k-row draw), the 5M rung's first 1,000,000 rows were already
    resident from the preceding 1M rung. Those 200,000 clusters were no longer
    brand-new, so they fell off the bulk fast path (resolve.py:915) into
    1,000,000 per-row `upsert_record` calls -- which read exactly like a
    Postgres-specific engine bug and got filed as one (#2894) before the
    arithmetic gave it away.
    """
    from goldenmatch.identity.store import IdentityStore

    if backend != "postgres":
        store = IdentityStore(backend="sqlite", path=path)
        try:
            yield store
        finally:
            store.close()
        return

    if not dsn:
        raise SystemExit("--backend postgres requires --dsn")
    import uuid
    from urllib.parse import urlparse, urlunparse

    import psycopg

    db_name = f"gm_cp_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{db_name}"')
    scoped = urlunparse(urlparse(dsn)._replace(path=f"/{db_name}"))
    store = IdentityStore(backend="postgres", connection=scoped)
    try:
        yield store
    finally:
        store.close()
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')


def run_phase(
    *,
    label: str,
    store: Any,
    probe: StoreProbe,
    clusters: dict[int, dict],
    df: Any,
    dataset: str,
    run_name: str,
) -> dict:
    from goldenmatch.core.bench import bench_capture
    from goldenmatch.identity.resolve import resolve_clusters

    probe.reset()
    sampler = Sampler()
    rss_before = _vmrss_mb()
    sampler.start()
    t0 = time.perf_counter()
    # apply_batch's prep phases carry `stage()` markers as of the #2893
    # follow-up; without a recorder active they no-op, so capture one to get
    # the breakdown of the backend-independent wall that ADR 0064 could only
    # see by subtraction.
    with bench_capture() as bench_rec:
        summary = resolve_clusters(
            clusters=clusters,
            df=df,
            store=store,
            run_name=run_name,
            dataset=dataset,
            # SQLite carries a singleton ceiling warning (store.py:118) and
            # singletons are not what the write path is being judged on here.
            emit_singletons=False,
        )
    wall = time.perf_counter() - t0
    sampler.halt()
    try:
        _bd = bench_rec.to_dict()
    except Exception as exc:  # noqa: BLE001 -- diagnostic only
        _bd = {"_capture_error": repr(exc)[:120]}
    _stage_wall = _bd.get("stage_timings_seconds") or {}
    _stage_rss = _bd.get("stage_peak_rss_kb") or {}

    n_members = sum(len(v.get("members") or []) for v in clusters.values())
    total_store_calls = sum(probe.counts.values())
    rec = {
        "phase": label,
        "clusters": len(clusters),
        "member_rows": n_members,
        "wall_s": round(wall, 3),
        "rss_before_mb": round(rss_before, 1),
        "peak_rss_sampled_mb": round(sampler.peak, 1),
        "peak_over_before_mb": round(sampler.peak - rss_before, 1),
        "store_calls_total": total_store_calls,
        "store_calls_per_member_row": round(total_store_calls / max(n_members, 1), 3),
        "store_wall_s": round(sum(probe.wall.values()), 3),
        "store_share_of_wall": round(sum(probe.wall.values()) / wall, 3) if wall else None,
        "by_method": probe.report(),
        "stage_wall_s": {k: round(v, 3) for k, v in sorted(
            _stage_wall.items(), key=lambda kv: -float(kv[1] or 0))},
        "stage_peak_rss_mb": {k: round(v / 1024.0, 1) for k, v in _stage_rss.items()},
        "summary": {
            k: v for k, v in vars(summary).items() if isinstance(v, (int, float, str, bool))
        },
    }
    if resource is not None:
        rec["ru_maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

    # A profiler that measured NOTHING must fail loudly, never report zeros as
    # if they were a finding -- the "green run measuring nothing" class. This
    # already caught one real bug: without `__row_id__` on the frame,
    # `apply_batch` skips every row (resolve.py:654) and the whole thing
    # reports 0 wall / 0 writes, which reads exactly like "the control plane
    # is free".
    wrote = sum(
        int(getattr(summary, k, 0) or 0)
        for k in ("created", "absorbed_records", "merged", "records_upserted")
    )
    if wrote == 0:
        raise SystemExit(
            f"[identity-cp] phase {label!r} resolved NOTHING "
            f"({n_members} member rows in, 0 writes out, {total_store_calls} store calls). "
            "Refusing to report a zero measurement as a result. Check that the frame "
            "carries __row_id__ and that cluster members are row ids present in it."
        )

    # The cold phase is only "cold" if the store really was empty: every cluster
    # must be brand-new, so `created` must equal the clusters processed. Anything
    # less means residue from an earlier run made some clusters look
    # already-known, which silently pushes them off the bulk fast path
    # (resolve.py:915) and into the per-row branch -- inflating wall and call
    # counts on whichever backend was not isolated. That is exactly how the
    # first 5M profile produced a Postgres-only 200,000-cluster "degradation"
    # that was really a shared-database artifact (filed, then withdrawn, as
    # #2894). Loud failure beats a plausible-looking wrong number.
    if label == "cold":
        expected = sum(1 for v in clusters.values() if len(v.get("members") or []) > 1)
        created = int(getattr(summary, "created", 0) or 0)
        if created != expected:
            raise SystemExit(
                f"[identity-cp] cold phase created {created} identities for "
                f"{expected} clusters -- the store was NOT empty. {expected - created} "
                "cluster(s) resolved against pre-existing records, so this run is "
                "contaminated and its bulk-vs-per-row split is meaningless. "
                "Postgres needs a freshly provisioned database per invocation."
            )
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Identity control-plane wall/RSS profiler")
    p.add_argument("--n", type=int, default=1_000_000, help="row count for this rung")
    p.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    p.add_argument("--dsn", default=os.environ.get("GOLDENMATCH_TEST_DATABASE_URL"))
    p.add_argument("--phase", choices=["cold", "incremental", "both"], default="both")
    p.add_argument("--pair-scores", choices=["none", "spanning", "full"], default="spanning")
    p.add_argument(
        "--incremental-frac",
        type=float,
        default=0.01,
        help="fraction of clusters re-resolved against the populated store",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default="identity_control_plane.json")
    args = p.parse_args(argv)

    sys.path.insert(0, str(REPO / "scripts"))
    import quality_invariant_scale as qis  # noqa: PLC0415

    t0 = time.perf_counter()
    table, gt = qis.generate_with_gt(args.n, seed=args.seed, shape="realistic")
    # `apply_batch` skips any row without `__row_id__` (resolve.py:654), and the
    # QIS generator does not emit one -- the real pipeline stamps it at ingest.
    # Without this the whole run silently resolves NOTHING and reports zeros.
    import pyarrow as pa  # noqa: PLC0415

    table = table.append_column("__row_id__", pa.array(range(table.num_rows), type=pa.int64()))
    t_gen = time.perf_counter() - t0

    clusters = build_clusters(gt, args.pair_scores)
    t_clusters = time.perf_counter() - t0 - t_gen

    out: dict[str, Any] = {
        "meta": {
            "n_rows_requested": args.n,
            "n_rows_actual": table.num_rows,
            "backend": args.backend,
            "pair_scores": args.pair_scores,
            "seed": args.seed,
            "t_generate_s": round(t_gen, 2),
            "t_build_clusters_s": round(t_clusters, 2),
            "note": (
                "Clusters come straight from QIS ground truth -- no dedupe run. "
                "This isolates the control-plane WRITE path; compute is not measured."
            ),
        },
        "phases": [],
    }

    with (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td,
        open_store(args.backend, args.dsn, str(Path(td) / "identity.db")) as store,
    ):
        probe = StoreProbe(store)
        dataset = f"bench_cp_{args.n}"

        if args.phase in ("cold", "both"):
            out["phases"].append(
                run_phase(
                    label="cold",
                    store=store,
                    probe=probe,
                    clusters=clusters,
                    df=table,
                    dataset=dataset,
                    run_name="cold_load",
                )
            )

        if args.phase in ("incremental", "both"):
            # A slice of the SAME clusters, re-resolved against the now-populated
            # store: every cluster overlaps an existing entity, which is exactly
            # the condition that disqualifies the bulk fast path (resolve.py:915).
            keys = list(clusters)[: max(1, int(len(clusters) * args.incremental_frac))]
            subset = {k: clusters[k] for k in keys}
            out["phases"].append(
                run_phase(
                    label="incremental",
                    store=store,
                    probe=probe,
                    clusters=subset,
                    df=table,
                    dataset=dataset,
                    run_name="incremental_pass",
                )
            )

    Path(args.out_json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"[identity-cp] wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
