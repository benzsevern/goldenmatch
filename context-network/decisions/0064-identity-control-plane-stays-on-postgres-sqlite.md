# 0064 — The identity control plane stays on Postgres/SQLite; the cost is not the store

**Status:** accepted (2026-09-07, Ben) • **Measured:** `scripts/bench_identity_control_plane.py`, [run 34154603961](https://github.com/benseverndev-oss/goldenmatch/actions/runs/34154603961) • **Shipped from it:** #2893 (fix in #2895) • **Frame:** [../architecture/one-product-two-engines.md](../architecture/one-product-two-engines.md), decision [0047](0047-one-product-two-engines-architecture.md)

## Context
Standing question, raised repeatedly: is Postgres the right long-horizon store
for the identity control plane? The framing was speed and memory — "Postgres is
heavy and can't do zero-copy" — with a proposed alternative of building a
custom relational database inside the Arrow-native Rust framework, tailored to
ER access patterns (merge/split, provenance, append-only event log).

It had never been measured. Every scale win in the repo's recent history came
from somewhere else (jemalloc page-decay −33% peak RSS, the bucket scorer, the
FS stage profile, blocking selection) — none from storage. So the question was
turned into a measurement before it was turned into architecture.

## Decision
**Keep Postgres/SQLite. Do not build a storage engine.** The measurement does
not support it, on either axis the proposal was aimed at.

At 5M rows, cold load, co-located `services: postgres:16` (network latency
removed on purpose — the honest way to isolate engine cost):

| | postgres | sqlite | gap |
|---|---|---|---|
| wall | 528.5 s | 452.1 s | +17% |
| store wall | 324.7 s | 230.5 s | +41% |
| **non-store wall** | **203.9 s** | **221.6 s** | **~0%** |
| **peak RSS** | **12,568 MB** | **12,556 MB** | **0.1%** |

1. **Roughly 210 s of wall is backend-independent** — Python-side batch prep in
   `apply_batch` (row dicts, record-id derivation, payload hashing) that runs
   before the store is touched. SQLite's figure is marginally *higher* than
   Postgres's, which is the clearest available evidence it has nothing to do
   with the storage engine.
2. **Peak RSS differs by 0.1% across two completely different engines.** Memory
   is entirely Python-side row materialization. A new store would not move it
   at all.
3. Therefore an *infinitely fast* store buys ~40% of wall and **zero** memory.
   A custom RDBMS is a multi-year project aimed at the smaller half of one axis
   and none of the other, and it would make the store the product — which
   contradicts 0047 (backends are replaceable; none is synonymous with
   GoldenMatch) and the North Star (renting storage is what lets us say *point
   it at the Postgres you already run*).
4. **Postgres's real price is 17%** against an embedded in-process database
   with the network already removed. That is the honest number, and it is not
   what makes this slow.

## Consequence
- The Arrow/zero-copy instinct was right about the *technique* and wrong about
  the *target*. The zero-copy opportunity is `apply_batch`'s materialization of
  5M rows into Python dicts — Arrow-shaped work **inside** the control plane,
  not underneath it. That is the sanctioned follow-on, in this order:
  1. `bulk_upsert_records` — 200 s (PG) / 110 s (SQLite), the largest single
     item on both, though already COPY/staged and possibly near its floor.
  2. The ~210 s prep path and the 12.6 GB, both untouched and with obvious
     headroom. Riskier than it looks: `apply_batch` is load-bearing correctness
     code with many branches.
  3. Residual `lookup_entity_ids` gap: 10.4 s (PG) vs 1.9 s (SQLite) post-#2893.
- **One real defect came out of the profile.** #2893: `lookup_entity_ids`
  chunked its IN-list at 900 — `SQLITE_MAX_VARIABLE_NUMBER` (#670) — for *every*
  SQL backend, turning one bulk pre-flight into ~5,556 Postgres round trips.
  50.82 s → 10.40 s after switching Postgres to a single `= ANY(array)`
  parameter. It also exposed that `python_goldenmatch_postgres` had no
  `identity/**` path filter, so the identity store's Postgres backend had never
  had PR-time Postgres coverage.
- **The incremental path is not the disaster it looked like.** It fires a
  consistent 1.2 store calls per member row at every scale (`upsert_record` per
  member, `upsert_identity` per cluster), but costs 125 µs/row against cold's
  140 µs/row. Worth batching eventually; not the reason to change stores.

## The measurement was harder to trust than the architecture
Three separate harness bugs, each of which produced a *plausible, explainable,
wrong* number rather than an obvious failure:

1. The QIS generator emits no `__row_id__`, and `apply_batch` skips every row
   lacking one (`resolve.py:654`) — the first run resolved nothing and reported
   0 wall / 0 writes, which reads exactly like "the control plane is free".
2. Postgres reused one shared database across rungs while SQLite got a fresh
   temp file. Because the QIS generator is prefix-stable, the 5M rung's first
   1,000,000 rows were already resident from the 1M rung — exactly 200,000
   clusters at `ROWS_PER_CLUSTER=5`, which correctly fell off the bulk fast
   path into exactly 1,000,000 per-row writes. Filed as a Postgres engine bug
   (#2894) and withdrawn once the arithmetic matched too well to be a
   coincidence. It also inflated the headline gap from 17% to 25%.
3. A sibling bench (#2633's) exited 0 with 2 of 4 rungs unmeasured.

**Every quantitative claim in this decision post-dates all three fixes.** The
generalisable lesson, and the reason it is recorded here rather than in a PR
description: a profiling harness that measures nothing must *fail*, not report
zeros — and a result that confirms your prior is the one to re-derive from the
other direction before filing it. Each guard now asserts the thing it assumed:
the zero-work check, and the cold-phase check that `created` equals the clusters
processed (verified to fire, not assumed to).
