# Network Updates Log

Newest first. One entry per meaningful change to the network.

## 2026-09-07 -- ADR 0064: the identity control plane stays on Postgres/SQLite
- Added **[ADR 0064](../decisions/0064-identity-control-plane-stays-on-postgres-sqlite.md)**,
  closing the standing "should we build a custom Arrow-native relational store for the
  control plane?" question with a measurement instead of an argument
  (`scripts/bench_identity_control_plane.py`, 5M rows, both backends).
- **The answer is no, on both axes the proposal targeted.** At 5M cold: ~210 s of wall is
  backend-INDEPENDENT Python-side prep in `apply_batch` (SQLite's figure is marginally
  *higher* than Postgres's), and peak RSS differs by **0.1%** across two entirely different
  engines -- 12,568 MB vs 12,556 MB. An infinitely fast store buys ~40% of wall and zero
  memory. Postgres's real price is **+17%** against an embedded in-process DB with the
  network removed.
- **Redirects the Arrow/zero-copy instinct at the right target**: `apply_batch`'s
  materialization of 5M rows into Python dicts -- Arrow-shaped work *inside* the control
  plane, not underneath it.
- Produced one real fix (#2893 / PR #2895: `lookup_entity_ids` was chunking at SQLite's
  900-parameter cap on Postgres, ~5,556 round trips -> 50.82 s, now 10.40 s) and exposed
  that `python_goldenmatch_postgres` had no `identity/**` path filter, so the identity
  store's Postgres backend had never had PR-time Postgres coverage.
- The ADR also records **three harness bugs** found along the way, each of which produced a
  plausible-but-wrong number rather than an obvious failure -- including a Postgres "engine
  bug" (#2894) that was a shared-database artifact and was withdrawn. Every figure in the
  ADR post-dates all three fixes.

## 2026-08-24 -- ADR 0063: generated API reference (Python static, REST OpenAPI, TS deferred)
- Added **[ADR 0063](../decisions/0063-generated-api-reference.md)** + design
  `docs/superpowers/specs/2026-08-24-generated-api-reference-design.md`. The derived-docs battery
  covers capability surfaces exhaustively and the API not at all: **330 exported symbols** across the
  six documented packages (202 of them goldenmatch's) with no generated reference, and three FastAPI
  services with no committed OpenAPI spec -- `docs_render` runs `--disable-openapi` because nothing
  feeds it. Decision: generate the Python side STATICALLY with `griffe` (no importing -- the
  `gen_suite_matrix` lesson, where importing `<pkg>.mcp.server` needs the `[mcp]` extra and flaps
  between a dev box and CI; at 330 symbols behind optional torch/aiohttp/ray/spark deps an
  import-based generator inherits that trap everywhere); scope to `__all__`, one page per package,
  driven by `config_matrix.roster.DOCUMENTED`, wired as a `regen_docs.py` WRITE step. REST: emit
  OpenAPI from the apps and drop `--disable-openapi`. TypeScript: DEFERRED explicitly -- its public
  surface is already gated cross-language by `parity/<pkg>.yaml` + `api_parity`. Four stages, with
  the documented-symbol floor LAST (a ratchet on an unmeasured backlog is red on arrival).

## 2026-07-28 -- ADR 0048: own the string-matching primitives (goldenfuzz, goldenphonetic)
- Added **[ADR 0048](../decisions/0048-own-string-matching-primitives.md)** recording the decision
  to OWN the two third-party string-matching deps as byte-identical, pyo3-free, published products:
  `goldenfuzz` (replaces rapidfuzz -- distances + the full `fuzz.*` composites + extract/cdist) and
  `goldenphonetic` (replaces jellyfish -- soundex/metaphone/nysiis/match-rating, nysiis 6.8x faster).
  Both on PyPI + crates.io; rapidfuzz + jellyfish evicted from every suite RUNTIME dep, retained only
  as workspace test-only oracles. Governing principle: **own it, don't clone it forever** -- remove the
  incumbent from runtime, pin correctness to the ALGORITHM (never `assert x == rapidfuzz`), prove speed
  in benches, keep the competitor only as a port-fidelity oracle in the owning kernel's own suite.
- Doc sweep: README gained an "Owned libraries (standalone)" table; `llms.txt` lists both; goldenphonetic
  graduated from the badge exception into `PYPI_PACKAGES` + the README pepy `?q=` download list.
- Applies [ADR 0047](../decisions/0047-one-product-two-engines-architecture.md)'s "one authoritative
  semantic owner per capability" test to the primitives. Next ownership target: ANN (faiss/hnswlib -> hnsw-core).

## 2026-07-27 -- ADR 0047 amended: conformance v2 (behavioral + temporal)
- Amended **[ADR 0047](../decisions/0047-one-product-two-engines-architecture.md)** after the
  thesis-conformance board hit its floor (10 low, 0 undeclared, T5 silent). The five tenets'
  WORDING is unchanged; two decision TESTS + the audit process are sharpened:
  T1 -> default-routing (a correct-but-unwired kernel is a *latent* second source; the shared
  owner must be the DEFAULT); T2 -> behavioral equivalence on the workload (dedupe F1-neutrality),
  not byte-parity on a fixture -- fixture parity is necessary, not sufficient; T3 -> deferrals
  carry an explicit un-defer trigger and are re-validated (a lifted premise is an OPEN divergence,
  not a low). Process: the audit hunts emergent/behavioral drift + retires vacuous items.
- Grounded in this session's work: the fs-default kernel was byte-parity-proven yet opt-in
  (latent second source, since fixed by the batteries default); the goldenanalysis frame-kernel
  deferral's Arrow premise had silently rotted (#1788); the deferral-provenance weakness went
  vacuous (all deferrals model-backed) yet stayed live.

## 2026-07-24 -- Cross-language phase-handoff conformance harness + published limits
- Added **[ADR 0046](../decisions/0046-cross-language-phase-handoff-conformance.md)**:
  cross-language (Python↔TS) phase-handoff is a MEASURED property, not assumed
  from surface parity. Surface parity (same ops exist) != artifact interop
  (a phase's output round-trips byte-for-byte).
- Shipped a runnable **conformance harness** (Python oracle → TS parity test):
  `tests/parity/cluster-conformance.parity.test.ts` (identical scored pairs →
  identical partition, incl. oversized-cluster MST split with tied edges) and
  `tests/parity/split-run.parity.test.ts` (a REAL half-Python/half-TS pipeline
  reproduces the single-language run; independent all-TS run agreed).
- **Verdict table** published on README, `docs-site/concepts/cross-language-parity`,
  and `llms.txt`: byte-safe (identity graph +crypto, `score→cluster`, split-run,
  cluster JSON, config, memory, fingerprints); tolerance-bounded (string scoring
  4dp — can flip a threshold); divergent (standardize/dates, embeddings,
  controller); Python-only (distributed/VLM/routing). Design note:
  `docs/design/2026-07-24-cross-language-phase-conformance.md`.
- Known limit stated honestly: the split-run's clean independent-run agreement is
  dataset-specific; the next extension is a corrupted-on-threshold split-run to
  hunt the flipping case.

## 2026-07-18 -- goldenmatch 3.5.0: `date` scorer + cross-surface consistency hardening
- Cut **goldenmatch 3.5.0** (docs sweep + release). Headline: the `date` scorer
  (#1858) -- `jaro_winkler` scores unrelated ISO birthdays 0.80+, so `date`
  compares by Damerau-Levenshtein over the canonical digits (typo 0.90 /
  unrelated 0.00), canonical impl in the Rust `score-core` kernel funneled to
  native + pure-Python + TypeScript (byte-identical). No new ADR: it is an
  instance of the established "Rust is the reference, mirror cross-surface"
  scorer pattern, not a new architectural decision. Documented in
  `docs-site/goldenmatch/scoring.mdx` (scorer table + "Date fields" section).
- Also in the release train: fused FS multi-pass + pipeline routing, Postgres
  identity-resolution single-transaction fix (#1886).
- Consistency-hardening wave (post-release, not gating 3.5.0): NE fast/slow-path
  parity (#1888), api_parity/native_symbols made blocking + scorer/transform
  surfaces added (#1889), BUCKET_HASH_SEED single-sourced + native scorer-id
  cross-surface gate (#1890), and the version_consistency gate extended to
  TypeScript (#1885). Same through-line: gates that guarded a cross-surface
  invariant while watching only one surface.

## 2026-07-17 -- Context-network sweep: the FS scale + hardening wave (ADRs 0041-0045)
- Swept the 2026-07-16..18 window (~40 PRs, shipped as **goldenmatch 3.4.0** +
  the suite minor train #1849) that the network had not yet captured. Five new
  ADRs, all Fellegi-Sunter-centric:
  - **0041 -- FS missing-value semantics** (`unobserved` vs `disagree`,
    per-dataset, library-does-not-impose). #1834 established `unobserved`
    (missing = no evidence, EM excludes the field, normalize over observed);
    it regressed historical_50k 0.83->0.33 because that data's missingness is
    informative, so #1851 exposed both modes + `_pick_missing_semantics`
    (auto-config picks `disagree` at >=20% max field null-rate) + env
    `GOLDENMATCH_FS_MISSING`; #1872 declines the native kernel on `disagree`
    (neutral-only). Ride-alongs #1856/#1861/#1835->#1836.
  - **0042 -- Native kernel owns 100% FS coverage** (`fs-core` crate #1869,
    embeddings-native #1871, `date` scorer #1876, exclude-Arc/Arrow #1808).
    numpy retained as the reference-mode fallback + parity oracle; deletion
    deferred. Confirms native IS the person default and no longer silently
    falls to numpy on name scorers.
  - **0043 -- Bucket is the default FS route** (#1810) + FS in
    distributed/chunked/strategy lanes (#1843/#1844), one shared `EMResult`;
    block-size safety #1784/#1790/#1829/#1857. person-1M `MemoryError` -> 139s.
  - **0044 -- Learned-blocking compiler** lowers to `multi_pass` via per-field
    transform chains (#1831/#1841/#1840/#1838). Compiler landed but INERT
    (wiring #1845 closed invalid).
  - **0045 -- quality_gate watches its own surfaces + is required** (#1847/#1877).
    Root cause of the whole wave being painful: historical_50k f1_probabilistic
    0.83->0.33 merged red on `main` because the gate skipped `probabilistic.py`
    and was non-blocking.
- Updated `architecture/fellegi-sunter-splink-parity.md`: native is no longer
  "default OFF" (100% coverage, bucket-default route); the "Splink 3-19x faster"
  caveat's re-bench is now in flight (`bench-er-headtohead.yml`, run 29628869394,
  person 100k/1M native lane).
- Not ADR'd (routine): dep bumps (#1850/#1862/#1866/#1873), CI perf
  (#1865/#1870), timestamp-flake pins (#1853/#1867), the #1849 version train.

## 2026-07-15 -- Embeddings first-class on Fellegi-Sunter (ADR 0040)
- FS-audit batch (#1800/#1801/#1806). `embedding` / `record_embedding` did not
  work on the probabilistic path -- they CRASHED (`score_field` has no embedding
  branch, so both EM training and scalar scoring raised `Unknown scorer`); the
  vectorized path could build embedding matrices but was gated off.
- Fix (#1806): vectorized EM E-step (`comparison_vector(field_sims=...)` +
  `_build_comparison_matrix` precompute via `_embedding_pair_sims`, train<->score
  level parity), un-gate `vectorized_scorer_supported`, `record_embedding`
  branch in `score_probabilistic_vectorized`, `probabilistic_block_scorer`
  forces vectorized for model-backed scorers regardless of
  `GOLDENMATCH_FS_VECTORIZED=0`, and the TUI routes FS through the block-scorer
  selector. `score_field` untouched (embeddings never reach it). NE +
  model-backed still forces scalar (deferred).
- Sibling fixes on the same batch: #1801 (TF `tf_adjustment` now applies on the
  scalar FS path — route parity), #1800 (distributed + chunked lanes raise
  `NotImplementedError` on FS matchkeys instead of silently dropping them).
- Docs: scoring.mdx (embedding-on-FS note), tuning.mdx (`GOLDENMATCH_FS_VECTORIZED`
  scope), backends-and-scale.mdx (FS is single-box only), CHANGELOG `[Unreleased]`.

## 2026-07-15 -- GoldenMatch precision-anchor threshold raise + commit demotion (ADR 0039)
- New default auto-config rule `rule_precision_anchor_threshold_raise`: raises
  the weighted threshold to 0.9 on the precision-collapse shape
  (mass_above_threshold >= 0.95 + name-only weighted matchkey + strong exact
  identity anchor + live TF table + threshold < 0.9). Crafted-fixture precision
  0.009 -> 0.9868 at recall 1.0; NCVR unaffected. Closes #1207 / #1319.
- Commit dynamics (both required; either alone still commits the over-merge):
  the dip gate needs >= 30 scored pairs before a flat dip reads RED
  (`_MIN_DIP_SUPPORT`), and `pick_committed` rank-demotes entries the rule's
  trigger still flags (`demote_suspect`, controller-passed only when the rule
  fired). Ride-along: fixed the `n_rows` shadow that silently disabled the
  `REFUSE_AT_N` >= 100k RED-refuse gate. ADR 0039.
- Same-day rollout ride-alongs: bucket fast path threads
  `MatchkeyField.tf_freqs` so the #1318 TF name downweight now bites on the
  default scoring path (#1782); anomaly diagnostics with prefilled issue URLs,
  `GOLDEN_DIAGNOSTICS=0` to disable (#1791); goldenpipe check stage scans the
  in-memory frame + goldensuite-mcp `clean_and_dedupe` is one in-process
  `goldenpipe.run()` (#1789); `from_splink` recognizes `IS NOT NULL` blocking
  guards (#1793).

## 2026-07-15 — GoldenMatch 3.3.0: FS negative evidence, EM-learned (ADR 0038)
- `negative_evidence` now works on `type: probabilistic` matchkeys: each NE field
  joins `train_em` as a constrained 2-state EM-learned `__ne__<field>` dimension
  (storage-only `[w_fired, 0]` clamp; fires when both present + strictly below
  threshold; `penalty_bits` fixed override). No labels needed — supersedes the
  Wave D deferral. Loud declines: continuous path (both surfaces), TS pipeline,
  fused `derive_from`.
- Splink migration upgrade pass gains the `fan_out` lever (risk-gated NE
  suggestion + `golden_rules.max_cluster_size` tuning from reference clusters;
  calibration lever now NE-aware) plus `--sample-cap` / `--no-measure` /
  `--id-column`.
- `goldenmatch-native 0.1.15` scores NE in the Rust kernels (`FS_SUPPORTS_NE`)
  and the fused kernel scores custom `level_thresholds`
  (`FUSED_FS_SUPPORTS_LEVEL_THRESHOLDS`); older wheels keep the pure-Python
  fallback. `goldenmatch-js 1.3.0` is a full FS-NE mirror (parity pinned
  bit-exact via committed Python-generated fixtures). `golden-suite 0.2.5`
  floors. ADR 0038.

## 2026-07-13 — GoldenFlow 2.1.0: owned auto-detect profile kernel (ADR 0037)
- Zero-config's type-inference *decision* is now an owned `goldenflow_core::profile`
  kernel: `infer_type(values, hint)` on every surface (Polars columnar, Polars-free
  list/dict, `goldenflow-native`, `goldenflow-wasm`/TS) + a fused `profile_column`
  (Path 1) that returns `inferred_type` + null/unique/samples in one Polars-free FFI
  call. Pure-Python/TS refs stay as byte-matched fallbacks; native-first, opt-out
  `GOLDENFLOW_NATIVE=0`.
- Cross-surface byte-parity via `tests/parity/profile_corpus.jsonl` (oracle =
  goldenflow-core). Wired via a new `profile` `_native_loader` component (floor
  `infer_type_list_arrow`).
- Documented as a distinct owned SURFACE (not a `@register_transform` entry, so out
  of the `test_owned_kernel_boundary.py` buckets) in the owned-kernel boundary doc.
  Accepted known edge: pure-TS strip-then-slice vs Python/Rust slice-then-strip on
  the ≤100 sample — a corpus-unexercised follow-up.
- `goldenflow-native 0.27.0` (base floor bumped) / `goldenflow 2.1.0`. ADR 0037.

## 2026-07-13 — GoldenMatch 3.1.0: Arrow-native engine, polars optional

The full train merged (#1720-#1736): D5d bucket scaffolding, the D2s spine
descent, the Frame-lane flip + widening (W-1..7: zero feature declines left),
the endgame arrow ports (A1-A9; bridge ledger 21 -> 6), deep-D2 fused-golden
dual-rep, the zero-polars gate, and D6 (polars -> optional [polars] extra,
3.1.0 lockstep). K1/K2 kernel candidates: measured NO-GO (precompute 1.1s /
13.9% of a 1M wall). Hotspot loop: 1M frame-lane wall 7.6s -> 5.5s, byte-
identical. ADR 0036.

## 2026-07-08 -- GoldenCheck: denial-constraint discovery, Stage 1 (ADR 0035)
- New opt-in discovered-rule family: mines denial constraints `¬(p1 ∧ … ∧ pm)` — if-then /
  cross-tuple invariants like `¬(status=shipped ∧ ship_date<order_date)` — from a single
  table and surfaces the violating rows. NOT in the default scan. Public API
  `discover_denial_constraints(df, ...)` + exported `DenialConstraint`; new
  `goldencheck denial-constraints` CLI + `--denial` flag on `scan` (`--deep` widens Pass-1
  to the full population). Findings surface as `check="denial_constraint"` (WARNING violated
  / INFO strict).
- LOAD-BEARING: sample-then-validate; two evidence passes (row-level exact O(n) + pairwise
  sampled over S²); order-preserving RANK encoding (deliberately NOT `intern_column`, whose
  hash-order breaks `<`/`≤` predicates); null operand ⇒ predicate false; FastDC minimal-cover;
  native pyo3-free `goldencheck-core::dc.rs` evidence kernel gated on `GOLDENCHECK_NATIVE`,
  set/byte-parity with pure Python, measure-first (~1.5–1.8× over a Polars cross-join,
  ~60–96× over pure Python at m=1500 → native default-on). Stage-1 gates: `arity_bound=2`,
  `require_order_comparison=True` (pure all-equality DCs suppressed), self-column cross dropped.
- Stage 1 of a 5-stage program; deferred: cross-table DCs, numeric-threshold literals,
  config/baseline pinning + DC drift, DuckDB/Postgres/WASM/MCP surfaces.

## 2026-07-08 -- GoldenFlow: Polars eviction FUNCTIONALLY COMPLETE (Phase 4, extends ADR 0034)
- The "Great Polars Eviction" (ADR 0034) is done on the transform + I/O surface. `import
  goldenflow` imports no Polars; the public `transform(data, config)` runs the native/Arrow
  substrate by DEFAULT (no `GOLDENFLOW_ENGINE=columnar` opt-in), and ALL 113 transforms are
  columnar (9/9 gaps closed) via a `scalar=`/`scalar_dtype`/`scalar_factory` registry
  mechanism + 3 new columnar op-shapes (multi-input `merge_name`, flag-only
  `initial_expand`, whole-column `category_auto_correct`) + a synthetic-`AsFloat`
  numeric-INPUT path. `transform()` reads `.csv`/`.parquet`/`.xlsx` + any DBAPI connection
  Polars-free; zero-config (`config=None`) is Polars-free too.
- LOAD-BEARING DECISIONS (design doc `docs/design/2026-07-07-phase4-polars-optional-scoping.md`
  §5a): (1) `transform()` is the Polars-free primary, `transform_df(pl.DataFrame)` the
  optional Polars-backend adapter (tautological — needs Polars to hold a pl.DataFrame);
  (2) CSV zero-config OWNS its inference (profiles columns as text, `"01234"` stays a zip)
  — an intentional `pl.read_csv` divergence, dates-style. New extras `[polars]`/`[parquet]`;
  `goldenflow-native` republished 0.26.0 (adds `format_f64` + AsFloat). Weight win measured
  ~185 MB installed / ~35 MB wheel.
- REMAINING: only the 2.0 major that drops `polars` from base deps -> `[polars]` (fully
  de-risked: `[polars]` extra + `goldenflow_nopolars` CI lane staged, polars-absent
  validated).

## 2026-07-06 -- GoldenFlow: fused columnar apply, Pillar-1 execution fusion, default-on (ADR 0034)
- First real step of the "Great Polars Eviction": fuse a column's run of owned no-arg
  total string→string kernels (25: text/email/name) into ONE native Arrow pass instead
  of crossing the Python/Polars/Arrow boundary once per transform. Parity by construction
  (byte-identical output + audit trail); generic over offset width because Polars exports
  LargeUtf8. Shipped `goldenflow-native 0.12.0` (fused kernel on PyPI) + `goldenflow 1.14.0`;
  `GOLDENFLOW_FUSED_APPLY` default-ON (opt-out `=0`).
- HONEST VERDICT (measured, `bench-goldenflow-fused`): wall 1.07–1.27× (config-dependent,
  *diluted* by compute-heavy kernels) but **peak RSS −22% at scale** — the durable win.
  Positioned as a memory play, not a speed play. See ADR 0034 for the two "measure beats
  intuition" lessons + the two CI footguns (YAML startup failure; stale rust-cache).

## 2026-07-06 -- GoldenAnalysis: cutover waves + the cut-vs-fixture rule + Wave 1b deferral (ADR 0033)
- Extended the `analysis-core` cutover across GoldenAnalysis and wrote down **when a
  cutover is the right tool vs a cross-surface parity fixture**: cut only for MUSCLE with
  a clean `Float64Array`-shaped boundary. Cutovers: numeric reductions `mean`/`min`/`max`
  (#1472) and `cluster_size_histogram` (#1478, anti-drift not speed) — all three surfaces
  (Python native + WASM); `_GATED_ON` now 9 primitives.
- Everything else got a **data-driven parity fixture** (byte-identical copy in both
  packages, Python + TS lock tests): frame-kernel equality semantics (`-0.0`/`NaN`/null,
  #1481), `quality.rollup`, and `regressions` (#1482) — trivial-compute and/or object-shaped
  inputs with no clean Rust boundary. The frame-kernel adversarial fixture **caught and
  fixed a real bug**: TS `duplicateRowRatio` conflated `NaN` and null.
- **Wave 1b (WASM for the frame kernels) consciously deferred** — no clean WASM boundary
  (Arrow-specific interning vs JSON-string keys), speculative browser payoff. Revisit only
  on a measured real workload.
- Discipline: Python ground truth + a `node` mirror of the TS impl on adversarial inputs
  *before* writing each fixture. Decision: ADR
  [0033](../decisions/0033-goldenanalysis-cutover-waves-and-parity-fixtures.md).

## 2026-07-05 -- GoldenFlow: compiled zero-Python DuckDB extension + all surfaces gated (ADR 0032)
- New surface `goldenflow-duckdb`: a compiled Rust DuckDB **loadable extension**
  linking `goldenflow-core` directly (no CPython in the process). 74 UDFs
  `goldenflow_<kernel>` = essentially the whole single-record transform surface;
  parity proven by threading the same shared `identifiers_corpus.jsonl` through
  real in-process DuckDB. Portable to **DuckDB >= 1.3.0** (stable C API v1.2.0,
  proven by a version-sweep), released `goldenflow-duckdb-v0.1.1` as 5 per-platform
  zips. Distinct from the pre-owned-kernel `duckdb/goldenmatch_duckdb/goldenflow.py`
  (which dispatches the Python registry per value).
- **Lockstep seam closed:** every `goldenflow-core` consumer is now a required
  `ci-required` lane — `rust`, `python_goldenflow_fallback`, `wasm_flow`,
  `rust_pgrx`, `goldenflow_duckdb`, `native_flow`. A core change can't merge with
  any surface's parity red; the committed corpus reds the gates on un-propagated
  drift.
- Decision: ADR [0032](../decisions/0032-goldenflow-duckdb-compiled-extension.md).
  Gotchas (filename→init-symbol coupling that shipped a broken v0.1.0, C-API vs
  DuckDB version, scalar NULL-propagation) in `packages/rust/extensions/CLAUDE.md`.

## 2026-07-04 -- GoldenFlow Wave D sweep (part 1): url, numeric, categorical migrated to owned kernels
- Three more existing transform families migrated to the owned-kernel
  pattern (not new additions -- registry stays at 92): `url_normalize` /
  `url_extract_domain`; the numeric parsers `currency_strip` /
  `percentage_normalize` / `to_integer` / `comma_decimal` /
  `scientific_to_decimal` plus the numeric-array ops `round` / `clamp` /
  `abs_value` / `fill_zero`; and the categorical family `boolean_normalize` /
  `gender_standardize` / `null_standardize` (fully owned) plus
  `category_standardize` / `category_from_file` (logic/data split -- the
  caller-supplied variant->canonical mapping stays in Python/TS, only the
  shared key-derivation is a Rust kernel). Same cross-surface pattern as
  prior waves (native + WASM/TS + pure-Python fallback), Rust is the
  reference implementation.
- **Behavior change (reference-mode, resolved in Rust's favor):** numeric
  `round` now uses round-half-away-from-zero (was Python's round-half-to-even
  banker's rounding); the five numeric string parsers return null on
  unparseable input on both Python and TS (TS parsers previously passed
  through unparsed input).
- Versions: goldenflow 1.7.0 -> 1.8.0 / npm 0.7.0 -> 0.8.0 / goldenflow-native
  0.5.0 -> 0.6.0.

## 2026-07-04 — GoldenFlow Wave D1: email transform family migrated to owned kernels
- The email transform family (`email_lowercase`, `email_normalize`,
  `email_extract_domain`, `email_validate`) is now backed by owned Rust
  kernels in `goldenflow-core`, cross-surface (native + WASM/TS +
  pure-Python fallback), byte-parity to the Rust oracle. Existing transforms
  migrated to native-first dispatch, not new additions -- registry stays at
  92. Versions: goldenflow 1.7.0 / npm 0.7.0 / goldenflow-native 0.5.0.

## 2026-07-04 — GoldenFlow Wave B: name_transliterate + name_script i18n name kernels
- Extension of ADR [../decisions/0031-goldenflow-reference-mode-identifiers-wasm.md](../decisions/0031-goldenflow-reference-mode-identifiers-wasm.md)
  (no new ADR needed) — two new owned i18n name kernels: `name_transliterate`
  (deterministic Unicode-to-ASCII fold via an explicit curated map, NOT NFD, for
  guaranteed cross-surface byte parity) and `name_script` (dominant-script
  detection via Unicode code point ranges). Same cross-surface pattern as
  Waves 0/A (native + WASM/TS + pure-Python fallback, byte-parity to the
  goldenflow-core Rust oracle), both `auto_apply=False`.
- Registry moves 90 → **92**. Versions: goldenflow 1.6.0 / npm 0.6.0 /
  goldenflow-native 0.4.0.

## 2026-07-04 — GoldenFlow Wave A: SWIFT/ABA/IMEI identifier families
- Extension of ADR [../decisions/0031-goldenflow-reference-mode-identifiers-wasm.md](../decisions/0031-goldenflow-reference-mode-identifiers-wasm.md)
  (no new ADR needed) — three new owned checksummed/structural identifier
  families: `swift_validate`/`swift_format` (SWIFT/BIC), `aba_validate` (US ABA
  routing), `imei_validate` (IMEI Luhn). Same cross-surface pattern as Wave 0
  (native + WASM/TS + pure-Python fallback, byte-parity to the goldenflow-core
  Rust oracle), all `auto_apply=False`.
- Registry moves 86 → **90**. Versions: goldenflow 1.5.0 / npm 0.5.0 /
  goldenflow-native 0.3.0.

## 2026-07-03 — GoldenFlow Wave 0: core split + reference-mode + checksummed identifiers + cross-surface WASM
- New ADR [../decisions/0031-goldenflow-reference-mode-identifiers-wasm.md](../decisions/0031-goldenflow-reference-mode-identifiers-wasm.md):
  GoldenFlow adopts the suite-standard `-core`/`-native`/`-wasm` layout — new
  pyo3-free `goldenflow-core` OWNS the phone + identifier kernels, `native-flow`
  becomes a thin PyO3 shim, and a new `goldenflow-wasm` wasm-bindgen crate
  surfaces the identifier kernels to the edge.
- Loader moves to reference-mode (`_has_symbol` + `_COMPONENT_SYMBOLS` +
  `_FALLBACK_ONLY`, mirroring goldenmatch): `GOLDENFLOW_NATIVE=auto` runs native
  wherever a kernel symbol exists; `_GATED_ON` retained only as documentation.
  `phone_validate` is `_FALLBACK_ONLY` (native symbol implements the wrong spec).
- 10 new checksummed-identifier transforms (cc/iban/isbn/ean/vat, all
  `auto_apply=False`) take the registry 76 → **86**. EU VAT checksum is bounded to
  DE + IT this wave (structural-only for the other 25 prefixes).
- Cross-surface: TS keeps pure-TS default + opt-in `enableWasm()`; one shared
  oracle corpus (`goldenflow-core`) asserted byte-identical across native /
  WASM-TS / pure-Python via `gen_identifiers_corpus.py --check` + a `wasm_flow`
  CI lane. Versions: goldenflow 1.4.0 / npm 0.4.0 / goldenflow-native 0.2.0.
- Architecture node [../architecture/goldenflow-native-kernel.md](../architecture/goldenflow-native-kernel.md)
  updated with a Wave 0 header note.

## 2026-06-28 — Perceptual image pHash: cross-platform determinism + wasm/TS
- New ADR [../decisions/0030-perceptual-cross-platform-determinism.md](../decisions/0030-perceptual-cross-platform-determinism.md):
  the image pHash computed its DCT basis cos() at runtime, so it was libm-fragile
  across platforms/surfaces (committed golden test passed on Linux CI, FAILED on
  Windows native; a wasm build was 6 bits off). Fix: a COMMITTED 8x32 DCT table
  that the Rust kernel (native + wasm) AND the Python reference both read (one
  generator, bit-exact in both languages) -> byte-identical everywhere. New opt-in
  `goldenmatch/core/perceptual-wasm` TS subpath (phashImage / hamming), byte-exact
  parity-gated + a `perceptual_wasm` CI drift lane. Image only; radial + audio stay
  Python/native (non-cos fragility). Fixture rebased 2 borderline phash values.

## 2026-06-28 — GoldenGraph TS: bitemporal store shipped (0.2.0)
- The store deferred in [0029](../decisions/0029-goldengraph-wasm-ts.md) now ships:
  `appendBatch` / `asOf` / `history` over a portable JSON `Snapshot` (the kernel's
  `store_*` ops). Parity fixtures cover the append → as_of → history flow; 9 total
  goldengraph parity cases. Also: the `goldenprofile → goldengraph` composition
  helpers (`resolutionFromClusters` / `mentionsFromProfiles`) landed (#1306).
- Gotcha recorded in 0029: wasm-bindgen maps the kernel's i64/u64 params to BigInt;
  the public API takes `number` and converts at the boundary.

## 2026-06-28 — GoldenGraph (KG engine) on the TS/WASM surface (v1: graph+query)
- New ADR [../decisions/0029-goldengraph-wasm-ts.md](../decisions/0029-goldengraph-wasm-ts.md):
  the GoldenGraph knowledge-graph engine gets a TS/JS surface via `goldengraph-wasm`
  — the second fold after goldenprofile (ADR 0028), same pattern.
- New **standalone** `packages/typescript/goldengraph` npm package: pure-by-default
  base + opt-in `goldengraph/wasm`; query fns REFUSE (throw) until enabled. **v1 = the
  4 graph+query ops** (`buildGraph`/`neighborhood`/`seedsByName`/`communities`);
  the bitemporal store (`store_append/as_of/history`) is DEFERRED (heaviest surface).
- Zero runtime deps (the resolution input is a `{mention:entity}` map / `["native",…]`,
  not goldenprofile's `Resolution` — composition is a future adapter).
- Enabling change: `goldengraph-core` → `graph-core { default-features = false }`
  (build-time gating; same accurate rationale as 0028).
- CI: `goldengraph_wasm` path-filter + drift guard in the `typescript` lane; publish
  wired-unfired (`publish-goldengraph-js.yml`). No Python cross-parity (native API is
  OO, not the JSON boundary; not in the py matrix — see issue #1304). Plan:
  [../../docs/superpowers/plans/2026-06-28-goldengraph-wasm-ts-parity.md](../../docs/superpowers/plans/2026-06-28-goldengraph-wasm-ts-parity.md).

## 2026-06-28 — GoldenProfile (Virtual Fingerprint) on the TS/WASM surface
- New ADR [../decisions/0028-goldenprofile-wasm-ts.md](../decisions/0028-goldenprofile-wasm-ts.md):
  the GoldenProfile Virtual Fingerprint engine (cross-document entity resolution,
  kernel ADR [0023](../decisions/0023-semantic-signature-virtual-fingerprint-engine.md))
  now has a TS/JS surface — the last unfinished leg of its one-kernel-many-surfaces
  matrix (Python/`-native` + C/`-cabi` + `-wasm` already existed).
- New **standalone** `packages/typescript/goldenprofile` npm package: pure-by-default
  base entry (zero wasm bytes, edge-safe) + an opt-in `goldenprofile/wasm` subpath that
  loads the kernel. Follows the **healer precedent** (opt-in-or-absent, no pure-TS
  resolver), not the score/analysis acceleration track. `resolveProfiles()` REFUSES
  (throws) when the backend isn't enabled — never a fake empty Resolution.
- Enabling change: `graph-core`'s `arrow` became an **opt-in default-on feature** so
  `goldenprofile-core` links on `wasm32`; native/pgrx/datafusion compile byte-identically.
- Cross-surface contract corrected to **partition + edge set + scores(4dp)**, NOT
  byte-ordering (the kernel's cluster/edge ordering is `HashMap`-seed nondeterministic).
  Canonical, idempotent fixtures gate both the TS-wasm and Python-native parity tests.
- CI: `goldenprofile_wasm` path-filter + a drift guard in the `typescript` lane; publish
  wired-but-unfired (`publish-goldenprofile-js.yml`). Plan:
  [../../docs/superpowers/plans/2026-06-28-goldenprofile-wasm-ts-parity.md](../../docs/superpowers/plans/2026-06-28-goldenprofile-wasm-ts-parity.md).

## 2026-06-27 — Config-suggestion healer on the TS/WASM surface
- New ADR [../decisions/0027-healer-wasm-ts.md](../decisions/0027-healer-wasm-ts.md):
  the healer (config-suggestion engine) now runs on the TypeScript/JS surface via
  WebAssembly. The existing pyo3-free `suggest-core` kernel is compiled to a
  `suggest-wasm` cdylib (mirroring the autoconfig `-core → -wasm → TS` precedent):
  arrow is feature-gated and an arrow-free `suggest_from_json` entry is shared by the
  Python native path and the wasm path (single source of truth, **zero Python change**).
- Full default-pipeline parity: wired into `dedupe({suggest, heal})` with the free
  trigger + verify path + bounded heal loop, on every TS surface (core / CLI / MCP
  `review_config` → MCP tool count 44→45 / A2A `review_config` skill). Opt-in
  `enableSuggestWasm()` (the `[native]` analog) registers the kernel; with no backend
  every surface is gracefully empty (`[]`/undefined, never throws). Kill-switch
  `GOLDENMATCH_SUGGEST_ON_DEDUPE`.
- One cross-surface golden-vector contract: the `suggest-core` BLESS oracle authors the
  fixtures; the TS parity test and a Python native cross-surface test run the SAME
  fixtures (TS == Rust == Python). CI: a `suggest_wasm` path filter gates a drift-guard
  step in the `typescript` lane + `suggest-core`/`suggest-wasm` steps in the `rust` lane.
- Docs: new [config-suggestions.mdx](../../docs-site/goldenmatch/config-suggestions.mdx)
  page (+ nav), TS README/CLAUDE.md healer sections, llms.txt notes.

## 2026-06-26 — Healer wired into the default pipeline (advisory, every surface)
- New ADR [../decisions/0026-healer-default-pipeline.md](../decisions/0026-healer-default-pipeline.md):
  the healer (`review_config`) is now part of the default `dedupe_df` pipeline as a
  **two-stage cost-bounded advisory surface**. (1) A *free* controller trigger reads the
  run's existing `postflight_report` (RED/YELLOW health or scoring dip ≥ 0.05) — no kernel
  call. (2) Only on a trigger does a new artifacts-in `suggest_from_result(result, df)`
  reuse the run's `scored_pairs`/`clusters` to attach raw candidate suggestions to
  `result.suggestions` (no re-run). Healthy result = byte-identical timing (no-op parity).
- Opt-in deeper paths: `dedupe_df(suggest=True)` (expensive verified gate),
  `dedupe_df(heal=True)` (full apply-and-re-run loop → `result.heal_trail` + healed config).
- Present on every surface: Python, CLI (`--suggest`/`--heal` + free default hint), MCP
  (`review_config` tool), A2A (`review_config` skill), REST (`GET /suggest`), web
  (`GET /api/v1/suggest`), TUI (Suggestions tab). Graceful-degrade without `goldenmatch[native]`
  (attaches nothing, never raises). Kill-switch `GOLDENMATCH_SUGGEST_ON_DEDUPE=0`.
- Stacked on the verify-gate flip (#1272): the default surface only delivers wins because
  the gate now keeps precision fixes ([0025](../decisions/0025-healer-verify-gate-proxy.md)).

## 2026-06-26 — Healer self-verify gate: default proxy -> cohesion (#1272) + healing-loop docs
- New ADR [../decisions/0025-healer-verify-gate-proxy.md](../decisions/0025-healer-verify-gate-proxy.md):
  the healer (`review_config`) self-verify gate's default health proxy flipped legacy -> cohesion
  (`cohesion_min_edge_cap50`), chosen by a proxy bake-off (`scripts/suggest_quality/bakeoff.py`)
  against the F1 oracle. Closes the raw-vs-live gap: suggester-gym live recovery 0.151 -> 0.543
  (== raw ceiling), zero net-negatives on real perturbations. Supersedes the earlier
  "cohesion fails / escalate to pseudo-labels" finding.
- Established the **healing-loop thesis** across the docs (new `/goldenmatch/config-suggestions`
  page, README "The healing loop" sections, overview, project-definition product-loop section,
  tuning healer-knobs): zero-config -> returned config -> healer suggests self-verified tweaks ->
  apply -> improve -> repeat. Healer stays opt-in (`goldenmatch[native]`).

## 2026-06-24 — Auto-config quality harness + probabilistic-routing lever (#1216/#1226/#1254)
- New ADR [../decisions/0024-autoconfig-probabilistic-routing.md](../decisions/0024-autoconfig-probabilistic-routing.md):
  a decision-kernel **quality harness** (`scripts/autoconfig_quality/`, report/gate/bless)
  that runs auto-config over a committed corpus (real benchmarks FEBRL3 / NCVR /
  historical_50k + synthetic anchors) and diffs the resulting decisions against a
  pinned baseline — 2-tier (host-independent config signals, hard-gated; slow F1 +
  attribution) with a CI `quality_gate` job. It exists to make a kernel change's
  quality impact measurable in one run and to **nominate the next lever on evidence**.
- The first lever it nominated: **probabilistic routing**. A dual-strategy scorecard
  column (`f1` default vs `f1_probabilistic` forced Fellegi-Sunter) measured a +0.36
  F1 gap on historical_50k. Auto-config now routes to FS when
  `GOLDENMATCH_AUTOCONFIG_ROUTE_PROBABILISTIC=1` (default-off) and the shape is
  probabilistic (no surviving exact matchkey on `{identifier,email,phone}` + ≥2 fuzzy).
  Flag-on proof: historical_50k 0.466→0.829, ncvr 0.983→0.990 routed; febrl3 +
  anchor_person_match unchanged (strong key blocks). Zero regression; default-flip
  deferred to a broader sweep. Flag in `docs-site/goldenmatch/tuning.mdx`.

## 2026-06-20 — Corpus-dedup throughput benchmark + perf gate (#1086)
- Closes the training-data-dedup epic's "defend the throughput claim" item. New ADR
  [../decisions/0022-corpus-dedup-throughput-benchmark.md](../decisions/0022-corpus-dedup-throughput-benchmark.md):
  a `scripts/bench_corpus_dedup/` harness (pluggable FineWeb/C4/Wikipedia/offline
  corpus adapters + injected ground-truth near-dups so recall is measurable on real
  text) and a **deterministic** per-PR `throughput-gate` (machine-independent cost —
  candidate-pairs / reduction-ratio / measured-recall — vs a committed baseline, so it
  can't flake on shared-runner wall-clock).
- **Published number:** ~1,192 docs/sec · 3.6 MB/sec on a 70k-doc FineWeb slice at
  ~0.43 measured LSH recall (docs/sec is auto-config-bound at this scale; raw sketch
  dedup ≈7,800/s). Documented in `docs-site/goldenmatch/tuning.mdx`.
- **The bench did its job:** it proved the #1083 throughput tier was overstated at
  scale (validated only on 10-row tests). Walking it up on real FineWeb surfaced+fixed
  four at-scale bugs — the GoldenCheck O(N²) quality scan on doc text, web-text
  mis-classification (tier refused), the ≥100k RED-config refusal, and (open) an O(N)
  survivorship `iter_rows` ceiling above ~70k. datatrove recall-parse + the 100k+
  ceiling are tracked follow-ups. SHIPPED #1134/#1139/#1142/#1144/#1147.

## 2026-06-19 — Semantic SimHash near-dup blocking (#1082 Phase B)
- Extends the sketch tier (ADR [../decisions/0020-minhash-lsh-sketch-tier.md](../decisions/0020-minhash-lsh-sketch-tier.md))
  with a *semantic* near-duplicate path. New pyo3-free SimHash (random ±1
  hyperplane) LSH kernel over embedding vectors, exposed as blocking
  `strategy="simhash"`: `SimHashKeyConfig` (`column`, `num_planes=256`, `seed=0`,
  `threshold | num_bands`, `model`, re-exported from the top level) +
  `core.simhash_blocker.SimHashLSHBlocker`, which embeds a text column and buckets
  cosine-similar vectors. **Auto-config routes a text corpus to `simhash`** when an
  embedder is reachable (`inhouse_embedding_available()` or a configured provider),
  else falls back to lexical `lsh` (#1082 Phase A) — `dedupe_df(corpus)` picks the
  semantic near-dup path automatically when embeddings are available. SimHash
  catches the semantic paraphrases that lexical MinHash/LSH (#1081, ~0.21 on QQP)
  misses. New native component `simhash`: shipped available but NOT in `_GATED_ON`
  (reachable via `GOLDENMATCH_NATIVE=1`, same posture as `sketch`/`pprl_bloom`),
  tuned by `GOLDENMATCH_NATIVE_SKETCH_RAYON_MIN_ROWS` (shared with the MinHash
  kernel). Cross-language byte parity via golden vectors (pure-Python reference +
  Rust `sketch-core` + pure-TS port; the kernel is f64, the semantic blocker is
  Python-primary since TS has no real embedder). Measured: a synthetic recall gate
  (`num_planes=256`/`num_bands=32` → recall 1.0 / reduction 0.86 on cosine≥0.89
  variants) + a QQP lexical-vs-semantic A/B (`bench-lsh-recall.yml --method
  semantic`). Treated as an addendum to ADR 0020, not a new decision (it reuses the
  Approach-A kernel/host split and parity-by-construction contract). Docs
  (blocking/configuration/tuning Mintlify) + CHANGELOGs (py + ts) updated. Part of
  the dedup epic #1080.

## 2026-06-19 — goldenmatch-kg: drop-in ER for KG frameworks (#1127)
- New decision [../decisions/0021-goldenmatch-kg-integrations.md](../decisions/0021-goldenmatch-kg-integrations.md):
  a new standalone `goldenmatch-kg` package drops goldenmatch in as the
  entity-resolution stage of three KG frameworks — neo4j-graphrag (a real
  `GoldenMatchResolver` subclassing `BasePropertySimilarityResolver`), LlamaIndex
  PropertyGraphIndex (an additive name-canonicalizing `TransformComponent`), and
  Graphiti (a post-ingestion `propose_entity_merges` pass; no public seam). A
  framework-agnostic `resolve_entities` core (zero-config `dedupe_df` → groups +
  canonical maps) is the only goldenmatch-touching code; base-free decision helpers
  are locally testable, the framework bindings are import-gated. The package is
  EXCLUDED from the uv workspace so its three heavy framework extras never enter the
  main `uv.lock` (the `goldenmatch[native]` sync-break footgun); it ships its own
  `goldenmatch-kg` CI lane (core + a fresh-venv-per-framework matrix). The
  per-framework lift is read off ER-KG-Bench (neo4j 0.322→0.969, graphiti
  0.379→0.969), not re-scored. Shim tests inject a deterministic stub for the
  goldenmatch decision because a zero-config merge on a ~3-row toy frame is
  version-flaky (1.30 merged "Apple"/"Apple Inc"; 2.2 did not). Docs site
  (`goldenmatch/kg-integrations`) + root `CLAUDE.md` updated.

## 2026-06-19 — MinHash/LSH sketch tier (#1081)
- New decision [../decisions/0020-minhash-lsh-sketch-tier.md](../decisions/0020-minhash-lsh-sketch-tier.md):
  phase 1 of the training-data dedup epic (#1080). A new pyo3-free
  `goldenmatch-sketch-core` Rust crate (shingling → MinHash → banded LSH) exposed
  on Python (pyo3 native + pure-Python fallback) and TypeScript, with a
  `MinHashLSHBlocker` + `BlockingConfig(strategy="lsh")`. Cross-language byte
  parity via a hand-rolled hash + shared golden vectors. Approach A: the kernel
  does per-record sketching, the host language groups `(band, bucket)` into blocks.
  The `sketch` native component ships available but not gated-on (reachable via
  `GOLDENMATCH_NATIVE=1`), like `pprl_bloom`. Measured recall: an always-on
  synthetic gate (recall 0.978 / reduction 0.989) plus a Quora-QQP bench job
  (`bench-lsh-recall.yml`). Tuning docs + CHANGELOGs (py + ts) updated.

## 2026-06-18 — goldenmatch 2.1.0 released + immutable-releases publish flow
- Released **goldenmatch 2.1.0** (PR #1060, tag `v2.1.0`). Shipped since v2.0.0:
  correlated survivorship — lock-step field groups + `anchor`/`allow_fill` (#1047/#1055),
  GroupProvenance surfaced end-to-end across lineage/explain/MCP/review-queue (#1053),
  chunked PPRL trusted-third-party linkage (`PPRLConfig.chunk_size`, ~9-14× lower peak
  memory, #1054), native-dispatch telemetry on the result object (`result.native`,
  #1048/#957), plus collective ER (0018) and the Sail WCC perf change. The CHANGELOG
  `[Unreleased]` section was incomplete (2 of 8 entries) and was finalized to `[2.1.0]`.
- New decision [../decisions/0019-immutable-releases-publish-flow.md](../decisions/0019-immutable-releases-publish-flow.md):
  `publish-goldenmatch.yml` now owns the GitHub Release lifecycle (push a `v*` tag → build →
  PyPI → sign+attest → **draft** release with assets attached → publish). Fixes the v2.1.0
  `Cannot upload asset ... to an immutable release` red. **New SOP: cut a release by pushing
  the tag only**, documented in `processes/development-workflow.md` + root `CLAUDE.md`.
  SHIPPED PR #1063.
- Docs: Mintlify `goldenmatch/pprl.mdx` + `goldenmatch/python-api.mdx` (chunk_size,
  `result.native`); `configuration.mdx` already covered the survivorship/provenance surfaces.

## 2026-06-16 — Collective ER via neighborhood similarity (Phase 0+1)
- New decision [../decisions/0018-collective-er-neighborhood-similarity.md](../decisions/0018-collective-er-neighborhood-similarity.md)
  (linked from the discovery hub).
- **`run_graph_er(propagation_mode="relational")`** (new `core/collective.py`) blends
  attribute similarity with neighbor-cluster overlap (Jaccard/Adamic-Adar) and iterates
  to a synchronous (Jacobi) fixpoint — the Bhattacharya-Getoor collective-ER algorithm.
  Candidate set = attribute-blocked ∪ capped co-neighbor pairs. Measured pairwise F1
  ~0.66 (attribute-only) → ~0.87 (collective) across seeds 7/8/9 on a fixture built so
  attributes alone can't resolve homonyms; the old flat-boost `additive` mode is shown
  actively harmful (~0.05). Existing `additive`/`multiplicative` modes unchanged
  (default still `additive`; output byte-identical). SHIPPED PR #1030 (Phase 0+1);
  Phase 2 (negative evidence) + Phase 3 (learned weights) deferred to their own plans.
- Docs: `docs-site/goldenmatch/python-api.mdx` (Graph ER section) + the package CHANGELOG.

## 2026-06-15 — arrow 55→59 + pyo3 0.28 across the pyarrow crates
- **Shipped (#1003 Phase 1, #1005 Phase 2):** `graph-core`, `native`, `analysis-native`,
  `goldencheck-native`, `native-flow` all moved to arrow 59 + pyo3 0.28 (native also numpy
  0.28). `datafusion-udf` deliberately left on arrow 58 / datafusion 53 — insulated via the
  arrow-free slice-kernel boundary, so the datafusion 53→5x major bump was declined as
  disproportionate.
- **Why it wasn't a one-liner:** arrow 59's `pyarrow` feature pins pyo3 0.28.2, forcing a
  pyo3 0.23→0.28 bump (+ numpy 0.28, matched to arrow's pyo3 — NOT the latest 0.29, since
  arrow lags one pyo3 version). The code migration was mechanical: `allow_threads`→`detach`,
  `downcast`→`cast`.
- **Architecture node updated:** `architecture/sql-native-extensions.md` now reflects
  graph-core on arrow 59 (the datafusion-udf insulation boundary is now 58↔59, mechanism
  unchanged). #999 (the original graph-core-only dependabot bump) closed/superseded.

## 2026-06-15 — Single-kernel-collapse R2: first slice shipped + value recalibration (R5 retired)
- **R2 first slice SHIPPED (#980, merged):** field scoring brought under the reversible
  `GOLDENMATCH_NATIVE` gate. `_native_field_matrix` had preferred the `score-core` kernel
  ungoverned (ignored the flag), so `GOLDENMATCH_NATIVE=0` did not force the pure path — a
  latent reversibility gap. Now gates on `native_enabled("field_scoring")` + `field_scoring`
  in `_GATED_ON` (default unchanged, output byte-identical, reversible + telemetered).
  Verified 138/138 + 11/11 parity in-env.
- **Recalibration (recorded in ADR 0016):** Python is already kernel-default; TS can't flip
  (edge-safety keeps pure-TS the default + fallback); the pure paths are load-bearing
  fallbacks, so **R5 "decommission" is retired** — the realistic end-state is "kernel =
  governed canonical fast path; pure = fallback; parity harnesses stay as kernel-vs-pure
  gates." The reward is narrower than the original "delete N reimplementations" framing.
- **Two cheap follow-ups investigated → no code needed:** (1) the **SQL-binding equivalence
  gate already EXISTS and runs live in CI** (`test_datafusion_ffi_udf.py` asserts the
  DataFusion FFI UDFs == rapidfuzz at 1e-6; `ci.yml` installs `datafusion` + builds the
  wheel + hard-imports) — so kill-criterion (1)'s SQL surface was never pending, a correction
  to the R1 verdict; all three bindings are gated. (2) **`pprl_bloom` HELD default-off** —
  26 parity tests green + 7.08× faster byte-identical, BUT `bloom.rs` uses an unguarded
  `par_iter()` (the #688 futex-park class) on the EPYC runner that won't provision; flipping
  it default-on is an unvalidatable #688 risk. Needs the #692-style threshold guard first.

## 2026-06-15 — Single-kernel-collapse R1 COMPLETE: GO to R2 (two documented residuals)
- **Kill-criterion (2) cleared — all four JS targets GREEN in CI.** `r1-kernel-js-targets.yml`
  run [#27518182208](https://github.com/benseverndev-oss/goldenmatch/actions/runs/27518182208)
  (workflow_dispatch on `main`): node / deno / browser (chromium) / **workers (workerd)** each
  build `score-wasm` and reproduce the frozen pure-TS reference at 4dp. node/deno/browser via the
  ONE universal base64 loader (no per-target hack); **workers via the build-time CompiledWasm
  module** — mandatory because workerd bans runtime WASM codegen, a supported per-target load
  *mechanism* not a hack. The first dispatch red'd on a host-vite `import-analysis` parse of the
  static `.wasm?module` import (0 tests); fixed in #977 (plain `.wasm` import + `assetsInclude`).
- **R1 overall: GO.** All four kill-criteria now have positive tracer evidence — (1) pure==kernel
  4dp across Python + TS/WASM, (2) cross-JS-target WASM PASS, (3) all-platform abi3 wheels PASS for
  the four mainstream arches (R1-B), (4) kernel measured not-slower (1.44×). Two carried residuals,
  both infra/mechanism not code: macOS-x86_64 wheel build-only; Workers needs the build-time module
  form. R2 (first scorer collapse behind a reversible default-flip flag) may proceed — additive,
  parity-gated, one reversible flag per step. ADR 0016 holds the verdict.

## 2026-06-14 — Single-kernel-collapse R1 Workstream A: cross-JS-target WASM (kill-criterion 2)
- **Shipped as additive, workflow_dispatch-only infra** — flips no default (the TS
  WASM path stays opt-in via `enableWasm()`; pure-TS stays default + fallback),
  deletes nothing, touches no default scoring path. New
  `.github/workflows/r1-kernel-js-targets.yml`: four jobs (node / deno / browser /
  workers) that build `score-wasm` once and run the SAME pure-TS-vs-kernel 4dp
  equivalence assertion under each JS runtime, each writing a per-target PASS/FAIL to
  `$GITHUB_STEP_SUMMARY`. Mirrors `r1-kernel-wheels.yml` (Workstream B).
- **Universal loader (the A1 decision): base64-INLINE (Option i)**, implemented behind
  the existing opt-in seam as `enableWasm({ universal: true })`. `build_wasm.sh` now
  also emits a gitignored `score_wasm_base64.js`; `goldenmatch-wasm-runtime` gained
  `decodeWasmBase64` + a `wasmBase64` LoadOption. No fetch/fs/`import.meta.url` asset
  resolution — the only edge-safe-everywhere path. Cost: base64 ~+33% over the raw
  `.wasm` (115,155 B → a 153,540-B string). Default `enableWasm()` unchanged; default
  users load zero wasm bytes. Note:
  [../../docs/superpowers/notes/2026-06-14-wasm-universal-loader.md](../../docs/superpowers/notes/2026-06-14-wasm-universal-loader.md).
- **Per-target harnesses** reuse the spike's assertion via a runtime-agnostic
  `kernel-equivalence-core.ts` + frozen `fixtures/pure-ts-reference.json`. In-env:
  **node / deno / browser RAN-GREEN** (deno + browser via the universal base64 loader,
  no per-target hack; browser in real chromium, fault-injection-verified). **workers:
  PENDING-RUN** — the pool runs in real workerd in-env and surfaced a genuine
  constraint (**workerd bans runtime WASM codegen** — `instantiate`/`Module`-from-bytes
  both throw "Wasm code generation disallowed by embedder"), so Workers needs a
  build-time CompiledWasm `.wasm` import; the green run is pending the dispatched
  `workers` job (vitest-4 host-transform vs pool-0.16 resolver friction in-env). ADR
  0016 gained an R1-A evidence section (per-target table + the Workers finding); the R1
  plan's Workstream A section now points at the workflow.

## 2026-06-14 — Single-kernel-collapse R1 Workstream B: all-platform wheel + #688 perf-cliff gate
- New [../architecture/single-kernel-collapse-R1-plan.md](../architecture/single-kernel-collapse-R1-plan.md)
  (the R1 go/no-go plan of record), linked from [../discovery.md](../discovery.md).
  Scopes both R1 workstreams — A (WASM across Node/browser/Workers/Deno =
  kill-criterion 2; scoped, harness pending) and B (all-platform abi3 wheels + the
  #688 perf cliff = kill-criterion 3; THIS change) — each with goal / verification
  design / per-target evidence / kill checkpoint, the "one generalized gate run in
  more places" + `kernel-targets` report idea, the additive/no-default-flip-in-R1
  rule, and the honestly-PARTIAL expected outcome (kernel-default where the
  platform reliably supports it + a thin pure fallback elsewhere).
- **Workstream B shipped as additive, workflow_dispatch-only infra** — flips no
  default, deletes nothing, touches no product/Rust source. New
  `.github/workflows/r1-kernel-wheels.yml`: a `wheels` matrix (linux x86_64
  manylinux 2_28 / linux aarch64 / macOS arm64 / macOS x86_64 cross / windows x64)
  that builds the abi3 wheel with the same SHA-pinned `maturin-action` + manifest
  as `publish-goldenmatch-native.yml`, then on a clean Python 3.11 runs the
  equivalence gate in **REQUIRE-KERNEL** mode (FAIL not skip if the just-built
  wheel is absent) + the bench with **ASSERT-NOT-SLOWER**; and a `perf_cliff` job
  on `ubuntu-latest-xlarge` (the 8-core AMD EPYC shape #688 wedged on; parameterized
  via `cliff_runner`) that runs the per-pair bench AND `scripts/bench_issue_688.py`,
  asserting no rayon-LockLatch futex-park regression. `fail-fast: false`;
  per-platform PASS/FAIL + wall-ratio to the step summary.
- **Two backward-compatible script flags added** (default behavior preserved — both
  skip-on-absent + exit 0 with neither flag): `--require-kernel` (already present on
  `check_kernel_equivalence.py`) and a new `--require-kernel`/`--assert-not-slower`
  pair on `scripts/bench_kernel_levenshtein.py` (exit 1 if the kernel is more than a
  small tolerance slower than pure — a #688-class cliff). ADR 0016's Go/No-Go
  evidence section gained an R1 row: kill-criterion #3 + the #688 cliff are now
  PROBED BY `r1-kernel-wheels.yml`, status PENDING-RUN.

## 2026-06-14 — Single-kernel-collapse feasibility spike (R0 + levenshtein tracer)
- New [../decisions/0016-single-kernel-collapse-spike.md](../decisions/0016-single-kernel-collapse-spike.md)
  (ADR, linked from [../discovery.md](../discovery.md)) + two architecture nodes:
  [../architecture/single-kernel-collapse-inventory.md](../architecture/single-kernel-collapse-inventory.md)
  (R0 duplication census, ranked) and
  [../architecture/single-kernel-collapse-roadmap.md](../architecture/single-kernel-collapse-roadmap.md)
  (R0–R5 stages + the two hard constraints + the four additive/reversible rules).
- **Fully-additive spike** — changes no default path, deletes nothing, flips no
  flag. New compare-only artifacts: `scripts/check_kernel_equivalence.py` (a
  generalizable `pure==kernel` 4dp gate, scorer-name parameterized),
  `scripts/bench_kernel_levenshtein.py` (measure-first wall-clock), and
  `packages/typescript/goldenmatch/tests/spike/kernel-equivalence.test.ts`
  (pure-TS vs WASM, skip-guarded on the artifact). None imported by a default path.
- **Verbatim 4-item kill criterion** recorded. In-env evidence on the levenshtein
  tracer: Python `pure==native` bit-identical (max diff 0.0 over 2028 pairs; also
  jaro_winkler 5.5e-17, token_sort 0.0); TS `pure==WASM` 4dp GREEN (built the WASM
  artifact in-env, ran un-skipped); kernel 1.44x faster than pure (per-pair, its
  pessimal shape). PENDING + load-bearing: cross-JS-target WASM (Node-only
  verified) and all-platform abi3 wheels (the #688 class). Existing scorer/parity
  tests stay green (TS 63, Python 138); `check_docs_consistency.py` PASS.
- New [../decisions/0015-goldenmatch-2.0-deprecation-cut.md](../decisions/0015-goldenmatch-2.0-deprecation-cut.md);
  linked from [../discovery.md](../discovery.md). Records the scope decision (cut
  exactly the four prepared items; keep the universal scorer `list[tuple]` path;
  `build_clusters` stays public as a frames-backed adapter; `:hash:` removal is
  asymmetric — un-fingerprintable rows keep it) and the verification lesson.
- **2.0.0 is live** on PyPI + the MCP registry (first backwards-incompatible major;
  `1.30.0 → 2.0.0`, merged `93193ccb`). Removed: the `:hash:` identity bridge +
  `GOLDENMATCH_IDENTITY_ID_SCHEME`, the `GOLDENMATCH_CLUSTER_FRAMES_OUT` gate +
  legacy dict cluster path, the `cheapest_healthy` / `_scale_aware_backend` shims.
- **Docs synced to 2.0 reality** (this follow-up PR): dropped the two removed flags
  from `docs-site/goldenmatch/tuning.mdx` (kept `GOLDENMATCH_SAIL_IDENTITY_ID_SCHEME`);
  rewrote the `identity-graph.mdx` back-compat/removal section (the `:h1:`→`:hash:`
  dual-candidate fallback + once-per-process warning are gone); added a `2.0.0`
  `<!-- README-callout -->` to the CHANGELOG and regenerated both READMEs via
  `scripts/sync_readme_callouts.py`. Migration guide: `migrating-to-v2.mdx`.

## 2026-06-14 — v1.0->2.0 evolution docs + reusable docs-sweep workflow (PR #946)
- Two new Mintlify Guides pages: `docs-site/goldenmatch/v1-to-v2.mdx` (the full
  1.0->2.0 capability arc) and `v1-vs-v2.mdx` (the at-a-glance comparison tables),
  cross-linked to each other and to the existing `migrating-to-v2.mdx`.
- New `.claude/doc-surfaces.md` — the repo's documentation-surface inventory (docs
  site + nav, READMEs + the CHANGELOG-sourced callout sync, version lockstep, this
  context network, examples, llms.txt/server.json). It is the thin repo-local list
  the global `rollout-docs-sweep` skill delegates to, so the end-of-rollout doc pass
  (the one this entry is part of) becomes a repeatable checklist instead of an ad-hoc
  prompt. The skill greps the repo for every removed/renamed symbol first.
- **Durable fact:** when a rollout ships, sweep ALL doc surfaces, not just the
  CHANGELOG/migration guide. The 2.0.0 cut shipped with stale `tuning.mdx` /
  `identity-graph.mdx` / READMEs until a follow-up caught them; the inventory + skill
  exist so that does not recur.

## 2026-06-13 — Discoverability / public-surface audit (PR #883)
- New workstream node [../planning/discoverability.md](../planning/discoverability.md);
  linked from [../discovery.md](../discovery.md) and promoted in
  [../planning/roadmap.md](../planning/roadmap.md) as an adjacent arc.
- An accuracy-first, value-forward pass over every pre-install surface. **README:**
  fixed DBLP-ACM `97.2→96.4`, dropped the `DQBench 95.30` headline badge (competitor
  ceiling, not a GM score — keep 91.04 in benchmarks only), `478→~940` tests,
  `36+→50+` MCP tools, Identity-Graph-v2-is-a-feature-shipped-in-v1.15 (not package
  v2.0); led the accuracy story with beats-Splink + surfaced GoldenAnalysis/WASM.
- **`llms.txt` family is now complete:** added the missing **goldenanalysis** +
  **infermap** files and a **suite-level root `/llms.txt`**; corrected the stale A2A
  skill count `10`/`12` → **`31`** (the agent card's `_SKILLS` grew via an MCP-parity
  pass) and the ~500K-cap performance framing → the verified 100M run; retargeted all
  archived-repo cross-links to the live monorepo paths.
- **Registry + citation:** PyPI keywords were behind npm (added `splink`,
  `record-linkage`, `fuzzy-matching`, `pprl`, … across the 6; `golden-suite` on all);
  new root **`CITATION.cff`**. **GitHub About** description + topics refreshed
  (`+splink`, `-negative-evidence`).
- **Archived sibling repos redirected** (their About now points to the monorepo).
- **Three durable facts recorded in the node:** (1) the homepage "what's new" block is
  single-sourced from `<!-- README-callout -->` markers in goldenmatch's CHANGELOG via
  `scripts/sync_readme_callouts.py` (`--check` is a CI gate) — edit the CHANGELOG, not
  the README; (2) the docs site is **Mintlify**, which auto-serves `/llms.txt` +
  `/llms-full.txt` at `docs.bensevern.dev` (the repo-root file is the GitHub/raw
  supplement); (3) an **archived repo is fully API-read-only** (`gh repo edit` → HTTP
  403), so editing its About needs unarchive → edit → re-archive.
- **Open / handed off:** the GitHub social-preview image (manual, no API) and external
  awesome-list PRs (`awesome-mcp-servers` / `Awesome-Entity-Resolution` /
  `awesome-data-quality` — entries drafted, pending go-ahead).

## 2026-06-13 — Opt-in WASM acceleration arc (TypeScript) — #878/#879/#880/#881
- New [../architecture/wasm-acceleration.md](../architecture/wasm-acceleration.md)
  and [../decisions/0014-opt-in-wasm-acceleration.md](../decisions/0014-opt-in-wasm-acceleration.md);
  linked both from [../discovery.md](../discovery.md).
- The TS packages now optionally reach the same pyo3-free `*-core` Rust kernels
  via WebAssembly — opt-in, pure-TS stays the default + fallback, edge-safe,
  `.wasm` built in CI (never committed). Two cores shipped:
  - **#878** `score-core` → goldenmatch `scoreMatrix` (`enableWasm()`; jaro_winkler/
    levenshtein/exact at first).
  - **#880** `analysis-core` → goldenanalysis `histogram`/`quantile`
    (`enableAnalysisWasm()`) **+** extracted the shared `goldenmatch-wasm-runtime`
    workspace package (byte loader + generic enable skeleton + registry) that both
    consumers ride.
  - **#879** aligned the hand-rolled pure-TS scorers with rapidfuzz (the parity
    prerequisite): codepoint iteration, Winkler `>0.7` boost, floored transposition
    `t//2` — empirically settled as integer-vs-float halving, not bit-parallel
    matching (0/50000 vs rapidfuzz incl. non-BMP).
  - **#881** added `token_sort` WASM coverage (new `score-core::token_sort_normalized_ratio`,
    distinct from the pinned un-normalized `score_one(2)`) + validated the bundled
    dist artifact path (defensive multi-location copy; flipped the benches to a
    dist-path gate, which caught a real 1M-element `histogram` `Math.min(...vals)`
    stack overflow).
- **Measure-first parked `graph-core`** without building it (UnionFind is one O(N)
  step among several in `buildClusters`, marshaling N pairs is O(N) →
  boundary-bound); `fingerprint-core` / `goldencheck-core` stay parked by design.

## 2026-06-12 — #855 goldencheck TS port: module parity + hardened golden harness (#873/#874)
- New [../decisions/0013-goldencheck-ts-parity-hardening.md](../decisions/0013-goldencheck-ts-parity-hardening.md);
  added the `#855` subsection to [../planning/surface-hardening.md](../planning/surface-hardening.md).
- **#873** ported the goldencheck TS gaps the 2026-06-11 audit found: 2 profilers
  (`freshness`, `fuzzy_values`), 4 relations (`approx_duplicate`, `approx_fd`,
  `composite_key`, `functional_dependency`), and the `validate` MCP tool —
  registries now 12 column profilers / 9 relations / 18 MCP tools, each mirroring
  the Python **fallback** (native kernels stay Python-only by design).
- **#874** hardened `tests/parity/parity.test.ts`: it now asserts confidence (4 dp)
  + affected_rows and FAILS on a missing manifest/golden, where it previously
  checked only (column, check, severity) and skipped silently. Goldens were
  regenerated on a clean `ubuntu-latest` runner (`regen-855-parity-goldens.yml`,
  artifact-download) because the dev box OOMs on Polars.
- **The hardening immediately caught a pre-existing bug.** TS `TemporalOrderProfiler`
  used `new Date(s)`, which parses bare integers (`"7"`) as dates, so it fired
  `temporal_order` on integer column pairs Python never flags. Gated TS
  `tryParseDate` on `YYYY-MM-DD` to match Python's `str.to_date('%Y-%m-%d')`; all 6
  parity cases now match Python byte-for-byte. **#855 CLOSED.**
- **By design:** `freshness` is unit-test-only (the CSV-roundtrip harness reads
  dates as `Utf8`, so Python's date-gated profiler can't fire through it).

## 2026-06-12 — FS block-scoring perf + the "native is slow" red herring (PR #869)
- New [../decisions/0012-fs-block-scoring-perf.md](../decisions/0012-fs-block-scoring-perf.md);
  added a perf section + corrected the "3-19x faster" framing in
  [../architecture/fellegi-sunter-splink-parity.md](../architecture/fellegi-sunter-splink-parity.md).
- **The bake-off's "Splink 3-19x faster" measured GM's NUMPY path, not native** —
  it never set `GOLDENMATCH_FS_NATIVE`, and probabilistic mode doesn't refuse on a
  missing kernel. Added a `gm_prob_native` bake-off column (native built +
  `score_block_pairs_fs` asserted in CI): **native ≈ numpy, no wall change.** The
  wall is per-block fan-out (historical_50k: 31,735 blocks, 79% ≤8 rows, ~222k tiny
  FFI calls), not scoring math — so the Rust kernel can't move it.
- **Three output-identical optimizations** on the numpy path, each gated by a
  fixed-`em_result` pair-set diff (200,058 pairs byte-identical), NOT the cluster
  hash (pipeline is non-deterministic ±3 clusters run-to-run): value-dedup (−32%),
  block-batching into shared S×S matrices (−48%, native calls 222k→4.3k), batch
  row-cap 512→256 (−20%). **historical_50k 86.5s → 24.6s (−72%) local.** All three
  CI-green on PR #869.
- Also refreshed [../../docs/er-vendor-comparison.md](../../docs/er-vendor-comparison.md)
  to v1.30.0 (refdata, identity graph, Splink-parity flip) earlier in the same PR.
- **Flagged, not fixed:** EM-sampling cluster-count nondeterminism (±3) and the
  pre-optimization bake-off table (re-bench pending) are recorded in 0012.

## 2026-06-11 — #844 FINISH LINE: 100M validated, per-group scoring fixed, default flipped (#864/#867)
- Updated [../architecture/distributed-wcc.md](../architecture/distributed-wcc.md)
  + [../decisions/0011-distributed-wcc-randomized-contraction.md](../decisions/0011-distributed-wcc-randomized-contraction.md)
  from "specs shipped, operator-deferred" to VALIDATED + default-flipped. **#844 CLOSED.**
- **The binding 100M run is done.** Self-provisioned 5-node `e2-standard-16` GCP
  cluster, 100M synthetic phase-5 dataset in GCS: full recall-complete e2e in
  **554.5 s (9.2 min, under the 30-min kill), 20,000,000 clusters recovered
  exactly, driver RSS 0.36 GB**, no head-wedge / no Ray deadlock. The WCC alone
  cleared a 200M-edge graph in 266 s in isolation.
- **The e2e wall was per-group scoring, NOT the WCC.** `_score_colocated_groups`
  looped `group_by` + a full per-partition kernel call per ~5-row group (~20M
  fixed-overhead calls at 100M; 0 of 64 score-tasks finished in 25 min). #864
  vectorizes it — score the whole partition once (the `bucket` backend already
  groups by the blocking key); parity-tested. That single change made the e2e viable.
- **#864 (merged)** also fixed auto-config `DuplicateError: __row_id__` on a
  `__row_id__`-carrying input (`_add_row_ids` guard) and gave the e2e bench an
  explicit-config + `allow_red_config` path (it always auto-configured before,
  which is slow + RED-degenerate at 100M).
- **#867 (open, reviewable)** flips `GOLDENMATCH_DISTRIBUTED_BLOCK_SHUFFLE` default
  `0→1` + adds `_assert_scratch_shared_if_multinode` (multi-node + node-local WCC
  scratch → raises instead of silently diverging).
- Deferred/optional: (b) project-to-scoring-columns-before-shuffle (a wide-record
  shuffle win, not needed for viability).

## 2026-06-11 — TS parity: refdata name scorers + autoconfig blocking (#857, from the #856 audit)
- Extended the parity workstream node
  [../planning/surface-hardening.md](../planning/surface-hardening.md):
  a new "Fixtures rot silently — the #856/#857 lesson" subsection under the
  parity-fixture methodology, plus the #857 entry.
- **The #856 audit found real cross-language drift sitting in `main`:** the
  TS `typescript` CI lane runs only on TS path changes, so a pure-Python
  autoconfig change leaves the committed parity vectors stale and the TS
  test green against a fixture that no longer reflects Python. The
  controller-stoppoint fixture had drifted (`ensemble` where Python now
  emits the refdata name scorers + evolved multi-pass blocking).
- **#857 (merged) closed it by porting, not pinning.** The edge-safe TS core
  gained `given_name_aliased_jw` (alias-aware JW) and `name_freq_weighted_jw`
  (Census surname-IDF-weighted JW), the first/last-name auto-config refine
  (`refineNameScorer`, last-before-first; `multi_name` unrefined), and a
  faithful port of `build_blocking`'s selection (exact gate at
  `cardinality_ratio ≤ 0.5` on the exact pool only; secondary-name passes).
  Scope grew twice (regen forced the 186KB surname table; the residual red
  was a separate blocking-evolution gap). Both refdata tables are generated
  TS modules synced from the Python source (`scripts/sync_ts_refdata.mjs`)
  and drift-guarded; numeric parity is pinned by Python-computed values in
  `tests/parity/scorer-ground-truth.test.ts` (4dp).
- **Two durable methodology guards recorded:** generate-and-drift-guard
  bundled data (never hand-copy), and pin scorer parity to Python ground
  truth rather than a TS self-mirror (a self-mirror passes even if both
  sides diverge from Python).
- Deferred: refdata transform packs + geo/date blocking branches. Follow-up
  **#860**: TS `buildWeightedMatchkey` still drops `nullRate>0.5` name
  columns while the blocking path now keeps them (why `sparse_people` stays
  loose-shape). Docs-site: `goldenmatch/typescript.mdx` (scorer list/table +
  Python-comparison row) and `goldenmatch/reference-data.mdx` (TS-parity note).

## 2026-06-10 — Distributed WCC for #844: randomized contraction + recall-complete Phase 5 (#851/#852)
- New nodes: [../architecture/distributed-wcc.md](../architecture/distributed-wcc.md)
  + [../decisions/0011-distributed-wcc-randomized-contraction.md](../decisions/0011-distributed-wcc-randomized-contraction.md).
- **Problem (#844):** the Phase-5 distributed pipeline under-merged at scale. PR
  #845's opt-in block-shuffle co-locates duplicates but makes components cross
  partitions, which the per-partition `local_cc_assignments` Union-Find
  under-merges. The two existing distributed WCCs both die at 100M:
  `two_phase_wcc` driver-collects + runs a cpython-loop UnionFind (head-wedge);
  `distributed_wcc` deadlocks Ray's streaming executor on iterative joins.
- **Fix (both specs SHIPPED):** Spec 1 (PR #851) = `randomized_contraction_wcc`
  (Bögeholz–Brand–Todor 2018, arXiv:1802.09478) — relational, chain-robust,
  O(log|V|) rounds, no driver UF, per-round parquet checkpoint to dodge the
  deadlock; pure-Polars reference gated vs `scipy.csgraph`. Spec 2 (PR #852) =
  wires it into `_run_phase5_pipeline` (block-shuffle on -> distributed WCC, off ->
  `local_cc`; same predicate the scorer uses) via a new `algorithm` kwarg on
  `build_clusters_distributed`; join + golden tail unchanged (shared contract).
  Opt-in (`GOLDENMATCH_DISTRIBUTED_BLOCK_SHUFFLE=1`), **default unchanged**.
- **Two un-locally-testable Ray Data join rules now recorded** (distinct-named keys
  + `ReadParquet` inputs — both surfaced as `ArrowInvalid` on the CI ray lane).
  The `distributed` job `timeout-minutes` went 20 -> 30 to fit the new blocking gate.
- **Deferred (operator):** the binding multi-node 100M run + the default-flip (need
  a BYO Ray cluster; `GOLDENMATCH_DISTRIBUTED_WCC_SCRATCH` must be a `gs://` prefix).
  Parallel to the [Sail tier](../architecture/sail-tier.md) (the Spark-Connect track
  that retires Ray); whichever binds 100M first is go-forward. Mintlify scale page
  (`docs-site/goldenmatch/backends-and-scale.mdx`) updated with the recall-complete
  path.

## 2026-06-10 — publish-containers flake hardening (ghcr buildkit mirror, #846)
- New decision: [../decisions/0010-publish-containers-ghcr-mirror.md](../decisions/0010-publish-containers-ghcr-mirror.md).
- **Audit:** `publish-containers` went red ~1 run in 18 over 30 days (11 fails,
  6 different packages) — every one a transient registry timeout, never a code
  bug. Dominant: `setup-buildx` pulls `moby/buildkit:buildx-stable-1` from Docker
  Hub *anonymously*; 7 legs pulling in parallel each main push race into Docker
  Hub's shared-runner-IP throttle → `context deadline exceeded`. Secondary: ghcr
  502s + GHA-cache blob copy errors at `Build and push`.
- **Fix (PR #846, merged):** a prereq `mirror` job republishes buildkit + binfmt
  into `ghcr.io/<owner>/{buildkit,binfmt}` once per run (retried); the legs pull
  the helper images from ghcr via `setup-buildx driver-opts:` / `setup-qemu
  image:` (ghcr login moved ahead of buildx). Docker Hub off the hot path: 7
  unguarded parallel pulls → 1 retried read. Native retry-once twins
  (`continue-on-error` + `outcome=='failure'`, no third-party action) backstop
  the residual ghcr/cache blips; `publish` still runs on a stale ghcr copy if
  `mirror` flakes.
- **Verified on `main`:** run `27284102426` — 8/8 jobs green, zero retry twin
  fired (the mirror eliminated the Docker Hub pulls outright, not just retried
  them). Operational detail recorded in root `CLAUDE.md` (`## publish-containers
  flakes`); a red leg is cosmetic (content-addressed, self-heals next push).

## 2026-06-09 — Rust test-coverage arc: make the tests real, then measure (#827/#830/#832)
- New nodes: [../architecture/rust-test-coverage.md](../architecture/rust-test-coverage.md)
  (per-crate testing map + the measured baseline) and
  [../decisions/0009-rust-test-coverage.md](../decisions/0009-rust-test-coverage.md).
- An audit found the Rust tree's CI claims were partly fictional. Closed the real
  gaps: standalone `graph-core`/`score-core` tests now run (#827); `native` got 18
  Rust unit tests behind an `extension-module` feature-gate (#827); the pgrx graph/
  fingerprint surface is psql-asserted vs a real `CREATE EXTENSION` (#827); the
  `bridge` went from **6 silent self-skipping no-ops to 42 real marshalling tests**
  (#830, install goldenmatch into the embedded interpreter + a `REQUIRE_PY` gate);
  and a `cargo-llvm-cov` `rust_coverage` job posts a per-crate baseline (#832).
- Two load-bearing facts recorded so they aren't re-litigated: **`cargo pgrx test`
  is a structural dead-end** for `goldenmatch_pg` (schema-gen broken → psql smoke
  instead), and **`native`'s 26% measured coverage is an artifact** (it's
  Python-parity-tested; llvm-cov only sees `cargo test`). Fixed the now-stale
  `cargo pgrx test`/`pg_test` claim in
  [../architecture/sql-native-extensions.md](../architecture/sql-native-extensions.md).
- Verdict: no compelling remaining Rust gap; the structural work is done and now
  measurable + regression-guardable.
## 2026-06-09 — FS auto-config v2: GoldenMatch now BEATS Splink on accuracy (#823)
- Updated the architecture node + decision:
  [../architecture/fellegi-sunter-splink-parity.md](../architecture/fellegi-sunter-splink-parity.md)
  (new "Accuracy arc — beating Splink" section + softened "Where Splink still
  leads": Splink no longer leads on PII accuracy),
  [../decisions/0008-fellegi-sunter-splink-parity.md](../decisions/0008-fellegi-sunter-splink-parity.md)
  (appended an accuracy-arc update; Status line bumped).
- **The probabilistic (Fellegi-Sunter) auto-config now beats Splink head-to-head**
  (#823, "FS auto-config v2", default-ON; kill-switch
  `GOLDENMATCH_FS_AUTOCONFIG_V2=0` restores legacy byte-identically). Scope is the
  probabilistic auto-config path only (`auto_configure_probabilistic_df` /
  `build_probabilistic_matchkeys`); weighted/DQbench + zero-config `dedupe_df` are
  untouched. Four levers: (1) admit `dob`/date columns as a `levenshtein`
  discriminator (v1 discarded them); (2a) drop redundant person-name composites
  when atomic given+family exist; (2b) low-cardinality fuzzy floor;
  (3) `_diversify_probabilistic_blocking` — *additive*, recall-positive blocking
  diversification onto orthogonal stable keys (date-year + postcode/zip);
  (4) admit description (title) + multi_name (authors) as `token_sort` (lifts the
  DBLP-ACM venue-only mega-match — 0.003 → 0.377, still recall-bound). #821 built
  the shared head-to-head evaluator (`scripts/bench_er_headtohead`) the claim rests
  on.
- **Measured (pairwise F1, shared evaluator) — deterministic as of #829:**
  historical_50k (Splink's flagship) 0.647 → 0.778 vs Splink 0.757; febrl3
  0.983 → 0.991 vs 0.965; synthetic_person 0.972 → 0.998 vs 0.996; dblp_acm
  0.003 → 0.377 (Splink skips it). GM also wins historical_50k at the cluster
  level (B-cubed F1 0.844 vs 0.789). Full bake-off:
  `docs/benchmarks/2026-06-09-splink-bakeoff.md`.
- **#829 (determinism fix):** the original #823 numbers rested on a
  non-deterministic EM training-pair sample — three invocations of the identical
  GM-prob path gave historical_50k 0.805 / 0.779 / 0.643 on one CI run. #829 sorts
  blocks by their stable `block_key` before the seeded shuffle; post-fix three
  harnesses agree within 0.002. The earlier `dblp_acm = 0.879` was a lucky draw
  that does not reproduce (deterministic 0.377). For bibliographic data use the
  *weighted* path (0.964 on DBLP-ACM), not probabilistic.
- **Honest framing preserved:** these are *pairwise* F1 under one shared harness.
  The often-cited ~0.97 Splink historical_50k number is a *cluster*-level metric,
  not exhaustive pairwise; a local diagnostic ran Splink 4.0.16 and it scores
  ~0.75 pairwise here (recall-bound — 5156 clusters, mean size ~10, no field
  exceeds 0.60 recall → ~0.93 pairwise ceiling for any engine). Claim is
  "matches/beats Splink on every dataset Splink scores on the same evaluator,"
  not "0.97 pairwise." Splink is also 3-19x faster on these datasets.

## 2026-06-08 — Fellegi-Sunter → Splink parity (+ EM perf, scale-out, vendor reposition)
- New architecture node + decision:
  [../architecture/fellegi-sunter-splink-parity.md](../architecture/fellegi-sunter-splink-parity.md),
  [../decisions/0008-fellegi-sunter-splink-parity.md](../decisions/0008-fellegi-sunter-splink-parity.md).
- **The `type: probabilistic` (Fellegi-Sunter) matchkey went from accuracy-competitive
  scorer to a Splink-class probabilistic-linkage engine** (PR #800, Phases 0–4 +
  the 3c bench): model lifecycle (`save_json`/`load_json` + `model_path`),
  supervised m from labels (`estimate_m_from_labels` + review/memory adapters),
  match-weight waterfall (`explain_pair_fs`, surfaced in `explain`/lineage),
  posterior calibration, label-driven threshold/accuracy analysis (`evaluate
  --threshold-sweep`), and scale-out on the shared `score_buckets` path (Phase 3a)
  + an opt-in native Rust kernel (Phase 3b, `GOLDENMATCH_FS_NATIVE=1`). The
  scale-out bet held: only one greenfield kernel function; everything else reused
  the existing bucket/Ray/DataFusion substrate.
- **The scale gate exposed the real bottleneck.** The 6M FS bench was
  `train_em`-bound, not scoring-bound (the native kernel had already cut
  `bucket_score` to ~14 s). `_sample_blocked_pairs` enumerated every within-block
  pair (`O(Σ size_i²)`, ~140M tuples) before sampling 10K; fixed to a
  block-stratified early-exit (PR #803, 13.7× `train_em`, ~100 s off the 6M wall,
  peak RSS halved). A separate fix corrected the *bench's* ground truth from a
  star to the entity clique (PR #802) — F1 was 0.825 only because the harness was
  scoring true matches as false; corrected to 1.000.
- **Measured 6M / `bucket` / 16c-64GB:** numpy 288.5 s / native 162.6 s (was 269 s),
  11.3 GB peak RSS, F1 1.000. Recorded in `docs/scale-envelope.md`.
- Docs-site: `goldenmatch/scoring.mdx` FS section rewritten (parity surface +
  scale + corrected DBLP-ACM 0.968, dropping the stale 57.6%-recall artifact);
  `reference/vendor-comparison.mdx` Splink row repositioned (FS feature parity
  closed; Splink retains distributed-1B+ and interactive-charting edge).
- Also corrects the goldencheck-integration node: #798 (quality-gated review)
  shipped.

## 2026-06-07 — GoldenCheck Arrow-native expansion + GoldenCheck→GoldenMatch integration
- Two new architecture nodes + one decision:
  [../architecture/goldencheck-native-kernel.md](../architecture/goldencheck-native-kernel.md),
  [../architecture/goldencheck-goldenmatch-integration.md](../architecture/goldencheck-goldenmatch-integration.md),
  [../decisions/0007-goldencheck-goldenmatch-integration.md](../decisions/0007-goldencheck-goldenmatch-integration.md).
- **GoldenCheck went from zero Rust to an optional `goldencheck-native` runtime**
  (#793, merged) + a deep-profiling wave: Benford (~16×), composite-key (1.7×, after
  the "naive kernel lost to Polars at 0.4× → u128 packing" fix), strict FD (12.8×),
  fuzzy value clustering (76×), approximate-FD violations (15.5×) — each parity-exact
  AND measured-to-beat-Polars. Plus `--deep` full-population mode, `refs` cross-file
  referential integrity, freshness/staleness, and two bridge APIs (`cell_quality`,
  `functional_dependencies`). Features Polars already wins (duplicate rows, refs,
  freshness) stay pure-Polars on purpose.
- **That quality signal now feeds GoldenMatch** through fail-open, default-OFF,
  benchmark-gated bridges in `core/quality.py` — four doors: quality-weighted
  survivorship (#794 ✅, wired the no-op `quality_weighting`), quality-aware blocking
  (#795 ✅, recall), FD-driven negative evidence (#797 🟡, precision), quality-gated
  review routing (#798 🟡, trust). Boundary held: value-level DQ in GoldenCheck,
  entity resolution in GoldenMatch.
- Process note logged in the decision: stacked PRs across squash-merges go `dirty`;
  recovery is merge-base-to-main then rebase-child-onto-main (done #794→#797).
- Docs-site: new `goldencheck/native.mdx` + `goldenmatch/data-quality.mdx`.

## 2026-06-07 — GoldenFlow Arrow-native kernel shipped
- New architecture node: [../architecture/goldenflow-native-kernel.md](../architecture/goldenflow-native-kernel.md)
  and decision: [../decisions/0006-goldenflow-native-nanp-gating.md](../decisions/0006-goldenflow-native-nanp-gating.md).
- Measured that GoldenFlow's `date_iso8601` + `phone_e164` were ~92 % of a 1M-row
  run (per-row `dateutil`/`phonenumbers`). Shipped: (1) **vectorized Polars fast
  paths** with a per-row fallback (`transforms/_fastpath.py::apply_with_residual`),
  76× date / 19× phone, ~14× end-to-end, parity-safe; (2) the optional
  `goldenflow-native` Rust/PyO3 abi3 kernel (`packages/rust/extensions/native-flow`)
  for the phone residual, Arrow zero-copy, **NANP-only gated** so it's byte-identical
  to `phonenumbers` by construction (the Rust port diverges on int'l/ambiguous
  numbers; two gates confine it to country-code-1 + canonical NANP).
- Infra mirrors `goldenmatch-native`: loader (`goldenflow/core/_native_loader.py`,
  `GOLDENFLOW_NATIVE` 0/auto/1), `publish-goldenflow-native.yml` (per-platform abi3
  wheels on `goldenflow-native-v*`), and two `native_flow` CI lanes (build + parity).
- Promoted in [../planning/roadmap.md](../planning/roadmap.md) as an adjacent
  Arrow-native arc. Docs-site updated in the same change: new
  `goldenflow/performance.mdx` (+ overview card + nav). Code-level notes in
  `packages/python/goldenflow/CLAUDE.md` + `packages/rust/extensions/native-flow/README.md`.

## 2026-06-06 — Auto-config search strategy after the engine speedup (v1.28.0)
- New planning node: [../planning/autoconfig-search-strategy.md](../planning/autoconfig-search-strategy.md)
  — the thesis (the controller's search strategy was calibrated to a cost model the
  perf arc falsified), the four-phase arc, and what shipped in 1.28.0 vs. what's staged.
- Shipped (1.28.0): a **planning-effort tier** (`fast`/`normal`/`thinking`/`einstein`)
  on `GoldenMatchConfig` + `dedupe_df`/`match_df`/`auto_configure_df` +
  `GOLDENMATCH_PLANNING_EFFORT`; `ControllerBudget.for_dataset(n_rows, effort)`;
  **measure-don't-extrapolate** at thinking+ (`blocker.measure_blocking_profile`);
  **provider-aware in-house embedding** (`_check_remote_assets` no longer demotes the
  local model). Default `normal` is byte-for-byte unchanged.
- Staged for a follow-up (behind the thinking/einstein seam): full successive-halving
  over a candidate grid (Phase 2) and an LLM-judge labeling objective (Phase 4).
- Spec: `docs/superpowers/specs/2026-06-06-autoconfig-search-strategy-after-engine-speedup-design.md`.
  Docs-site: `goldenmatch/auto-config.mdx` (Planning effort + in-house embeddings).

## 2026-06-05 — Security hardening arc (42 alerts cleared, Scorecard 6.1->7.3)
- New workstream node: [../planning/security-hardening.md](../planning/security-hardening.md)
  — Dependabot 7/7 (#761/#762), code-scanning 35/35 (log-injection #764,
  path-injection #768, TokenPermissions #760/#772, dismissals), Scorecard
  climbs (Token-Permissions->10, Signed-Releases->8, Fuzzing->10 via
  #770/#778/#783), the CodeQL Autofix incident, the 4-bug property-test
  ledger, and 3 open actions.
- Promoted in [../planning/roadmap.md](../planning/roadmap.md) as an adjacent arc.
- Docs-site updated in the same change: `GOLDENMATCH_ALLOWED_ROOT` in
  configuration + MCP path-sandbox section; cosign release-verification
  snippet in installation.

## 2026-06-05 — Surface hardening + parity arc (Waves 0-4)
- New workstream node: [../planning/surface-hardening.md](../planning/surface-hardening.md)
  — the four-surface audit, the merged auth/bug/TUI waves (#766/#767/#769), the
  ten-PR open queue (#771-#782), the Railway `GOLDENMATCH_MCP_TOKEN` open action,
  and the parity-fixture methodology (structure-not-ids; emitter-asserted margins).
- Promoted in [../planning/roadmap.md](../planning/roadmap.md) as an adjacent arc.
- Docs-site updated in the same change (MCP/REST/A2A auth, review command, TUI
  8-tabs + auto-config screen, REST readiness health, A2A streaming:false).

## 2026-06-05 — SQL-native graph + embedding UDFs shipped (#509, all 3 PRs)
- #509 fully delivered across PRs #740 (graph half — DuckDB + Postgres), #743 (embed
  half — `goldenmatch-embed` wheel + repoint + bridge cleanup), #745 (DataFusion FFI
  graph UDFs). The graph + embed SQL surface is now **native-direct** (pure-Rust
  `graph-core` + `goldenembed-rs`, no embedded-CPython JSON bridge) across all three
  backends; the #503 bridge placeholder + its dead `bridge::api` fns are gone.
- New nodes: [../architecture/sql-native-extensions.md](../architecture/sql-native-extensions.md),
  [../decisions/0005-sql-native-direct-udfs.md](../decisions/0005-sql-native-direct-udfs.md).
  Noted the adjacent SQL surface in [../planning/roadmap.md](../planning/roadmap.md).
- New crates/packages: `graph-core` (pyo3-free shared kernel), `goldenmatch-embed`
  (maturin wheel over goldenembed-rs). `goldenmatch_pg` + `goldenmatch-duckdb` bumped
  0.5.0→0.6.0 (new handwritten SQL surface + upgrade script). Spec/plan:
  `docs/superpowers/specs/2026-06-04-sql-native-graph-embed-udfs-design.md`,
  `docs/superpowers/plans/2026-06-04-sql-native-graph-embed-udfs.md`.

## 2026-06-04 — Sail tier S4 harness shipped (buildable tier COMPLETE)
- S4 harness merged (PR #717): chain-robust O(log n) WCC via pointer-jumping (the blind
  large-star/small-star attempt was wrong, caught by plan-review hand-trace + replaced),
  `run_sail_pipeline` end-to-end, and the 100M bench scaffold. The `sail` lane has 6 green gates.
  The BUILDABLE Sail tier is COMPLETE; only the real 100M cluster run + Ray retirement remain
  (need a BYO Sail cluster). Updated [../architecture/sail-tier.md](../architecture/sail-tier.md) +
  [../planning/roadmap.md](../planning/roadmap.md).

## 2026-06-03 — Sail tier Stage S3 (golden) shipped
- S3 golden merged (PR #714): distributed survivorship on Sail (collect_list + merge_field UDF),
  content-parity green. SCOPE DECISION: S3 scoped to golden only; identity split to its own next
  stage (stateful graph subsystem, not a relational op). Updated
  [../architecture/sail-tier.md](../architecture/sail-tier.md) +
  [../planning/roadmap.md](../planning/roadmap.md). Identity-on-Sail is next, then S4 (real cluster).

## 2026-06-03 — Sail tier Stage S2 shipped (make-or-break gate)
- S2 merged (PR #712): WCC on Sail via min-label propagation, partition-parity green. The
  existential "WCC-on-Sail at all" risk is CLOSED. Marked S2 done in
  [../architecture/sail-tier.md](../architecture/sail-tier.md) +
  [../planning/roadmap.md](../planning/roadmap.md). S3 (golden + identity on Sail) is next.

## 2026-06-03 — Sail tier Stage S1 shipped
- S1 merged (PR #709): the `goldenmatch.sail` harness + scorer pandas UDF + score/dedup,
  parity-green on a new `sail` CI lane. Marked S1 done in
  [../architecture/sail-tier.md](../architecture/sail-tier.md) +
  [../planning/roadmap.md](../planning/roadmap.md). S2 (WCC on Sail) is the next gate.

## 2026-06-03 — Sail tier specced + roadmapped
- Added the Sail-tier design (`docs/superpowers/specs/2026-06-03-sail-tier-design.md`,
  spec-reviewer approved) — the distributed Sail-native pipeline that replaces Ray.
- New nodes: [../architecture/sail-tier.md](../architecture/sail-tier.md),
  [../decisions/0004-sail-tier-scope.md](../decisions/0004-sail-tier-scope.md); promoted
  the Sail tier in [../planning/roadmap.md](../planning/roadmap.md) (S1-S4, WCC as the gate).

## 2026-06-03 — Network created
- Seeded the context network and the root `.context-network.md` discovery file.
- Captured the DataFusion-spine workstream end-to-end: Stages A-E status
  ([../architecture/datafusion-spine.md](../architecture/datafusion-spine.md)), and the
  three decisions that had no prior home:
  - [0001 gate reframe — engine portability](../decisions/0001-gate-reframe-engine-portability.md)
  - [0002 scale-mode contract](../decisions/0002-scale-mode-contract.md) (PR #702)
  - [0003 Stage E spill HONEST-NULL](../decisions/0003-stage-e-spill-honest-null.md) (PRs #705/#706)
- Recorded the development workflow + environment constraints
  ([../processes/development-workflow.md](../processes/development-workflow.md)) and the
  roadmap ([../planning/roadmap.md](../planning/roadmap.md)).
- Committed to git on branch `chore/context-network`.

---
**Classification:** meta/log • **Last updated:** 2026-06-13
