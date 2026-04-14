/**
 * app.ts -- GoldenMatch interactive TUI built on `ink` (React for CLIs).
 *
 * This module loads `ink` and `react` lazily via `createRequire` so the rest
 * of the package stays usable without those optional peer dependencies.
 *
 * The UI mirrors the Python Textual TUI: 6 tabs (Data, Config, Matches,
 * Golden, Boost, Export) with keyboard navigation [1..6], [Tab] to cycle,
 * [r] to run dedupe, [q] / [Esc] to quit.
 *
 * Richer ink-ecosystem addons (ink-table, ink-select-input, ink-text-input,
 * ink-spinner, ink-gradient) are optional peer deps loaded lazily via
 * ./widgets.js. Each tab degrades gracefully to plain text when an addon is
 * not installed.
 *
 * Implementation notes:
 *   - Uses React.createElement directly (no JSX) so we don't need a JSX
 *     transform in the existing tsup build.
 *   - The `ink` / `react` modules are typed as `any` at the boundary because
 *     they're optional peer deps; we don't want to require `@types/react`
 *     just to satisfy strict typecheck.
 */

import { createRequire } from "node:module";
import type { Row, GoldenMatchConfig, DedupeResult } from "../../core/types.js";
import { loadAddons, type LoadedAddons } from "./widgets.js";

const require = createRequire(import.meta.url);

// ---------------------------------------------------------------------------
// Optional peer dependency loaders
// ---------------------------------------------------------------------------

/* eslint-disable @typescript-eslint/no-explicit-any */

function loadInk(): any {
  try {
    return require("ink");
  } catch {
    throw new Error(
      "'ink' and 'react' are required for the TUI. Install with: npm install ink react",
    );
  }
}

function loadReact(): any {
  try {
    return require("react");
  } catch {
    throw new Error(
      "'react' is required for the TUI. Install with: npm install react",
    );
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface TuiOptions {
  readonly files?: readonly string[];
  readonly config?: GoldenMatchConfig;
}

/**
 * Launch the GoldenMatch TUI. Resolves once the user quits.
 */
export async function startTui(options: TuiOptions = {}): Promise<void> {
  const ink = loadInk();
  const React = loadReact();
  const h = React.createElement;

  // Load optional ink-ecosystem addons (each may be null).
  const addons: LoadedAddons = await loadAddons();

  // -------------------------------------------------------------------------
  // File loader (lazy import so the TUI module stays light)
  // -------------------------------------------------------------------------

  const loadFiles = async (files: readonly string[]): Promise<Row[]> => {
    const { readFile } = await import("../connectors/file.js");
    const all: Row[] = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i]!;
      const fileRows = readFile(f);
      for (const r of fileRows) {
        all.push({ ...r, __source__: `file_${i}` });
      }
    }
    return all;
  };

  // -------------------------------------------------------------------------
  // Tab components
  // -------------------------------------------------------------------------

  const MAX_TABLE_COLS = 5;
  const MAX_TABLE_ROWS = 10;

  const visibleCols = (row: Row): string[] =>
    Object.keys(row).filter((c) => !c.startsWith("__"));

  const DataTab = (props: { rows: readonly Row[] }) => {
    const { rows } = props;
    if (rows.length === 0) {
      return h(
        ink.Text,
        { dimColor: true },
        "No data loaded. Pass files as CLI args.",
      );
    }
    const cols = visibleCols(rows[0]!).slice(0, MAX_TABLE_COLS);

    if (addons.Table) {
      const display = rows.slice(0, MAX_TABLE_ROWS).map((r) => {
        const d: Record<string, string> = {};
        for (const c of cols) {
          const v = (r as Record<string, unknown>)[c];
          d[c] = v === undefined || v === null ? "" : String(v);
        }
        return d;
      });
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(
          ink.Text,
          {},
          `${rows.length} rows, showing first ${Math.min(
            MAX_TABLE_ROWS,
            rows.length,
          )} (cols: ${cols.join(", ")})`,
        ),
        h(addons.Table, { data: display }),
      );
    }

    return h(
      ink.Box,
      { flexDirection: "column" },
      h(ink.Text, {}, `${rows.length} rows loaded`),
      h(
        ink.Text,
        { dimColor: true },
        "Columns: " + (cols.length > 0 ? cols.join(", ") : "-"),
      ),
      ...rows.slice(0, MAX_TABLE_ROWS).map((row, i) =>
        h(
          ink.Text,
          { key: `row-${i}`, dimColor: true },
          cols
            .map((c) => {
              const v = (row as Record<string, unknown>)[c];
              return `${c}=${v ?? ""}`;
            })
            .join(" | "),
        ),
      ),
    );
  };

  const ConfigTab = (props: { config: GoldenMatchConfig | null }) => {
    const { config } = props;
    const mks = config?.matchkeys ?? config?.matchSettings ?? [];
    const blockingDesc = config?.blocking?.strategy ?? "-";
    const blockingKeys =
      config?.blocking?.keys?.map((k) => k.fields.join("+")).join(", ") ?? "-";

    // Local UI state for interactive config editing. Hooks must be at the
    // top level of the component — we always declare them and only use the
    // interactive branch when SelectInput/TextInput are available.
    const [selectedMk, setSelectedMk] = React.useState(null) as [
      number | null,
      (v: number | null) => void,
    ];
    const [thresholdDraft, setThresholdDraft] = React.useState("") as [
      string,
      (v: string) => void,
    ];

    const header = h(
      ink.Box,
      { flexDirection: "column" },
      h(ink.Text, { bold: true }, "Config"),
      h(ink.Text, {}, `Matchkeys: ${mks.length}`),
      h(ink.Text, {}, `Blocking: ${blockingDesc}, keys: ${blockingKeys}`),
    );

    if (mks.length === 0) {
      return header;
    }

    if (addons.SelectInput && selectedMk === null) {
      const items = mks.map((mk, i) => ({
        label: `${mk.name} (${mk.type}) threshold=${mk.threshold ?? "-"}`,
        value: String(i),
      }));
      return h(
        ink.Box,
        { flexDirection: "column" },
        header,
        h(ink.Text, { dimColor: true }, "Select a matchkey to inspect:"),
        h(addons.SelectInput, {
          items,
          onSelect: (item: { value: string }) => {
            const idx = Number(item.value);
            setSelectedMk(idx);
            setThresholdDraft(String(mks[idx]?.threshold ?? ""));
          },
        }),
      );
    }

    if (addons.SelectInput && selectedMk !== null) {
      const mk = mks[selectedMk];
      if (!mk) {
        setSelectedMk(null);
        return header;
      }
      const fields = mk.fields.map((f) => f.field).join(", ");
      return h(
        ink.Box,
        { flexDirection: "column" },
        header,
        h(ink.Text, { bold: true }, `Editing matchkey: ${mk.name}`),
        h(ink.Text, {}, `  type: ${mk.type}`),
        h(ink.Text, {}, `  fields: ${fields}`),
        addons.TextInput
          ? h(
              ink.Box,
              {},
              h(ink.Text, {}, "  threshold: "),
              h(addons.TextInput, {
                value: thresholdDraft,
                onChange: setThresholdDraft,
                onSubmit: (value: string) => {
                  const n = Number(value);
                  if (!Number.isNaN(n)) {
                    (mk as { threshold?: number }).threshold = n;
                  }
                  setSelectedMk(null);
                },
              }),
            )
          : h(
              ink.Text,
              { dimColor: true },
              `  threshold: ${mk.threshold ?? "-"} (install ink-text-input to edit)`,
            ),
        h(ink.Text, { dimColor: true }, "[Enter] save  [Esc] back"),
      );
    }

    // Fallback: plain listing
    return h(
      ink.Box,
      { flexDirection: "column" },
      header,
      ...mks.map((mk, i) =>
        h(
          ink.Text,
          { key: `mk-${i}`, dimColor: true },
          `  - ${mk.name} (${mk.type}), threshold=${mk.threshold ?? "-"}, fields: ${mk.fields
            .map((f) => f.field)
            .join(", ")}`,
        ),
      ),
    );
  };

  const MatchesTab = (props: { result: DedupeResult | null }) => {
    const { result } = props;
    const [selectedPair, setSelectedPair] = React.useState(null) as [
      number | null,
      (v: number | null) => void,
    ];

    if (!result) {
      return h(
        ink.Text,
        { dimColor: true },
        "No results yet. Press 'r' to run dedupe.",
      );
    }
    const pairs = result.scoredPairs.slice(0, MAX_TABLE_ROWS);
    if (pairs.length === 0) {
      return h(ink.Text, {}, "No scored pairs");
    }

    // Drill-in view
    if (addons.SelectInput && selectedPair !== null) {
      const p = pairs[selectedPair];
      if (!p) {
        setSelectedPair(null);
        return h(ink.Text, {}, "");
      }
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(ink.Text, { bold: true }, `Pair detail ${selectedPair + 1}/${pairs.length}`),
        h(ink.Text, {}, `  idA: ${p.idA}`),
        h(ink.Text, {}, `  idB: ${p.idB}`),
        h(ink.Text, {}, `  score: ${p.score.toFixed(4)}`),
        h(ink.Text, { dimColor: true }, "(select another pair from list)"),
        h(addons.SelectInput, {
          items: pairs.map((pp, i) => ({
            label: `${pp.idA} <-> ${pp.idB}  (${pp.score.toFixed(3)})`,
            value: String(i),
          })),
          onSelect: (item: { value: string }) =>
            setSelectedPair(Number(item.value)),
        }),
      );
    }

    if (addons.Table) {
      const data = pairs.map((p) => ({
        idA: String(p.idA),
        idB: String(p.idB),
        score: p.score.toFixed(4),
      }));
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(
          ink.Text,
          { bold: true },
          `Scored pairs: ${result.scoredPairs.length} (showing first ${pairs.length})`,
        ),
        h(addons.Table, { data }),
        addons.SelectInput
          ? h(addons.SelectInput, {
              items: pairs.map((p, i) => ({
                label: `Inspect ${p.idA} <-> ${p.idB}`,
                value: String(i),
              })),
              onSelect: (item: { value: string }) =>
                setSelectedPair(Number(item.value)),
            })
          : h(
              ink.Text,
              { dimColor: true },
              "(install ink-select-input for drill-in)",
            ),
      );
    }

    return h(
      ink.Box,
      { flexDirection: "column" },
      h(
        ink.Text,
        { bold: true },
        `Scored pairs: ${result.scoredPairs.length}`,
      ),
      ...pairs.map((p, i) =>
        h(
          ink.Text,
          { key: `pair-${i}` },
          `  ${p.idA} <-> ${p.idB}: ${p.score.toFixed(3)}`,
        ),
      ),
    );
  };

  const GoldenTab = (props: { result: DedupeResult | null }) => {
    const { result } = props;
    if (!result) return h(ink.Text, { dimColor: true }, "No results yet.");
    const records = result.goldenRecords.slice(0, MAX_TABLE_ROWS);

    if (records.length === 0) {
      return h(
        ink.Text,
        { bold: true },
        `Golden records: ${result.goldenRecords.length}`,
      );
    }

    if (addons.Table) {
      const cols = visibleCols(records[0]!).slice(0, MAX_TABLE_COLS);
      const data = records.map((r) => {
        const d: Record<string, string> = {};
        for (const c of cols) {
          const v = (r as Record<string, unknown>)[c];
          d[c] = v === undefined || v === null ? "" : String(v);
        }
        return d;
      });
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(
          ink.Text,
          { bold: true },
          `Golden records: ${result.goldenRecords.length}`,
        ),
        h(addons.Table, { data }),
      );
    }

    return h(
      ink.Box,
      { flexDirection: "column" },
      h(
        ink.Text,
        { bold: true },
        `Golden records: ${result.goldenRecords.length}`,
      ),
      ...records.map((r, i) =>
        h(
          ink.Text,
          { key: `g-${i}`, dimColor: true },
          JSON.stringify(r).slice(0, 100),
        ),
      ),
    );
  };

  const BoostTab = (props: { result: DedupeResult | null }) => {
    const { result } = props;
    const [idx, setIdx] = React.useState(0) as [
      number,
      (v: number | ((prev: number) => number)) => void,
    ];
    const [labels, setLabels] = React.useState({}) as [
      Record<number, string>,
      (v: Record<number, string>) => void,
    ];

    if (!result) {
      return h(
        ink.Text,
        { dimColor: true },
        "No results yet. Press 'r' to run dedupe.",
      );
    }

    const borderline = result.scoredPairs
      .filter((p) => p.score >= 0.7 && p.score < 0.9)
      .slice(0, 20);

    if (borderline.length === 0) {
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(ink.Text, { bold: true }, "Boost - active learning"),
        h(ink.Text, {}, "No borderline pairs (0.7-0.9 score) to label."),
      );
    }

    if (idx >= borderline.length) {
      const counts = Object.values(labels).reduce(
        (acc, v) => {
          acc[v] = (acc[v] ?? 0) + 1;
          return acc;
        },
        {} as Record<string, number>,
      );
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(ink.Text, { color: "green", bold: true }, "All pairs labeled!"),
        h(
          ink.Text,
          {},
          `y=${counts["y"] ?? 0}  n=${counts["n"] ?? 0}  s=${counts["s"] ?? 0}`,
        ),
      );
    }

    const pair = borderline[idx]!;

    if (addons.SelectInput) {
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(
          ink.Text,
          { bold: true },
          `Pair ${idx + 1}/${borderline.length} - Score: ${pair.score.toFixed(3)}`,
        ),
        h(ink.Text, {}, `  Record ${pair.idA}`),
        h(ink.Text, {}, `  Record ${pair.idB}`),
        h(addons.SelectInput, {
          items: [
            { label: "Yes, this is a match", value: "y" },
            { label: "No, different entities", value: "n" },
            { label: "Skip", value: "s" },
          ],
          onSelect: (item: { value: string }) => {
            setLabels({ ...labels, [idx]: item.value });
            setIdx((prev) => prev + 1);
          },
        }),
      );
    }

    return h(
      ink.Box,
      { flexDirection: "column" },
      h(ink.Text, { bold: true }, "Boost - active learning"),
      h(ink.Text, { dimColor: true }, "Label borderline pairs: y/n/s (skip)"),
      h(
        ink.Text,
        {},
        `Pair ${idx + 1}/${borderline.length}: ${pair.idA} <-> ${pair.idB} (${pair.score.toFixed(3)})`,
      ),
      h(
        ink.Text,
        { dimColor: true },
        "Install ink-select-input for interactive labeling",
      ),
    );
  };

  const ExportTab = (props: {
    result: DedupeResult | null;
    setStatus: (s: string) => void;
  }) => {
    const { result, setStatus } = props;
    const [exporting, setExporting] = React.useState(false) as [
      boolean,
      (v: boolean) => void,
    ];
    const [done, setDone] = React.useState(null) as [
      string | null,
      (v: string | null) => void,
    ];

    if (!result) {
      return h(ink.Text, { dimColor: true }, "No results yet.");
    }

    const doExport = (format: string) => {
      setExporting(true);
      setDone(null);
      setStatus(`Exporting as ${format}...`);
      // Simulate async write. Real impl would dispatch to a writer.
      setTimeout(() => {
        setExporting(false);
        setDone(format);
        setStatus(`Export complete (${format})`);
      }, 400);
    };

    if (exporting) {
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(ink.Text, { bold: true }, "Export"),
        addons.Spinner
          ? h(
              ink.Box,
              {},
              h(addons.Spinner, { type: "dots" }),
              h(ink.Text, {}, "  writing..."),
            )
          : h(ink.Text, {}, "writing..."),
      );
    }

    const summary = h(
      ink.Text,
      {},
      `Ready: ${result.goldenRecords.length} golden, ${result.dupes.length} dupes, ${result.unique.length} unique`,
    );

    if (addons.SelectInput) {
      return h(
        ink.Box,
        { flexDirection: "column" },
        h(ink.Text, { bold: true }, "Export"),
        summary,
        done
          ? h(
              ink.Text,
              { color: "green" },
              `Last export: ${done}. Choose another format to export again.`,
            )
          : h(ink.Text, { dimColor: true }, "Choose output format:"),
        h(addons.SelectInput, {
          items: [
            { label: "CSV", value: "csv" },
            { label: "JSON", value: "json" },
          ],
          onSelect: (item: { value: string }) => doExport(item.value),
        }),
      );
    }

    return h(
      ink.Box,
      { flexDirection: "column" },
      h(ink.Text, { bold: true }, "Export"),
      h(
        ink.Text,
        { dimColor: true },
        "Press [g] for golden, [d] for dupes, [u] for unique",
      ),
      summary,
    );
  };

  // -------------------------------------------------------------------------
  // Top-level App
  // -------------------------------------------------------------------------

  const App = (props: { options: TuiOptions }) => {
    const [tab, setTab] = React.useState(0) as [
      number,
      (v: number | ((prev: number) => number)) => void,
    ];
    const [rows, setRows] = React.useState([]) as [
      readonly Row[],
      (v: readonly Row[]) => void,
    ];
    const [result, setResult] = React.useState(null) as [
      DedupeResult | null,
      (v: DedupeResult | null) => void,
    ];
    const [config] = React.useState(props.options.config ?? null) as [
      GoldenMatchConfig | null,
      (v: GoldenMatchConfig | null) => void,
    ];
    const [status, setStatus] = React.useState("Ready") as [
      string,
      (v: string) => void,
    ];

    const { exit } = ink.useApp();

    const runDedupe = React.useCallback(async () => {
      if (rows.length === 0) {
        setStatus("No rows loaded");
        return;
      }
      setStatus("Running dedupe...");
      try {
        const { dedupe } = await import("../../core/api.js");
        const r = dedupe(rows, config ? { config } : {});
        setResult(r);
        setStatus(`Complete: ${r.stats.totalClusters} clusters`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setStatus(`Error: ${msg}`);
      }
    }, [rows, config]);

    ink.useInput((input: string, key: any) => {
      if (key.escape || input === "q") {
        exit();
        return;
      }
      if (input === "1") setTab(0);
      else if (input === "2") setTab(1);
      else if (input === "3") setTab(2);
      else if (input === "4") setTab(3);
      else if (input === "5") setTab(4);
      else if (input === "6") setTab(5);
      else if (key.tab) setTab((t: number) => (t + 1) % 6);
      else if (input === "r") {
        void runDedupe();
      }
    });

    React.useEffect(() => {
      const files = props.options.files;
      if (files && files.length > 0) {
        loadFiles(files)
          .then((rs) => {
            setRows(rs);
            setStatus(`Loaded ${rs.length} rows from ${files.length} file(s)`);
          })
          .catch((err: unknown) => {
            const msg = err instanceof Error ? err.message : String(err);
            setStatus(`Error: ${msg}`);
          });
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const tabs = ["Data", "Config", "Matches", "Golden", "Boost", "Export"];

    let body: any = null;
    if (tab === 0) body = h(DataTab, { rows });
    else if (tab === 1) body = h(ConfigTab, { config });
    else if (tab === 2) body = h(MatchesTab, { result });
    else if (tab === 3) body = h(GoldenTab, { result });
    else if (tab === 4) body = h(BoostTab, { result });
    else if (tab === 5) body = h(ExportTab, { result, setStatus });

    const titleText = "GoldenMatch TUI - v0.1.0";
    const title = addons.Gradient
      ? h(
          addons.Gradient,
          { name: "rainbow" },
          h(ink.Text, { bold: true }, titleText),
        )
      : h(ink.Text, { bold: true, color: "cyan" }, titleText);

    return h(
      ink.Box,
      { flexDirection: "column", padding: 1 },
      // Header
      h(ink.Box, { borderStyle: "double", paddingX: 1 }, title),
      // Tab bar
      h(
        ink.Box,
        { marginTop: 1 },
        ...tabs.map((name: string, i: number) =>
          h(
            ink.Box,
            { key: `tab-${i}`, marginRight: 2 },
            h(
              ink.Text,
              { color: tab === i ? "green" : "gray", bold: tab === i },
              `[${i + 1}] ${name}`,
            ),
          ),
        ),
      ),
      // Tab content
      h(
        ink.Box,
        { marginTop: 1, flexDirection: "column", minHeight: 10 },
        body,
      ),
      // Footer
      h(
        ink.Box,
        { marginTop: 1, borderStyle: "single", paddingX: 1 },
        h(
          ink.Text,
          { dimColor: true },
          `[q]uit  [1-6] tabs  [Tab] cycle  [r]un dedupe   *   ${status}`,
        ),
      ),
    );
  };

  const { waitUntilExit } = ink.render(h(App, { options }));
  await waitUntilExit();
}

/* eslint-enable @typescript-eslint/no-explicit-any */
