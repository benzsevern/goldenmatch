import { describe, it, expect } from "vitest";
import {
  runGraphER,
  type TableSchema,
  type Relationship,
  type GraphERScorer,
} from "../../src/core/graph-er.js";
import { scorePair } from "../../src/core/scorer.js";
import { makeMatchkeyField } from "../../src/core/types.js";
import type { Row, ScoredPair, MatchkeyField } from "../../src/core/types.js";

// Build a scorer that compares every pair in a row list using a weighted
// matchkey on the supplied fields. runGraphER expects pair idA/idB to be
// 0-based row indices (it's how clustersFromPairs seeds its Union-Find).
function allPairsScorer(fields: readonly MatchkeyField[]): GraphERScorer {
  return (rows: readonly Row[]): readonly ScoredPair[] => {
    const pairs: ScoredPair[] = [];
    for (let i = 0; i < rows.length; i++) {
      for (let j = i + 1; j < rows.length; j++) {
        const score = scorePair(rows[i]!, rows[j]!, fields);
        pairs.push({ idA: i, idB: j, score });
      }
    }
    return pairs;
  };
}

describe("runGraphER", () => {
  it("produces clusters per table for 2 tables + 1 relationship", () => {
    const customers: Row[] = [
      { id: 1, name: "John Smith", company_id: 100 },
      { id: 2, name: "Jon Smith", company_id: 100 },
      { id: 3, name: "Jane Doe", company_id: 200 },
    ];
    const companies: Row[] = [
      { id: 100, name: "Acme Inc" },
      { id: 200, name: "Widgets LLC" },
    ];

    const tables: TableSchema[] = [
      { name: "customers", rows: customers, idColumn: "id" },
      { name: "companies", rows: companies, idColumn: "id" },
    ];
    const relationships: Relationship[] = [
      { tableA: "customers", tableB: "companies", fkColumn: "company_id" },
    ];

    const nameField = [
      makeMatchkeyField({ field: "name", scorer: "jaro_winkler", transforms: ["lowercase"] }),
    ];
    const scorerByTable = new Map<string, GraphERScorer>([
      ["customers", allPairsScorer(nameField)],
      ["companies", allPairsScorer(nameField)],
    ]);

    const result = runGraphER(tables, relationships, {
      scorerByTable,
      threshold: 0.85,
      maxIterations: 5,
    });

    expect(result.clustersByTable.has("customers")).toBe(true);
    expect(result.clustersByTable.has("companies")).toBe(true);
    // Companies are distinct -> 2 singleton clusters
    expect(result.clustersByTable.get("companies")!.size).toBe(2);
    // John/Jon Smith pair above threshold -> 1 cluster of size 2; Jane is singleton
    const custClusters = result.clustersByTable.get("customers")!;
    const sizes = [...custClusters.values()].map((c) => c.size).sort();
    expect(sizes).toEqual([1, 2]);
  });

  it("evidence propagation pulls rows with shared FK cluster toward same cluster", () => {
    // Two customer rows whose names are borderline (below base threshold) but
    // whose company_id points to the same company row. Evidence boost should
    // push them over the threshold once companies are clustered.
    const customers: Row[] = [
      { id: 1, name: "J Smith", company_id: 100 },
      { id: 2, name: "John Smyth", company_id: 100 },
    ];
    const companies: Row[] = [{ id: 100, name: "Acme Inc" }];

    const tables: TableSchema[] = [
      { name: "customers", rows: customers, idColumn: "id" },
      { name: "companies", rows: companies, idColumn: "id" },
    ];
    const relationships: Relationship[] = [
      { tableA: "customers", tableB: "companies", fkColumn: "company_id" },
    ];
    const nameField = [makeMatchkeyField({ field: "name", scorer: "jaro_winkler" })];
    const scorerByTable = new Map<string, GraphERScorer>([
      ["customers", allPairsScorer(nameField)],
      ["companies", allPairsScorer(nameField)],
    ]);

    // Choose threshold right above the raw JW score for "J Smith" vs "John Smyth"
    // so that only the boost can merge them.
    const rawScore = scorePair(customers[0]!, customers[1]!, nameField);
    const threshold = Math.min(0.99, rawScore + 0.05);

    const result = runGraphER(tables, relationships, {
      scorerByTable,
      threshold,
      similarityBoost: 0.5,
      maxIterations: 5,
    });

    const custClusters = result.clustersByTable.get("customers")!;
    // After at least one propagation iteration, the two customers should merge.
    const merged = [...custClusters.values()].some((c) => c.size === 2);
    expect(merged).toBe(true);
  });

  it("converges within maxIterations for a small tractable dataset", () => {
    const customers: Row[] = [
      { id: 1, name: "Alice", company_id: 100 },
      { id: 2, name: "Bob", company_id: 200 },
    ];
    const companies: Row[] = [
      { id: 100, name: "Acme" },
      { id: 200, name: "Widget" },
    ];
    const tables: TableSchema[] = [
      { name: "customers", rows: customers, idColumn: "id" },
      { name: "companies", rows: companies, idColumn: "id" },
    ];
    const nameField = [makeMatchkeyField({ field: "name", scorer: "jaro_winkler" })];
    const scorerByTable = new Map<string, GraphERScorer>([
      ["customers", allPairsScorer(nameField)],
      ["companies", allPairsScorer(nameField)],
    ]);

    const result = runGraphER(tables, [], { scorerByTable, maxIterations: 10 });
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThanOrEqual(10);
  });

  it("respects the maxIterations cap when propagation never stabilizes", () => {
    // Construct inputs that force at least one iteration — we just need to
    // confirm the loop terminates at the cap even if convergence isn't reached.
    const customers: Row[] = [
      { id: 1, name: "A", company_id: 100 },
      { id: 2, name: "B", company_id: 100 },
    ];
    const companies: Row[] = [{ id: 100, name: "Acme" }];
    const tables: TableSchema[] = [
      { name: "customers", rows: customers, idColumn: "id" },
      { name: "companies", rows: companies, idColumn: "id" },
    ];
    const nameField = [makeMatchkeyField({ field: "name", scorer: "jaro_winkler" })];
    const scorerByTable = new Map<string, GraphERScorer>([
      ["customers", allPairsScorer(nameField)],
      ["companies", allPairsScorer(nameField)],
    ]);

    const cap = 2;
    const result = runGraphER(tables, [], {
      scorerByTable,
      maxIterations: cap,
      convergenceThreshold: 0, // never accept convergence except exact match
    });
    // Iterations (+1 if converged) must not exceed the cap + 1.
    expect(result.iterations).toBeLessThanOrEqual(cap + 1);
  });

  it("throws when a scorer is missing for a table", () => {
    const tables: TableSchema[] = [
      { name: "t", rows: [{ id: 1 }], idColumn: "id" },
    ];
    expect(() =>
      runGraphER(tables, [], { scorerByTable: new Map() }),
    ).toThrow(/Missing scorer/);
  });
});
