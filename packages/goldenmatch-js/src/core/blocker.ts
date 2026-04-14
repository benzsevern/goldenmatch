/**
 * blocker.ts — Groups records into blocks for pairwise comparison.
 *
 * Edge-safe: no Node.js imports. Pure TypeScript only.
 *
 * Ports `goldenmatch/core/blocker.py`.
 */

import type {
  BlockingConfig,
  BlockingKeyConfig,
  BlockResult,
  Row,
  SortKeyField,
} from "./types.js";
import { applyTransforms } from "./transforms.js";
import { buildANNBlocks, buildANNPairBlocks } from "./ann-blocker.js";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Build a composite block key string for a single row.
 *
 * For each field in `keyConfig.fields`, extracts the value, applies
 * transforms, and concatenates with "||". Returns `null` if any field
 * value is null/undefined or any transform returns null.
 */
function buildBlockKey(
  row: Row,
  keyConfig: BlockingKeyConfig,
): string | null {
  const parts: string[] = [];
  for (const field of keyConfig.fields) {
    const raw = row[field];
    if (raw === null || raw === undefined) return null;
    const str = String(raw);
    if (keyConfig.transforms.length > 0) {
      const val = applyTransforms(str, keyConfig.transforms);
      if (val === null || val === undefined) return null;
      parts.push(val);
    } else {
      parts.push(str);
    }
  }
  return parts.join("||");
}

/**
 * Build a sort key string for a row using SortKeyField config.
 * Returns `null` if any field value is null/undefined.
 */
function buildSortKey(
  row: Row,
  sortKeyFields: readonly SortKeyField[],
): string | null {
  const parts: string[] = [];
  for (const skf of sortKeyFields) {
    const raw = row[skf.column];
    if (raw === null || raw === undefined) return null;
    const str = String(raw);
    if (skf.transforms.length > 0) {
      const val = applyTransforms(str, skf.transforms);
      if (val === null || val === undefined) return null;
      parts.push(val);
    } else {
      parts.push(str);
    }
  }
  return parts.join("||");
}

// ---------------------------------------------------------------------------
// Static blocking
// ---------------------------------------------------------------------------

/**
 * Group rows by blocking key. Skip blocks with fewer than 2 rows.
 * Handle oversized blocks per `config.skipOversized`.
 */
export function buildStaticBlocks(
  rows: readonly Row[],
  config: BlockingConfig,
): BlockResult[] {
  if (rows.length < 2) return [];

  const results: BlockResult[] = [];

  for (const keyConfig of config.keys) {
    const groups = new Map<string, Row[]>();

    for (const row of rows) {
      const key = buildBlockKey(row, keyConfig);
      if (key === null) continue;
      let group = groups.get(key);
      if (!group) {
        group = [];
        groups.set(key, group);
      }
      group.push(row);
    }

    for (const [key, group] of groups) {
      if (group.length < 2) continue;

      if (group.length > config.maxBlockSize) {
        if (config.skipOversized) {
          // Skip oversized blocks
          continue;
        }
        // Process anyway (caller is warned via the oversized size)
      }

      results.push({
        blockKey: key,
        rows: group,
        strategy: "static",
        depth: 0,
      });
    }
  }

  return results;
}

// ---------------------------------------------------------------------------
// Multi-pass blocking
// ---------------------------------------------------------------------------

/**
 * Run multiple blocking passes using `config.passes`.
 *
 * Each pass uses a different `BlockingKeyConfig`. Blocks are deduplicated
 * by block key so each unique key appears only once.
 */
export function buildMultiPassBlocks(
  rows: readonly Row[],
  config: BlockingConfig,
): BlockResult[] {
  if (rows.length < 2) return [];

  const passes = config.passes ?? [];
  if (passes.length === 0) return [];

  const allBlocks: BlockResult[] = [];
  const seenKeys = new Set<string>();

  for (const passConfig of passes) {
    // Build a temporary config with just this pass's key
    const tempConfig: BlockingConfig = {
      ...config,
      strategy: "static",
      keys: [passConfig],
    };

    const blocks = buildStaticBlocks(rows, tempConfig);

    for (const block of blocks) {
      if (!seenKeys.has(block.blockKey)) {
        seenKeys.add(block.blockKey);
        allBlocks.push({
          ...block,
          strategy: "multi_pass",
        });
      }
    }
  }

  return allBlocks;
}

// ---------------------------------------------------------------------------
// Sorted neighborhood blocking
// ---------------------------------------------------------------------------

/**
 * Sort rows by a composite sort key, then slide a window of
 * `config.windowSize` through the sorted data.
 *
 * Each window position produces one block. Requires `config.sortKey`
 * to be configured.
 */
export function buildSortedNeighborhoodBlocks(
  rows: readonly Row[],
  config: BlockingConfig,
): BlockResult[] {
  if (rows.length < 2) return [];

  const sortKeyFields = config.sortKey;
  if (!sortKeyFields || sortKeyFields.length === 0) {
    throw new Error(
      "sorted_neighborhood strategy requires sortKey configuration.",
    );
  }

  const windowSize = config.windowSize ?? 10;

  // Build (sortKey, row) pairs, filter nulls, and sort
  const keyed: Array<{ key: string; row: Row }> = [];
  for (const row of rows) {
    const key = buildSortKey(row, sortKeyFields);
    if (key !== null) {
      keyed.push({ key, row });
    }
  }

  keyed.sort((a, b) => {
    if (a.key < b.key) return -1;
    if (a.key > b.key) return 1;
    return 0;
  });

  const n = keyed.length;
  if (n < 2) return [];

  const results: BlockResult[] = [];

  if (n <= windowSize) {
    // Dataset smaller than window -- single block
    results.push({
      blockKey: "sorted_window_0",
      rows: keyed.map((k) => k.row),
      strategy: "sorted_neighborhood",
      depth: 0,
    });
    return results;
  }

  // Slide window through sorted data
  for (let i = 0; i <= n - windowSize; i++) {
    const windowRows = keyed.slice(i, i + windowSize).map((k) => k.row);
    results.push({
      blockKey: `sorted_window_${i}`,
      rows: windowRows,
      strategy: "sorted_neighborhood",
      depth: 0,
    });
  }

  return results;
}

// ---------------------------------------------------------------------------
// Auto-split oversized block
// ---------------------------------------------------------------------------

/**
 * Split an oversized block by the column with the most unique values
 * that produces useful groups (>= 2 rows each).
 *
 * This is a zero-config fallback when no `subBlockKeys` are configured
 * for adaptive blocking.
 */
export function autoSplitBlock(
  blockRows: readonly Row[],
  maxBlockSize: number,
  parentKey: string,
): BlockResult[] {
  if (blockRows.length < 2) return [];

  // Find non-internal columns (not prefixed with __)
  const sampleRow = blockRows[0];
  if (!sampleRow) return [];

  const candidates = Object.keys(sampleRow).filter(
    (c) => !c.startsWith("__"),
  );

  if (candidates.length === 0) {
    // No non-internal columns -- return as-is
    return [
      {
        blockKey: parentKey,
        rows: blockRows,
        strategy: "adaptive",
        depth: 1,
        parentKey,
      },
    ];
  }

  // Pick column whose cardinality best splits blocks.
  // Score = number of groups with >= 2 rows (useful groups).
  let bestCol = candidates[0]!;
  let bestUsefulGroups = 0;
  let bestNunique = 0;

  for (const col of candidates) {
    const groups = new Map<string, number>();
    for (const row of blockRows) {
      const val = row[col];
      const key = val === null || val === undefined ? "__null__" : String(val);
      groups.set(key, (groups.get(key) ?? 0) + 1);
    }

    const nunique = groups.size;
    let usefulGroups = 0;
    for (const count of groups.values()) {
      if (count >= 2) usefulGroups++;
    }

    const avgGroup = nunique > 0 ? blockRows.length / nunique : blockRows.length;

    if (
      usefulGroups > bestUsefulGroups ||
      (usefulGroups === bestUsefulGroups &&
        avgGroup <= maxBlockSize &&
        nunique > bestNunique)
    ) {
      bestUsefulGroups = usefulGroups;
      bestNunique = nunique;
      bestCol = col;
    }
  }

  // Split by the chosen column
  const splitGroups = new Map<string, Row[]>();
  for (const row of blockRows) {
    const val = row[bestCol];
    const key = val === null || val === undefined ? "__null__" : String(val);
    let group = splitGroups.get(key);
    if (!group) {
      group = [];
      splitGroups.set(key, group);
    }
    group.push(row);
  }

  const results: BlockResult[] = [];
  for (const [key, group] of splitGroups) {
    if (key === "__null__") continue; // skip null groups
    if (group.length < 2) continue;
    results.push({
      blockKey: `${parentKey}||${key}`,
      rows: group,
      strategy: "adaptive",
      depth: 1,
      parentKey,
    });
  }

  // If no useful splits, return the block as-is
  if (results.length === 0) {
    return [
      {
        blockKey: parentKey,
        rows: blockRows,
        strategy: "adaptive",
        depth: 1,
        parentKey,
      },
    ];
  }

  return results;
}

// ---------------------------------------------------------------------------
// Adaptive blocking (static + auto-split for oversized)
// ---------------------------------------------------------------------------

/**
 * Build static blocks first, then auto-split any oversized blocks
 * using the highest-cardinality column.
 *
 * If `config.subBlockKeys` is configured, uses recursive sub-blocking
 * instead of auto-split.
 */
export function buildAdaptiveBlocks(
  rows: readonly Row[],
  config: BlockingConfig,
): BlockResult[] {
  if (rows.length < 2) return [];

  const primaryBlocks = buildStaticBlocks(rows, config);
  const subBlockKeys = config.subBlockKeys ?? [];

  const results: BlockResult[] = [];

  for (const block of primaryBlocks) {
    const size = block.rows.length;

    if (size > config.maxBlockSize && subBlockKeys.length > 0) {
      // Recursive sub-blocking with configured keys
      const subResults = subBlock(
        block.rows,
        subBlockKeys,
        config.maxBlockSize,
        1,
        block.blockKey,
      );
      results.push(...subResults);
    } else if (size > config.maxBlockSize && !config.skipOversized) {
      // Auto-split by highest-cardinality column
      const autoResults = autoSplitBlock(
        block.rows,
        config.maxBlockSize,
        block.blockKey,
      );
      results.push(...autoResults);
    } else {
      results.push(block);
    }
  }

  return results;
}

/**
 * Recursively sub-block an oversized block using configured sub-block keys.
 *
 * Max recursion depth is 3. If all keys are exhausted or depth exceeds 3,
 * the block is returned as-is.
 */
function subBlock(
  blockRows: readonly Row[],
  subBlockKeys: readonly BlockingKeyConfig[],
  maxBlockSize: number,
  depth: number,
  parentKey: string,
): BlockResult[] {
  if (depth > 3 || subBlockKeys.length === 0) {
    // Max depth or no more keys -- return as-is
    return [
      {
        blockKey: parentKey,
        rows: blockRows,
        strategy: "adaptive",
        depth,
        parentKey,
      },
    ];
  }

  const currentKey = subBlockKeys[0]!;
  const remainingKeys = subBlockKeys.slice(1);

  const groups = new Map<string, Row[]>();
  for (const row of blockRows) {
    const key = buildBlockKey(row, currentKey);
    if (key === null) continue;
    let group = groups.get(key);
    if (!group) {
      group = [];
      groups.set(key, group);
    }
    group.push(row);
  }

  const results: BlockResult[] = [];
  for (const [key, group] of groups) {
    if (group.length < 2) continue;

    if (group.length > maxBlockSize && remainingKeys.length > 0 && depth < 3) {
      // Recurse with next sub-block key
      const subResults = subBlock(
        group,
        remainingKeys,
        maxBlockSize,
        depth + 1,
        parentKey,
      );
      results.push(...subResults);
    } else {
      results.push({
        blockKey: key,
        rows: group,
        strategy: "adaptive",
        depth,
        parentKey,
      });
    }
  }

  return results;
}

// ---------------------------------------------------------------------------
// Best blocking key selection
// ---------------------------------------------------------------------------

/**
 * Evaluate candidate blocking keys and select the one with the smallest
 * max group size while maintaining >= 50% coverage.
 *
 * Coverage = fraction of rows that produce a non-null block key.
 * If only one key is provided, returns it directly.
 */
export function selectBestBlockingKey(
  rows: readonly Row[],
  keys: readonly BlockingKeyConfig[],
  maxBlockSize: number = 5000,
): BlockingKeyConfig {
  if (keys.length === 0) {
    throw new Error("selectBestBlockingKey requires at least one key.");
  }
  if (keys.length === 1) return keys[0]!;

  const total = rows.length;
  if (total === 0) return keys[0]!;

  let bestKey: BlockingKeyConfig = keys[0]!;
  let bestMaxSize = Infinity;

  for (const keyConfig of keys) {
    const groupSizes = new Map<string, number>();
    let nonNull = 0;

    for (const row of rows) {
      const key = buildBlockKey(row, keyConfig);
      if (key !== null) {
        nonNull++;
        groupSizes.set(key, (groupSizes.get(key) ?? 0) + 1);
      }
    }

    const coverage = nonNull / total;
    if (coverage < 0.5) continue; // Skip low-coverage keys

    // Find max group size
    let maxSize = 0;
    for (const size of groupSizes.values()) {
      if (size > maxSize) maxSize = size;
    }

    if (
      maxSize < bestMaxSize ||
      (maxSize === bestMaxSize && groupSizes.size > 0)
    ) {
      bestMaxSize = maxSize;
      bestKey = keyConfig;
    }
  }

  return bestKey;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Build blocks from rows based on blocking configuration.
 *
 * Routes by `config.strategy`:
 * - `"static"` -- hash-based grouping on blocking keys
 * - `"multi_pass"` -- multiple passes with deduplication
 * - `"sorted_neighborhood"` -- sliding window over sorted data
 * - `"adaptive"` -- static + auto-split for oversized blocks
 * - `"ann"`, `"ann_pairs"`, `"canopy"`, `"learned"` -- not yet implemented
 *
 * If `config.autoSelect` is true and multiple keys are configured,
 * automatically selects the best key before blocking.
 */
export function buildBlocks(
  rows: readonly Row[],
  config: BlockingConfig,
): BlockResult[] {
  if (rows.length < 2) return [];

  // Auto-select best key if enabled
  let effectiveConfig = config;
  if (config.autoSelect && config.keys.length > 1) {
    const bestKey = selectBestBlockingKey(
      rows,
      config.keys,
      config.maxBlockSize,
    );
    effectiveConfig = {
      ...config,
      keys: [bestKey],
      autoSelect: false,
    };
  }

  switch (effectiveConfig.strategy) {
    case "static":
      return buildStaticBlocks(rows, effectiveConfig);

    case "multi_pass":
      return buildMultiPassBlocks(rows, effectiveConfig);

    case "sorted_neighborhood":
      return buildSortedNeighborhoodBlocks(rows, effectiveConfig);

    case "adaptive":
      return buildAdaptiveBlocks(rows, effectiveConfig);

    case "ann":
    case "ann_pairs":
      throw new Error(
        `ANN blocking strategy "${effectiveConfig.strategy}" is not yet implemented in the TypeScript port. ` +
          "It requires FAISS or a similar approximate nearest neighbor library.",
      );

    case "canopy":
      throw new Error(
        'Canopy blocking strategy is not yet implemented in the TypeScript port. ' +
          "It requires TF-IDF vectorization.",
      );

    case "learned":
      throw new Error(
        'Learned blocking strategy is not yet implemented in the TypeScript port. ' +
          "It requires predicate learning from training pairs.",
      );

    default: {
      // Exhaustive check -- if a new strategy is added to the union type
      // but not handled here, this will cause a compile-time error.
      const _exhaustive: never = effectiveConfig.strategy;
      throw new Error(`Unknown blocking strategy: ${String(_exhaustive)}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Async entry point — required for ANN strategies that fetch embeddings.
// ---------------------------------------------------------------------------

/**
 * Async variant of `buildBlocks`. Required for `"ann"` and `"ann_pairs"`
 * strategies which need to fetch embeddings via HTTP. All other strategies
 * delegate to the synchronous `buildBlocks` path.
 */
export async function buildBlocksAsync(
  rows: readonly Row[],
  config: BlockingConfig,
): Promise<BlockResult[]> {
  if (rows.length < 2) return [];

  if (config.strategy === "ann") {
    if (!config.annColumn) {
      throw new Error('"ann" strategy requires `annColumn` in BlockingConfig.');
    }
    return await buildANNBlocks(rows, config.annColumn, {
      ...(config.annTopK !== undefined ? { topK: config.annTopK } : {}),
      ...(config.annModel !== undefined ? { model: config.annModel } : {}),
      ...(config.maxBlockSize !== undefined ? { maxBlockSize: config.maxBlockSize } : {}),
    });
  }

  if (config.strategy === "ann_pairs") {
    if (!config.annColumn) {
      throw new Error('"ann_pairs" strategy requires `annColumn` in BlockingConfig.');
    }
    return await buildANNPairBlocks(rows, config.annColumn, {
      ...(config.annTopK !== undefined ? { topK: config.annTopK } : {}),
      ...(config.annModel !== undefined ? { model: config.annModel } : {}),
    });
  }

  return buildBlocks(rows, config);
}
