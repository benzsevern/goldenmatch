/**
 * widgets.ts -- Optional ink ecosystem addon loaders.
 *
 * Each addon (ink-table, ink-select-input, ink-text-input, ink-spinner,
 * ink-gradient) is an optional peer dependency. We provide a uniform
 * mechanism to try-load them so callers can render rich UI when installed
 * and fall back to plain `ink.Text` / `ink.Box` output otherwise.
 *
 * Two loading styles are provided:
 *   - `tryLoad<T>(name)`: synchronous `require`-based load. Works for CJS
 *     packages (ink-select-input, ink-text-input, ink-spinner, ink-gradient
 *     in older versions).
 *   - `loadAddons()`: async dynamic `import()` load. Works for ESM-only
 *     packages (ink-table v3+).
 *
 * Both return `null` for a missing package rather than throwing, so the
 * caller can branch on presence.
 */

import { createRequire } from "node:module";

/* eslint-disable @typescript-eslint/no-explicit-any */

const require = createRequire(import.meta.url);

/**
 * Synchronously try to require a package. Returns `null` if the package is
 * not installed (or fails to load for any reason).
 */
export function tryLoad<T = any>(name: string): T | null {
  try {
    return require(name) as T;
  } catch {
    return null;
  }
}

/**
 * Lazy-getter bag of synchronously-loaded ink addons. Each access retries
 * the require in case the environment changed, but most callers will prefer
 * `loadAddons()` (async) since ink-table is ESM-only.
 */
export const inkAddons = {
  get table(): any {
    return tryLoad("ink-table");
  },
  get selectInput(): any {
    return tryLoad("ink-select-input");
  },
  get textInput(): any {
    return tryLoad("ink-text-input");
  },
  get spinner(): any {
    return tryLoad("ink-spinner");
  },
  get gradient(): any {
    return tryLoad("ink-gradient");
  },
};

/**
 * Collected addon components, each either the default export of its package
 * or `null` if the package isn't installed.
 */
export interface LoadedAddons {
  Table: any | null;
  SelectInput: any | null;
  TextInput: any | null;
  Spinner: any | null;
  Gradient: any | null;
}

/**
 * Asynchronously load all optional ink addons via dynamic `import()`.
 *
 * Uses `import()` rather than `require()` because ink-table v3+ ships as
 * ESM-only and cannot be loaded from CJS. The other addons work either
 * way; we standardise on import for consistency.
 *
 * Any addon that fails to load (missing package, import error, etc.) is
 * silently set to `null`. Callers should branch on each field.
 */
export async function loadAddons(): Promise<LoadedAddons> {
  const addons: LoadedAddons = {
    Table: null,
    SelectInput: null,
    TextInput: null,
    Spinner: null,
    Gradient: null,
  };

  // Wrap each import in its own try so a single missing addon doesn't
  // poison the others.
  try {
    const mod: any = await import("ink-table" as string);
    addons.Table = mod.default ?? mod;
  } catch {
    /* optional */
  }
  try {
    const mod: any = await import("ink-select-input" as string);
    addons.SelectInput = mod.default ?? mod;
  } catch {
    /* optional */
  }
  try {
    const mod: any = await import("ink-text-input" as string);
    addons.TextInput = mod.default ?? mod;
  } catch {
    /* optional */
  }
  try {
    const mod: any = await import("ink-spinner" as string);
    addons.Spinner = mod.default ?? mod;
  } catch {
    /* optional */
  }
  try {
    const mod: any = await import("ink-gradient" as string);
    addons.Gradient = mod.default ?? mod;
  } catch {
    /* optional */
  }

  return addons;
}

/* eslint-enable @typescript-eslint/no-explicit-any */
