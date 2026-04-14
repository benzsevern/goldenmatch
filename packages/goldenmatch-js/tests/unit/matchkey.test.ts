import { describe, it, expect } from "vitest";
import {
  computeMatchkeyValue,
  computeMatchkeys,
  addRowIds,
  addSourceColumn,
} from "../../src/core/index.js";
import type { MatchkeyConfig, Row } from "../../src/core/index.js";

describe("computeMatchkeyValue", () => {
  it("single field", () => {
    const row: Row = { email: "john@example.com" };
    const mk: MatchkeyConfig = {
      name: "email",
      type: "exact",
      fields: [{ field: "email", transforms: ["lowercase"], scorer: "exact", weight: 1.0 }],
    };
    expect(computeMatchkeyValue(row, mk)).toBe("john@example.com");
  });

  it("multiple fields joined with ||", () => {
    const row: Row = { first: "John", last: "Smith" };
    const mk: MatchkeyConfig = {
      name: "name",
      type: "exact",
      fields: [
        { field: "first", transforms: [], scorer: "exact", weight: 1.0 },
        { field: "last", transforms: [], scorer: "exact", weight: 1.0 },
      ],
    };
    expect(computeMatchkeyValue(row, mk)).toBe("John||Smith");
  });

  it("null field returns null matchkey", () => {
    const row: Row = { email: null };
    const mk: MatchkeyConfig = {
      name: "email",
      type: "exact",
      fields: [{ field: "email", transforms: [], scorer: "exact", weight: 1.0 }],
    };
    expect(computeMatchkeyValue(row, mk)).toBe(null);
  });

  it("applies transform chain", () => {
    const row: Row = { email: "  JOHN@X.COM  " };
    const mk: MatchkeyConfig = {
      name: "email",
      type: "exact",
      fields: [
        { field: "email", transforms: ["lowercase", "strip"], scorer: "exact", weight: 1.0 },
      ],
    };
    expect(computeMatchkeyValue(row, mk)).toBe("john@x.com");
  });
});

describe("computeMatchkeys", () => {
  it("adds __mk_{name}__ columns", () => {
    const rows: Row[] = [{ email: "a@x.com" }, { email: "b@x.com" }];
    const mks: MatchkeyConfig[] = [
      {
        name: "email_mk",
        type: "exact",
        fields: [{ field: "email", transforms: [], scorer: "exact", weight: 1.0 }],
      },
    ];
    const out = computeMatchkeys(rows, mks);
    expect(out[0]!.__mk_email_mk__).toBe("a@x.com");
    expect(out[1]!.__mk_email_mk__).toBe("b@x.com");
  });
});

describe("addRowIds", () => {
  it("adds sequential __row_id__ starting at 0", () => {
    const rows: Row[] = [{ a: 1 }, { a: 2 }];
    const out = addRowIds(rows);
    expect(out[0]!.__row_id__).toBe(0);
    expect(out[1]!.__row_id__).toBe(1);
  });

  it("supports offset", () => {
    const rows: Row[] = [{ a: 1 }, { a: 2 }];
    const out = addRowIds(rows, 10);
    expect(out[0]!.__row_id__).toBe(10);
    expect(out[1]!.__row_id__).toBe(11);
  });
});

describe("addSourceColumn", () => {
  it("adds __source__ to every row", () => {
    const rows: Row[] = [{ a: 1 }, { a: 2 }];
    const out = addSourceColumn(rows, "crm");
    for (const r of out) {
      expect(r.__source__).toBe("crm");
    }
  });
});
