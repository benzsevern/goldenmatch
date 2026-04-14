import { describe, it, expect } from "vitest";
import { detectDomain, extractFeatures } from "../../src/core/domain.js";
import type { Row } from "../../src/core/types.js";

describe("detectDomain", () => {
  it("classifies electronics-like columns as 'product'", () => {
    const profile = detectDomain(["brand", "model", "sku", "price"]);
    expect(profile.name).toBe("product");
    expect(profile.confidence).toBeGreaterThan(0);
    expect(profile.featureColumns).toContain("brand");
    expect(profile.featureColumns).toContain("model");
  });

  it("classifies person columns as 'person'", () => {
    const profile = detectDomain(["first_name", "last_name", "email", "phone"]);
    expect(profile.name).toBe("person");
    expect(profile.confidence).toBeGreaterThan(0);
  });

  it("classifies bibliographic columns as 'bibliographic'", () => {
    const profile = detectDomain(["title", "authors", "year", "venue"]);
    expect(profile.name).toBe("bibliographic");
    expect(profile.confidence).toBeGreaterThan(0);
  });

  it("returns 'generic' when no signatures match", () => {
    const profile = detectDomain(["foo", "bar", "baz"]);
    expect(profile.name).toBe("generic");
    expect(profile.confidence).toBe(0);
    expect(profile.featureColumns).toEqual([]);
  });

  it("reports text columns (description, title, notes, body)", () => {
    const profile = detectDomain(["title", "description", "notes", "brand"]);
    // All three TEXT_NAME_RE hits show up in textColumns regardless of winner.
    for (const col of ["title", "description", "notes"]) {
      expect(profile.textColumns).toContain(col);
    }
  });
});

describe("extractFeatures", () => {
  it("adds __brand__/__model__/__version__ columns for product rows with signal", () => {
    const rows: Row[] = [
      { brand: "Apple", model: "iPhone-12", description: "A product" },
      { brand: "Samsung", model: "SGH-M220", description: "Phone" },
    ];
    const profile = detectDomain(Object.keys(rows[0]!));
    const { rows: enriched, lowConfidenceIds } = extractFeatures(rows, profile);
    expect(enriched.length).toBe(2);
    expect(enriched[0]!["__brand__"]).toBe("apple");
    expect(enriched[0]!["__model__"]).toBe("IPHONE12");
    expect(enriched[1]!["__brand__"]).toBe("samsung");
    // Both rows have brand + model => not in lowConfidenceIds.
    expect(lowConfidenceIds).not.toContain(0);
    expect(lowConfidenceIds).not.toContain(1);
  });

  it("reports low-confidence rows when features can't be extracted", () => {
    const rows: Row[] = [
      { brand: null, model: null, description: "foo bar" },
      { brand: null, model: null, description: null },
    ];
    const profile = detectDomain(["brand", "model", "description"]);
    const { lowConfidenceIds } = extractFeatures(rows, profile, 0.5);
    // Both rows have 0/3 features extracted -> confidence 0 < 0.5.
    expect(lowConfidenceIds).toContain(0);
    expect(lowConfidenceIds).toContain(1);
  });

  it("generic domain: returns rows unchanged and no low-confidence ids", () => {
    const rows: Row[] = [{ foo: "a" }, { foo: "b" }];
    const profile = detectDomain(["foo"]);
    const { rows: out, lowConfidenceIds } = extractFeatures(rows, profile);
    expect(out.length).toBe(2);
    expect(lowConfidenceIds).toEqual([]);
    // No underscore columns injected.
    expect(Object.keys(out[0]!)).toEqual(["foo"]);
  });
});
