import { describe, it, expect } from "vitest";
import {
  registerConnector,
  loadConnector,
  listConnectors,
  type BaseConnector,
} from "../../src/node/connectors/base.js";

function makeFakeConnector(name: string): BaseConnector {
  return {
    name,
    async connect() {},
    async read() {
      return [];
    },
    async close() {},
  };
}

describe("connector registry", () => {
  it("registerConnector + loadConnector round-trips", () => {
    registerConnector("test_fake_a", () => makeFakeConnector("test_fake_a"));
    const c = loadConnector("test_fake_a", {});
    expect(c.name).toBe("test_fake_a");
  });

  it("loadConnector with unknown name throws including registered list", () => {
    registerConnector("known_one", () => makeFakeConnector("known_one"));
    expect(() => loadConnector("does_not_exist", {})).toThrow(/Unknown connector/);
    expect(() => loadConnector("does_not_exist", {})).toThrow(/known_one/);
  });

  it("listConnectors includes registered connectors", () => {
    registerConnector("test_fake_b", () => makeFakeConnector("test_fake_b"));
    const names = listConnectors();
    expect(names).toContain("test_fake_b");
  });

  it("factory receives passed config", async () => {
    let received: unknown = null;
    registerConnector("test_fake_c", (cfg) => {
      received = cfg;
      return makeFakeConnector("test_fake_c");
    });
    loadConnector("test_fake_c", { foo: "bar", n: 42 });
    expect(received).toEqual({ foo: "bar", n: 42 });
  });
});
