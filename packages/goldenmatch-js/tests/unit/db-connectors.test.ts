import { describe, it, expect } from "vitest";
import { createPostgresConnector } from "../../src/node/db/postgres.js";
import { createDuckDBConnector } from "../../src/node/backends/duckdb.js";

describe("createPostgresConnector", () => {
  it("throws clear install message when 'pg' is not installed", () => {
    // pg is an optional peer dep; not in this dev env.
    try {
      const c = createPostgresConnector({ host: "localhost" });
      // If pg happens to be installed, just validate the shape.
      expect(typeof c.connect).toBe("function");
      expect(typeof c.query).toBe("function");
      expect(typeof c.close).toBe("function");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      expect(msg).toMatch(/pg/);
      expect(msg).toMatch(/install/i);
    }
  });
});

describe("createDuckDBConnector", () => {
  it("throws clear install message when '@duckdb/node-api' is not installed", async () => {
    try {
      const c = await createDuckDBConnector();
      // If installed, validate connector shape.
      expect(typeof c.readTable).toBe("function");
      expect(typeof c.writeTable).toBe("function");
      expect(typeof c.close).toBe("function");
      c.close();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      expect(msg).toMatch(/@duckdb\/node-api/);
      expect(msg).toMatch(/install/i);
    }
  });
});
