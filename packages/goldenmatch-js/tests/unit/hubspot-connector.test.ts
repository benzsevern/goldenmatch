import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
} from "vitest";
import { createHubSpotConnector } from "../../src/node/connectors/hubspot.js";

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    text: async () => "",
    json: async () => body,
  } as Response;
}

describe("createHubSpotConnector", () => {
  it("returns connector with proper shape", () => {
    const c = createHubSpotConnector({ apiKey: "key" });
    expect(c.name).toBe("hubspot");
    expect(typeof c.connect).toBe("function");
    expect(typeof c.read).toBe("function");
    expect(typeof c.close).toBe("function");
  });

  it("connect() is a no-op (no fetch)", async () => {
    const c = createHubSpotConnector({ apiKey: "key" });
    await c.connect();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("throws when called with raw SQL string", async () => {
    const c = createHubSpotConnector({ apiKey: "key" });
    await expect(c.read("SELECT * FROM contacts")).rejects.toThrow(
      /object query/i,
    );
  });

  it("read() returns rows merging id + properties", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        results: [
          { id: "1", properties: { firstname: "Ada", lastname: "Lovelace" } },
          { id: "2", properties: { firstname: "Alan", lastname: "Turing" } },
        ],
      }),
    );
    const c = createHubSpotConnector({ apiKey: "key" });
    const rows = await c.read({
      table: "contacts",
      columns: ["firstname", "lastname"],
      limit: 50,
    });
    expect(rows.length).toBe(2);
    expect(rows[0]!.id).toBe("1");
    expect(rows[0]!.firstname).toBe("Ada");
    expect(rows[1]!.lastname).toBe("Turing");

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/crm/v3/objects/contacts");
    expect(url).toContain("limit=50");
    expect(url).toContain("properties=firstname,lastname");

    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect((init.headers as Record<string, string>).Authorization).toBe(
      "Bearer key",
    );
  });

  it("paginates via paging.next.link", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          results: [{ id: "1", properties: { name: "A" } }],
          paging: { next: { link: "https://api.hubapi.com/crm/v3/objects/contacts?after=abc" } },
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          results: [{ id: "2", properties: { name: "B" } }],
        }),
      );

    const c = createHubSpotConnector({ apiKey: "key" });
    const rows = await c.read({ table: "contacts" });
    expect(rows.length).toBe(2);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const secondUrl = fetchMock.mock.calls[1]![0] as string;
    expect(secondUrl).toContain("after=abc");
  });

  it("throws on non-2xx response", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 403,
      text: async () => "forbidden",
      json: async () => ({}),
    } as Response);
    const c = createHubSpotConnector({ apiKey: "key" });
    await expect(c.read({ table: "contacts" })).rejects.toThrow(
      /HubSpot query failed/,
    );
  });
});
