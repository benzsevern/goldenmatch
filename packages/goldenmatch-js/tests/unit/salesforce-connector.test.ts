import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
} from "vitest";
import { createSalesforceConnector } from "../../src/node/connectors/salesforce.js";

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

describe("createSalesforceConnector", () => {
  it("returns connector with name 'salesforce'", () => {
    const c = createSalesforceConnector({
      instanceUrl: "https://example.my.salesforce.com",
      accessToken: "token",
    });
    expect(c.name).toBe("salesforce");
    expect(typeof c.connect).toBe("function");
    expect(typeof c.read).toBe("function");
    expect(typeof c.close).toBe("function");
  });

  it("read() before connect() throws 'Not connected'", async () => {
    const c = createSalesforceConnector({
      instanceUrl: "https://example.my.salesforce.com",
      // no accessToken
    });
    await expect(c.read({ table: "Account" })).rejects.toThrow(/not connected/i);
  });

  it("connect() does OAuth password grant", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        access_token: "TOKEN_123",
        instance_url: "https://x.my.salesforce.com",
      }),
    );
    const c = createSalesforceConnector({
      instanceUrl: "https://example.my.salesforce.com",
      clientId: "cid",
      clientSecret: "secret",
      username: "u",
      password: "p",
      securityToken: "tok",
    });
    await c.connect();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/services/oauth2/token");
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(init.method).toBe("POST");
    // Body should contain combined password+token
    expect(String(init.body)).toContain("password=ptok");
  });

  it("connect() with pre-issued accessToken is no-op (no fetch)", async () => {
    const c = createSalesforceConnector({
      instanceUrl: "https://example.my.salesforce.com",
      accessToken: "preissued",
    });
    await c.connect();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("read() with object query builds SOQL and fetches paginated results", async () => {
    // First page returns nextRecordsUrl; second page returns final.
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          records: [{ Id: "001A", Name: "Acme" }],
          done: false,
          nextRecordsUrl: "/services/data/v61.0/query/01g0000NEXT",
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          records: [{ Id: "001B", Name: "Beta" }],
          done: true,
        }),
      );

    const c = createSalesforceConnector({
      instanceUrl: "https://example.my.salesforce.com",
      accessToken: "TOKEN",
    });
    const rows = await c.read({ table: "Account", columns: ["Id", "Name"], limit: 10 });
    expect(rows.length).toBe(2);
    expect(rows[0]!.Id).toBe("001A");
    expect(rows[1]!.Id).toBe("001B");
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const firstUrl = fetchMock.mock.calls[0]![0] as string;
    expect(firstUrl).toContain("/services/data/v61.0/query");
    expect(firstUrl).toContain("Id%2CName"); // url-encoded comma
    expect(firstUrl).toContain("LIMIT%2010"); // encodeURIComponent uses %20 for space

    const secondUrl = fetchMock.mock.calls[1]![0] as string;
    expect(secondUrl).toContain("01g0000NEXT");
  });

  it("read() with raw SOQL string passes it through", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ records: [{ Id: "X" }], done: true }),
    );
    const c = createSalesforceConnector({
      instanceUrl: "https://example.my.salesforce.com",
      accessToken: "T",
    });
    const rows = await c.read("SELECT Id FROM Account WHERE Name = 'Foo'");
    expect(rows.length).toBe(1);
    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("WHERE%20Name");
  });

  it("read() throws on non-2xx response", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: async () => "Unauthorized",
      json: async () => ({}),
    } as Response);
    const c = createSalesforceConnector({
      instanceUrl: "https://example.my.salesforce.com",
      accessToken: "T",
    });
    await expect(c.read({ table: "Account" })).rejects.toThrow(/Salesforce query failed/);
  });
});
