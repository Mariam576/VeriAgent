import { describe, expect, it } from "vitest";
import { formatValue, safeLink, titleCase } from "./format";

describe("formatValue", () => {
  it("writes amounts the way the reports do", () => {
    expect(formatValue(58000, "GBP")).toBe("£58,000");
    expect(formatValue(1234.5, "USD")).toBe("$1,234.50");
    expect(formatValue(-18000, "GBP")).toBe("-£18,000");
    expect(formatValue(1250000, "PKR")).toBe("PKR 1,250,000");
    expect(formatValue(30, "days")).toBe("30 days");
    expect(formatValue(45, "%")).toBe("45%");
    expect(formatValue(44, null)).toBe("44");
    expect(formatValue(null, "GBP")).toBe("");
  });
});

describe("safeLink", () => {
  it("accepts GitHub issue links and nothing else", () => {
    expect(safeLink("https://github.com/Mariam576/VeriAgent/issues/17")).toBe(
      "https://github.com/Mariam576/VeriAgent/issues/17",
    );
    expect(safeLink("https://evil.example/github.com/a/b/issues/1")).toBeNull();
    expect(safeLink("javascript:alert(1)")).toBeNull();
    expect(safeLink("https://github.com/a/b/issues/1?x=<script>")).toBeNull();
    expect(safeLink(null)).toBeNull();
  });
});

describe("titleCase", () => {
  it("makes machine names readable", () => {
    expect(titleCase("amount_mismatch")).toBe("Amount mismatch");
  });
});
