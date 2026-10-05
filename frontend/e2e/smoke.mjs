// A walk through the whole interface in a real browser: create an account, upload,
// investigate, approve with a follow-up ticket, inspect a file, compare, memory,
// reload, sign out and in again.
//
// It needs a running server with the offline model and the ticket server:
//
//   LLM_PROVIDER=mock EMBEDDING_PROVIDER=hash ACTION_SERVER=tickets uvicorn app.main:app --port 8000
//
// and Playwright, which is not a dependency of this project (it downloads a browser):
//
//   npm install --no-save playwright && npx playwright install chromium
//   node e2e/smoke.mjs                      (BASE=http://localhost:8000 by default)
//
// Screenshots are written to e2e/screenshots.
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const base = process.env.BASE ?? "http://localhost:8000";
const samples = resolve(here, "../../samples");
const out = process.env.OUT ?? resolve(here, "screenshots");
mkdirSync(out, { recursive: true });
const email = `e2e-${Date.now()}@example.com`;
const password = "a-long-test-password";

const browser = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
const context = await browser.newContext({ viewport: { width: 1500, height: 950 } });
const page = await context.newPage();
const problems = [];
// Two 401 answers are expected: the first visit has no session yet, and the wrong password below.
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("401")) problems.push("console: " + m.text()); });
page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
const step = (text) => console.log("- " + text);

await page.goto(base);
await page.getByRole("heading", { name: "VeriAgent" }).waitFor();
await page.screenshot({ path: `${out}/01-signin.png` });
step("sign-in page shown");

await page.getByRole("tab", { name: "Create account" }).click();
await page.getByLabel("Name").fill("Mariam");
await page.getByLabel("Email").fill(email);
await page.getByLabel("Password").fill(password);
await page.getByRole("button", { name: "Create account" }).click();
await page.getByRole("heading", { name: "Documents", exact: true }).waitFor();
step("account created, workspace shown");

await page.locator('input[type="file"]').setInputFiles([
  `${samples}/Contract.pdf`, `${samples}/Invoice_102.pdf`, `${samples}/Procurement_Policy.pdf`,
]);
await page.getByText("Procurement_Policy.pdf").waitFor();
await page.waitForFunction(() => document.querySelectorAll(".document").length === 3 && !document.body.innerText.includes("Reading..."), null, { timeout: 30000 });
step("three documents uploaded and ready");

await page.getByRole("button", { name: /Review this invoice against the contract/ }).click();
await page.getByText("Your decision is needed").waitFor({ timeout: 60000 });
await page.getByText("Risk level", { exact: true }).waitFor();
step("investigation paused for a decision; result panel filled");
await page.screenshot({ path: `${out}/02-result-pending.png` });

// Approval with the follow-up action ticked.
await page.getByLabel(/Create an issue in the local ticket list/).check();
await page.getByText("Read exactly what would be sent").click();
await page.getByLabel("Note (optional, shown in the report)").fill("Checked with the supplier by phone.");
await page.screenshot({ path: `${out}/03-approval.png` });
await page.getByRole("button", { name: "Send decision" }).click();
await page.getByText(/Approved by Mariam/).waitFor({ timeout: 60000 });
await page.getByText("Created", { exact: true }).waitFor();
if (await page.getByText("Waiting for your decision").count()) problems.push("the chat still says the answer is waiting for a decision");
step("approved; the ticket was created through MCP");
await page.screenshot({ path: `${out}/04-result-final.png` });

// Upload the altered invoice and look at its inspection.
await page.locator('input[type="file"]').setInputFiles([`${samples}/forensics/Invoice_205_altered.pdf`]);
await page.getByText("Signs of editing").waitFor({ timeout: 30000 });
await page.getByRole("button", { name: /Invoice_205_altered\.pdf/ }).first().click();
await page.getByText("File inspection").waitFor();
await page.getByText(/covered by a shape/).waitFor();
await page.getByText("Key facts").waitFor();
await page.getByText("INV-205").first().waitFor({ timeout: 30000 });
await page.screenshot({ path: `${out}/05-document.png` });
await page.keyboard.press("Escape");
step("document details: inspection and key facts shown");

// Compare contract and the two invoices.
for (const name of ["Contract.pdf", "Invoice_102.pdf", "Invoice_205_altered.pdf"]) {
  await page.getByLabel(`Select ${name} for comparison`).check();
}
await page.getByRole("button", { name: "Compare 3" }).click();
await page.getByText("Key facts, side by side").waitFor({ timeout: 30000 });
await page.getByText("Differences found").waitFor();
await page.screenshot({ path: `${out}/06-compare.png` });
await page.keyboard.press("Escape");
step("comparison shown");

// Memory.
await page.getByRole("button", { name: "Memory" }).click();
await page.getByRole("heading", { name: "Preferences" }).waitFor();
await page.getByLabel("New note").fill("Brightline invoices are paid quarterly in arrears.");
await page.getByRole("button", { name: "Add" }).click();
await page.getByText("Brightline invoices are paid quarterly in arrears.").waitFor();
await page.screenshot({ path: `${out}/07-memory.png` });
await page.keyboard.press("Escape");
step("memory: note added");

// History opens a stored report.
await page.locator(".history-item").first().click();
await page.getByText("Why this risk level").waitFor();
step("history opens a stored report");

// A reload keeps the session (refresh cookie), without any token in storage.
const stored = await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }));
if (/eyJ/.test(stored)) problems.push("a token was found in browser storage");
await page.reload();
await page.getByRole("heading", { name: "Documents", exact: true }).waitFor();
step("still signed in after a reload; no token in browser storage");

// Sign out and in again.
await page.getByRole("button", { name: /Mariam/ }).click();
await page.getByRole("button", { name: "Sign out" }).click();
await page.getByRole("tab", { name: "Sign in" }).waitFor();
await page.getByLabel("Email").fill(email);
await page.getByLabel("Password").fill("wrong-password-123");
await page.getByRole("button", { name: "Sign in" }).last().click();
await page.getByRole("alert").waitFor();
await page.getByLabel("Password").fill(password);
await page.getByRole("button", { name: "Sign in" }).last().click();
await page.getByRole("heading", { name: "Documents", exact: true }).waitFor();
await page.getByText("Contract.pdf").first().waitFor();
step("signed out, wrong password refused, signed in again; documents still there");

// Narrow screen.
await page.setViewportSize({ width: 420, height: 900 });
await page.screenshot({ path: `${out}/08-narrow.png`, fullPage: true });

await browser.close();
if (problems.length) { console.log("PROBLEMS:\n" + problems.join("\n")); process.exit(1); }
console.log("OK - no console errors");
