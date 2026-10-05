// Vite builds the interface into frontend/dist, which the API then serves itself
// (see app/main.py). One address for pages and API means: no CORS, and the
// sign-in cookie works without special settings.
//
// While developing ("npm run dev", http://localhost:5173) the page comes from
// Vite and every API path is passed on to the API on port 8000.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const apiPaths = ["/auth", "/documents", "/investigations", "/chat", "/memory", "/comparisons", "/actions", "/health"];
const target = process.env.VERIAGENT_API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: Object.fromEntries(apiPaths.map((path) => [path, { target, changeOrigin: false }])) },
  build: { outDir: "dist", sourcemap: false },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
