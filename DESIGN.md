# Design System — Store & SKU Operations Bayes + SPC Operate Workbench

## Aesthetic Direction & Usage Scene
- **Mode:** `Operate` (precision fulfillment & process-control ledger with automatic `light dark` color-scheme parity and a manual theme toggle).
- **Dark Palette (Default / `prefers-color-scheme: dark`):** Slate obsidian canvas `#0b1120`, primary surface `#111827`, secondary control layer `#172135`, structural hairlines `#29374d`, primary ink `#f4f7fb`, secondary slate `#9db1cc`, muted `#7d93b2`.
- **Light Palette (`prefers-color-scheme: light`):** Crisp paper canvas `#f8fafc`, primary surface `#ffffff`, secondary control layer `#f1f5f9`, structural hairlines `#cbd5e1`, primary ink `#0f172a`, secondary slate `#475569`, muted `#64748b`.
- **Semantic Operational Accents (Dark / Light):**
  - In-Control / Low-Risk Benchmark: `#34d399` / `#047857`
  - Baseline / Information / Shrinkage Prior: `#38bdf8` / `#0284c7`
  - Watch / Nelson Run-Rule Shift: `#fbbf24` / `#b45309`
  - Out-of-Control / High-Risk SLA Breach: `#f472b6` / `#be185d`

## Typography & Hierarchy
- **Single Tuned Sans Family:** `"Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif` with `font-family: "Cascadia Mono", Consolas, monospace` reserved strictly for SKU IDs, lane codes, SQL DDL, and tabular figures.
- **Fixed `rem` Scale (No Fluid `clamp()`):** `0.75rem` (12px table badges), `0.8125rem` (13px dense cells), `0.9375rem` (15px body copy, 68ch max measure), `1.125rem` (18px panel title), `1.5rem` (24px workspace title).
- **Tabular Numerals:** `font-variant-numeric: tabular-nums` enforced across all ledgers, KPI strips, and comparison tables.

## Craft Floor & Modern Web Guidance Compliance
- **Zero Eyebrows / Kickers / Numbered Heading Prefixes.**
- **Zero Nested Cards:** Top KPI summary uses a continuous `1px` bordered horizontal ledger strip rather than disconnected hero cards.
- **Dual-Theme Charts (`<picture>` + `light-dark()`):** Every Matplotlib figure ships in both `images/<name>.png` (dark) and `images/<name>_light.png` (light), paired in `index.html` and `README.md` via `<picture>` and `prefers-color-scheme`.
- **Browser Surface Theming:** Custom `::selection`, `caret-color`, `:focus-visible` outline, `text-underline-offset: 0.22em`, and `scrollbar-color`.
- **Responsive Container-Query Tables & Layout Containment:** `.table-wrapper` uses `container-type: inline-size` with sticky headers and `@container (width < 600px)` stacked label injection; independent sections use `content-visibility: auto; contain-intrinsic-size: auto 300px auto 520px`.
