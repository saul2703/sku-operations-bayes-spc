import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";

const ROOT = process.cwd();
const summaryPath = path.join(ROOT, "data", "sku_ops_summary.json");
const htmlPath = path.join(ROOT, "index.html");
const readmePath = path.join(ROOT, "README.md");
const sqlPath = path.join(ROOT, "sql", "sku_ops_star_schema.sql");
const productPath = path.join(ROOT, "PRODUCT.md");
const designPath = path.join(ROOT, "DESIGN.md");

assert.ok(fs.existsSync(summaryPath), "Missing data/sku_ops_summary.json");
assert.ok(fs.existsSync(htmlPath), "Missing index.html");
assert.ok(fs.existsSync(readmePath), "Missing README.md");
assert.ok(fs.existsSync(sqlPath), "Missing sql/sku_ops_star_schema.sql");
assert.ok(fs.existsSync(productPath), "Missing PRODUCT.md");
assert.ok(fs.existsSync(designPath), "Missing DESIGN.md");

const data = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));

// 1. Verify exact dataset reconciliation
assert.equal(data.dataset.total_order_items, 58894, "Total order items must equal 58,894");
assert.equal(data.dataset.unique_orders, 31160, "Unique orders must equal 31,160");
assert.equal(data.skus.length, 116, "Must have exactly 116 SKUs");
assert.equal(data.lanes.length, 92, "Must have exactly 92 Region x Shipping Mode lanes");
assert.equal(data.monthly_spc.length, 37, "Must have exactly 37 monthly SPC periods");

const skuOrderSum = data.skus.reduce((acc, s) => acc + s.n_orders, 0);
const laneOrderSum = data.lanes.reduce((acc, l) => acc + l.n_orders, 0);
const spcOrderSum = data.monthly_spc.reduce((acc, m) => acc + m.n_orders, 0);
assert.equal(skuOrderSum, 58894, "SKU order counts must reconcile to 58,894");
assert.equal(laneOrderSum, 58894, "Lane order counts must reconcile to 58,894");
assert.equal(spcOrderSum, 58894, "Monthly SPC order counts must reconcile to 58,894");

// 2. Verify Empirical Bayes posterior and credible interval integrity
for (const s of data.skus) {
  assert.ok(
    s.ci_low_pct <= s.eb_late_pct && s.eb_late_pct <= s.ci_high_pct,
    `Credible interval violation on SKU ${s.sku_id}`
  );
}
assert.equal(data.empirical_bayes.standard_class_sku_prior.true_between_variance, 0, "Within Standard Class, between-SKU variance collapses to zero");

// 3. Verify all 6 Matplotlib charts exist in both dark and light modes and are non-empty
const expectedCharts = [
  "ops_overview_preview",
  "empirical_bayes_shrinkage",
  "spc_p_chart_late_delivery",
  "spc_xbar_lead_time_drift",
  "lane_risk_heatmap_shipping_mode",
  "sku_financial_exposure_pareto",
];
for (const name of expectedCharts) {
  for (const suffix of ["", "_light"]) {
    const img = `${name}${suffix}.png`;
    const p = path.join(ROOT, "images", img);
    assert.ok(fs.existsSync(p), `Missing chart: images/${img}`);
    assert.ok(fs.statSync(p).size > 30000, `Chart too small: images/${img}`);
  }
}

// 4. Verify Impeccable craft-floor, dual light/dark theme parity, and JS syntax in index.html & README.md
const html = fs.readFileSync(htmlPath, "utf-8");
const readme = fs.readFileSync(readmePath, "utf-8");
assert.ok(!html.includes('class="eyebrow"'), "Impeccable craft-floor ban: no eyebrow/kicker above headings");
assert.ok(html.includes('<meta name="color-scheme" content="light dark"'), "index.html must declare light dark color-scheme");
assert.ok(html.includes("light-dark("), "index.html must use CSS light-dark() tokens");
assert.ok(html.includes("btnThemeToggle") && html.includes("<picture data-chart="), "index.html must include theme toggle and dual-mode <picture> charts");
assert.ok(html.includes("liveChartShrinkage") && html.includes("liveChartExposure"), "index.html must include interactive filter-reactive SVG charts");
assert.ok(html.includes("content-visibility: auto"), "modern-web-guidance: must include content-visibility: auto containment");
assert.ok(html.includes("container-type: inline-size"), "modern-web-guidance: must include container-type: inline-size for responsive tables");
assert.ok(html.includes("skuGrid") && html.includes("spcGrid") && html.includes("laneGrid"), "index.html must include all three AG Grid containers");
assert.ok(html.includes("empty-filter-banner") && html.includes("btnResetFilters"), "index.html must include recoverable empty filter state");
assert.ok(html.includes('typeof agGrid !== "undefined"'), "index.html must guard against offline AG Grid CDN failure");
assert.ok(html.includes("sku_ops_star_schema.sql") && html.includes("fact_order_items"), "index.html must include Star Schema SQL DDL");
assert.ok(readme.includes("prefers-color-scheme: light") && readme.includes("prefers-color-scheme: dark"), "README.md must pair dark and light <picture> sources");
new Function(html.split("<script>").pop().split("</script>")[0]);

console.log("ALL VERIFICATION CHECKS PASSED:");
console.log(`  - Reconciled 58,894 order items across 116 SKUs, 92 lanes, and 37 SPC months`);
console.log(`  - Empirical Bayes Beta(${data.empirical_bayes.sku_prior.alpha0}, ${data.empirical_bayes.sku_prior.beta0}) credible intervals verified`);
console.log(`  - Standard Class between-SKU variance = 0.0 (proves SLA promise tier dominates SKU differences)`);
console.log(`  - Impeccable Operate Workbench + Star Schema SQL + 12 Dark/Light PNG charts verified`);
