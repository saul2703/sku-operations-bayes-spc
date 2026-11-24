# Product Context — Global Store & SKU Operations (`sku-operations-bayes-spc`) Workbench

## Purpose & Audience
An operational fulfillment, shrinkage-adjusted risk ranking, and statistical process control (SPC) workbench built for supply-chain directors, territory managers, and data science hiring managers reviewing **58,894 real-world order-item records** across **116 SKUs**, **49 categories**, **23 global regions**, and **4 shipping modes** (`DataCoSupplyChainDataset.csv`).

## Surface Mode
**Operate** (with structured **Read** reference tabs). Analysts use the workbench to:
1. Separate real operational failure from small-sample noise across all **116 SKUs** and **92 Region x Shipping Mode lanes** using **Empirical Bayes Beta-Binomial Shrinkage** and 95% credible intervals.
2. Monitor **37 monthly fulfillment periods** (`2015-01` through `2018-01`) with **Shewhart p-charts** (late-delivery rate) and **X-bar / S charts** (shipping lead-time variance) flagged by Western Electric / Nelson run rules.
3. Inspect the Top 10 vs. Bottom 10 shrinkage-ranked SKUs and lanes, plus the normalized Star Schema SQL DDL and analytical views.
