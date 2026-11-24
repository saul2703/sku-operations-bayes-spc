-- =============================================================================
-- Global Store & SKU Operations (DataCo Supply Chain — 58,894 Order Items)
-- Relational Star Schema DDL & Analytical Views for Empirical Bayes + SPC
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS sku_ops_analytics;

-- 1. DIMENSION TABLE: SKU CATALOG (116 Unique Products across 49 Categories)
CREATE TABLE sku_ops_analytics.dim_sku (
    sku_id          INTEGER      PRIMARY KEY, -- Order Item Cardprod Id
    product_name    VARCHAR(160) NOT NULL,
    category_id     INTEGER      NOT NULL,
    category_name   VARCHAR(80)  NOT NULL,
    department_id   INTEGER      NOT NULL,
    department_name VARCHAR(80)  NOT NULL,
    unit_price      NUMERIC(12,2) NOT NULL
);

-- 2. DIMENSION TABLE: ORDER GEOGRAPHY (5 Markets -> 23 Order Regions)
CREATE TABLE sku_ops_analytics.dim_geography (
    geo_key         SERIAL       PRIMARY KEY,
    market          VARCHAR(40)  NOT NULL,    -- Europe, LATAM, Pacific Asia, USCA, Africa
    order_region    VARCHAR(60)  NOT NULL,    -- 23 regions (e.g., Western Europe, Central America)
    order_country   VARCHAR(80)  NOT NULL,
    order_state     VARCHAR(80),
    order_city      VARCHAR(80),
    UNIQUE (market, order_region, order_country, order_state, order_city)
);

-- 3. DIMENSION TABLE: CUSTOMERS (3 Segments: Consumer, Corporate, Home Office)
CREATE TABLE sku_ops_analytics.dim_customer (
    customer_id      INTEGER     PRIMARY KEY,
    customer_segment VARCHAR(30) NOT NULL,
    customer_country VARCHAR(80) NOT NULL,
    customer_state   VARCHAR(40),
    customer_city    VARCHAR(80)
);

-- 4. FACT TABLE: ORDER ITEMS (58,894 Lines across 31,160 Orders, Jan 2015 - Jan 2018)
CREATE TABLE sku_ops_analytics.fact_order_items (
    order_item_id           INTEGER       PRIMARY KEY,
    order_id                INTEGER       NOT NULL,
    sku_id                  INTEGER       NOT NULL REFERENCES sku_ops_analytics.dim_sku(sku_id),
    geo_key                 INTEGER       NOT NULL REFERENCES sku_ops_analytics.dim_geography(geo_key),
    customer_id             INTEGER       NOT NULL REFERENCES sku_ops_analytics.dim_customer(customer_id),
    order_timestamp         TIMESTAMP     NOT NULL, -- Normalized from mixed MM/DD/YYYY & MM-DD-YYYY
    shipping_timestamp      TIMESTAMP     NOT NULL,
    shipping_mode           VARCHAR(30)   NOT NULL, -- Standard, Second Class, First Class, Same Day
    payment_type            VARCHAR(20)   NOT NULL, -- DEBIT, TRANSFER, PAYMENT, CASH
    order_status            VARCHAR(30)   NOT NULL,
    delivery_status         VARCHAR(30)   NOT NULL,
    days_shipping_real      SMALLINT      NOT NULL,
    days_shipment_scheduled SMALLINT      NOT NULL,
    lead_time_variance_days SMALLINT      NOT NULL, -- real - scheduled (>0 means behind SLA)
    is_late_delivery        BOOLEAN       NOT NULL, -- Late_delivery_risk (1 -> TRUE, 0 -> FALSE)
    is_canceled_shipping    BOOLEAN       NOT NULL, -- Delivery Status = 'Shipping canceled'
    quantity                SMALLINT      NOT NULL,
    gross_sales             NUMERIC(12,2) NOT NULL,
    discount_amount         NUMERIC(12,2) NOT NULL,
    net_item_total          NUMERIC(12,2) NOT NULL,
    profit_per_order        NUMERIC(12,2) NOT NULL
);

-- =============================================================================
-- ANALYTICAL VIEW 1: SKU-LEVEL EMPIRICAL BAYES INPUTS (116 SKUs)
-- Separates noisy low-volume spikes from statistically credible high-volume bottlenecks
-- =============================================================================
CREATE OR REPLACE VIEW sku_ops_analytics.vw_sku_empirical_bayes_inputs AS
SELECT
    s.sku_id,
    s.product_name,
    s.category_name,
    s.department_name,
    COUNT(f.order_item_id)                                              AS n_orders,
    SUM(CASE WHEN f.is_late_delivery THEN 1 ELSE 0 END)                 AS late_orders,
    ROUND(100.0 * AVG(f.is_late_delivery::INT)::NUMERIC, 2)             AS raw_late_pct,
    ROUND(AVG(f.lead_time_variance_days)::NUMERIC, 3)                   AS mean_delay_days,
    ROUND(SUM(f.gross_sales)::NUMERIC, 2)                               AS total_sales,
    ROUND(SUM(CASE WHEN f.is_late_delivery THEN f.gross_sales ELSE 0 END)::NUMERIC, 2) AS late_sales_exposure,
    ROUND(SUM(f.profit_per_order)::NUMERIC, 2)                          AS total_profit
FROM sku_ops_analytics.dim_sku s
JOIN sku_ops_analytics.fact_order_items f ON f.sku_id = s.sku_id
GROUP BY s.sku_id, s.product_name, s.category_name, s.department_name
ORDER BY late_sales_exposure DESC;

-- =============================================================================
-- ANALYTICAL VIEW 2: MONTHLY STATISTICAL PROCESS CONTROL (SPC) SERIES (37 MONTHS)
-- Feeds Shewhart p-chart (late rate) and X-bar / S chart (lead-time variance)
-- =============================================================================
CREATE OR REPLACE VIEW sku_ops_analytics.vw_monthly_spc_control_inputs AS
SELECT
    TO_CHAR(DATE_TRUNC('month', f.order_timestamp), 'YYYY-MM')          AS order_month,
    COUNT(f.order_item_id)                                              AS n_orders,
    SUM(CASE WHEN f.is_late_delivery THEN 1 ELSE 0 END)                 AS late_orders,
    ROUND(100.0 * AVG(f.is_late_delivery::INT)::NUMERIC, 2)             AS late_pct,
    ROUND(100.0 * AVG(f.is_canceled_shipping::INT)::NUMERIC, 2)         AS canceled_pct,
    ROUND(AVG(f.lead_time_variance_days)::NUMERIC, 3)                   AS mean_delay_days,
    ROUND(STDDEV_SAMP(f.lead_time_variance_days)::NUMERIC, 3)           AS stddev_delay_days,
    ROUND(SUM(f.gross_sales)::NUMERIC, 2)                               AS monthly_sales
FROM sku_ops_analytics.fact_order_items f
GROUP BY DATE_TRUNC('month', f.order_timestamp)
ORDER BY order_month;
