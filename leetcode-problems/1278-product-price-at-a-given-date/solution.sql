/* Write your T-SQL query statement below */
WITH LastChangedPrices AS (
    SELECT 
        product_id,
        new_price,
        ROW_NUMBER() OVER(PARTITION BY product_id ORDER BY change_date DESC) as rn
    FROM Products
    WHERE change_date <= '2019-08-16'
),
UniqueProducts AS (
    SELECT DISTINCT product_id FROM Products
)
SELECT 
    p.product_id,
    ISNULL(lcp.new_price, 10) AS price
FROM UniqueProducts p
LEFT JOIN LastChangedPrices lcp 
    ON p.product_id = lcp.product_id AND lcp.rn = 1;
