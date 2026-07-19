SELECT 
    p.product_id,
    -- ISNULL catches products with 0 sales. 
    -- CAST prevents SQL Server from cutting off the decimals during division.
    ISNULL(
        ROUND(
            CAST(SUM(p.price * u.units) AS DECIMAL(18,2)) / SUM(u.units), 
            2
        ), 
        0
    ) AS average_price
FROM 
    Prices p
LEFT JOIN 
    UnitsSold u ON p.product_id = u.product_id 
    AND u.purchase_date BETWEEN p.start_date AND p.end_date
GROUP BY 
    p.product_id;
