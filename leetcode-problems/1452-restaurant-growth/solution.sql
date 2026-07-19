/* Write your T-SQL query statement below */
WITH DailySales AS (
    -- Step 1: Combine multiple customer orders into a single sum per day
    SELECT 
        visited_on,
        SUM(amount) AS daily_amount
    FROM Customer
    GROUP BY visited_on
),
RollingMetrics AS (
    -- Step 2: Use window functions to look back exactly 6 rows (days) + current row
    SELECT 
        visited_on,
        SUM(daily_amount) OVER(
            ORDER BY visited_on 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS amount,
        ROUND(
            AVG(CAST(daily_amount AS FLOAT)) OVER(
                ORDER BY visited_on 
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 
            2
        ) AS average_amount,
        -- Generate a row number to easily filter out the first 6 incomplete days
        ROW_NUMBER() OVER(ORDER BY visited_on) AS row_num
    FROM DailySales
)
-- Step 3: Fetch only the rows that have a complete 7-day history window
SELECT 
    visited_on, 
    amount, 
    average_amount
FROM RollingMetrics
WHERE row_num >= 7
ORDER BY visited_on ASC;
