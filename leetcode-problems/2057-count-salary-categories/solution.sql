/* Write your T-SQL query statement below */
WITH Categories AS (
    SELECT 'Low Salary' AS category
    UNION ALL
    SELECT 'Average Salary' AS category
    UNION ALL
    SELECT 'High Salary' AS category
)
SELECT 
    c.category,
    COUNT(a.account_id) AS accounts_count
FROM Categories c
LEFT JOIN Accounts a ON c.category = CASE 
    WHEN a.income < 20000 THEN 'Low Salary'
    WHEN a.income >= 20000 AND a.income <= 50000 THEN 'Average Salary'
    ELSE 'High Salary'
END
GROUP BY c.category;
