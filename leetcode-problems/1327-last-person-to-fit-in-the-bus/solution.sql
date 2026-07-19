/* Write your T-SQL query statement below */
WITH RunningWeights AS (
    SELECT 
        person_name,
        turn,
        -- Calculate the cumulative weight up to the current person's turn
        SUM(weight) OVER (ORDER BY turn ASC) AS total_weight
    FROM Queue
)
SELECT TOP 1 person_name
FROM RunningWeights
WHERE total_weight <= 1000
ORDER BY turn DESC;
