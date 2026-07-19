/* Write your T-SQL query statement below */
WITH PolicyCounts AS (
    SELECT 
        tiv_2016,
        -- Counts how many policyholders share the same 2015 value
        COUNT(*) OVER(PARTITION BY tiv_2015) AS tiv_2015_count,
        -- Counts how many policyholders are at this exact location
        COUNT(*) OVER(PARTITION BY lat, lon) AS location_count
    FROM Insurance
)
SELECT 
    ROUND(SUM(tiv_2016), 2) AS tiv_2016
FROM PolicyCounts
WHERE 
    tiv_2015_count > 1     -- Condition 1: Must be shared with someone else
    AND location_count = 1; -- Condition 2: Must be a completely unique city/spot
