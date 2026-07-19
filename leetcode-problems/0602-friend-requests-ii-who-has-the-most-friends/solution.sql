/* Write your T-SQL query statement below */
WITH AllConnections AS (
    -- Select all IDs who sent a request
    SELECT requester_id AS id 
    FROM RequestAccepted
    
    UNION ALL
    
    -- Select all IDs who accepted a request
    SELECT accepter_id AS id 
    FROM RequestAccepted
)
SELECT TOP 1 
    id, 
    COUNT(*) AS num
FROM AllConnections
GROUP BY id
ORDER BY num DESC;
