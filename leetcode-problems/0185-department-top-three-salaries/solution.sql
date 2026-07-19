/* Write your T-SQL query statement below */
select distinct
d.name as Department,
e.name as Employee,
e.salary as Salary
FROM (
    SELECT
        name,
        salary,
        departmentId,
        DENSE_RANK() OVER (PARTITION BY departmentId ORDER BY salary DESC) AS rnk
    FROM Employee
) e
JOIN Department d
    ON e.departmentId = d.id
WHERE e.rnk <= 3;
