/* Write your T-SQL query statement below */
SELECT 
    id,
    CASE 
        -- If it's an ODD seat number, get the NEXT student's name
        WHEN id % 2 <> 0 THEN ISNULL(LEAD(student) OVER(ORDER BY id), student)
        -- If it's an EVEN seat number, get the PREVIOUS student's name
        ELSE LAG(student) OVER(ORDER BY id)
    END AS student
FROM Seat
ORDER BY id ASC;
