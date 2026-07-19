/* Write your T-SQL query statement below */
select
*
from users
where mail like '%@leetcode.com' COLLATE SQL_Latin1_General_CP1_CS_AS
AND LEFT(mail, 1) LIKE '[a-zA-Z]'
AND LEFT(mail, LEN(mail) - 13) NOT LIKE '%[^a-zA-Z0-9_.-]%';
