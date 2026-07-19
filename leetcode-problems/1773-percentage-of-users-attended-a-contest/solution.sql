-- /* Write your T-SQL query statement below */

-- select
-- contest_id,
-- -- round(avg(cast(r.user_id as float) / count(u.user_id) * 100), 2) percentage
-- round(count(user_id) * 100 / (select count(*) from users), 2) percentage
-- from register
-- group by contest_id
-- order by percentage desc, contest_id


WITH total_users AS (
    SELECT COUNT(DISTINCT user_id) AS [total]
    FROM Users
)
SELECT DISTINCT
    contest_id,
    ROUND(COUNT(user_id) OVER(PARTITION BY contest_id) * 100.0 / total, 2) AS [percentage]
FROM Register
CROSS JOIN total_users
ORDER BY percentage DESC, contest_id;
