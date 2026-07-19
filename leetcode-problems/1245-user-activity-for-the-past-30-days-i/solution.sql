/* Write your T-SQL query statement below */
select
activity_date day,
count(distinct user_id) active_users
from activity
WHERE DATEDIFF(day, activity_date, '2019-07-27') < 30
  AND activity_date <= '2019-07-27'
group by activity_date 
