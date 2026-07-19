/* Write your T-SQL query statement below */
select
m.employee_id,
m.name,
count(e.reports_to) reports_count,
round(avg(e.age * 1.0), 0) average_age
from employees e
join employees m
on e.reports_to = m.employee_id
group by m.employee_id, m.name
order by m.employee_id;
