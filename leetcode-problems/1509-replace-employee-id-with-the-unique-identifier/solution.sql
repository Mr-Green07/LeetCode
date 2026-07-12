/* Write your T-SQL query statement below */
select
unique_id,
name
from employees ei
left join EmployeeUNI e
    on ei.id = e.id
 
