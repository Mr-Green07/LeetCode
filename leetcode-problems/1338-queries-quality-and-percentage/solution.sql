/* Write your T-SQL query statement below */
select
query_name,
round(avg(rating * 1.0 / position), 2) quality,
round(sum(
    case when rating<3 then 1 else 0 end
    ) * 100.0 /count(*), 2
) poor_query_percentage
from Queries 
group by query_name

