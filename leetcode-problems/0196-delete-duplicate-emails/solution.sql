/* Write your T-SQL query statement below */
with cte as (
    select id, email,
        row_number() over(partition by email order by id) as rn
    from person
)
delete from cte where rn > 1;
