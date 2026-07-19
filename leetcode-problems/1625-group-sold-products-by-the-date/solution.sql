/* Write your T-SQL query statement below */
select
sell_date,
count(product) as num_sold,
string_agg(product, ',') within group(order by product asc) as products
from (
    select distinct sell_date, product
    from activities
) as s
group by sell_date
order by sell_date asc;
