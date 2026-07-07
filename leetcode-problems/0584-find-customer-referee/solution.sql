# Write your MySQL query statement below
-- select name 
-- from Customer 
-- where referee_id != 2 or referee_id = null;
SELECT name FROM customer WHERE referee_id <> 2 OR referee_id IS NULL;
