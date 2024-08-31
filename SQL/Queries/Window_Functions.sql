-- For allowing GROUP BY operations on dates
SET sql_mode=(SELECT REPLACE(@@sql_mode, 'ONLY_FULL_GROUP_BY', ''));


-- DB Zomato
create database zomato;
use zomato;
show tables;

select * from orders;
-- drop table orders;


-- DB Swiggy
create database swiggy;
use swiggy;
show tables;
drop database swiggy;

select * from orders;
-- drop table orders;


-- DB IPL
create database ipl_08_22;
use ipl_08_22;
show tables;

-- drop database ipl_08_22;
select count(*) as 'number_of_records' from ipl;


-- DB Students
use students;
show tables;

create table marks(
	student_id integer primary key auto_increment,
    student_name varchar(255) not null,
    branch varchar(255) not null,
    marks_obtained integer not null
);

insert into marks (student_name, branch, marks_obtained) values
("Nitish", "EEE", 82),
("Rishabh", "EEE", 91),
("Anukant", "EEE", 69),
("Rupesh", "EEE", 55),
("Shubham", "CSE", 78),
("Ved", "CSE", 43),
("Deepak", "CSE", 98),
("Arpan", "CSE", 95),
("Vinay", "ECE", 95),
("Ankit", "ECE", 88),
("Anand", "ECE", 81),
("Rohit", "ECE", 95),
("Prashant", "MECH", 74),
("Amit", "MECH", 69),
("Sunny", "MECH", 39),
("Gautum", "MECH", 51);

select * from marks;
-- select * from student_records;

-- Aggregate Function
select branch, avg(marks_obtained)
from marks
group by branch;

-- Window Functions - Introduction

-- Complete summary - Min, Max and Avg marks of all students by branch
select *,
min(marks_obtained) over(partition by branch) as min_marks,
max(marks_obtained) over(partition by branch) as max_marks,
round(avg(marks_obtained) over(partition by branch), 2) as avg_marks
from marks
order by student_id;

-- Students who scored above average by branch
select * from (select *,
round(avg(marks_obtained) over(partition by branch), 2) as avg_marks
from marks) as summary
where marks_obtained > summary.avg_marks
order by student_id;

-- Rankings across all branches
select *,
rank() over(order by marks_obtained desc) as ranking
from marks;

-- Rankings w.r.t. branches
select *,
rank() over(partition by branch order by marks_obtained desc) as branch_ranking
from marks;

-- Dense Ranking - all branches
select *,
dense_rank() over(order by marks_obtained desc) as ranking
from marks;

-- Dense Ranking - individual branches
select *,
dense_rank() over(partition by branch order by marks_obtained desc) as ranking
from marks;

-- Row Number - all records get a unique row number
select *,
row_number() over(order by marks_obtained desc) as row_id
from marks;

-- Obtaining branch-wise row numbers of students based on marks
select *,
concat(branch, '-', row_number() over(partition by branch order by marks_obtained desc)) as branch_row_id
from marks;


-- Questions

-- Q.1) Find out the top 2 customers by expenditure from each month.
select * from (select monthname(date) as 'month', user_id, sum(amount) as 'monthly_total',
					rank() over(partition by monthname(date) order by sum(amount) desc) as 'monthly_rank'
				from orders
				group by monthname(date), user_id
				order by month(date)) as t
where t.monthly_rank < 3;

-- Q.2) Find the FIRST_VALUE() and LAST_VALUE() of marks obtained by branch in descending order.
select *,
first_value(marks_obtained) over(partition by branch order by marks_obtained desc) as 'highest_score',
last_value(marks_obtained) over(partition by branch order by marks_obtained desc 
			rows between unbounded preceding and unbounded following) as 'lowest_score'
from marks;

-- Alternatively
select *,
first_value(marks_obtained) over w as 'highest_score',
last_value(marks_obtained) over w as 'lowest_score'
from marks
window w as (partition by branch order by marks_obtained desc rows between unbounded preceding and unbounded following);

-- Q.3) Find the 2nd and 3rd highest scores in each branch.
select *,
nth_value(marks_obtained, 2) over(partition by branch order by marks_obtained desc
			rows between unbounded preceding and unbounded following) as 'second_highest',
nth_value(marks_obtained, 3) over(partition by branch order by marks_obtained desc
			rows between unbounded preceding and unbounded following) as 'third_highest'
from marks;

-- Q.4) Find the branch toppers.
select student_name, branch, marks_obtained from (select *,
					first_value(student_name) over(partition by branch order by marks_obtained desc) as 'topper_name',
					first_value(marks_obtained) over(partition by branch order by marks_obtained desc) as 'topper_marks'
					from marks) as t
where t.marks_obtained = t.topper_marks and t.student_name = t.topper_name;

-- Q.5) Find the LEAD() and LAG() by 1 record per branch.
select *,
lead(marks_obtained) over(partition by branch order by student_id) as 'lead',
lag(marks_obtained) over(partition by branch order by student_id) as 'lag'
from marks;

-- Q.6) Find the Month-on-Month revenue growth of Zomato.
select monthname(date) as 'month', sum(amount) as 'monthly_total',
coalesce(round((((sum(amount) - lag(sum(amount)) over(order by month(date))) 
		/ lag(sum(amount)) over(order by month(date))) * 100), 2), 0.00) as 'mom_growth'
from orders
group by monthname(date)
order by month(date);

-- Q.7) Finding the second last scorer in each branch.
select student_name, branch, marks_obtained, branch_rank from (
	select *,
		row_number() over(partition by branch order by marks_obtained) as 'branch_rank'
		from marks
	) as t
where t.branch_rank = 2;

-- Q.8) Find the lowest scorers in each branch.
select student_id, student_name, branch, marks_obtained 
from (select *,
		last_value(marks_obtained) over(partition by branch order by marks_obtained desc
			rows between unbounded preceding and unbounded following) as 'lowest_score'
		from marks) as t
where t.marks_obtained = t.lowest_score;

-- Q.9) Find the second highest scorers in each branch.
select * from (select *,
				rank() over(partition by branch order by marks_obtained) as 'branch_rank'
				from marks) as t
where t.branch_rank = 2;

-- Q.10) Find the top 5 batsmen from each team.
select * from (select BattingTeam, batter, sum(batsman_run) as 'total_runs',
					dense_rank() over(partition by BattingTeam order by sum(batsman_run) desc) as 'rank_within_team'
				from ipl
				group by BattingTeam, batter) as t
where t.rank_within_team < 6
order by t.BattingTeam, t.rank_within_team;

-- Q.11) Find the number of runs scored by Virat Kohli up until his 50th, 100th and 200th match. [Cumulative Sum]
select * from (select
			concat("Match - ", row_number() over(order by ID)) as 'match_no',
            sum(batsman_run) as 'runs_scored',
			sum(sum(batsman_run)) over(rows between unbounded preceding and current row) as 'career_runs'
		from ipl
		where batter = "V Kohli"
		group by ID
) as t
where t.match_no = 'Match - 50' or t.match_no = 'Match - 100' or t.match_no = 'Match - 200';

-- Q.12) Find the Cumulative Average of the runs scored by Virat Kohli over all the seasons.
select concat("Match - ", row_number() over(order by ID)) as 'match_no',
	sum(batsman_run) as 'runs_scored',
    round(avg(sum(batsman_run)) over(rows between unbounded preceding and current row), 2) as 'career_avg',
	sum(sum(batsman_run)) over(rows between unbounded preceding and current row) as 'career_runs'
from ipl
where batter = "V Kohli"
group by ID;

-- Q.13) Find the Running Average of the runs scored by Virat Kohli over all the seasons.
select concat("Match - ", row_number() over(order by ID)) as 'match_no',
	sum(batsman_run) as 'runs_scored',
    round(avg(sum(batsman_run)) over(rows between unbounded preceding and current row), 2) as 'career_avg',
    round(avg(sum(batsman_run)) over(rows between 9 preceding and current row), 2) as 'rolling_avg_10',
	sum(sum(batsman_run)) over(rows between unbounded preceding and current row) as 'career_runs'
from ipl
where batter = "V Kohli"
group by ID;

-- Q.14) Find out the percent of total amount earned from all the foods for restaurant #1.
select food_id, t3.f_name as 'food_name', round(total_amount/sum(total_amount) over() * 100, 2) as 'percent_of_total' 
from (select f_id as 'food_id', sum(amount) as 'total_amount' from orders t1
join order_details t2
on t1.order_id = t2.order_id
where r_id = 1
group by f_id) as t
join food t3
on t.food_id = t3.f_id
order by percent_of_total desc;

