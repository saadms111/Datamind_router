# Evaluation report: base_qwen15b_untuned

- Created: 2026-10-10T19:14:00+00:00
- Models: qwen2.5-coder-1.5b-base
- **Execution accuracy: 66.0%** (33/50)
- Latency ms: mean 1591, p50 1045, **p95 3849**, max 6778
- Tokens: 24,906 prompt + 2,129 completion
- Total cost: $0.0116; **$0.23 per 1,000 queries**
- Average LLM calls per query: 1.16

## Cost assumptions
- gpu_concurrent_requests: 1.0
- gpu_hourly_usd: 0.526
- local_model: qwen2.5-coder-1.5b-base

## Outcome by status
- rejected: 6
- success: 44

## Failure categories
- wrong_row_count: 7
- guard_rejected: 6
- wrong_values: 2
- wrong_column_count: 1
- wrong_ordering: 1

## Accuracy by gold-SQL complexity (heuristic buckets)
- easy: 73.1% (n=26)
- hard: 80.0% (n=5)
- medium: 52.6% (n=19)

## Accuracy by route
- local: 66.0% (n=50)

## Failures (17)

### spider-dev-003 · tvshow · wrong_row_count
Question: What is the episode for the TV series named "Sky Radio"?
Gold: `SELECT T2.Episode FROM TV_Channel AS T1 JOIN TV_series AS T2 ON T1.id = T2.Channel WHERE T1.series_name = "Sky Radio";`
Predicted: `SELECT T2.Episode FROM Cartoon AS T1 INNER JOIN TV_series AS T2 ON T1.id = T2.Channel WHERE T1.Title = 'Sky Radio'`

### spider-dev-011 · pets_1 · wrong_values
Question: How many dog pets are raised by female students?
Gold: `SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T2.petid  =  T3.petid WHERE T1.sex  =  'F' AND T3.pettype  =  'dog'`
Predicted: `SELECT COUNT(T1.PetID) FROM Has_Pet AS T1 INNER JOIN Pets AS T2 ON T1.PetID = T2.PetID WHERE T2.PetType = 'Dog' AND T1.StuID IN (SELECT StuID FROM Student WHERE Sex = 'F')`
Structural differences: joins, nested_selects

### spider-dev-016 · flight_2 · wrong_row_count
Question: Which city is the most frequent source airport?
Gold: `SELECT T1.City FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.SourceAirport GROUP BY T1.City ORDER BY count(*) DESC LIMIT 1`
Predicted: `SELECT City FROM airports GROUP BY City ORDER BY COUNT(*) DESC LIMIT 1`
Structural differences: joins

### spider-dev-017 · dog_kennels · wrong_values
Question: What is the average age of the dogs who have gone through any treatments?
Gold: `SELECT avg(age) FROM Dogs WHERE dog_id IN ( SELECT dog_id FROM Treatments )`
Predicted: `SELECT AVG(T1.age) FROM Dogs AS T1 INNER JOIN Treatments AS T2 ON T1.dog_id = T2.dog_id`
Structural differences: joins, nested_selects

### spider-dev-020 · wta_1 · guard_rejected
Question: What is the name of the winner with the most rank points who participated in the Australian Open tournament?
Gold: `SELECT winner_name FROM matches WHERE tourney_name  =  'Australian Open' ORDER BY winner_rank_points DESC LIMIT 1`
Predicted: `SELECT T1.winner_name FROM matches AS T1 INNER JOIN rankings AS T2 ON T1.winner_id = T2.player_id WHERE T1.tournament_id = 'Australian Open' ORDER BY T2.ranking_points DESC LIMIT 1;`
Error: unknown_column: Column 'tournament_id' not found in matches
Structural differences: joins

### spider-dev-021 · employee_hire_evaluation · guard_rejected
Question: Which district has both stores with less than 3000 products and stores with more than 10000 products?
Gold: `SELECT district FROM shop WHERE Number_products  <  3000 INTERSECT SELECT district FROM shop WHERE Number_products  >  10000`
Predicted: `SELECT T2.District FROM shop AS T1 INNER JOIN district AS T2 ON T1.District = T2.District WHERE T1.Number_products < 3000 INTERSECT SELECT T2.District FROM shop AS T1 INNER JOIN district AS T2 ON T1.District = T2.District WHERE T1.Number_products > 10000`
Error: unknown_table: Unknown table 'district'
Structural differences: joins

### spider-dev-022 · student_transcripts_tracking · wrong_column_count
Question: List all the student details in reversed lexicographical order.
Gold: `SELECT other_student_details FROM Students ORDER BY other_student_details DESC`
Predicted: `SELECT first_name, middle_name, last_name FROM Students ORDER BY first_name DESC, middle_name DESC, last_name DESC`

### spider-dev-023 · tvshow · wrong_ordering
Question: What are all of the episodes ordered by ratings?
Gold: `SELECT Episode FROM TV_series ORDER BY rating`
Predicted: `SELECT Episode FROM TV_series ORDER BY Rating DESC`

### spider-dev-025 · world_1 · wrong_row_count
Question: What are the different government forms and what is the total population of each for government forms that have an average life expectancy greater than 72?
Gold: `SELECT sum(Population) ,  GovernmentForm FROM country GROUP BY GovernmentForm HAVING avg(LifeExpectancy)  >  72`
Predicted: `SELECT GovernmentForm, SUM(Population) AS TotalPopulation FROM country WHERE LifeExpectancy > 72 GROUP BY GovernmentForm`
Structural differences: aggregates, having

### spider-dev-027 · singer · wrong_row_count
Question: What are the names of the singers who are not French citizens?
Gold: `SELECT Name FROM singer WHERE Citizenship != "France"`
Predicted: `SELECT Name FROM singer WHERE Citizenship <> 'French'`

### spider-dev-028 · car_1 · guard_rejected
Question: What is the car model with the highest mpg ?
Gold: `select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;`
Predicted: `SELECT Model FROM car_names ORDER BY MPG DESC LIMIT 1;`
Error: unknown_column: Unknown column 'mpg'
Structural differences: joins

### spider-dev-030 · concert_singer · guard_rejected
Question: Show countries where a singer above age 40 and a singer below 30 are from.
Gold: `SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30`
Predicted: `SELECT DISTINCT T2.Country FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID WHERE T1.Age > 40 INTERSECT SELECT DISTINCT T2.Country FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID WHERE T1.Age < 30;`
Error: unknown_column: Column 'country' not found in singer_in_concert; unknown_column: Column 'country' not found in singer_in_concert
Structural differences: distinct, joins

### spider-dev-031 · pets_1 · guard_rejected
Question: What major is every student who does not own a cat as a pet, and also how old are they?
Gold: `SELECT major ,  age FROM student WHERE stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')`
Predicted: `SELECT T1.Major, T1.Age FROM Student AS T1 INNER JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID WHERE T2.PetType != 'Cat'`
Error: unknown_column: Column 'pettype' not found in has_pet
Structural differences: joins, nested_selects

### spider-dev-033 · course_teach · guard_rejected
Question: What are the names of the teachers and the courses they teach in ascending alphabetical order by the name of the teacher?
Gold: `SELECT T3.Name ,  T2.Course FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID  =  T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID  =  T3.Teacher_ID ORDER BY T3.Name`
Predicted: `SELECT T1.Name AS Teacher_Name, T2.Course FROM teacher AS T1 INNER JOIN course_arrange AS T2 ON T1.Teacher_ID = T2.Teacher_ID ORDER BY T1.Name ASC;`
Error: unknown_column: Column 'course' not found in course_arrange
Structural differences: joins

### spider-dev-034 · real_estate_properties · wrong_row_count
Question: Show the property type descriptions of properties belonging to that code.
Gold: `SELECT T2.property_type_description FROM Properties AS T1 JOIN Ref_Property_Types AS T2 ON T1.property_type_code  =  T2.property_type_code GROUP BY T1.property_type_code`
Predicted: `SELECT T2.property_type_description FROM Properties AS T1 INNER JOIN Ref_Property_Types AS T2 ON T1.property_type_code = T2.property_type_code`
Structural differences: group_by

### spider-dev-041 · employee_hire_evaluation · wrong_row_count
Question: Which shops' number products is above the average? Give me the shop names.
Gold: `SELECT name FROM shop WHERE number_products  >  (SELECT avg(number_products) FROM shop)`
Predicted: `SELECT T2.Name FROM shop AS T1 INNER JOIN employee AS T2 ON T1.Manager_name = T2.Name WHERE T1.Number_products > (SELECT AVG(Number_products) FROM shop)`
Structural differences: joins

### spider-dev-043 · tvshow · wrong_row_count
Question: What is Weekly Rank of TV series with Episode "A Love of a Lifetime"?
Gold: `SELECT Weekly_Rank FROM TV_series WHERE Episode = "A Love of a Lifetime";`
Predicted: `SELECT T2.Weekly_Rank FROM Cartoon AS T1 INNER JOIN TV_series AS T2 ON T1.id = T2.Channel WHERE T1.Title = 'A Love of a Lifetime'`
Structural differences: joins
