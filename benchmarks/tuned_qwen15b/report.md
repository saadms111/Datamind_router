# Evaluation report: tuned_qwen15b

- Created: 2026-10-11T00:37:19+00:00
- Models: qwen2.5-coder-1.5b-tuned
- **Execution accuracy: 74.0%** (37/50)
- Latency ms: mean 1337, p50 970, **p95 3251**, max 4760
- Tokens: 23,120 prompt + 1,815 completion
- Total cost: $0.0098; **$0.20 per 1,000 queries**
- Average LLM calls per query: 1.08

## Cost assumptions
- gpu_concurrent_requests: 1.0
- gpu_hourly_usd: 0.526
- local_model: qwen2.5-coder-1.5b-tuned

## Outcome by status
- rejected: 4
- success: 46

## Failure categories
- wrong_row_count: 6
- guard_rejected: 4
- wrong_values: 2
- wrong_column_count: 1

## Accuracy by gold-SQL complexity (heuristic buckets)
- easy: 84.6% (n=26)
- hard: 60.0% (n=5)
- medium: 63.2% (n=19)

## Accuracy by route
- local: 74.0% (n=50)

## Failures (13)

### spider-dev-003 · tvshow · wrong_row_count
Question: What is the episode for the TV series named "Sky Radio"?
Gold: `SELECT T2.Episode FROM TV_Channel AS T1 JOIN TV_series AS T2 ON T1.id = T2.Channel WHERE T1.series_name = "Sky Radio";`
Predicted: `SELECT T2.Episode FROM Cartoon AS T1 JOIN TV_series AS T2 ON T1.id = T2.channel WHERE T1.Title = "Sky Radio"`

### spider-dev-008 · car_1 · wrong_values
Question: What is the name of the country with the most car makers?
Gold: `SELECT T2.CountryName FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId GROUP BY T1.Country ORDER BY Count(*) DESC LIMIT 1;`
Predicted: `SELECT Country FROM car_makers GROUP BY Country ORDER BY COUNT(*) DESC LIMIT 1`
Structural differences: joins

### spider-dev-016 · flight_2 · wrong_row_count
Question: Which city is the most frequent source airport?
Gold: `SELECT T1.City FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.SourceAirport GROUP BY T1.City ORDER BY count(*) DESC LIMIT 1`
Predicted: `SELECT City FROM airports GROUP BY City ORDER BY COUNT(*) DESC LIMIT 1`
Structural differences: joins

### spider-dev-017 · dog_kennels · wrong_values
Question: What is the average age of the dogs who have gone through any treatments?
Gold: `SELECT avg(age) FROM Dogs WHERE dog_id IN ( SELECT dog_id FROM Treatments )`
Predicted: `SELECT AVG(T1.age) FROM Dogs AS T1 JOIN Treatments AS T2 ON T1.dog_id = T2.dog_id`
Structural differences: joins, nested_selects

### spider-dev-020 · wta_1 · guard_rejected
Question: What is the name of the winner with the most rank points who participated in the Australian Open tournament?
Gold: `SELECT winner_name FROM matches WHERE tourney_name  =  'Australian Open' ORDER BY winner_rank_points DESC LIMIT 1`
Predicted: `SELECT T1.winner_name FROM matches AS T1 JOIN players AS T2 ON T1.winner_id = T2.player_id WHERE T1.tourney_name = "Australian Open" ORDER BY T2.ranking_points DESC LIMIT 1`
Error: unknown_column: Column 'ranking_points' not found in players
Structural differences: joins

### spider-dev-022 · student_transcripts_tracking · wrong_column_count
Question: List all the student details in reversed lexicographical order.
Gold: `SELECT other_student_details FROM Students ORDER BY other_student_details DESC`
Predicted: `SELECT * FROM students ORDER BY last_name DESC, first_name DESC`

### spider-dev-025 · world_1 · wrong_row_count
Question: What are the different government forms and what is the total population of each for government forms that have an average life expectancy greater than 72?
Gold: `SELECT sum(Population) ,  GovernmentForm FROM country GROUP BY GovernmentForm HAVING avg(LifeExpectancy)  >  72`
Predicted: `SELECT GovernmentForm, SUM(Population) FROM country WHERE LifeExpectancy > 72 GROUP BY GovernmentForm`
Structural differences: aggregates, having

### spider-dev-027 · singer · wrong_row_count
Question: What are the names of the singers who are not French citizens?
Gold: `SELECT Name FROM singer WHERE Citizenship != "France"`
Predicted: `SELECT Name FROM singer WHERE Citizenship <> 'French'`

### spider-dev-028 · car_1 · guard_rejected
Question: What is the car model with the highest mpg ?
Gold: `select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;`
Predicted: `SELECT Model FROM cars_data WHERE MPG = (SELECT max(MPG) FROM cars_data)`
Error: unknown_column: Unknown column 'model'
Structural differences: aggregates, joins, limit, nested_selects, order_by

### spider-dev-031 · pets_1 · guard_rejected
Question: What major is every student who does not own a cat as a pet, and also how old are they?
Gold: `SELECT major ,  age FROM student WHERE stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')`
Predicted: `SELECT T1.Major ,  T2.age FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID WHERE T2.PetID NOT IN (SELECT T3.petid FROM Pets AS T3 WHERE T3.pettype  =  'cat')`
Error: unknown_column: Column 'age' not found in has_pet
Structural differences: joins

### spider-dev-033 · course_teach · guard_rejected
Question: What are the names of the teachers and the courses they teach in ascending alphabetical order by the name of the teacher?
Gold: `SELECT T3.Name ,  T2.Course FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID  =  T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID  =  T3.Teacher_ID ORDER BY T3.Name`
Predicted: `SELECT T2.name ,  T1.course FROM course AS T1 JOIN course_arrange AS T2 ON T1.course_id  =  T2.course_id ORDER BY T2.name ASC`
Error: unknown_column: Column 'name' not found in course_arrange; unknown_column: Column 'name' not found in course_arrange
Structural differences: joins

### spider-dev-034 · real_estate_properties · wrong_row_count
Question: Show the property type descriptions of properties belonging to that code.
Gold: `SELECT T2.property_type_description FROM Properties AS T1 JOIN Ref_Property_Types AS T2 ON T1.property_type_code  =  T2.property_type_code GROUP BY T1.property_type_code`
Predicted: `SELECT T2.property_type_description FROM Properties AS T1 JOIN Ref_Property_Types AS T2 ON T1.property_type_code = T2.property_type_code WHERE T1.property_id = 1`
Structural differences: group_by

### spider-dev-038 · network_1 · wrong_row_count
Question: What is the average grade of students who have friends?
Gold: `SELECT avg(grade) FROM Highschooler WHERE id IN (SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id)`
Predicted: `SELECT AVG(T1.grade) FROM highschooler AS T1 JOIN friend AS T2 ON T1.id = T2.friend_id GROUP BY T2.student_id HAVING COUNT(*) >= 1`
Structural differences: aggregates, group_by, having, nested_selects
