# Evaluation report: routed_conf_m30

- Created: 2026-10-11T01:11:40+00:00
- Models: claude-sonnet-5-5, qwen2.5-coder-1.5b-tuned
- **Execution accuracy: 80.0%** (40/50)
- Latency ms: mean 1416, p50 1050, **p95 3981**, max 6069
- Tokens: 25,877 prompt + 2,289 completion
- Total cost: $0.0198; **$0.40 per 1,000 queries**
- Average LLM calls per query: 1.16

## Cost assumptions
- claude_model: claude-sonnet-5-5
- claude_usd_per_mtok_in_out: 2.0 / 10.0
- gpu_concurrent_requests: 1.0
- gpu_hourly_usd: 0.526
- local_model: qwen2.5-coder-1.5b-tuned
- min_confidence: -0.3

## Outcome by status
- success: 50

## Failure categories
- wrong_row_count: 6
- wrong_values: 3
- wrong_column_count: 1

## Accuracy by gold-SQL complexity (heuristic buckets)
- easy: 84.6% (n=26)
- hard: 60.0% (n=5)
- medium: 78.9% (n=19)

## Accuracy by route
- claude: 75.0% (n=4)
- local: 80.4% (n=46)

## Failures (10)

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

### spider-dev-028 · car_1 · wrong_values
Question: What is the car model with the highest mpg ?
Gold: `select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;`
Predicted: `SELECT cn.Model FROM cars_data AS cd JOIN car_names AS cn ON cd.Id = cn.MakeId ORDER BY CAST(cd.MPG AS REAL) DESC LIMIT 1`

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
