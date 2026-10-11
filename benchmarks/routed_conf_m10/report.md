# Evaluation report: routed_conf_m10

- Created: 2026-10-11T01:09:09+00:00
- Models: claude-sonnet-5-5, qwen2.5-coder-1.5b-tuned
- **Execution accuracy: 92.0%** (46/50)
- Latency ms: mean 1846, p50 1285, **p95 4465**, max 9622
- Tokens: 35,123 prompt + 3,353 completion
- Total cost: $0.0489; **$0.98 per 1,000 queries**
- Average LLM calls per query: 1.38

## Cost assumptions
- claude_model: claude-sonnet-5-5
- claude_usd_per_mtok_in_out: 2.0 / 10.0
- gpu_concurrent_requests: 1.0
- gpu_hourly_usd: 0.526
- local_model: qwen2.5-coder-1.5b-tuned
- min_confidence: -0.1

## Outcome by status
- success: 50

## Failure categories
- wrong_row_count: 2
- wrong_values: 2

## Accuracy by gold-SQL complexity (heuristic buckets)
- easy: 96.2% (n=26)
- hard: 80.0% (n=5)
- medium: 89.5% (n=19)

## Accuracy by route
- claude: 93.3% (n=15)
- local: 91.4% (n=35)

## Failures (4)

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

### spider-dev-025 · world_1 · wrong_row_count
Question: What are the different government forms and what is the total population of each for government forms that have an average life expectancy greater than 72?
Gold: `SELECT sum(Population) ,  GovernmentForm FROM country GROUP BY GovernmentForm HAVING avg(LifeExpectancy)  >  72`
Predicted: `SELECT GovernmentForm, SUM(Population) FROM country WHERE LifeExpectancy > 72 GROUP BY GovernmentForm`
Structural differences: aggregates, having

### spider-dev-028 · car_1 · wrong_values
Question: What is the car model with the highest mpg ?
Gold: `select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;`
Predicted: `SELECT cn.Model FROM cars_data AS cd JOIN car_names AS cn ON cd.Id = cn.MakeId ORDER BY CAST(cd.MPG AS REAL) DESC LIMIT 1`
