# Evaluation report: claude_baseline

- Created: 2026-10-05T21:00:54+00:00
- Models: claude-sonnet-5-5
- **Execution accuracy: 96.0%** (48/50)
- Latency ms: mean 1548, p50 1339, **p95 2487**, max 6247
- Tokens: 38,405 prompt + 3,894 completion
- Total cost: $0.1157; **$2.31 per 1,000 queries**
- Average LLM calls per query: 1.00

## Outcome by status
- success: 50

## Failure categories
- wrong_values: 2

## Accuracy by gold-SQL complexity (heuristic buckets)
- easy: 96.2% (n=26)
- hard: 100.0% (n=5)
- medium: 94.7% (n=19)

## Accuracy by route
- claude: 96.0% (n=50)

## Failures (2)

### spider-dev-002 · student_transcripts_tracking · wrong_values
Question: What is the average transcript date?
Gold: `SELECT avg(transcript_date) FROM Transcripts`
Predicted: `SELECT DATETIME(AVG(JULIANDAY(transcript_date))) AS average_transcript_date FROM Transcripts`

### spider-dev-028 · car_1 · wrong_values
Question: What is the car model with the highest mpg ?
Gold: `select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;`
Predicted: `SELECT cn.Model FROM cars_data AS cd JOIN car_names AS cn ON cd.Id = cn.MakeId ORDER BY CAST(cd.MPG AS REAL) DESC LIMIT 1`
