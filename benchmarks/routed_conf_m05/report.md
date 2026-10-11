# Evaluation report: routed_conf_m05

- Created: 2026-10-11T01:07:33+00:00
- Models: claude-sonnet-5-5, qwen2.5-coder-1.5b-tuned
- **Execution accuracy: 98.0%** (49/50)
- Latency ms: mean 2062, p50 1795, **p95 5032**, max 6149
- Tokens: 44,634 prompt + 4,105 completion
- Total cost: $0.0755; **$1.51 per 1,000 queries**
- Average LLM calls per query: 1.60

## Cost assumptions
- claude_model: claude-sonnet-5-5
- claude_usd_per_mtok_in_out: 2.0 / 10.0
- gpu_concurrent_requests: 1.0
- gpu_hourly_usd: 0.526
- local_model: qwen2.5-coder-1.5b-tuned
- min_confidence: -0.05

## Outcome by status
- success: 50

## Failure categories
- wrong_values: 1

## Accuracy by gold-SQL complexity (heuristic buckets)
- easy: 100.0% (n=26)
- hard: 100.0% (n=5)
- medium: 94.7% (n=19)

## Accuracy by route
- claude: 96.2% (n=26)
- local: 100.0% (n=24)

## Failures (1)

### spider-dev-028 · car_1 · wrong_values
Question: What is the car model with the highest mpg ?
Gold: `select t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;`
Predicted: `SELECT cn.Model FROM car_names AS cn JOIN cars_data AS cd ON cn.MakeId = cd.Id ORDER BY CAST(cd.MPG AS REAL) DESC LIMIT 1`
