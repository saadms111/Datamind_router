# Train, serve and benchmark on a free Colab T4

Target: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, LoRA fine-tuned on Spider, served with vLLM,
benchmarked against the Claude baseline. Use **Runtime > Change runtime type > T4 GPU**.
Keep everything that must survive a disconnect on Google Drive.

The T4 has no native bf16, so this guide uses fp16 throughout. Train first, install
vLLM afterwards: vLLM pins its own PyTorch build and replacing Colab's torch mid-session
is the usual source of breakage.

## 1. Setup (training session)

```python
from google.colab import drive
drive.mount("/content/drive")
```
```bash
# clone your GitHub repo (or upload the folder), then:
%cd /content/datamind-router
!pip install -q transformers peft accelerate datasets sqlglot pydantic huggingface-hub httpx anthropic
```

## 2. Data and training

```bash
# 50-example eval sample first, so the training-data builder can verify no database overlaps it
!PYTHONPATH=src python -m eval.loader
!PYTHONPATH=src python -m router.data

# smoke test (a few steps), then the real run
!PYTHONPATH=src python -m router.train_lora --max-steps 5 --limit-train 200 --output-dir /content/smoke
!PYTHONPATH=src python -m router.train_lora --output-dir /content/drive/MyDrive/datamind/qwen15b --resume
```

If Colab disconnects, re-run the same command: `--resume` continues from the latest
checkpoint in the output directory. If a run is too slow for your session limit, pass
`--epochs 1`. Out of memory: `--batch-size 2 --grad-accum 16`.

```bash
!PYTHONPATH=src python -m router.merge_lora \
    --adapter-dir /content/drive/MyDrive/datamind/qwen15b/adapter \
    --output-dir  /content/drive/MyDrive/datamind/qwen15b/merged --dtype float16
```

## 3. Serve with vLLM

**Easiest path:** open `notebooks/colab_baseline.ipynb` in a fresh T4 runtime. It does
steps 3 and 4 below for you (upload the repo zip, install, start vLLM, smoke test, full
run, download results). Use it for the untuned baseline, and again for the fine-tuned
model by changing the settings cell.

By hand, in a fresh runtime (T4), from the repo root:

```bash
!pip install -q vllm
!pip install -q sqlglot pydantic httpx anthropic datasets huggingface-hub
```
Start the server in the background and wait until it answers (the first start downloads
the model):
```bash
!pkill -9 -f -i '[v]llm' ; sleep 3
!nohup vllm serve /content/drive/MyDrive/datamind/qwen15b/merged \
    --served-model-name datamind-qwen --dtype half \
    --max-model-len 4096 --gpu-memory-utilization 0.80 --enforce-eager \
    --port 8000 > /content/vllm.log 2>&1 &
!sleep 120 && curl -s localhost:8000/v1/models && tail -n 20 /content/vllm.log
```

Rules that avoid every failure seen so far:

- `--local-model` in `eval.run_eval` must equal `--served-model-name` exactly, or every
  request returns 404.
- `VLLM_USE_V1=0` and similar are environment variables, never CLI flags.
- Kill any old server before starting a new one (the `pkill` line above); a stale server
  holds the GPU memory and the port.
- `--enforce-eager` skips CUDA-graph capture. It starts reliably on the T4 but makes
  decoding somewhat slower, so quote latency with that caveat and use the same flags for
  every local run.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named eval.run_eval` / `datamind` | Run from the repo root with `PYTHONPATH=src python -m eval.run_eval ...` |
| `--gpu-hourly-usd is required` | By design: local and routed runs must state a GPU rate (0 is allowed) |
| HTTP 404 model not found | `--local-model` differs from `--served-model-name` |
| `unrecognized arguments: --VLLM-USE-V1=0` | Remove it; env vars are not CLI flags |
| `Cannot use FA version 2 ... TRITON_ATTN` | Harmless fallback on T4, not the failure; read the other ERROR lines |
| `Engine core initialization failed` | Kill stale servers, keep `--max-model-len 4096 --enforce-eager`, lower `--gpu-memory-utilization` to 0.70, read `/content/vllm.log` |
| `TypeError ... assumptions` at the end of a run | Stale `src/eval/report.py` or `runner.py`; both belong in `src/eval/`, not `src/datamind/` |

## 4. Benchmark (run in the same notebook, against localhost)

```python
import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

# Use an equivalent cloud GPU's on-demand $/hour and keep it fixed for every run.
# A free Colab T4 costs you $0, but a $0 local model is not a fair cost story.
GPU_RATE = float(input("USD per GPU-hour: "))
OUT = "/content/drive/MyDrive/datamind/results"
```
```bash
!PYTHONPATH=src python -m eval.run_eval --backend local  --gpu-hourly-usd {GPU_RATE} --run-name local_qwen15b --output-dir {OUT}
!PYTHONPATH=src python -m eval.run_eval --backend routed --gpu-hourly-usd {GPU_RATE} --run-name routed_fail_only --output-dir {OUT}
!PYTHONPATH=src python -m eval.run_eval --backend routed --gpu-hourly-usd {GPU_RATE} --min-confidence -0.15 --run-name routed_conf_015 --output-dir {OUT}
```

`routed` without `--min-confidence` escalates only on failed SQL. Sweep a few
`--min-confidence` values (for example -0.05, -0.10, -0.15, -0.30) and keep the one that
holds accuracy near the Claude baseline at the lowest cost. Every report records the GPU
rate under "Cost assumptions"; quote it next to any cost number.

## Reading the cost numbers honestly

- Local cost is `latency x GPU rate`, with the whole GPU charged to each query (a
  sequential run). A busy production server batches requests, so real per-query cost
  would be lower.
- A routed query that escalates pays for the wasted local attempt plus Claude, so it is
  slower and dearer than calling Claude directly. The savings come from the fraction
  that stays local.
- With 50 examples, one query is 2 percentage points. Treat small accuracy differences
  as noise and say so.
