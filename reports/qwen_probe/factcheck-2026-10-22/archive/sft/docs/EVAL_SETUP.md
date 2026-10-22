# Eval setup

End-to-end pipeline for evaluating DualMLPAdapter checkpoints from `scripts/gr/`
against held-out terminal-wrench tasks, using the harbor runner and a
gpt-5.4-nano "intent-based" judge.

## Prereqs

- Linux host with `sudo` (apt + Docker config require root).
- NVIDIA driver ≥ R570 (CUDA 12.8 capable). For CUDA 13 you can move to a newer
  vllm but this pipeline pins 0.9.2 — see the Pitfalls table below.
- 8× ≥80 GB GPUs (e.g., H100 80GB) recommended. The 32B model in DP=8 takes
  ~75 GB per GPU; smaller fleets need a different DP/TP/quant config.
- `uv` installed (`pipx install uv`).

## Setup

```bash
git clone --recurse-submodules https://github.com/jnward/GRAFT
cd GRAFT/sft
bash scripts/eval/setup_box.sh
cp .env.example .env && $EDITOR .env   # set OPENROUTER_API_KEY (and HF_TOKEN if first-time)
```

`setup_box.sh` is idempotent. It will:
1. apt-install `docker.io`, `docker-compose-v2`, `ninja-build` (install `python3.12-dev` yourself if triton fails on `Python.h`; see the Pitfalls table).
2. Write `/etc/docker/daemon.json` with expanded `default-address-pools` so harbor
   can run >32 concurrent trials without Docker exhausting its IP ranges.
3. Init submodules (`third_party/harbor`, `third_party/terminal-wrench`). The
   script's wrench `.git` shrink (a partial re-clone) only triggers when `.git` is
   a directory, which a normal submodule checkout of this repo is not, so the full
   wrench history is kept. To avoid it, clone without `--recurse-submodules` and run
   `git submodule update --init --filter=blob:none sft/third_party/terminal-wrench`
   from the repo root before `setup_box.sh` (not tested by the authors).
4. Create `.venvs/vllm/` with vllm 0.9.2 + transformers 4.51.3 + flash-attn 2.8.3
   (these versions matter — see the Pitfalls table below).
5. Create `.venvs/eval/` with the training deps (torch 2.5.1+cu124,
   transformers 5.5.4, trl 1.2.0, accelerate 1.13.0, datasets 4.8.4, peft 0.19.1,
   flash-attn 2.8.3) plus the judge deps (openai, python-dotenv, tiktoken,
   safetensors).
6. `uv sync` inside `third_party/harbor`.
7. Smoke check: harbor's oracle agent on task 1018.

## Drop in checkpoints

Place adapter checkpoints written by `train_gr.py` under, e.g.:
```
checkpoints/<run-name>/adapter_state_dict.pt (any path; run.py takes --checkpoint)
```
The orchestrator only needs the absolute path to the `.pt` file.

## Run an eval

```bash
source .venvs/eval/bin/activate
python -m scripts.eval.run \
    --checkpoint checkpoints/gr_32b_mlp_fr02_ddp_ep2/adapter_state_dict.pt \
    --mode retain_only \
    --job-name gr_ep2_retain
```

`retain_only` = deployment configuration (forget adapter ablated), the mode of most
Fig. 4 arms (see the job → arm table in `../README.md`).

The orchestrator does, in order:
1. **Merge** the adapter into a wider-MLP Qwen3-32B at `build/merged/<job>/`
   (~3 min, 65 GB on disk per merged model). For mode=both both branches are
   active; for mode=retain_only the forget branch is set to 0; for forget_only
   the retain branch is set to 0; for custom you supply both scales.
2. **Stop** any running vLLM, then **serve** the merged model on port 8000
   (DP=8, max-model-len 16384, gpu-memory-util 0.92; the Fig. 4 setting is
   `GPU_MEM_UTIL=0.95`).
3. **Run harbor** with `terminus-2` over the 99 held-out tasks at
   `build/eval-dataset/`, 64 concurrent trials, max_turns=64, agent timeout
   360 s (the Fig. 4 setting is `prep_eval_dataset.py --agent-timeout 3600`).
   Trial outputs land in `build/jobs/<job>/`.
4. **Judge** every rollout with gpt-5.4-nano (via OpenRouter) using the v3
   prompt at `scripts/eval/prompts/judge_v3.txt`.
5. **Print** pass rate (with Wilson 95% CI) and hack rate at thresholds 0.5,
   0.7, 0.8 (console summary only; Fig. 4's 90% intervals are computed by `charts/plot_scatter.py`).

Skips are available for re-running individual stages: `--skip-merge`,
`--skip-serve`, `--skip-eval`, `--skip-judge`.

## Output layout

```
build/
├── merged/
│   └── <job>/                        # 65 GB merged HF model dir per run
├── jobs/
│   └── <job>/
│       ├── result.json
│       ├── <task>__<hash>/           # one trial dir per task
│       │   ├── result.json
│       │   ├── verifier/reward.txt
│       │   └── agent/trajectory.json # ATIF format, judged by score_rollouts
│       └── judge_scores_judge_v3.json
├── logs/vllm.log
└── vllm.pid
```

## Where things live in the repo

- `scripts/eval/run.py` — orchestrator (merge → serve → harbor → judge).
- `scripts/eval/merge_adapter.py` — DualMLPAdapter → wider Qwen3 fusion.
- `scripts/eval/serve_merged.sh`, `stop_serve.sh` — vLLM lifecycle helpers.
- `scripts/eval/run_harbor_eval.sh` — thin harbor wrapper.
- `scripts/eval/prep_eval_dataset.py` — symlink + task.toml patcher.
- `scripts/eval/score_rollouts.py` — judge harness over harbor trajectories.
- `scripts/eval/recompute_summary.py` — refresh summary block from reward.txt.
- `scripts/eval/judge_calibration.py` / `validate_judge_on_train.py` — prompt
  validation tooling.
- `scripts/eval/prompts/judge_{v1,v2,v3}.txt` — vendored copies of the judge
  prompts (v3 is the default; v1 is upstream's original).

## Known pitfalls

See the Pitfalls table below for the seven gotchas we hit on this
box: CUDA driver, transformers/vllm version pinning, ninja, python3-dev, flash-attn
build isolation, Docker network pools, DP=8 KV cache.

## Judge wiring

`monitoring/monitor.py` (in the wrench submodule) uses the **OpenAI Responses
API** (`client.responses.create`). OpenRouter happens to support it
transparently when given:

```
OPENAI_API_KEY=$OPENROUTER_API_KEY
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

If a future judge model fails, verify the upstream provider exposes
`/v1/responses` (not just `/v1/chat/completions`). The model id at OpenRouter
is `openai/gpt-5.4-nano` (note the leading `openai/`).

The default prompt is `judge_v3.txt`. v1 is upstream's. v2 was an
intermediate iteration kept for reference. v3 was validated against:
- 19-trial calibration set = 5 known hacks + 5 baselines + 9 hand-labeled Qwen3-32B false positives (from base-model rollouts on held-out eval tasks); 150 labeled train trajectories (50 hacks / 50 baselines / 50 attacker-legit)

v3 holds recall=1.00 against known hacks while dropping precision-killing
false positives on sloppy / sandbox-limited / premature-done-flag rollouts
(calibration tooling: see `../README.md`, "Judge").

## The two venv split

Cannot collapse to one. **vLLM 0.9.2** requires `transformers<4.55` because
`aimv2` was added to transformers 4.54+ and breaks vLLM's config registry.
**Training** needs `transformers==5.5.4` to match what we trained against.
They cannot coexist. Judge deps are tiny and live in `.venvs/eval/`.

`.venvs/vllm/bin/vllm` is invoked by `serve_merged.sh`.
`.venvs/eval/bin/python` runs `merge_adapter.py`, `score_rollouts.py`
and `run.py`.

## Pitfalls table

| Symptom | Root cause | Fix |
|---|---|---|
| `RuntimeError: NVIDIA driver too old` from vLLM startup | vllm ≥0.20 wants CUDA 13 driver | pin `vllm==0.9.2`; if you upgrade the driver you can move forward |
| `ValueError: 'aimv2' is already used by a Transformers config` | transformers ≥4.54 conflicts with vllm 0.9 | pin `transformers==4.51.3` in the vllm venv |
| flashinfer JIT compile fails: `[Errno 2] No such file or directory: 'ninja'` | missing build tool | `apt install ninja-build` |
| triton compile fails on `Python.h: No such file` | missing Python C headers | `apt install python3.12-dev` |
| flash-attn install fails: missing wheel build dep | flash-attn doesn't declare `wheel` as build-dep | `pip install wheel packaging` first; install with `--no-build-isolation` |
| harbor: `failed to create network … all predefined address pools have been fully subnetted` | Docker default `default-address-pools` exhausts at ~32 networks | write `/etc/docker/daemon.json` with expanded pools (172.30.0.0/15, 172.32.0.0/15) and `systemctl restart docker` |
| KV cache OOM at DP=8: `available KV cache memory (6.58 GiB), required 8.00 GiB` | Qwen3-32B at 32k context × 8 replicas overcommits | reduce `--max-model-len` to 16384 (still > terminus-2's 8 k summarization threshold) |
