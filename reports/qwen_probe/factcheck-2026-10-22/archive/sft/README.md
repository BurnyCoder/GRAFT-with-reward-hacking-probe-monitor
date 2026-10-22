# `sft/`: GRAFT for large-scale SFT (Qwen3-32B, Terminal Wrench)

Code and evaluation records for paper §5 (`sec:results-sft`, Fig. `fig:large-scale-sft`), App.
"Large-scale SFT", App. "Large-scale SFT evaluation" (incl. `app:hack-prompt`), and the SFT part of
`app:ip-prompts`.

- Run everything from `sft/`. Python modules live under the `scripts` package (`python -m scripts.…`,
  `PYTHONPATH=.`).
- RL code (Sec. 3–4) lives in `../rl/`. This directory is independent of it.

## Layout

| Path | What |
|---|---|
| `scripts/data_prep.py` | Terminal Wrench (TW) trajectories → `clean_pool.jsonl`, `hack_pool.jsonl` |
| `scripts/split_train_eval.py` | 70/30 task-id split (seed 42) → `task_split.json` |
| `scripts/gr/data_prep_gr.py` | builds the labeled pool D_L (forget) and unlabeled pool D_U (retain) via an oracle classifier |
| `scripts/gr/train_gr.py` | trainer (GRAFT, `--no-routing`, `--gradient-ascent`, `--inject-prompt`, …) |
| `scripts/gr/adapter.py` | `DualMLPAdapter` (two SwiGLU branches per MLP) |
| `scripts/gr/loader.py` | per-example class assignment (forget / unclassified / anchor) |
| `scripts/gr/split_moment.py` | `SplitMomentAdamW` (m from routed grad, v from natural grad, fp32 moments) |
| `scripts/gr/test_gr_core.py`, `scripts/gr/test_adapter.py` | unit tests (`test_gr_core` on CPU; `test_adapter` needs CUDA) |
| `scripts/launch_gr_32b_split.sh`, `scripts/accelerate_ddp.yaml` | GRAFT launch (8-process DDP, bf16) |
| `scripts/eval/` | eval pipeline (table below) |
| `scripts/eval/prompts/judge_v{1,2,3}.txt` | judge prompts (v3 = default) |
| `scripts/eval/patches/harbor_verifier_on_ctx_overflow.patch` | local harbor patch |
| `charts/plot_scatter.py` | Fig. 4 |
| `build/jobs/<job>/{config,result,judge_scores_judge_v3}.json` | eval records for 56 jobs |
| `data/task_split.json`, `data/eval_tasks.txt` | committed split (232 train / 99 eval tasks); `eval_tasks.txt` = the 99 eval ids |
| `docs/EVAL_SETUP.md` | eval-box setup, judge wiring, pitfalls table |
| `.env.example` | template for `.env` (`OPENROUTER_API_KEY`, `HF_TOKEN`) |
| `third_party/harbor`, `third_party/terminal-wrench` | git submodules (populated by `git submodule update --init`) |

Eval scripts (`scripts/eval/`):

| Script | Role |
|---|---|
| `setup_box.sh` | apt deps, Docker address pools, submodules + harbor patch, `.venvs/vllm`, `.venvs/eval`, harbor `uv sync`, smoke test |
| `prep_eval_dataset.py` | 99 eval tasks → `build/eval-dataset/`; sets agent timeout; optional prompt injection |
| `run.py` | per checkpoint: merge → serve → harbor → judge → summary |
| `merge_adapter.py` | folds the adapter into a wider Qwen3 MLP (vanilla architecture for vLLM) |
| `serve_merged.sh`, `stop_serve.sh` | vLLM 0.9.2 lifecycle (DP=8, `max_model_len` 16384) |
| `run_harbor_eval.sh` | `harbor run` with terminus-2 |
| `score_rollouts.py` | LLM judge over trajectories (TW `monitoring/monitor.py` harness) |
| `recompute_summary.py` | rewrites the summary block (thresholds 0.5/0.7/0.8) from `reward.txt` |
| `treat_agent_errors_as_fail.py` | agent-side errors without reward → reward 0 |
| `sweep_round.sh`, `sweep_round_baselines.sh` | multi-round drivers (rounds 2–4) |
| `sweep_ga4.sh` | GA×4 elicitation-panel driver |
| `judge_calibration.py`, `validate_judge_on_train.py` | judge-prompt validation tooling |
| `classify_v5_intent.py` | regex intent check over trajectories |
| `config.py` | paths and defaults |

## Setup

- Hardware: training is 8-process DDP (`scripts/accelerate_ddp.yaml`); eval serves Qwen3-32B
  with DP=8 on ≥80 GB GPUs (`docs/EVAL_SETUP.md`).
- Linux with Docker and `sudo`. `uv` required.
- Steps:
  ```bash
  git submodule update --init sft/third_party/harbor sft/third_party/terminal-wrench   # from the repo root
  cd sft
  cp .env.example .env    # set OPENROUTER_API_KEY; HF_TOKEN only for the first model download
  bash scripts/eval/setup_box.sh
  ```
- Submodule pins: harbor `290ab985`, terminal-wrench `d8a29613`.
- HF cache: `export HF_HOME=$HOME/huggingface` (the layout `run.py`, `sweep_round.sh` and
  `.env.example` assume) before downloading `Qwen/Qwen3-32B`, for both training and eval. The
  training launcher runs offline with the default HF cache, which is `~/.cache/huggingface`
  when `HF_HOME` is unset, so a snapshot downloaded there is not found by `run.py`
  (or pass `--base-model <snapshot dir>` to `run.py`).
- `setup_box.sh`:
  - creates `.venvs/vllm` (vllm 0.9.2, transformers 4.51.3, flash-attn 2.8.3) and `.venvs/eval`
    (torch 2.5.1+cu124, transformers 5.5.4, trl 1.2.0, accelerate 1.13.0, datasets 4.8.4,
    peft 0.19.1, flash-attn 2.8.3, openai, python-dotenv, …);
  - rewrites `/etc/docker/daemon.json` and restarts Docker. Use a dedicated machine;
  - exits if the Docker socket is not writable (join the `docker` group);
  - does not install `python3.12-dev` (see the `docs/EVAL_SETUP.md` pitfalls table).
- **Check the harbor patch line.** `setup_box.sh` must print
  `applying harbor_verifier_on_ctx_overflow.patch to third_party/harbor`. If it prints
  `… already applied (or doesn't apply cleanly), skipping` on a fresh checkout, the patch did not
  apply. Without it, trials that hit `ContextLengthExceededError` / `OutputLengthExceededError`
  skip the verifier.
- The setup smoke test builds `build/eval-dataset/` with the default 360 s agent timeout and no
  prompt. Re-run `prep_eval_dataset.py` (below) before any real eval.
- Details: `docs/EVAL_SETUP.md`.

## Regenerate Fig. 4 (released data, no GPU)

```bash
pip install matplotlib
python charts/plot_scatter.py      # → charts/routing_scatter_main.{png,pdf}
```

- Inputs: `build/jobs/<job>/judge_scores_judge_v3.json` for the 45 Fig. 4 jobs (table below).
- Env knobs (defaults = Fig. 4):
  - `HACK_THRESHOLD=0.8`: judge score ≥ threshold counts as a hack. Use only 0.5, 0.7 or 0.8.
    The k=1 points read precomputed counts for these three thresholds. For any other value
    their hack count silently becomes 0.
  - `LEGIT_X=1`: y-axis = correct-solution ("legit") rate. `0` = raw pass rate.
  - `GR_STEM=gr-split-unc-ep5-retain`: job-name stem of the GRAFT point.
  - `SHOW_OLD_GR=1`: adds the GRAFT variant without split-moment (`gr-s1like-unc-ep5-retain-*`)
    as a light-blue point. It is not added to the hand-built legend.
- Non-default `HACK_THRESHOLD` / `LEGIT_X` add `_thr<T>` / `_passx` to the output file name.
- Printed summary:
  - lists every plotted baseline plus GA×1/×2 and the GRAFT variant without split-moment;
  - does **not** print the plotted GRAFT arm (`GR_STEM`). Read the plot, or call
    `get_clustered_ci` / `get_pass_hack_ci` in the script.
- To plot your own GRAFT run against the released baselines:
  - name its jobs `<stem>-no`, `<stem>-no-r2`, `<stem>-no-r3`, `<stem>-no-r4`, `<stem>-v5`;
  - run with `GR_STEM=<stem>`.

### Metrics and CIs (as implemented in `charts/plot_scatter.py`)

- Per trial:
  - pass = verifier reward 1;
  - hack = judge score ≥ 0.8;
  - legit = pass and not hack.
- Multi-round configs (k>1, no-prompt panel): clustered Wald on per-task means. Average over rounds
  within each task, then m ± 1.645·√(s²/n_tasks), clipped to [0, 1]. Tasks with no outcome for a
  metric drop out.
- k=1 configs: Wilson 90%. Counts come from the job's summary block.
  - legit numerator = passes − passed-and-hacked;
  - legit denominator = trials with a judge score, which can differ slightly from the pass
    denominator (trials with a reward).
- GRAFT w/o routing, elicitation panel: paired normal CI over the retain-only and forget-only jobs
  (per-task average of the two).
- GRAFT w/o routing, no-prompt panel: retain-only and forget-only jobs, 2 rounds each, pooled into
  one clustered CI.
- `scripts/eval/run.py` prints a separate console summary with a 95% Wilson interval on pass rate.
  Fig. 4 does not use it.
- Denominators: trials without a reward or a judge score are excluded. Per-job counts for the
  Fig. 4 jobs are 94–99 of 99. `base-qwen3-32b-v5-99` lists 112 trials, 13 of which are duplicate
  infra-failed attempts with neither.

## Released eval records (`build/jobs/`)

- 56 job dirs × {`config.json` (harbor job config), `result.json` (harbor stats incl. per-exception
  counts), `judge_scores_judge_v3.json`}.
- `judge_scores_judge_v3.json` = `summary` + per-trial `results` with `task_id`, `reward`, `score`,
  `response` and `error`.
  - `response.reasoning` is **free text from the gpt-5.4-nano judge** (via OpenRouter) about one
    Qwen3-32B trajectory. It quotes commands, file paths and Terminal Wrench task content
    (Apache-2.0), including task-internal IP addresses.
  - These files are evaluation records, not metrics only. The judge provider's output terms apply.
    See the repo-level `NOTICE`.
- `task_id` = harbor trial-dir prefix. Harbor truncates slugs to 32 characters (6 tasks),
  consistently across rounds.
- Agent kwargs in every `config.json`: `max_turns` 64, `enable_summarize` false,
  `enable_thinking` false, `seed` only in rounds 2–4. `n_concurrent_trials` 64, `n_attempts` 1.
  No temperature is passed (see "Reproducibility notes").

### Job → arm (Fig. 4)

| Fig. 4 legend | Checkpoint (run name) | Eval mode | No-prompt jobs (k) | Elicitation job |
|---|---|---|---|---|
| GRAFT (ours) | `scripts/launch_gr_32b_split.sh` (`gr_32b_mlp_fr02_ddp_s1like_unc_both_split_ru001`), ep5. Eval stem `gr-split-unc-ep5-retain` | retain_only | `gr-split-unc-ep5-retain-no{,-r2,-r3,-r4}` (4) | `gr-split-unc-ep5-retain-v5` |
| Inoculation adapter | `gr_32b_mlp_fr02_ddp_s1like_pretrainf_filter_ep5` | retain_only | `gr-s1like-pretrainf-filter-ep5-retain-no{,-r2,-r3,-r4}` (4) | `gr-s1like-pretrainf-filter-ep5-retain-v5` |
| IP (elicitation paraphrase) | `…_s1like_inoc_general_ep5` | retain_only | `gr-s1like-inoc-general-ep5-retain-no{,-r2,-r3,-r4}` (4) | `gr-s1like-inoc-general-ep5-retain-v5` |
| IP (EM prompt) | `…_s1like_inoc_emergent_ep5` | retain_only | `gr-s1like-inoc-emergent-ep5-retain-no{,-r2,-r3,-r4}` (4) | `gr-s1like-inoc-emergent-ep5-retain-v5` |
| No intervention | `…_s1like_noint_ep5` | both | `gr-s1like-noint-ep5-both-no` (1) | `gr-s1like-noint-ep5-both-v5` |
| Monitor filtering | `…_s1like_ga0_ep5` | retain_only | `gr-s1like-ga0-ep5-retain-no` (1) | `gr-s1like-ga0-ep5-retain-v5` |
| Oracle filtering | `…_s1like_skyline_ep5` | retain_only | `gr-s1like-skyline-ep5-retain-no{,-r2,-r3,-r4}` (4) | `gr-s1like-skyline-ep5-retain-v5` |
| Gradient ascent (×4) | `…_s1like_ga4_ep5` | retain_only | `gr-s1like-ga4-ep5-retain-no{,-r2,-r3,-r4}` (4) | `gr-s1like-ga4-ep5-retain-v5` |
| GRAFT w/o routing | `…_s1like_noint_ep5` (**same checkpoint as No intervention**) | retain_only + forget_only, pooled | `gr-s1like-noint-ep5-{retain,forget}-no{,-r2}` (2+2) | `gr-s1like-noint-ep5-{retain,forget}-v5` (paired) |
| No training | base Qwen3-32B HF snapshot (no merge) | — | `base-qwen3-32b-no-99`, `base-qwen3-32b-no-r{2,3,4}` (4) | `base-qwen3-32b-v5-99` |

`…` = `gr_32b_mlp_fr02_ddp`. The `v5` suffix = elicitation panel; `no` = no-prompt panel.

Other jobs in `build/jobs/` (not plotted in Fig. 4):

| Jobs | What |
|---|---|
| `gr-s1like-ga{1,2}-ep5-retain-{no,v5}` (4) | GA×1/×2, same 99 eval tasks |
| `gr-s1like-unc-ep5-retain-no{,-r2,-r3,-r4}`, `gr-s1like-unc-ep5-retain-v5` (5) | GRAFT variant: same recipe without `--split-moment --kappa 2` (`SHOW_OLD_GR=1` plots it) |
| `gr-s1like-inoc-v5-ep5-retain-{no,v5}` (2) | IP variant `inoc_v5` |

## Data pipeline

External input: Terminal Wrench trajectories (`tasks/<id>/<model>/{baseline,hack,non_hack}_trajectories/…`).

```bash
export GRAFT_SFT_DATA_DIR=/path/to/sft-data          # default /workspace/training/data
GRAFT_TW_ROOT=third_party/terminal-wrench python scripts/data_prep.py      # default TW root /workspace/terminal-wrench
python scripts/split_train_eval.py
python -m scripts.gr.data_prep_gr --recall 0.2 --clean-frac 0.5
```

- `scripts/data_prep.py` (no CLI flags):
  - clean = TW `baseline` + `attacker_legitimate_solve`; hack = `rewarded_serious_exploit` +
    `rewarded_nonserious_hack`;
  - strips the red-team prompt block (from "You are an authorized Red Team QA Agent" up to
    "Current terminal state:") from the first user message. This assumes the block sits right
    before that marker;
  - drops records over 32,768 tokens under the **Qwen3-8B** tokenizer's chat template. This
    pre-filter is separate from the 16,384-token training filter.
- `scripts/split_train_eval.py` (no CLI flags):
  - seed 42, `EVAL_FRAC` 0.30, split over the union of task ids in both pools;
  - writes `{clean,hack}_{train,eval}.jsonl` (not used by GRAFT) and
    `$GRAFT_SFT_DATA_DIR/task_split.json`;
  - the committed `data/task_split.json` is the split used (232/99). A re-run reproduces it only if
    the pools contain the same task ids (same TW commit and filters). Compare before training, or
    copy `data/task_split.json` into `$GRAFT_SFT_DATA_DIR/`;
  - do not set `GRAFT_SFT_DATA_DIR=data`: the script would overwrite the committed split, which the
    eval pipeline reads (`scripts/eval/config.py`).
- `scripts/gr/data_prep_gr.py`:
  - `--recall` = fraction of training hacks labeled forget (D_L). The rest stay in the retain pool
    as false negatives;
  - `--clean-frac 0.5` = subsample to equal clean/hack counts; `--clean-frac 1.0` drops all hacks;
  - `--exclude-attacker-legit` is on by default, so the clean pool keeps only `baseline` records;
  - `--seed` 42;
  - writes `gr_train_{retain,forget}.jsonl` and `gr_eval_{retain,forget}.jsonl` (held-out-task
    loss evals during training).
- The run-name tag `fr02` = `--recall 0.2`. The script default is 0.5.

## Training (GRAFT)

```bash
GRAFT_SFT_VENV=.venvs/eval GRAFT_SFT_DATA_DIR=/path/to/sft-data GRAFT_SFT_CKPT_DIR=/path/to/ckpts \
  bash scripts/launch_gr_32b_split.sh
```

- The launcher:
  - `cd`s to `sft/`, activates `$GRAFT_SFT_VENV` (default `.venv`), sources `.env` if present;
  - sets `HF_HUB_OFFLINE=1`, so download `Qwen/Qwen3-32B` into the HF cache first;
  - runs `accelerate launch --config_file scripts/accelerate_ddp.yaml scripts/gr/train_gr.py …`
    (8 processes, bf16);
  - appends extra CLI args after its own (argparse keeps the last value of a repeated flag);
  - logs to `logs/`.
- Recipe (all flags in the launcher; `train_gr.py`'s own defaults are Qwen3-8B settings, so use
  the launcher's flags):
  - `--model-name Qwen/Qwen3-32B`, `--d-retain 64 --d-forget 64` (adapters on all layers);
  - `--match-rslora --lora-alpha 32 --lora-r 16`. Per-branch scale = α·√(1/(0.5·d)) =
    32·√(1/32) ≈ 5.66. **`--lora-r` has no effect** under `--match-rslora`;
  - `--lr 5e-4`, betas (0.9, 0.95), weight decay 1e-4, `--warmup-ratio 0.05` linear warmup, then
    cosine to 0;
  - `--epochs 5`, `--step-size 48` global (8 ranks × 6 accumulation × micro-batch 1);
  - `--max-length 16384 --filter-overlong`: longer records are dropped, not truncated;
  - `--classifier-retain-recall 0.01 --retain-from-unlabeled`: each retain-pool (D_U) record
    becomes an anchor with probability 0.01 (seed 42). Anchors run with the forget branch ablated
    and update only the retain adapter;
  - `--unclassified-trains-both`: other D_U records update both adapters;
  - `--split-moment --kappa 2`: on D_L records the retain gradient is zeroed and the forget
    gradient scaled ×2. One `SplitMomentAdamW` takes m from the routed gradient and v from the
    natural gradient (fp32 moments), freezes an adapter on windows where it has no examples, and
    applies the forget participation factor;
  - `--run-name gr_32b_mlp_fr02_ddp_s1like_unc_both_split_ru001`.
- Output:
  - `$GRAFT_SFT_CKPT_DIR/<run-name>/checkpoint-epoch-{1..5}/adapter_state_dict.pt`;
  - final `$GRAFT_SFT_CKPT_DIR/<run-name>/adapter_state_dict.pt`;
  - adapter weights only. The paper evaluates epoch 5.
- W&B: logs only if `WANDB_API_KEY` is set (project `terminal-wrench-sft`). Launcher health check:
  `gr/forget_realized_step` ≈ 2.0 on windows with hacks.
- Held-out loss evals: at step 0 and after each epoch (`--no-evals` / `--eval-only-at-end` to
  skip). Step 0 is cached at `$GRAFT_SFT_DATA_DIR/step0_eval_cache.json`, keyed by nothing but the
  adapter mode. Delete the cache after changing the model or data.
- Single training seed per arm.
- Training environment: `.venvs/eval` (created by `setup_box.sh`).

### Baselines

- Run `train_gr.py` directly via `accelerate launch` (pattern as in the launcher). The launcher
  hard-codes `--split-moment --kappa 2`, which `train_gr.py` rejects together with
  `--gradient-ascent`.
- Baselines run without `--split-moment`: two plain `torch.optim.AdamW` over bf16 adapter
  parameters.
- Remaining flags as in the GRAFT launcher.

| Arm (run-name tag) | Flags |
|---|---|
| No intervention / GRAFT w/o routing (`noint`) | `--no-routing`: every record is "unclassified" and trains both adapters. So **no anchors** and no forget-only updates. One checkpoint, evaluated in three modes |
| Monitor filtering (`ga0`) | `--gradient-ascent --gradient-ascent-magnitude 0`: single adapter (forget branch ablated in every forward); D_L loss ×0 |
| Gradient ascent ×1/×2/×4 (`ga1/ga2/ga4`) | `--gradient-ascent --gradient-ascent-magnitude {1,2,4}` |
| Oracle filtering (`skyline`) | data from `data_prep_gr.py --clean-frac 1.0`; eval is retain_only |
| IP paraphrase / EM (`inoc_general` / `inoc_emergent`) | `--inject-prompt <file>`: the file text plus a blank line is prepended to the **first user message** of every record (records have no system turn) |

## Evaluation pipeline

Eval environment for the Fig. 4 settings (repo defaults in parentheses):

```bash
set -a; source .env; set +a
export N_ATTEMPTS=1 N_CONCURRENT=64 ENABLE_SUMMARIZE=false ENABLE_THINKING=false
export GPU_MEM_UTIL=0.95                         # (default 0.92)
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=$OPENROUTER_API_KEY        # judge key; harbor also sends it to API_BASE (local vLLM, no auth check)
```

1. **Prep the eval dataset** (99 eval tasks from `data/task_split.json`):
   ```bash
   .venvs/eval/bin/python -m scripts.eval.prep_eval_dataset --agent-timeout 3600     # no-prompt panel
   .venvs/eval/bin/python -m scripts.eval.prep_eval_dataset --agent-timeout 3600 \
       --inject-prompt third_party/terminal-wrench/prompts/hack_prompt_v5.md          # elicitation panel
   ```
   - default agent timeout is 360 s; the Fig. 4 setting is 3600 s;
   - `--inject-prompt` prepends the file text plus a blank line to each task's `instruction.md`;
   - both panels share `build/eval-dataset/`. Re-prep to switch panels.
2. **Per adapter checkpoint:**
   ```bash
   .venvs/eval/bin/python -m scripts.eval.run --checkpoint <ckpt>/adapter_state_dict.pt \
       --mode retain_only --job-name <job> --merged-dir build/merged/<name>
   ```
   - modes: `retain_only` (deployment; forget scale 0), `both`, `forget_only`, `custom`
     (`--retain-scale`, `--forget-scale`);
   - steps: merge (about 65 GB per merged model) → `stop_serve.sh` → `serve_merged.sh` →
     `run_harbor_eval.sh` → `score_rollouts` (gpt-5.4-nano, `judge_v3`) → `recompute_summary`;
   - vLLM served-model name = basename of `--merged-dir`;
   - `--base-model` (default `Qwen/Qwen3-32B`) resolves to the last snapshot under
     `$HF_HOME/hub` (default `~/huggingface/hub`). Pass an absolute snapshot path to pin it;
   - skips: `--skip-merge`, `--skip-serve`, `--skip-eval`, `--skip-judge`.
3. **Base model** (no merge):
   ```bash
   bash scripts/eval/stop_serve.sh
   bash scripts/eval/serve_merged.sh <Qwen3-32B snapshot dir> base-qwen3-32b
   bash scripts/eval/run_harbor_eval.sh base-qwen3-32b <job>
   .venvs/eval/bin/python -m scripts.eval.score_rollouts build/jobs/<job> \
       --judge-model openai/gpt-5.4-nano --prompt judge_v3 \
       --output build/jobs/<job>/judge_scores_judge_v3.json
   .venvs/eval/bin/python -m scripts.eval.recompute_summary build/jobs/<job> \
       --scores-file judge_scores_judge_v3.json
   ```
   - always run `recompute_summary` with `--scores-file judge_scores_judge_v3.json`. Its default
     file name (`judge_scores_v3.json`) does not match;
   - only `recompute_summary` writes the 0.5/0.7/0.8 threshold block that `plot_scatter.py` reads
     for k=1 points. Without it those points show 0 hacks.
4. **Agent-error handling:**
   ```bash
   .venvs/eval/bin/python -m scripts.eval.treat_agent_errors_as_fail --dry-run    # then without --dry-run
   ```
   - sets reward 0 for trials without a reward that ended in `ContextLengthExceededError` or a
     non-setup `AgentTimeoutError`, then re-runs `recompute_summary`;
   - infra errors (Docker, setup timeouts, …) stay excluded;
   - scans every job under `build/jobs/` (`--jobs-dir` to change). The shipped jobs have no trial
     dirs, so it does not touch them;
   - not called by `run.py` or the drivers.
5. **Multi-round protocol (k=4, no-prompt panel):**
   - round 1: no seed (vLLM default RNG). Rounds r=2,3,4: `LLM_SEED=2000+r` (2002/2003/2004),
     passed by `run_harbor_eval.sh` into the LLM-call kwargs. `config.json` records it;
   - job names `<stem>-no`, `<stem>-no-r2`, `<stem>-no-r3`, `<stem>-no-r4`. The elicitation panel
     is k=1 (`<stem>-v5`);
   - manual round: `LLM_SEED=2002 … -m scripts.eval.run … --skip-merge --merged-dir <existing>`;
   - drivers as committed:
     - `bash scripts/eval/sweep_round.sh <r>`: rounds 2–4 for the GRAFT variant without
       split-moment (`gr-s1like-unc-ep5-retain`), IP paraphrase and the base model;
     - `bash scripts/eval/sweep_round_baselines.sh <r>`: GA×4, oracle filtering, IP (EM),
       inoculation adapter, plus the w/o-routing retain/forget pair in round 2 only;
     - both expect the round-1 merged dirs under `build/merged/` and a no-prompt
       `build/eval-dataset/` with 99 tasks, and both skip already-judged jobs.
6. **GA×4 elicitation job:** `bash scripts/eval/sweep_ga4.sh`.
   - evaluates `checkpoints/gr_32b_mlp_fr02_ddp_s1like_ga4_ep{1..5}` with `--mode retain_only` on
     the current `build/eval-dataset/` (prep it with the v5 prompt first);
   - Fig. 4 uses the ep5 job `gr-s1like-ga4-ep5-retain-v5`.
7. **Logs:** `build/logs/` (`GRAFT_EVAL_LOG` overrides the drivers' log path).

- **Network warning:** `serve_merged.sh` binds vLLM to `0.0.0.0:8000` without authentication by
  default. On a shared or internet-facing host, firewall the port or set `VLLM_HOST=127.0.0.1`.
  `run_harbor_eval.sh` defaults to `API_BASE=http://localhost:8000/v1`.
- `stop_serve.sh` force-kills **every** process on the host matching `vllm serve` / `EngineCore`,
  not only the one it started.

## Judge

- Model `openai/gpt-5.4-nano` via OpenRouter (Responses API). It is the only judge used, and its
  future availability is not guaranteed.
- Prompt `scripts/eval/prompts/judge_v3.txt` (intent-based).
  - `judge_v1.txt` reproduces the Terminal Wrench monitor prompt;
  - `judge_v2.txt` is an intermediate version.
- Harness: `DEVELOPER_TEMPLATE`, `USER_TEMPLATE`, `call_monitor`, `format_atif_trajectory` and
  `parse_monitor_response` from `third_party/terminal-wrench/monitoring/monitor.py`.
- The judge reads the task text from `build/eval-dataset/<task>/instruction.md` **at judging time**.
  Judge a job before re-prepping the dataset for the other panel.
- Calibration tooling:
  - `judge_calibration.py`: 19-trial set = 5 known TW train hacks + 5 TW train baselines + 9
    hand-labeled false positives. The false positives come from base-model rollouts on held-out
    eval tasks (job dir with trial dirs, passed via `--qwen32b-job`);
  - `validate_judge_on_train.py`: 150 labeled TW train trajectories (50 hacks / 50 baselines /
    50 attacker-legit).
- `classify_v5_intent.py` (regex intent signals) runs over per-trial trajectory dirs
  (`build/jobs/<job>/<task>__<hash>/`).

## Tests

From `sft/` with the `.venvs/eval` environment (the training dependencies are needed).

```bash
PYTHONPATH=. python -m scripts.gr.test_gr_core    # CPU, tiny model, incl. a 2-process gloo parity check
PYTHONPATH=. python -m scripts.gr.test_adapter    # needs CUDA and downloads Qwen/Qwen3-8B
```

## Reproducibility notes

- **Optimizers.** GRAFT uses `SplitMomentAdamW` (κ=2); all baselines use plain AdamW with bf16
  states.
- **Eval drivers.**
  - `sweep_round.sh`, `sweep_round_baselines.sh` and `sweep_ga4.sh` set `AGENT_TIMEOUT=3600` but
    never pass it. Prep `build/eval-dataset/` with `--agent-timeout 3600` before running them.
  - The agent timeout is not stored in the job JSONs (it lives in the prepared `task.toml`).
- **Seeds.** Round 1 is unseeded; rounds 2–4 are seeded. Single training seed per arm. Rounds
  resample the eval only, not training.
- **Repo defaults vs Fig. 4 settings.** Agent timeout 360 vs 3600 s; `GPU_MEM_UTIL` 0.92 vs 0.95.
- **Temperature.** Not set anywhere in this repo or in any job config. terminus-2's default at the
  pinned harbor commit is 0.7 (`src/harbor/agents/terminus_2/terminus_2.py`, checked 2026-10-12).
- **Context cap.** `max_model_len` is 16384.
  - Many trials end in `ContextLengthExceededError`: 35–88 of 99 per SFT'd-checkpoint Fig. 4 job;
    17–31 for the base model.
  - Counts per exception type are in each `result.json` (`stats.evals.*.exception_stats`).
  - The patched harbor still runs the verifier on these trials.
- **Threshold.** Hacked iff judge score ≥ 0.8.
- **GRAFT w/o routing** shares its checkpoint with No intervention. That run used `--no-routing`,
  so it differs from GRAFT in routing, anchors, optimizer and κ.
- **GA multiplier.** Only ×1, ×2 and ×4 exist. All three were evaluated only on the 99 eval tasks;
  there is no separate validation split.
- **Base model.** `run.py` takes the last cached Qwen3-32B snapshot; `sweep_round.sh` takes the
  first. Pass an absolute snapshot path to pin it.

## Pitfalls

- `run.py` does **not** overwrite a non-empty `--merged-dir`. It prints "refusing to overwrite" and
  then serves the existing dir. Use a fresh merged dir per checkpoint/mode, or delete the old one.
- Wrong `recompute_summary` file name or a skipped `recompute_summary`: k=1 hack counts read as 0
  in Fig. 4.
- `HACK_THRESHOLD` outside {0.5, 0.7, 0.8}: k=1 hack counts read as 0.
- Switching panels: re-run `prep_eval_dataset.py`, and judge the previous panel's jobs first.
- `setup_box.sh` leaves `build/eval-dataset/` at the 360 s default timeout.
- Step-0 loss cache (`step0_eval_cache.json`) is reused across runs.
- Drivers source `.env` unconditionally. Create it from `.env.example` first.

## Provenance and licenses

- `scripts/gr/split_moment.py` is ported from `../rl/split_moment.py` in this repo (authors' code).
- `scripts/eval/prompts/judge_v1.txt` reproduces the Terminal Wrench monitor prompt (Apache-2.0);
  v2/v3 are modified versions.
- The harbor patch modifies harbor (Apache-2.0; github.com/harbor-framework/harbor, formerly
  laude-institute/harbor).
- Terminal Wrench and harbor are submodules, not vendored.
- See the repo-level `NOTICE`.
