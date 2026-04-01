# Learning to Learn-at-Test-Time: Language Agents with Learnable Adaptation Policies

This repository contains code for **Meta-TTL**, a bi-level framework that learns adaptation policies for test-time learning (TTL) in language agents. Instead of relying on fixed, hand-crafted adaptation rules, Meta-TTL formulates the discovery of effective adaptation policies as a meta-learning problem: an outer loop uses evolutionary optimization over a distribution of training tasks to find a meta-prompt that teaches agents how to improve across episodes, while an inner loop executes the standard TTL process under that meta-prompt.

**Key contributions:**
- We formalize TTL as a **meta-learning problem over adaptation policies**, providing a principled framework for optimizing how agents update themselves across episodes.
- We propose **Meta-TTL**, which uses evolutionary optimization on a task distribution to learn an adaptation policy that generalizes to unseen environments. The policy is realized as a natural-language meta-prompt that turns generic self-correction into concrete adaptation instructions.
- We evaluate on **Jericho** (interactive fiction games) and **WebArena-Lite** (web navigation), demonstrating consistent improvements over hand-crafted baselines on both in-distribution and out-of-distribution tasks.

## Repository Structure

```
meta-ttl/
  jericho_agent/          # Jericho evaluation (baselines + meta-agent)
  jericho_meta_training/  # Jericho evolutionary meta-training (outer loop)
  webarena/               # WebArena evaluation (baselines + meta-agent)
  webarena_meta_training/ # WebArena evolutionary meta-training (outer loop)
  rllm/                   # Shared library: agents, environments, runners
  requirements.md         # Python dependency snapshot
```

## Environment Setup

The validated environment uses the `jericho_min` conda env.

```bash
conda activate jericho_min
cd /path/to/meta-ttl
```

Additional references:
- Frozen Python dependency snapshot: [`requirements.md`](./requirements.md)
- WebArena env vars: [`webarena/env_setup.txt`](./webarena/env_setup.txt)

WebArena additionally requires the sites referenced in `webarena/env_setup.txt` to be reachable, BrowserGym / Playwright installed, and browser automation allowed on the machine.


## Jericho

### Baseline Evaluation

Evaluates actor-only baselines (Static, Reflexion, Memory Agent) without a meta-agent. Uses [`jericho_agent/evaluate_baselines.py`](./jericho_agent/evaluate_baselines.py).

```bash
python jericho_agent/evaluate_baselines.py \
  --agent_type reflexion \
  --actor_model google/gemini-3-flash-preview \
  --games detective zork1 temple \
  --seeds 3 \
  --repeats 5 \
  --num_episodes 6 \
  --env_step_limit 50 \
  --max_concurrent 16 \
  --save_dir results/baselines \
  --save_and_plot true
```

Supported `--agent_type`: `static`, `reflexion`, `crossmem`.

### Meta-Agent Evaluation (Inner Loop)

Evaluates a meta-agent that provides cross-episode adaptation guidance to the actor via `JerichoMetaEnv`. This corresponds to the inner-loop TTL evaluation described in the paper. Uses [`jericho_agent/evaluate.py`](./jericho_agent/evaluate.py).

```bash
python jericho_agent/evaluate.py \
  --meta_model z-ai/glm-5 \
  --actor_model google/gemini-3-flash-preview \
  --games detective zork1 temple \
  --seeds 3 \
  --repeats 5 \
  --env_step_limit 30 \
  --max_episodes 3 \
  --prompt_tag DEFAULT \
  --save_and_plot true
```

Use `--prompt_tag` to select which meta-prompt to deploy (e.g., `DEFAULT` for the naive baseline, or an optimized variant).

### Evolutionary Meta-Training (Outer Loop)

Runs the outer-loop evolutionary optimization to learn the adaptation policy (meta-prompt) over a distribution of Jericho training tasks. Uses [`jericho_meta_training/train_meta_agent.py`](./jericho_meta_training/train_meta_agent.py).

```bash
python jericho_meta_training/train_meta_agent.py
```

The script builds a mixed-game train/val split from `detective`, `zork1`, and `temple`, then optimizes the meta-prompt through iterative proposal, evaluation, and selection. Configuration (dataset composition, meta model, actor model) is currently set inside the file. Outputs include trajectory pickles under `train_results/` and optional W&B logging.


## WebArena-Lite

### Baseline Evaluation

Evaluates actor-only baselines on WebArena-Lite tasks. Uses [`webarena/evaluate_baselines_webarena.py`](./webarena/evaluate_baselines_webarena.py).

```bash
python webarena/evaluate_baselines_webarena.py \
  --agent_types reflexion crossmem static \
  --splits id ood \
  --model google/gemini-3-flash-preview \
  --repeat 5 \
  --workers 4 \
  --max_steps 10 \
  --log_base logs/baselines_webarena
```

### Meta-Agent Evaluation (Inner Loop)

Evaluates a meta-agent over `WebArenaMetaEnv` with site-aware parallel scheduling. Uses [`webarena/evaluate_webarena.py`](./webarena/evaluate_webarena.py).

Before running, ensure `webarena/env_setup.txt` is valid and the WebArena sites are live.

```bash
python webarena/evaluate_webarena.py \
  --meta_model openai/gpt-5 \
  --actor_model google/gemini-3-flash-preview \
  --tasks 0,1,2,25 \
  --max_episodes 4 \
  --env_step_limit 10 \
  --max_concurrent 4 \
  --config_dir config_files_lite
```

### Evolutionary Meta-Training (Outer Loop)

Runs the outer-loop evolutionary optimization for WebArena. Uses [`webarena_meta_training/train_meta_agent.py`](./webarena_meta_training/train_meta_agent.py).

```bash
python webarena_meta_training/train_meta_agent.py \
  --meta_model google/gemini-3-flash-preview \
  --n_val_per_domain 2 \
  --n_id_eval_per_domain 10 \
  --max_concurrent 4 \
  --max_metric_calls 100 \
  --max_episodes 3 \
  --env_step_limit 10
```

The script builds domain-aware train/val/eval splits, then optimizes the meta-prompt through evolutionary search. Outputs include run artifacts under `train_results_webarena/` and W&B logging.


## Notes

- Baseline evaluation and meta-agent evaluation are intentionally separate workflows. In Jericho, `evaluate_baselines.py` runs actor-only agents while `evaluate.py` runs the full actor + meta-agent loop. The same separation applies to WebArena.
- The evolutionary meta-training scripts optimize the meta-prompt (adaptation policy) offline. At test time, the learned meta-prompt is frozen and applied zero-shot to unseen tasks.
- For a fully scripted environment bootstrap, use [`requirements.md`](./requirements.md) as the Python package snapshot and add the machine-specific WebArena/browser setup on top.
