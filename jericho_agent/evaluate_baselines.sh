#!/bin/bash

echo "Starting evaluation jobs..."

python jericho_agent/evaluate_baselines.py \
  --agent_type reflexion \
  --actor_model google/gemini-3-flash-preview \
  --games detective zork1 temple balances library zork3 \
  --seeds 1 \
  --repeats 5 \
  --num_episodes 6 \
  --env_step_limit 50 > eval_jericho_baselines_1.log 2>&1

python jericho_agent/evaluate_baselines.py \
  --agent_type crossmem \
  --actor_model google/gemini-3-flash-preview \
  --games detective zork1 temple balances library zork3 \
  --seeds 1 \
  --repeats 10 \
  --num_episodes 6 \
  --env_step_limit 50 > eval_jericho_baselines_2.log 2>&1

wait

echo "Both evaluation jobs have finished."
