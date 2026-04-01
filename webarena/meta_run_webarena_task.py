#!/usr/bin/env python
"""
Subprocess entry point: runs the full meta-loop for ONE WebArena task.
Receives config via CLI JSON args, returns Trajectory via pickle on stdout.

Called by WebArenaRunnerMP._worker_run_group() — not meant to be run directly.
"""
import sys
import os

# === Redirect stdout to stderr BEFORE any imports that might print ===
# Save original stdout fd for pickle output at the end.
sys.stdout.flush()
_pickle_fd = os.dup(sys.stdout.fileno())
_PICKLE_OUT = os.fdopen(_pickle_fd, 'wb')
# Redirect fd 1 (stdout) to fd 2 (stderr) so all prints go to stderr
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

import json
import pickle
import time
import argparse
import contextlib
import traceback
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================================
# Load WebArena Environment Variables (same as evaluate_webarena.py)
# ============================================================================

def load_webarena_env():
    """Load WebArena environment variables from env_setup.txt."""
    env_file = Path(__file__).parent / "env_setup.txt"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[7:]
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if '$BASE_URL' in value and 'BASE_URL' in os.environ:
                        value = value.replace('$BASE_URL', os.environ['BASE_URL'])
                    os.environ[key] = value

    if 'OPENROUTER_API_KEY' in os.environ:
        base = "https://openrouter.ai/api/v1"
        os.environ["OPENAI_BASE_URL"] = base
        os.environ["OPENAI_API_BASE"] = base
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

load_webarena_env()

from rllm.environments.webarena.meta_webarena_env import WebArenaMetaEnv
from rllm.agents.ttl_cum_agent import TTLCumulativeAgent
from rllm.agents.agent import Trajectory
from rllm.environments.jericho.openai_helpers import chat_completion_with_retries, init_global_client


# ============================================================================
# Log redirect (same as webarena_mp_runner.py)
# ============================================================================

@contextlib.contextmanager
def _task_log_redirect(log_path):
    """Redirect stdout/stderr to a task-specific log file."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    log_file = open(log_path, 'w', buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        yield log_file
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()


# ============================================================================
# Core: run the full meta-loop for one task
# ============================================================================

def run_single_task(task_config, env_args, agent_args, meta_llm_cfg, api_cfg, log, log_dir):
    """Run complete meta-loop for one task. Returns Trajectory.

    This is the same logic as the original _worker_run_group() inner loop
    (webarena_mp_runner.py lines 189-266), but isolated in its own process.
    """
    # Ensure CWD is webarena/ so relative paths (config_files_lite/, etc.) resolve correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if api_cfg:
        init_global_client(base_url=api_cfg.get('base_url'), api_key=api_cfg.get('api_key'))

    task_start = time.time()

    log_path = None
    if log_dir:
        tid = task_config.get('task_id', 0)
        rid = task_config.get('run_id', 0)
        log_path = os.path.join(log_dir, f"task_{tid}_r{rid}.log")

    log_ctx = _task_log_redirect(log_path) if log_path else contextlib.nullcontext(None)

    with log_ctx:
        try:
            # 1. Instantiate env and agent
            full_env_args = {**task_config, **env_args}
            env = WebArenaMetaEnv.from_dict(full_env_args)
            agent = TTLCumulativeAgent(**agent_args)

            # 2. Reset (runs episode 0)
            obs, info = env.reset()
            agent.reset()
            agent.update_from_env(observation=obs, reward=0, done=False, info=info)
            agent.trajectory.ep0_score = float(info.get('raw_info', {}).get('score', 0)) if 'raw_info' in info else 0.0

            ep0_score = info.get('raw_info', {}).get('score', 0) if 'raw_info' in info else 0
            if log:
                print(f"\n{'='*80}")
                print(f"EPISODE 0 | Score: {ep0_score} | Task: {task_config.get('task_name', '?')}")
                print(f"{'='*80}\n")

            # 3. Run full Meta Loop
            max_meta_steps = env.meta_cfg.get("max_episodes", 3)

            for step in range(max_meta_steps):
                messages = agent.chat_completions

                if log:
                    print(f"\n{'='*80}")
                    print(f"META AGENT RESPONSE  (episode {step} -> {step + 1})")
                    print(f"{'='*80}")

                response_text = chat_completion_with_retries(
                    model=meta_llm_cfg["model"],
                    messages=messages,
                    temperature=meta_llm_cfg.get("temperature", 0.7),
                    extra_body=meta_llm_cfg.get("extra_body", None)
                ).choices[0].message.content

                if log:
                    print(response_text)
                    print(f"{'='*80}\n")

                action = agent.update_from_model(response_text)

                next_obs, reward, done, info = env.step(action.action)
                agent.update_from_env(observation=next_obs, reward=reward, done=done, info=info)

                if log:
                    print(f"\n{'='*80}")
                    print(f"EPISODE {step + 1} | Score: {reward} | Task: {task_config.get('task_name', '?')}")
                    print(f"{'='*80}\n")

                if agent.trajectory.steps:
                    last_step = agent.trajectory.steps[-1]
                    last_step.reward = reward
                    last_step.info.update(info)

            if log:
                print(f"\nFinal Reward: {agent.trajectory.reward}")

            agent.trajectory.task = task_config
            agent.trajectory.full_history = agent.messages
            agent.trajectory.task_duration = time.time() - task_start
            return agent.trajectory

        except Exception as e:
            print(f"Task failed: {e}")
            traceback.print_exc()
            sentinel = Trajectory(task=task_config, steps=[], reward=0.0)
            sentinel.task_duration = time.time() - task_start
            sentinel.ep0_score = 0.0
            sentinel.error = str(e)
            return sentinel


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--env_args", type=str, required=True)
    parser.add_argument("--agent_args", type=str, required=True)
    parser.add_argument("--meta_llm_cfg", type=str, required=True)
    parser.add_argument("--api_cfg", type=str, required=True)
    parser.add_argument("--log", type=str, default="true")
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    trajectory = run_single_task(
        task_config=json.loads(args.task_config),
        env_args=json.loads(args.env_args),
        agent_args=json.loads(args.agent_args),
        meta_llm_cfg=json.loads(args.meta_llm_cfg),
        api_cfg=json.loads(args.api_cfg),
        log=args.log.lower() == "true",
        log_dir=args.log_dir,
    )

    # Write Trajectory as pickle to original stdout (saved before redirect)
    _PICKLE_OUT.write(pickle.dumps(trajectory))
    _PICKLE_OUT.flush()
    _PICKLE_OUT.close()
