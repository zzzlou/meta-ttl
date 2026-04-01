"""
WebArenaMetaTrainingAdapter — mirrors JerichoMetaTrainingAdapter from jericho_meta_training/jericho_adapter.py.

Plugs WebArenaRunnerMP + WebArenaMetaEnv into the evolutionary meta-training loop.
"""

import os
import datetime
import pickle
import numpy as np
from typing import List, Dict, Any, Sequence
from collections import defaultdict

from gepa.core.adapter import GEPAAdapter, EvaluationBatch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'webarena'))
from evaluate_webarena import run_evaluation_batch

from webarena_agent.utils import (
    calculate_binary_auc,
    format_meta_trajectory_webarena,
)

# ---------------------------------------------------------------------------
# Default configs (override via constructor args or task-level fields)
# ---------------------------------------------------------------------------

WEBARENA_META_ENV_CONFIG = {
    "max_episodes": 4,
    "env_step_limit": 15,
    "initial_prompt": "You are a helpful web agent. Complete the task carefully.",
}

WEBARENA_ACTOR_CONFIG = {
    "llm_model": "google/gemini-3-flash-preview",
    "llm_temperature": 0.5,
    "max_memory": 0,
    "agent_type": "memory",
}

# System-prompt scaffolding (mirrors FIXED_HEADER / FIXED_FOOTER from Jericho).
# The optimizer mutates the "strategy_section" component between runs.
WEBARENA_META_SYSTEM_PROMPT_HEADER = """You are an AI Optimization Assistant.
## Input Format
The input is a chronological log of a web-agent interacting with a browser to complete a task. Each step in the trajectory follows this structure:
- [OBS]: The current webpage state (accessibility tree or screenshot description).
- [ACTION]: The browser action executed by the agent (e.g., click, type, select_option, send_msg_to_user).

The trajectory ends with a [FINAL_OBS] and a final score (0 = fail, 1 = success).
"""

WEBARENA_META_SYSTEM_PROMPT_FOOTER = """
## Output Format:
   <think>reasoning...</think>
   <learn>Your derived useful feedback. This will be used as the new system prompt for the actor agent</learn>

If no useful feedback can be derived, output an empty <learn></learn>.
"""


class WebArenaMetaTrainingAdapter(GEPAAdapter):
    """Adapter for WebArena evolutionary meta-training.

    The optimizer calls:
      1. evaluate(batch, candidate) -> EvaluationBatch
      2. make_reflective_dataset(candidate, eval_batch, components) -> records

    Args:
        meta_model_name: Model used as the meta-agent (e.g. 'gpt-4o').
        api_config: {'base_url': ..., 'api_key': ...}
        max_concurrent: Max parallel site-group workers in WebArenaRunnerMP.
        save_base_dir: Directory for saving trajectory pickles.
        meta_env_config: Override for WEBARENA_META_ENV_CONFIG.
        actor_config: Override for WEBARENA_ACTOR_CONFIG.
    """

    def __init__(
        self,
        meta_model_name: str,
        api_config: Dict[str, str],
        max_concurrent: int = 4,
        save_base_dir: str = "results",
        meta_env_config: Dict = None,
        actor_config: Dict = None,
        max_reflect_tasks: int = 1,
    ):
        self.meta_model_name = meta_model_name
        self.api_config = api_config
        self.max_concurrent = max_concurrent
        self.meta_env_config = meta_env_config or WEBARENA_META_ENV_CONFIG.copy()
        self.actor_config = actor_config or WEBARENA_ACTOR_CONFIG.copy()
        self.max_reflect_tasks = max_reflect_tasks

        tz_cn = datetime.timezone(datetime.timedelta(hours=8))
        timestamp = datetime.datetime.now(tz_cn).strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_base_dir, f"WebArena_MetaTrain_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Trajectories will be saved to: {self.save_dir}")

    # ------------------------------------------------------------------
    # Optimizer interface — evaluate()
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: List[Dict],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run all tasks and return scores.

        batch: list of task dicts, each with 'task_name' (e.g. 'webarena.42').
               Optionally include 'n_repeats' to run a task multiple times.
        candidate: dict with key 'strategy_section' — the optimisable prompt part.
        """
        strategy_part = candidate.get("strategy_section", "")
        full_meta_prompt = (
            f"{WEBARENA_META_SYSTEM_PROMPT_HEADER}\n\n"
            f"{strategy_part}\n\n"
            f"{WEBARENA_META_SYSTEM_PROMPT_FOOTER}"
        )

        # Expand tasks by n_repeats
        expanded_tasks = []
        for i, task in enumerate(batch):
            repeats = task.get('n_repeats', 1)
            for r in range(repeats):
                t_copy = task.copy()
                t_copy['_original_index'] = i
                t_copy['run_id'] = r
                t_copy['uid'] = f"{task.get('task_name', i)}_r{r}"
                expanded_tasks.append(t_copy)

        meta_llm_config = {
            "model": self.meta_model_name,
            "temperature": 0.7,
        }

        try:
            raw_results = run_evaluation_batch(
                tasks=expanded_tasks,
                meta_llm_config=meta_llm_config,
                meta_system_prompt=full_meta_prompt,
                api_config=self.api_config,
                meta_env_config=self.meta_env_config,
                actor_config=self.actor_config,
                max_concurrent=self.max_concurrent,
            )
        except Exception as e:
            print(f"Evaluation error: {e}")
            fallback = [0.0] * len(batch)
            return EvaluationBatch(outputs=fallback, scores=fallback, trajectories=None)

        # Group results by original task index
        grouped = defaultdict(list)
        for traj in raw_results:
            orig_idx = traj.task.get('_original_index')
            if orig_idx is not None:
                grouped[orig_idx].append(traj)

        aggregated_scores = []
        aggregated_outputs = []
        all_trajectories_payload = []
        batch_ts = datetime.datetime.now().strftime("%H%M%S%f")

        for i in range(len(batch)):
            trajs = grouped[i]
            if not trajs:
                aggregated_scores.append(0.0)
                aggregated_outputs.append(0.0)
                if capture_traces:
                    all_trajectories_payload.append([])
                continue

            single_run_scores = [calculate_binary_auc(t) for t in trajs]

            # Optionally save pickles
            for traj, score in zip(trajs, single_run_scores):
                try:
                    run_id = traj.task.get('run_id')
                    filename = f"score_{score:.4f}_task_{i}_run_{run_id}_{batch_ts}.pkl"
                    save_path = os.path.join(self.save_dir, filename)
                    with open(save_path, "wb") as f:
                        pickle.dump(traj, f)
                except Exception as e:
                    print(f"Failed to save pickle: {e}")

            avg_score = float(np.mean(single_run_scores))
            aggregated_scores.append(avg_score)
            aggregated_outputs.append(avg_score)

            if capture_traces:
                all_trajectories_payload.append({
                    "all_runs": trajs,
                    "all_scores": single_run_scores,
                    "avg_score": avg_score,
                    "n_repeats": len(trajs),
                })

        return EvaluationBatch(
            outputs=aggregated_outputs,
            scores=aggregated_scores,
            trajectories=all_trajectories_payload if capture_traces else None,
        )

    # ------------------------------------------------------------------
    # Optimizer interface — make_reflective_dataset()
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str],
    ) -> Dict[str, Sequence[Dict[str, Any]]]:
        """Build reflection records from the worst-performing run per task."""

        if not eval_batch.trajectories:
            return {}

        task_payloads = eval_batch.trajectories

        # Step 1: Find worst run per task
        worst_per_task = []
        for task_idx, payload in enumerate(task_payloads):
            if not payload:
                continue

            all_run_scores = payload["all_scores"]
            all_runs = payload["all_runs"]

            sorted_run_idxs = np.argsort(all_run_scores)  # ascending: worst first
            worst_traj = None
            worst_score = float('inf')
            for idx in sorted_run_idxs:
                cand = all_runs[idx]
                if hasattr(cand, 'full_history') and cand.full_history:
                    worst_traj = cand
                    worst_score = all_run_scores[idx]
                    break

            if worst_traj is not None:
                worst_per_task.append((worst_score, task_idx, worst_traj))

        # Step 2: Sort by worst run score (ascending) and take N worst tasks
        worst_per_task.sort(key=lambda x: x[0])
        selected = worst_per_task[:self.max_reflect_tasks]

        all_logs_str = ""
        for _, task_idx, worst_traj in selected:
            task_name = worst_traj.task.get('task_name', f"Task_{task_idx}")
            log_str = format_meta_trajectory_webarena(worst_traj)
            all_logs_str += f"=== Logs for Task: {task_name} ===\n"
            all_logs_str += log_str + "\n\n"

        if not all_logs_str.strip():
            return {}  # No usable traces — signal to proposer to skip this iteration

        task_context = (
            "The Meta-Agent's job is to read web-agent trajectories and provide "
            "test-time guidance to help the agent succeed in the next episode. "
            "The agent interacts with a web browser to complete tasks across "
            "multiple episodes per task."
        )
        record = {
            "Task Context": task_context,
            "Message History": all_logs_str.strip(),
        }
        print(record)

        return {"strategy_section": [record]}
