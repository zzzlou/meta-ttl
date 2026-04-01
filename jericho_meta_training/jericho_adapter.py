import asyncio
import numpy as np
from typing import List, Dict, Any, Sequence
import os
import pickle
import datetime
from gepa.core.adapter import GEPAAdapter, EvaluationBatch
# Import project-specific evaluation logic.
# Assume `run_optimization_session` returns `List[Dict]`, where each dict is a trajectory.
# Any concurrency should already be handled internally; this adapter calls it synchronously.
from jericho_agent.run_jericho_api import run_optimization_session 
from jericho_agent.evaluate import run_evaluation_batch
from jericho_agent.utils import format_meta_trajectory, calculate_auc_float,calculate_time_weighted_auc
from collections import defaultdict
from jericho_agent.prompts import FIXED_HEADER,FIXED_FOOTER,INITIAL_STRATEGY

DETECTIVE_MAX_SCORE = 360
ZORK1_MAX_SCORE = 350
TEMPLE_MAX_SCORE = 35
# DEFAULT_API_CONFIG = {
#     "base_url": "https://openrouter.ai/api/v1",
#     "api_key": "<set via environment>"
# }

# Fixed environment configuration for the meta loop.
META_ENV_CONFIG = {
    "max_episodes": 5,
    "env_step_limit": 50,
    "initial_prompt": "You are a helpful game player. Explore systematically." #initial prompt for actor agent, before test time learning
}

ACTOR_CONFIG = {
    "llm_model": "google/gemini-3-flash-preview",
    "llm_temperature": 0.5,
    "max_memory": 5
}
class JerichoMetaTrainingAdapter(GEPAAdapter):
    def __init__(
        self, 
        meta_model_name: str,
        api_config: Dict[str, str],
        max_concurrent: int = 32,
        save_base_dir: str = "results"
    ):
        """
        Args:
            meta_model_name: Model used by the meta agent (e.g. "gpt-4o").
            api_config: API credentials and endpoint settings.
            default_repeats: Default number of repeats.
            max_concurrent: Concurrency level for the multiprocessing runner.
        """
        self.meta_model_name = meta_model_name
        self.api_config = api_config
        self.max_concurrent = max_concurrent
        tz_cn = datetime.timezone(datetime.timedelta(hours=8))
        timestamp = datetime.datetime.now(tz_cn).strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_base_dir, f"MetaTrain_Run_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📂 All trajectories will be saved to: {self.save_dir}")

    def evaluate(
        self,
        batch: List[Dict], # The batch looks like [Task_Seed0, Task_Seed1, ...].
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        
        strategy_part = candidate.get("strategy_section", "")
        full_meta_prompt = f"{FIXED_HEADER}\n\n{strategy_part}\n\n{FIXED_FOOTER}"

        # --- Change 1: expand tasks dynamically. ---
        expanded_tasks = []
        
        for i, task in enumerate(batch):
            # Read the repeat requirement from each task dynamically.
            repeats = task.get('n_repeats', 1)
            
            for r in range(repeats):
                t_copy = task.copy()
                # Tag the original task ID so repeated runs can be regrouped later.
                t_copy['_original_index'] = i 
                # Add a run ID for debugging.
                t_copy['run_id'] = r
                expanded_tasks.append(t_copy)

        meta_llm_config = {
            "model": self.meta_model_name,
            "temperature": 0.7, # Evolutionary search usually benefits from some temperature.
        }
        # --- Change 2: run the expanded tasks in batch. ---
        try:
            raw_results = run_evaluation_batch(
                tasks=expanded_tasks,
                meta_llm_config=meta_llm_config,
                meta_system_prompt=full_meta_prompt,
                api_config=self.api_config,
                meta_env_config=META_ENV_CONFIG,
                actor_config=ACTOR_CONFIG,
                max_concurrent=self.max_concurrent
            )
        except Exception as e:
            print(f"Exec Error: {e}")
            fallback_scores = [0.0] * len(batch)
            return EvaluationBatch(outputs=fallback_scores, scores=fallback_scores, trajectories=None)
        # 3. Aggregate results.
        grouped_results = defaultdict(list)
        for traj in raw_results:
            orig_idx = traj.task.get('_original_index')
            if orig_idx is not None:
                grouped_results[orig_idx].append(traj)

        aggregated_scores = []
        aggregated_outputs = []
        # `trajectories` now stores a `List[traj]` per task instead of a single trajectory.
        all_trajectories_payload = [] 
        batch_ts = datetime.datetime.now().strftime("%H%M%S%f")
        for i in range(len(batch)):
            trajs = grouped_results[i]
            
            if not trajs:
                # Fallback when a task produces no result.
                aggregated_scores.append(0.0)
                aggregated_outputs.append(0.0)
                if capture_traces:
                    all_trajectories_payload.append([])
                continue

            # A. Compute the score for each individual run.
            single_run_scores = [calculate_time_weighted_auc(traj) for traj in trajs]
            
            # --- New feature: save pickles. ---
            for traj, score in zip(trajs, single_run_scores):
                try:
                    # Use `run_id` to avoid collisions across repeated runs of the same task.
                    run_id = traj.task.get('run_id', 0)
                    
                    # Filename format: score_taskIdx_runIdx.pkl
                    # Example: score_0.8521_task_0_run_1.pkl
                    filename = f"score_{score:.4f}_task_{i}_run_{run_id}_{batch_ts}.pkl"
                    save_path = os.path.join(self.save_dir, filename)
                    
                    # with open(save_path, "wb") as f:
                    #     pickle.dump(traj, f) # This would store the raw Trajectory object.
                except Exception as e:
                    print(f"⚠️ Failed to save pickle: {e}")
            # B. Report the average score back to the optimizer for this task.
            avg_score = float(np.mean(single_run_scores))
            aggregated_scores.append(avg_score)
            aggregated_outputs.append(avg_score) # Outputs are used for logging only.

            # C. Key change: return all run data as-is.
            
            if capture_traces:
                # Package every run and score for this task so reflection can choose freely.
                task_payload = {
                    "all_runs": trajs,                # All trajectory objects for this task.
                    "all_scores": single_run_scores,  # Per-run scores, using weighted AUC.
                    "avg_score": avg_score,
                    "n_repeats": len(trajs)
                }
                all_trajectories_payload.append(task_payload)

        return EvaluationBatch(
            outputs=aggregated_outputs, 
            scores=aggregated_scores, 
            # Use `None` when traces are not needed to save memory.
            trajectories=all_trajectories_payload if capture_traces else None
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str],
    ) -> Dict[str, Sequence[Dict[str, Any]]]:
        
        if not eval_batch.trajectories:
            return {}

        # `scores` contains the per-task average score.
        task_avg_scores = eval_batch.scores
        # `trajectories` contains the payloads assembled above.
        task_payloads = eval_batch.trajectories
        reflection_records = []

        # --- First pass: find the worst task at the task level. ---
        # This focuses reflection on the hardest task.
        worst_task_idx = np.argmin(task_avg_scores)
        worst_payload = task_payloads[worst_task_idx]
        
        # --- Second pass: find the worst run within that task. ---
        # This isolates the concrete failure case.
        all_run_scores = worst_payload["all_scores"]
        all_runs = worst_payload["all_runs"]
        
        worst_run_idx = np.argmin(all_run_scores)
        worst_traj = all_runs[worst_run_idx]
        worst_score = all_run_scores[worst_run_idx] # This is a single-run score, not the task average.

        # --- Format the log. ---
        # `format_meta_trajectory` must handle a single trajectory object.
        log_str = format_meta_trajectory(worst_traj)
        
        # breakpoint()
        task_context = (
            "The Meta-Agent's job is to read game logs from an Actor Agent and provide Test-time guidance to help the Actor Agent get a higher score in the next episode. The Actor agent interacts with the game for multiple episodes per task. This is a multi-game training setup where the specific game varies across different Tasks."
        )
        record = {
                "Task Context": task_context,
                "Message History": log_str,
            }
        print(record)
        reflection_records.append(record)

        return {"strategy_section": reflection_records}
