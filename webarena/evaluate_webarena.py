"""
WebArena Meta-Agent Evaluation Script.

Runs WebArena tasks with a meta-agent (teacher) that reflects on actor (student) trajectories
and provides updated guiding prompts across episodes.

Features:
- Meta-agent reflection loop via WebArenaMetaEnv
- Site-aware parallel scheduling (same-site tasks run serially)
- Comprehensive logging (incremental JSON, summary TXT, per-task logs)
- Supports memory, reference, and reflexion actor agents

Usage:
    # Single task, 2 episodes of meta-guidance
    python evaluate_webarena.py --meta_model openai/gpt-5 --actor_model google/gemini-3-flash-preview --tasks 0

    # Multiple tasks, parallel with site-aware scheduling
    python evaluate_webarena.py --meta_model openai/gpt-5 --actor_model google/gemini-3-flash-preview \\
        --tasks 0,1,2,25 --max_concurrent 4

    # All tasks, reflexion agent, 3 episodes per task
    python evaluate_webarena.py --meta_model openai/gpt-5 --actor_model google/gemini-3-flash-preview \\
        --all --agent_type reflexion --max_episodes 3
"""

import argparse
import datetime
import json
import os
import sys
import pickle
import time
import threading
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Ensure we can import from rllm and jericho_agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jericho_agent.utils import format_meta_trajectory


# ============================================================================
# Load WebArena Environment Variables (MUST be before rllm imports)
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

    # Global Patch: Force OpenRouter base URLs if OpenRouter key is detected
    # This ensures third-party WebArena evaluators don't crash hitting standard OpenAI.
    if 'OPENROUTER_API_KEY' in os.environ:
        base = "https://openrouter.ai/api/v1"
        os.environ["OPENAI_BASE_URL"] = base
        os.environ["OPENAI_API_BASE"] = base
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

# Load env vars at module level (required: autoeval/clients.py needs OPENAI_API_KEY at import time)
load_webarena_env()


from rllm.engine.webarena_mp_runner import WebArenaRunnerMP
from rllm.agents.ttl_cum_agent import TTLCumulativeAgent
from rllm.environments.webarena.meta_webarena_env import WebArenaMetaEnv
from rllm.environments.jericho.openai_helpers import init_global_client

from jericho_agent.prompts import DEFAULT_META_SYSTEM_PROMPT




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


DEFAULT_API_CONFIG = {
    "base_url": os.environ.get("OPENROUTER_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or "https://openrouter.ai/api/v1",
    "api_key": os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "EMPTY"),
}


# ============================================================================
# Task Discovery and Grouping (adapted from test_webarena_lite.py)
# ============================================================================

def get_available_tasks(config_dir=None, sites_filter=None):
    """Get list of available task IDs from webarena package or local config files."""
    if config_dir is None:
        try:
            import importlib.resources
            import webarena
            all_configs_str = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
            all_configs = json.loads(all_configs_str)

            if sites_filter:
                filtered = [c for c in all_configs
                           if 'task_id' in c and len(c.get('sites', [])) == 1 and c['sites'][0] == sites_filter]
                return sorted(set(c['task_id'] for c in filtered))
            else:
                return sorted(set(c.get('task_id') for c in all_configs if 'task_id' in c))
        except Exception as e:
            print(f"Error reading tasks from webarena package: {e}")
            return []
    else:
        config_path = Path(config_dir)
        if not config_path.exists():
            print(f"Error: Config directory {config_dir} not found")
            return []
        task_ids = []
        for json_file in config_path.glob("*.json"):
            try:
                task_id = int(json_file.stem)
                if sites_filter:
                    with open(json_file, 'r') as f:
                        conf = json.load(f)
                        sites = conf.get('sites', [])
                        if len(sites) == 1 and sites[0] == sites_filter:
                            task_ids.append(task_id)
                else:
                    task_ids.append(task_id)
            except (ValueError, json.JSONDecodeError):
                continue
        return sorted(task_ids)


def get_task_metadata(task_ids):
    """Get metadata for tasks including sites for grouping."""
    try:
        import importlib.resources
        import webarena
        all_configs_str = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
        all_configs = json.loads(all_configs_str)
        metadata = {}
        for conf in all_configs:
            tid = conf.get('task_id')
            if tid in task_ids:
                metadata[tid] = {
                    'intent_template': conf.get('intent_template', ''),
                    'sites': conf.get('sites', []),
                    'intent': conf.get('intent', '')
                }
        return metadata
    except Exception as e:
        print(f"Error reading task metadata: {e}")
        return {tid: {'intent_template': '', 'sites': [], 'intent': ''} for tid in task_ids}


def group_tasks_by_site(task_ids, metadata):
    """Group tasks by site and consecutive IDs for scheduling.
    
    Tasks within the same group must run serially.
    Tasks in different groups can run in parallel.
    """
    site_tasks = defaultdict(list)
    for tid in sorted(task_ids):
        sites = metadata.get(tid, {}).get('sites', [])
        site = sites[0] if sites else 'unknown'
        site_tasks[site].append(tid)

    groups = []
    for site in sorted(site_tasks.keys()):
        task_list = sorted(site_tasks[site])
        if not task_list:
            continue
        # Group consecutive task IDs
        current_group = [task_list[0]]
        for i in range(1, len(task_list)):
            if task_list[i] == task_list[i - 1] + 1:
                current_group.append(task_list[i])
            else:
                groups.append((site, current_group))
                current_group = [task_list[i]]
        if current_group:
            groups.append((site, current_group))

    return groups


# ============================================================================
# Logging (adapted from test_webarena_lite.py)
# ============================================================================

class EvalLogger:
    """Comprehensive logging for meta-agent evaluation runs."""
    
    def __init__(self, log_dir: Path, metadata: dict):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        # Initialize incremental results file
        self.results_file = log_dir / "results_incremental.json"
        self.summary_file = log_dir / "results_summary.txt"
        
        self.initial_data = {
            'metadata': metadata,
            'results': [],
            'statistics': {
                'completed_tasks': 0,
                'total_runs': 0,
                'total_success': 0,
                'total_failure': 0,
                'overall_accuracy': 0.0,
            }
        }
        with open(self.results_file, 'w') as f:
            json.dump(self.initial_data, f, indent=2)
        
        # Running counters
        self.total_runs = 0
        self.total_success = 0
        self.total_failure = 0
    
    def log_task_result(self, task_name: str, task_id: int, 
                        episodes: list, final_success: bool, 
                        best_score: float, duration: float):
        """Log result for a completed task (all episodes)."""
        with self.lock:
            self.total_runs += 1
            if final_success:
                self.total_success += 1
            else:
                self.total_failure += 1
            
            # Read, update, write
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            result_entry = {
                'task_id': task_id,
                'task_name': task_name,
                'success': final_success,
                'best_score': best_score,
                'num_episodes': len(episodes),
                'episodes': episodes,
                'duration': duration,
                'timestamp': datetime.datetime.now().isoformat()
            }
            data['results'].append(result_entry)
            
            # Update statistics
            data['statistics']['completed_tasks'] = len(data['results'])
            data['statistics']['total_runs'] = self.total_runs
            data['statistics']['total_success'] = self.total_success
            data['statistics']['total_failure'] = self.total_failure
            accuracy = 100.0 * self.total_success / self.total_runs if self.total_runs > 0 else 0.0
            data['statistics']['overall_accuracy'] = accuracy
            data['statistics']['last_update'] = datetime.datetime.now().isoformat()
            
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update summary text
            self._write_summary(data)
    
    def _write_summary(self, data):
        """Write human-readable summary file."""
        with open(self.summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WebArena Meta-Agent Evaluation - Real-time Results\n")
            f.write("=" * 80 + "\n\n")
            
            meta = data['metadata']
            f.write(f"Meta Model: {meta.get('meta_model', '?')}\n")
            f.write(f"Actor Model: {meta.get('actor_model', '?')}\n")
            f.write(f"Agent Type: {meta.get('agent_type', '?')}\n")
            f.write(f"Max Episodes: {meta.get('max_episodes', '?')}\n")
            f.write(f"Started: {meta.get('start_time', '?')}\n")
            
            stats = data['statistics']
            f.write(f"Last Update: {stats.get('last_update', '?')}\n\n")
            f.write(f"Progress: {stats['completed_tasks']} tasks completed\n")
            f.write(f"Success: {stats['total_success']}\n")
            f.write(f"Failure: {stats['total_failure']}\n")
            f.write(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Per-task results
            f.write("Results by Task:\n")
            f.write("-" * 80 + "\n")
            for r in sorted(data['results'], key=lambda x: x['task_id']):
                status_icon = "✓" if r['success'] else "✗"
                f.write(f"Task {r['task_id']}: {status_icon} (best_score={r['best_score']}, "
                        f"episodes={r['num_episodes']}, {r['duration']:.1f}s)\n")
                for ep in r.get('episodes', []):
                    ep_icon = "✓" if ep.get('success', False) else "✗"
                    f.write(f"  Episode {ep.get('episode', '?')}: {ep_icon} score={ep.get('score', 0)}\n")
    
    def write_final_results(self, results, total_duration):
        """Write final results.json with full summary."""
        final_file = self.log_dir / "results.json"
        
        # Read incremental data
        with open(self.results_file, 'r') as f:
            data = json.load(f)
        
        data['total_duration'] = total_duration
        data['statistics']['total_duration'] = total_duration
        
        with open(final_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Write final summary
        summary_file = self.log_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("WebArena Meta-Agent Evaluation Summary\n")
            f.write("=" * 80 + "\n\n")
            
            meta = data['metadata']
            f.write(f"Meta Model: {meta.get('meta_model', '?')}\n")
            f.write(f"Actor Model: {meta.get('actor_model', '?')}\n")
            f.write(f"Agent Type: {meta.get('agent_type', '?')}\n")
            f.write(f"Max Episodes: {meta.get('max_episodes', '?')}\n")
            f.write(f"Date: {datetime.datetime.now().isoformat()}\n\n")
            
            stats = data['statistics']
            f.write(f"Total tasks: {stats['completed_tasks']}\n")
            f.write(f"Successful: {stats['total_success']}\n")
            f.write(f"Failed: {stats['total_failure']}\n")
            f.write(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%\n")
            f.write(f"Total Duration: {total_duration:.1f}s\n\n")
            
            # Detailed per-task
            f.write("=" * 80 + "\n")
            f.write("Detailed Results:\n")
            f.write("-" * 80 + "\n")
            for r in sorted(data['results'], key=lambda x: x['task_id']):
                status_icon = "✓" if r['success'] else "✗"
                f.write(f"Task {r['task_id']:3d}: {status_icon} best_score={r['best_score']:4.0f} "
                        f"episodes={r['num_episodes']} ({r['duration']:.1f}s)\n")
        
        print(f"\nResults saved to:")
        print(f"  Final results (JSON): {final_file}")
        print(f"  Incremental results (JSON): {self.results_file}")
        print(f"  Real-time summary (TXT): {self.summary_file}")
        print(f"  Summary (TXT): {summary_file}")


# ============================================================================
# Score Extraction and Reporting
# ============================================================================

def extract_webarena_score_trajectories(results):
    """Returns dict[task_id][run_id] = [ep0_score, ep1_score, ...].

    ep0 comes from ep0_score attribute set by the runner after reset (preferred),
    or falls back to raw_info in the first step for old pkl files.
    ep1+ come from step.reward set by the runner after each env.step().
    """
    import numpy as np
    data = defaultdict(dict)
    for res in results:
        if not res:
            continue
        task_id = res.task.get('task_id', -1)
        run_id = res.task.get('run_id', 0)
        if hasattr(res, 'ep0_score'):
            ep0_score = float(res.ep0_score)
        elif res.steps:
            ep0_score = float(res.steps[0].info.get('raw_info', {}).get('score', 0))
        else:
            continue  # old pkl with no steps and no ep0_score — unrecoverable
        trajectory = [ep0_score] + [step.reward for step in res.steps]
        data[task_id][run_id] = trajectory
    return data


def _calculate_w_auc(scores, max_score=1):
    import numpy as np
    if not scores:
        return 0.0
    K = len(scores)
    weights = np.arange(1, K + 1)
    return float(np.sum(weights * np.array(scores)) / (np.sum(weights) * max_score))


def report_webarena_trajectories(results, meta_model="?", actor_model="?", max_episodes=None):
    """Print a hierarchical trajectory report with W-AUC, similar to jericho's report_hierarchical_text."""
    import numpy as np

    MAX_SCORE = 1
    data = extract_webarena_score_trajectories(results)

    print("\n" + "=" * 80)
    print(f"WebArena Meta-Agent Results | Meta: {meta_model} | Actor: {actor_model}")
    n_tasks = sum(len(runs) for runs in data.values())
    ep_label = f"Max Episodes: {max_episodes}" if max_episodes else ""
    print(f"Tasks: {n_tasks} | {ep_label} | Max Score: {MAX_SCORE}")
    print("=" * 80)
    print(f"{'Task ID':<10} {'Run':<6} {'Trajectory':<30} {'W-AUC':<10} {'Max Score':<10}")
    print("-" * 70)

    global_w_aucs = []
    global_maxs = []

    for task_id in sorted(data.keys()):
        runs = data[task_id]
        for run_id in sorted(runs.keys()):
            traj = runs[run_id]
            w_auc = _calculate_w_auc(traj, MAX_SCORE)
            run_max = max(traj) if traj else 0
            traj_str = str([int(s) for s in traj])
            if len(traj_str) > 30:
                traj_str = traj_str[:27] + "..."
            print(f"Task {task_id:<5} r{run_id:<4} {traj_str:<30} {w_auc:<10.4f} {run_max:<10.1f}")
            global_w_aucs.append(w_auc)
            global_maxs.append(run_max)

    print("-" * 70)
    if global_w_aucs:
        print(f"{'ALL':<10} {'GLOBAL':<6} {'==========':<30} {np.mean(global_w_aucs):<10.4f} {np.mean(global_maxs):<10.1f}")
    print("=" * 80 + "\n")


# ============================================================================
# Core Evaluation Logic
# ============================================================================

def run_evaluation_batch(
    tasks: List[Dict],
    meta_llm_config: Dict[str, Any],
    meta_system_prompt: str,
    api_config: Dict[str, str],
    meta_env_config: Dict[str, Any],
    actor_config: Dict[str, Any],
    max_concurrent: int = 4,
    task_log_dir: str = None,
):
    """Execute a batch of WebArena tasks via WebArenaRunnerMP."""
    IS_LOG = True  # task-specific output is captured in per-task log files

    runner = WebArenaRunnerMP(
        agent_class=TTLCumulativeAgent,
        env_class=WebArenaMetaEnv,
        agent_args={"system_prompt": meta_system_prompt},
        meta_llm_config=meta_llm_config,
        env_args={
            "meta_cfg": meta_env_config,
            "actor_cls": None,  # Unused - run.py handles agent creation
            "actor_args": actor_config,
            "log": IS_LOG,
        },
        log=IS_LOG,
    )

    print(f"Submitting {len(tasks)} tasks to WebArena MP Runner (Max Concurrent: {max_concurrent})...")
    results = runner.execute_tasks(tasks, max_concurrent=max_concurrent, api_cfg=api_config, log_dir=task_log_dir)
    
    # Sort by uid if available
    try:
        results.sort(key=lambda x: x.task['uid'])
    except Exception:
        pass
    
    return results


def evaluate(
    meta_llm_config: Dict[str, Any],
    system_prompt: str,
    api_config: Dict,
    task_ids: List[int],
    num_repeats: int = 1,
    meta_env_config: Dict = None,
    actor_config: Dict = None,
    max_concurrent: int = 4,
    log_dir: str = None,
):
    """Main evaluation entry point."""
    model_name_for_log = meta_llm_config.get("model", "unknown_model")
    actor_model = actor_config.get("llm_model", "unknown")
    
    print(f"\n🔥 STARTING WEBARENA META-AGENT EVALUATION")                                   
    print(f"   Meta LLM: {model_name_for_log}")                                              
    print(f"   Actor LLM: {actor_model}")                                                    
    print(f"   Agent Type: {actor_config.get('agent_type', 'memory')}")                      
    print(f"   Meta Config: {meta_llm_config}")                                              
    print(f"   Actor Config: {actor_config}")                                                
    print(f"   Env Config: {meta_env_config}")                                               
    print(f"   Task IDs: {task_ids} | Repeats: {num_repeats}") 
    
    print(f"Tasks: {len(task_ids)} | Max Episodes: {meta_env_config.get('max_episodes', 3)} | Repeats: {num_repeats}")

    start_time = time.perf_counter()

    # Set up logging
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_short = meta_llm_config.get("model", "unknown").split("/")[-1]
        actor_short = actor_model.split("/")[-1]
        agent_type = actor_config.get("agent_type", "memory")
        log_dir = f"logs/webarena_meta_{agent_type}/{timestamp}"
    
    log_path = Path(log_dir)
    print(f"Log directory: {log_path.resolve()}")

    task_log_path = log_path / "task_logs"
    task_log_path.mkdir(parents=True, exist_ok=True)

    logger = EvalLogger(log_path, metadata={
        'meta_model': model_name_for_log,
        'actor_model': actor_model,
        'agent_type': actor_config.get('agent_type', 'memory'),
        'max_episodes': meta_env_config.get('max_episodes'),
        'env_step_limit': meta_env_config.get('env_step_limit'),
        'start_time': datetime.datetime.now().isoformat(),
        'log_dir': str(log_path),
        'task_ids': task_ids,
        'num_repeats': num_repeats,
    })

    # Prepare tasks
    all_tasks = []
    for tid in task_ids:
        for run_i in range(num_repeats):
            task_name = f"webarena.{tid}"
            new_task = {
                'game_name': 'webarena',
                'uid': f"{task_name}_r{run_i}",
                'task_name': task_name,
                'task_id': tid,
                'run_id': run_i,
                'seed': run_i * 1000 + tid,
                'config_dir': actor_config.get('config_dir', 'config_files'),
            }
            all_tasks.append(new_task)

    # Run — wrapped in try-finally to guarantee summary output
    results = []
    try:
        results = run_evaluation_batch(
            tasks=all_tasks,
            meta_llm_config=meta_llm_config,
            meta_system_prompt=system_prompt,
            api_config=api_config,
            meta_env_config=meta_env_config,
            actor_config=actor_config,
            max_concurrent=max_concurrent,
            task_log_dir=str(task_log_path),
        )
    except Exception as e:
        print(f"\nERROR: run_evaluation_batch crashed: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to produce summary from any partial results...\n")

    elapsed_time = time.perf_counter() - start_time
    # Save raw results as pkl
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_short = meta_llm_config.get("model", "unknown").split("/")[-1]
        actor_short = actor_model.split("/")[-1]
        pkl_filename = f"eval_webarena_{meta_short}_{actor_short}_{timestamp}.pkl"

        pkl_dir = log_path / "raw"
        pkl_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = pkl_dir / pkl_filename

        print(f"\nSaving raw results to {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
    except Exception as e:
        print(f"Warning: Could not save pkl results: {e}")

    # Extract all trajectories post-hoc (single source of truth)
    traj_data = extract_webarena_score_trajectories(results)

    # Process results for logging — each result wrapped individually
    for res in results:
        try:
            if not hasattr(res, 'task') or not res.task:
                continue

            task_config = res.task
            task_name = task_config.get('task_name', '?')
            task_id = task_config.get('task_id', -1)
            run_id = task_config.get('run_id', 0)
            error_msg = getattr(res, 'error', None)

            traj = traj_data.get(task_id, {}).get(run_id, [])
            best_score = max(traj) if traj else 0
            final_success = best_score >= 1
            episodes_info = [
                {'episode': i, 'score': float(s), 'success': float(s) >= 1}
                for i, s in enumerate(traj)
            ]
            if error_msg and not episodes_info:
                episodes_info = [{'episode': 0, 'score': 0, 'success': False, 'error': error_msg}]

            actual_duration = getattr(res, 'task_duration', -1.0)
            logger.log_task_result(
                task_name=task_name,
                task_id=task_id,
                episodes=episodes_info,
                final_success=final_success,
                best_score=best_score,
                duration=actual_duration,
            )

            status = "FAIL" if error_msg else ("PASS" if final_success else "FAIL")
            w_auc = _calculate_w_auc(traj) if traj else 0.0
            dur_str = f"{actual_duration:.1f}s" if actual_duration >= 0 else "?s"
            error_note = f" ERROR: {error_msg[:80]}" if error_msg else ""
            print(f"  [{logger.total_runs}/{len(results)}] Task {task_id} (r{run_id}): "
                  f"{status} traj={[int(s) for s in traj]} | W-AUC: {w_auc:.3f} ({dur_str}){error_note}")
        except Exception as e:
            print(f"  Warning: Could not process result: {e}")

    # Write final results — always attempt
    logger.write_final_results(results, elapsed_time)


    # Print trajectory summary with W-AUC

    report_webarena_trajectories(
        results,
        meta_model=model_name_for_log,
        actor_model=actor_model,
        max_episodes=meta_env_config.get('max_episodes') if meta_env_config else None,
    )


    formatted_time = str(datetime.timedelta(seconds=int(elapsed_time)))
    print(f"Evaluation complete. Total time: {formatted_time}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Load WebArena environment variables (already loaded at module level,
    # but call again in case env vars were overridden)
    load_webarena_env()
    
    # Initialize global OpenAI client
    init_global_client(base_url=DEFAULT_API_CONFIG['base_url'], api_key=DEFAULT_API_CONFIG["api_key"])

    parser = argparse.ArgumentParser(description="Run WebArena Meta-Agent Evaluation")
    
    # Model arguments
    parser.add_argument("--meta_model", type=str, required=True, help="Meta Agent (Teacher) Model Name")
    parser.add_argument("--actor_model", type=str, required=True, help="Actor Agent (Student) Model Name")
    
    # Task selection
    parser.add_argument("--tasks", type=str, default="", help="Comma separated list of task IDs. Leave empty for all.")
    parser.add_argument("--start", type=int, default=None, help="Start task ID")
    parser.add_argument("--end", type=int, default=None, help="End task ID")
    parser.add_argument("--all", action='store_true', help="Run all available tasks")
    parser.add_argument("--sites", type=str, default=None,
                       help="Filter tasks by site (e.g., shopping_admin, map, reddit, gitlab)")
    
    # Agent & environment configuration
    parser.add_argument("--agent_type", type=str, default="memory",
                       choices=["memory", "reference", "reflexion"],
                       help="Actor agent implementation to use")
    parser.add_argument("--config_dir", type=str, default="config_files_lite",
                       choices=["config_files_lite", "config_files"],
                       help="Configuration directory for task configs")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per task")
    parser.add_argument("--env_step_limit", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--max_episodes", type=int, default=4, help="Max episodes of meta-agent guidance")
    parser.add_argument("--max_concurrent", type=int, default=4, help="Max parallel WebArena groups")
    
    # Output
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for logs (auto-generated if not specified)")
    parser.add_argument("--disable_memory", type=str2bool, default=True,
                       help="Disable intra-episode memory for the actor agent")
    
    args = parser.parse_args()

    # Determine which tasks to run
    if args.tasks:
        task_ids = [int(x.strip()) for x in args.tasks.split(",")]
    elif args.all:
        task_ids = get_available_tasks(sites_filter=args.sites)
    elif args.start is not None and args.end is not None:
        task_ids = list(range(args.start, args.end + 1))
    else:
        task_ids = get_available_tasks(args.config_dir, sites_filter=args.sites)

    if not task_ids:
        print("No valid tasks to run")
        sys.exit(1)

    # Print task grouping info
    metadata = get_task_metadata(task_ids)
    groups = group_tasks_by_site(task_ids, metadata)
    
    print(f"\n{'='*80}")
    print(f"WebArena Meta-Agent Evaluation")
    print(f"{'='*80}")
    print(f"Meta Model: {args.meta_model}")
    print(f"Actor Model: {args.actor_model}")
    print(f"Agent Type: {args.agent_type}")
    print(f"Tasks: {len(task_ids)} tasks")
    print(f"Max Episodes: {args.max_episodes}")
    print(f"Env Step Limit: {args.env_step_limit}")
    print(f"Config Dir: {args.config_dir}")
    if args.sites:
        print(f"Site Filter: {args.sites}")
    print(f"Repeats: {args.repeats}")
    print(f"Max Concurrent: {args.max_concurrent}")
    
    print(f"\nTask Grouping ({len(groups)} groups):")
    print(f"  - Tasks within same group run SERIALLY")
    print(f"  - Tasks from different groups run in PARALLEL")
    if len(groups) <= 30:
        for i, (site, group_tasks) in enumerate(groups):
            task_range = f"{group_tasks[0]}-{group_tasks[-1]}" if len(group_tasks) > 1 else str(group_tasks[0])
            print(f"  Group {i+1} ({site}, {len(group_tasks)} tasks): Tasks {task_range}")
    else:
        print(f"\n  Task Grouping: {len(groups)} groups (too many to display)")
        site_summary = defaultdict(list)
        for site, group in groups:
            site_summary[site].extend(group)
        print("\n  Task Summary by Site:")
        for site in sorted(site_summary.keys()):
            tasks = sorted(site_summary[site])
            print(f"    {site}: {len(tasks)} tasks - {tasks}")
    print(f"{'='*80}\n")

    # Build configs
    meta_llm_config = {
        "model": args.meta_model,
        "temperature": 0.7,
    }

    actor_config = {
        "llm_model": args.actor_model,
        "llm_temperature": 0.5,
        "max_memory": 30,
        "agent_type": args.agent_type,
        "disable_memory": args.disable_memory,
        "config_dir": args.config_dir,
    }

    meta_env_config = {
        "max_episodes": args.max_episodes,
        "env_step_limit": args.env_step_limit,
        "initial_prompt": "Explore systematically and examine objects to make progress."
    }

    # Run evaluation
    evaluate(
        meta_llm_config=meta_llm_config,
        system_prompt=DEFAULT_META_SYSTEM_PROMPT,
        api_config=DEFAULT_API_CONFIG,
        task_ids=task_ids,
        num_repeats=args.repeats,
        meta_env_config=meta_env_config,
        actor_config=actor_config,
        max_concurrent=args.max_concurrent,
        log_dir=args.log_dir,
    )
