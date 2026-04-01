"""
WebArena-specific MP Runner with site-aware group scheduling.

Key difference from SimpleRunnerMP:
- SimpleRunnerMP submits 1 process per TASK (fine for Jericho since games are independent)
- WebArenaRunnerMP submits 1 process per SITE GROUP (tasks on the same site run serially
  to avoid shared-state interference like shopping carts, browser sessions, etc.)

This mirrors the scheduling approach from test_webarena_lite.py's group_tasks_by_intent_template().
"""

import contextlib
import json
import multiprocessing
import concurrent.futures
import os
import pickle
import subprocess
import sys
import time
import traceback
from typing import Dict, Any, List
from collections import defaultdict


from rllm.agents.agent import Trajectory
from rllm.engine.simple_runner import SimpleRunner


@contextlib.contextmanager
def _task_log_redirect(log_path):
    """Redirect stdout/stderr to a task-specific log file within a worker process."""
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


def _get_task_site(task_config: Dict) -> str:
    """Extract site from task config for grouping.
    
    WebArena tasks have a 'task_name' like 'webarena.0'. We look up the site
    from the webarena package config.
    """
    task_name = task_config.get("task_name", "")
    
    # Try to extract task ID and look up site from config
    try:
        import re
        match = re.search(r'\.(\d+)$', task_name)
        if match:
            task_id = int(match.group(1))
            
            # Try reading from webarena package
            try:
                import json
                import importlib.resources
                import webarena
                all_configs_str = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
                all_configs = json.loads(all_configs_str)
                
                for conf in all_configs:
                    if conf.get('task_id') == task_id:
                        sites = conf.get('sites', [])
                        return sites[0] if sites else 'unknown'
            except Exception:
                pass
            
            # Fallback: try reading from local config_files
            config_dir = task_config.get("config_dir", "config_files")
            config_path = f"{config_dir}/{task_id}.json"
            try:
                import json, os
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        conf = json.load(f)
                        sites = conf.get('sites', [])
                        return sites[0] if sites else 'unknown'
            except Exception:
                pass
    except Exception:
        pass
    
    return 'unknown'


def _group_tasks_by_site(tasks: List[Dict]) -> List[List[Dict]]:
    """Group tasks by site and consecutive IDs.
    
    Tasks on the same site with consecutive IDs likely share dependencies or intent,
    so they must run serially. However, tasks on the same site with non-consecutive
    IDs can safely run in parallel across different processes.
    
    Returns list of groups, where each group is a list of tasks that must run serially.
    """
    site_tasks = defaultdict(list)
    for task in tasks:
        site = _get_task_site(task)
        site_tasks[site].append(task)
    
    groups = []
    import re
    
    def get_task_num(task_dict):
        # Extract integer N from 'webarena.N_rM' or 'webarena.N'
        name = task_dict.get('task_name', '')
        match = re.search(r'\.(\d+)', name)
        return int(match.group(1)) if match else -1

    for site in sorted(site_tasks.keys()):
        # Sort tasks by their numeric ID to group consecutive ones
        task_list = sorted(site_tasks[site], key=get_task_num)
        
        if not task_list:
            continue
            
        current_group = [task_list[0]]
        for i in range(1, len(task_list)):
            prev_num = get_task_num(task_list[i-1])
            curr_num = get_task_num(task_list[i])
            
            # If IDs are consecutive, or if they are the exact same ID (e.g. run_id 0 and 1)
            # we keep them in the same serial group to prevent interference.
            if curr_num == prev_num + 1 or curr_num == prev_num:
                current_group.append(task_list[i])
            else:
                groups.append(current_group)
                current_group = [task_list[i]]
                
        if current_group:
            groups.append(current_group)
            
    return groups


def _worker_run_group(
    group_id: int,
    group_tasks: List[Dict],
    env_cls: Any,
    agent_cls: Any,
    env_args: Dict,
    agent_args: Dict,
    meta_llm_cfg: Dict,
    api_cfg: Dict,
    log: bool,
    log_dir: str = None,
):
    """
    Worker process: runs a GROUP of tasks serially via subprocess isolation.

    Each task spawns a separate subprocess (meta_run_webarena_task.py) that runs
    the full meta-loop. The subprocess exits after each task, ensuring full
    browser/Playwright resource cleanup between tasks.

    env_cls and agent_cls are accepted for interface compatibility but unused —
    the subprocess imports the required classes directly.
    """
    group_results = []

    worker_script = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'webarena', 'meta_run_webarena_task.py'
    ))

    for task_idx, task_config in enumerate(group_tasks):
        task_id_str = f"G{group_id}T{task_idx}"
        task_start = time.time()

        cmd = [
            sys.executable, worker_script,
            "--task_config", json.dumps(task_config),
            "--env_args", json.dumps(env_args),
            "--agent_args", json.dumps(agent_args),
            "--meta_llm_cfg", json.dumps(meta_llm_cfg),
            "--api_cfg", json.dumps(api_cfg if api_cfg else {}),
            "--log", str(log).lower(),
        ]
        if log_dir:
            cmd.extend(["--log_dir", log_dir])

        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=1800)
            if proc.returncode == 0 and proc.stdout:
                trajectory = pickle.loads(proc.stdout)
                group_results.append(trajectory)
            else:
                stderr_tail = proc.stderr.decode(errors='replace')[-500:] if proc.stderr else ""
                sentinel = Trajectory(task=task_config, steps=[], reward=0.0)
                sentinel.ep0_score = 0.0
                sentinel.task_duration = time.time() - task_start
                sentinel.error = f"Subprocess failed (rc={proc.returncode}): {stderr_tail}"
                group_results.append(sentinel)
                print(f"Task {task_id_str} subprocess failed (rc={proc.returncode}): {stderr_tail}")
        except subprocess.TimeoutExpired:
            sentinel = Trajectory(task=task_config, steps=[], reward=0.0)
            sentinel.ep0_score = 0.0
            sentinel.task_duration = time.time() - task_start
            sentinel.error = "Subprocess timed out (1800s)"
            group_results.append(sentinel)
            print(f"Task {task_id_str} subprocess timed out (1800s)")
        except Exception as e:
            sentinel = Trajectory(task=task_config, steps=[], reward=0.0)
            sentinel.ep0_score = 0.0
            sentinel.task_duration = time.time() - task_start
            sentinel.error = str(e)
            group_results.append(sentinel)
            print(f"Task {task_id_str} failed: {e}")
            traceback.print_exc()

    return group_results


class WebArenaRunnerMP(SimpleRunner):
    """
    WebArena-specific MP Runner with site-aware scheduling.
    
    Key behavior:
    - Tasks are grouped by site (shopping_admin, reddit, gitlab, etc.)
    - Each group is submitted as a single process to the pool
    - Tasks within a group run serially (avoid shared-state interference)
    - Groups run in parallel
    
    This mirrors test_webarena_lite.py's approach but works within the BaseEnv/BaseAgent framework.
    """
    
    def execute_tasks(self, tasks, api_cfg, max_concurrent=4, log_dir=None):                                       
        """Execute tasks with site-aware group scheduling."""    

        # 1. Group tasks by site
        groups = _group_tasks_by_site(tasks)


        # Log grouping info
        print(f"Starting smart parallel execution: {len(groups)} groups, {len(tasks)} tasks (max {max_concurrent} concurrent)")
        print(f"  - Tasks within each group run serially")
        print(f"  - Groups run in parallel\n")

        # 2. Use Spawn mode for safety
        ctx = multiprocessing.get_context("spawn")

        

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(max_concurrent, len(groups)),
            mp_context=ctx
        ) as executor:
            future_to_group = {}
            for i, group in enumerate(groups):
                f = executor.submit(
                    _worker_run_group,
                    i, group,
                    self.env_class, self.agent_class,
                    self.env_args, self.agent_args,
                    self.meta_llm_config,
                    api_cfg,
                    self.log,
                    log_dir,
                )
                future_to_group[f] = (i, group)

            # Collect results with timeout
            all_results = []
            
            for f in concurrent.futures.as_completed(future_to_group):
                group_idx, group_tasks = future_to_group[f]
                try:
                    group_results = f.result()
                    if group_results:
                        all_results.extend(group_results)
                except Exception as e:
                    print(f"Warning: Worker group {group_idx} failed: {e}")
                    traceback.print_exc()

        return all_results
