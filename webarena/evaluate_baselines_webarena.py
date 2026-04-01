#!/usr/bin/env python
"""
WebArena Baseline Evaluation — Reflexion, Cross-Episode Memory, and Static agents.

Runs the evaluation task IDs (30 ID + 54 OOD) through
test_webarena_lite.py with --no_early_stop so all episodes execute.
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

ID_EVAL_TASKS = "1,3,7,19,20,28,29,38,42,43,44,46,49,52,53,55,62,69,78,81,89,91,96,99,103,104,112,118,122,156"
OOD_EVAL_TASKS = "0,2,5,8,13,14,17,21,26,27,30,32,33,40,47,48,50,51,59,70,80,88,97,98,101,106,107,108,110,113,114,115,116,125,126,131,132,133,134,135,136,137,138,139,140,141,145,149,150,151,152,153,154,160"

SPLITS = {
    "id": ("ID Eval (30 tasks)", ID_EVAL_TASKS),
    "ood": ("OOD Eval (54 tasks)", OOD_EVAL_TASKS),
}


def count_tasks(tasks_str):
    return len([t for t in tasks_str.split(",") if t.strip()])


def run_eval(agent_type, split_name, tasks_str, model, repeat, workers, max_steps, log_base):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_base, f"{agent_type}_{split_name}_{timestamp}")

    print("\n" + "=" * 70)
    print(f"  Agent:  {agent_type}")
    print(f"  Split:  {split_name}")
    print(f"  Model:  {model}")
    print(f"  Repeat: {repeat} episodes")
    print(f"  Tasks:  {count_tasks(tasks_str)} task(s)")
    print(f"  IDs:    {tasks_str[:80]}{'...' if len(tasks_str) > 80 else ''}")
    print(f"  Log:    {log_dir}")
    print(f"  Start:  {datetime.now().isoformat()}")
    print("=" * 70)

    cmd = [
        sys.executable, "test_webarena_lite.py",
        "--tasks", tasks_str,
        "--agent_type", agent_type,
        "--repeat", str(repeat),
        "--no_early_stop",
        "--model", model,
        "--max_steps", str(max_steps),
        "--workers", str(workers),
        "--log_dir", log_dir,
    ]

    print(f"CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"WARNING: test_webarena_lite.py exited with code {result.returncode}")
    else:
        print(f"  Finished: {datetime.now().isoformat()}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run WebArena baseline evaluations")
    parser.add_argument(
        "--agent_types",
        nargs="+",
        default=["reflexion", "crossmem"],
        choices=["reflexion", "crossmem", "static"],
        help="Baseline agent types to evaluate",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["id", "ood"],
        choices=["id", "ood"],
        help="Evaluation splits to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-3-flash-preview",
        help="Actor model",
    )
    parser.add_argument("--repeat", type=int, default=5, help="Episodes per task")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--max_steps", type=int, default=10, help="Steps per episode (default 10)")
    parser.add_argument(
        "--log_base",
        type=str,
        default="logs/baselines_webarena",
        help="Base log directory",
    )
    parser.add_argument(
        "--id_tasks",
        type=str,
        default=ID_EVAL_TASKS,
        help="Comma-separated task IDs for the id split",
    )
    parser.add_argument(
        "--ood_tasks",
        type=str,
        default=OOD_EVAL_TASKS,
        help="Comma-separated task IDs for the ood split",
    )

    args = parser.parse_args()

    split_map = {
        "id": ("ID Eval", args.id_tasks),
        "ood": ("OOD Eval", args.ood_tasks),
    }

    print(f"\nWebArena Baseline Evaluation")
    print(f"  Agent types: {args.agent_types}")
    print(f"  Splits:      {args.splits}")
    print(f"  Model:       {args.model}")
    print(f"  Episodes:    {args.repeat}")
    print(f"  Workers:     {args.workers}")
    print(f"  ID tasks:    {count_tasks(args.id_tasks)}")
    print(f"  OOD tasks:   {count_tasks(args.ood_tasks)}")

    for agent_type in args.agent_types:
        for split_key in args.splits:
            split_desc, tasks_str = split_map[split_key]
            run_eval(
                agent_type=agent_type,
                split_name=split_key,
                tasks_str=tasks_str,
                model=args.model,
                repeat=args.repeat,
                workers=args.workers,
                max_steps=args.max_steps,
                log_base=args.log_base,
            )

    print(f"\nAll evaluations complete at {datetime.now().isoformat()}")
    print(f"Results in: {args.log_base}/")


if __name__ == "__main__":
    main()
