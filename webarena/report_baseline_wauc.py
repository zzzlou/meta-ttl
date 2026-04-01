#!/usr/bin/env python
"""
Post-process WebArena baseline results.json files into W-AUC reports.
"""

import argparse
import json
import glob
import os
import re
from collections import defaultdict
import numpy as np

ID_DOMAINS = {
    'gitlab':   [43, 44, 49, 62, 81, 96, 99, 112, 122, 156],
    'map':      [1, 3, 7, 19, 20, 38, 52, 53, 69, 91],
    'shopping': [28, 29, 42, 46, 55, 78, 89, 103, 104, 118],
}
OOD_DOMAINS = {
    'reddit':         [5,14,97,98,131,132,133,134,135,136,137,138,139,140,141,151,152,153,154],
    'shopping_admin':  [0,2,8,13,17,21,26,27,30,32,33,40,47,48,50,51,59,70,80,88,101,106,107,108,110,113,114,115,116,125,126,145,149,150,160],
}
SPLIT_DOMAINS = {'id': ID_DOMAINS, 'ood': OOD_DOMAINS}

TASK_TO_DOMAIN = {}
for _d, _tids in {**ID_DOMAINS, **OOD_DOMAINS}.items():
    for _t in _tids:
        TASK_TO_DOMAIN[_t] = _d


def calculate_w_auc(scores, max_score=1):
    if not scores:
        return 0.0
    k = len(scores)
    weights = np.arange(1, k + 1)
    return float(np.sum(weights * np.array(scores)) / (np.sum(weights) * max_score))


def infer_label(path):
    dirname = os.path.basename(os.path.dirname(path))
    match = re.match(r"^(reflexion|crossmem|static)_(id|ood)_", dirname)
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"


def report(results_path):
    with open(results_path) as f:
        data = json.load(f)

    agent_type, split = infer_label(results_path)
    tasks = sorted(data["results"], key=lambda t: t["task_id"])
    num_episodes = data.get("repeats_per_task", "?")

    print("\n" + "=" * 80)
    print(f"WebArena Baseline Results | Agent: {agent_type} | Split: {split}")
    print(f"Tasks: {len(tasks)} | Episodes: {num_episodes} | Max Score: 1")
    print("=" * 80)
    print(f"{'Task ID':<10} {'Run':<6} {'Trajectory':<30} {'W-AUC':<10} {'Max Score':<10}")
    print("-" * 70)

    all_w_aucs = []
    all_maxs = []
    task_w_aucs = {}
    task_u_avgs = {}
    task_maxs = {}

    for task in tasks:
        traj = [int(r["success"] or 0) for r in task["repeats"]]
        w_auc = calculate_w_auc(traj)
        u_avg = sum(traj) / len(traj) if traj else 0.0
        run_max = max(traj) if traj else 0

        traj_str = str(traj)
        if len(traj_str) > 30:
            traj_str = traj_str[:27] + "..."

        print(f"Task {task['task_id']:<5} r0    {traj_str:<30} {w_auc:<10.4f} {float(run_max):<10.1f}")
        all_w_aucs.append(w_auc)
        all_maxs.append(run_max)
        task_w_aucs[task['task_id']] = w_auc
        task_u_avgs[task['task_id']] = u_avg
        task_maxs[task['task_id']] = float(run_max)

    domains = SPLIT_DOMAINS.get(split, {})
    domain_w = {}
    domain_u = {}
    domain_sr = {}
    if domains:
        print("-" * 70)
        domain_names = sorted(domains.keys())
        print(f"  {'Domain':<18} {'W-AUC':>8} {'AvgScore':>8} {'SR':>8}  (n/total)")
        for domain in domain_names:
            tids = domains[domain]
            w_scores = [task_w_aucs[t] for t in tids if t in task_w_aucs]
            u_scores = [task_u_avgs[t] for t in tids if t in task_u_avgs]
            sr_scores = [task_maxs[t] for t in tids if t in task_maxs]
            n = len(w_scores)
            if w_scores:
                w_avg = np.mean(w_scores)
                u_avg = np.mean(u_scores)
                sr_avg = np.mean(sr_scores)
                domain_w[domain] = w_avg
                domain_u[domain] = u_avg
                domain_sr[domain] = sr_avg
                print(f"  {domain:<18} {w_avg:>8.4f} {u_avg:>8.4f} {sr_avg:>8.4f}  ({n}/{len(tids)})")
            else:
                print(f"  {domain:<18} {'   -':>8} {'   -':>8} {'   -':>8}  (0/{len(tids)})")

    print("-" * 70)
    if domain_w:
        overall_w = np.mean(list(domain_w.values()))
        overall_u = np.mean(list(domain_u.values()))
        overall_sr = np.mean(list(domain_sr.values()))
        print(f"{'ALL':<10} {'GLOBAL':<6} {'==========':<18} W-AUC={overall_w:.4f}  AvgScore={overall_u:.4f}  SR={overall_sr:.4f}")
    elif all_w_aucs:
        print(f"{'ALL':<10} {'GLOBAL':<6} {'==========':<18} W-AUC={np.mean(all_w_aucs):.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Report W-AUC from baseline results.json")
    parser.add_argument("files", nargs="*", help="Path(s) to results.json")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover all results.json under logs/baselines_webarena/")
    args = parser.parse_args()

    paths = list(args.files)
    if args.auto:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "logs", "baselines_webarena")
        paths.extend(sorted(glob.glob(os.path.join(base, "*/results.json"))))

    if not paths:
        print("No results.json files found. Provide paths or use --auto.")
        return

    for p in paths:
        if not os.path.isfile(p):
            print(f"Skipping {p}: not found")
            continue
        report(p)


if __name__ == "__main__":
    main()
