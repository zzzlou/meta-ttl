import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import collections
import argparse
# Simple score mapping. Extend this as needed for additional games.
JERICHO_MAX_SCORES = {
    "905": 1, "acorncourt": 30, "advent": 350, "adventureland": 100, "afflicted": 75,
    "anchor": 100, "awaken": 50, "balances": 51, "deephome": 300, "detective": 360,
    "dragon": 25, "enchanter": 400, "gold": 100, "inhumane": 90, "jewel": 90,
    "karn": 170, "library": 30, "ludicorp": 150, "moonlit": 1, "omniquest": 50,
    "pentari": 70, "reverb": 50, "snacktime": 50, "sorcerer": 400, "spellbrkr": 600,
    "spirit": 250, "temple": 35, "tryst205": 350, "yomomma": 35, "zenon": 20,
    "zork1": 350, "zork3": 7, "ztuu": 100
}

def get_max_score(game_name):
    return JERICHO_MAX_SCORES[game_name]

def calculate_w_auc(scores, max_score):
    """Compute weighted AUC or a similar aggregate metric."""
    if not scores: return 0.0
    # Normalize by the maximum possible score.
    K = len(scores)
    # Compute a simple weighted AUC-style score.
    weights = np.array([t for t in range(1, K + 1)])
    weighted_sum = np.sum(weights * np.array(scores))
    max_weighted_sum = np.sum(weights * max_score)
    w_auc = weighted_sum / max_weighted_sum if max_weighted_sum > 0 else 0.0
    return w_auc

def extract_score_trajectories(results):
    """
    Extract clean score trajectories from raw results.
    Returns: dict[game][seed] = list of trajectories (list of floats)
    """
    hierarchical_data = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for traj in results:
        if not traj.steps: continue
        
        game = traj.task.get('game_name', 'unknown')
        seed = traj.task.get('seed_idx', 0)
        
        # 1. Read the baseline score (Ep0).
    
        ep0_score = float(traj.steps[0].info['raw_info']['score'])
        
            
        trajectory = [ep0_score]
        
        # 2. Read the following TTL-round scores.
        for step in traj.steps:
            trajectory.append(step.reward) # Assume `step.reward` is the final score for that round.
            
        hierarchical_data[game][seed].append(trajectory)
        
    return hierarchical_data

def report_hierarchical_text(data_map, metadata):
    """
    Print a hierarchical text report.
    metadata: {meta_model, actor_model, steps, prompt_tag}
    """
    game = list(data_map.keys())[0] # Assume each call handles one game, or loop externally.
    seeds_data = data_map[game]
    sorted_seeds = sorted(seeds_data.keys())
    max_score = get_max_score(game)
    
    print("\n" + "="*100)
    header = f"📊 HIERARCHICAL REPORT | Game: {game.upper()} | Seeds: {len(sorted_seeds)} | Max: {max_score}"
    sub_header = f"META: {metadata['meta_model']} | ACTOR: {metadata['actor_model']} | STEP: {metadata['steps']} | PROMPT: {metadata['prompt_tag']}"
    print(header)
    print(sub_header)
    print("="*100)
    
    # Define the output columns.
    # Seed, Run, Trajectory, W-AUC, Avg Score, Max Score
    print(f"{'Seed':<8} {'Run ID':<8} {'Trajectory':<30} {'W-AUC':<10} {'Avg Score':<12} {'Max Score':<10}")
    print("-" * 100)

    global_w_aucs = []
    global_avgs = []
    global_maxs = []

    for seed in sorted_seeds:
        runs = seeds_data[seed]
        seed_w_aucs = []
        seed_avgs = []
        seed_maxs = []
        
        for run_idx, traj in enumerate(runs):
            w_auc = calculate_w_auc(traj, max_score)
            run_avg = float(np.mean(traj))
            run_max = max(traj)
            
            # Format the trajectory string, e.g. [10, 20, ...].
            traj_str = str([int(s) for s in traj])
            if len(traj_str) > 30: traj_str = traj_str[:27] + "..."
            
            print(f"Seed {seed:<3} Run {run_idx:<4} {traj_str:<30} {w_auc:<10.4f} {run_avg:<12.4f} {run_max:<10.1f}")

            seed_w_aucs.append(w_auc)
            seed_avgs.append(run_avg)
            seed_maxs.append(run_max)
            global_w_aucs.append(w_auc)
            global_avgs.append(run_avg)
            global_maxs.append(run_max)

        # Seed Average
        print(f"Seed {seed:<3} {'AVG':<8} {'..........':<30} {np.mean(seed_w_aucs):<10.4f} {np.mean(seed_avgs):<12.4f} {np.mean(seed_maxs):<10.1f}")
        print("-" * 100) # Separator between seeds

    # Global Average
    print(f"{'ALL':<8} {'GLOBAL':<8} {'==========':<30} {np.mean(global_w_aucs):<10.4f} {np.mean(global_avgs):<12.4f} {np.mean(global_maxs):<10.1f}")
    print("=" * 100 + "\n")

def plot_learning_curve_internal(data_map, metadata, save_dir="results/plots"):
    """
    Plot directly from in-memory data.
    """
    game = list(data_map.keys())[0]
    seeds_data = data_map[game]
    
    # 1. Aggregate all run data.
    all_runs = []
    for seed, runs in seeds_data.items():
        all_runs.extend(runs)
    
    if not all_runs: return

    # Convert to a NumPy array.
    # If runs have different lengths, truncate to the shortest run.
    try:
        data_np = np.array(all_runs)
    except ValueError:
        # Handle length mismatch by taking the minimum shared length.
        min_len = min(len(r) for r in all_runs)
        data_np = np.array([r[:min_len] for r in all_runs])

    means = np.mean(data_np, axis=0)
    stds = np.std(data_np, axis=0)
    sems = stds / np.sqrt(len(all_runs))
    x_axis = range(len(means)) # 0 is baseline, 1 is Ep1...

    # 2. Plot.
    plt.figure(figsize=(10, 6))
    
    # Build the plot label.
    label_str = f"{metadata['meta_model'].split('/')[-1]} (Prompt: {metadata['prompt_tag']})"
    
    plt.plot(x_axis, means, marker='o', linewidth=2, label=label_str)
    plt.fill_between(x_axis, means - sems, means + sems, alpha=0.2)
    
    # 3. Decorate the chart.
    max_ep = metadata.get('max_episodes', '?')
    title = f"{game.upper()} Learning Curve\nMeta: {metadata['meta_model']} | Actor: {metadata['actor_model']}"
    plt.title(title, fontsize=14)
    plt.xlabel("Meta Episodes (0=Baseline)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 4. Save the figure.
    os.makedirs(save_dir, exist_ok=True)
    
    # Sanitize filename components.
    m_name = metadata['meta_model'].split('/')[-1]
    a_name = metadata['actor_model'].split('/')[-1]
    fname = f"{game}_{m_name}_{a_name}_{metadata['prompt_tag']}_{metadata['steps']}steps_{max_ep}eps.png"
    save_path = os.path.join(save_dir, fname)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to avoid memory leaks.
    
    print(f"🖼️  Plot saved successfully: {save_path}")

def plot_multi_run_comparison(experiment_list, save_dir="results/plots_comparison", file_suffix="compare"):
    """
    Plot a comparison chart across multiple experiments.
    Logic: Intersection of games -> Truncate to shortest length -> Plot with SEM.
    """
    if not experiment_list:
        print("[ERROR] No experiments to plot.")
        return

    # 1. Intersect the available game sets.
    game_sets = [set(exp["data_map"].keys()) for exp in experiment_list]
    common_games = set.intersection(*game_sets)
    games = sorted(list(common_games))

    if not games:
        print("[WARNING] No common games found across all loaded files (Intersection is empty).")
        return
    
    print(f"Found {len(games)} common games: {games}")
    os.makedirs(save_dir, exist_ok=True)

    for game in games:
        print(f"\n[Processing Game]: {game}")
        
        # --- Pass 1: Scan for lengths & repeat counts ---
        plot_buffer = []  
        global_min_len = float('inf') 
        
        for exp in experiment_list:
            data_map = exp["data_map"]
            label = exp["label"]
            
            if game not in data_map: 
                print(f"  [WARNING] {label}: Missing game data.")
                continue

            seeds_data = data_map[game]
            all_runs = []
            for seed, runs in seeds_data.items():
                all_runs.extend(runs)
            
            if not all_runs: 
                print(f"  [WARNING] {label}: Empty data.")
                continue

            # Print N, the number of trajectories.
            num_repeats = len(all_runs)
            print(f"  - {label:<40} | N={num_repeats} trajectories")

            internal_min_len = min(len(r) for r in all_runs)
            
            if internal_min_len < global_min_len:
                global_min_len = internal_min_len
            
            plot_buffer.append({
                "label": label,
                "data": all_runs, # Keep raw data first, then truncate during plotting.
                "linestyle": exp.get("linestyle", "-"),
                "color": exp.get("color", None)
            })

        if not plot_buffer:
            print(f"  [SKIP] No valid runs found for game {game}.")
            continue

        print(f"  > Alignment: Truncating all experiments to length {global_min_len} steps.")

        # --- Pass 2: Plotting ---
        plt.figure(figsize=(8.5, 5.5))

        for item in plot_buffer:
            # Truncate data to the shared minimum length.
            final_runs = [r[:global_min_len] for r in item["data"]]
            data_np = np.array(final_runs)

            means = np.mean(data_np, axis=0)
            x_axis = range(len(means))

            # Plot the mean curve.
            plt.plot(
                x_axis, means, 
                linewidth=2.2,
                marker='o',
                markersize=5,
                label=item["label"], linestyle=item["linestyle"],
                color=item["color"]
            )

        title = game.replace("_", " ").upper()
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(list(range(global_min_len)), fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
        plt.legend(fontsize=10, loc='best', frameon=False)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"compare_{game}_{file_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Saved]: {save_path}")
        

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
