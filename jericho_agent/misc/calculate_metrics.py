import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np

from jericho_agent.eval_utils import extract_score_trajectories, calculate_w_auc, get_max_score

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def process_configs(configs):
    for config in configs:
        label = config.get("label", "Unknown Label")
        paths = config.get("paths", [])
        
        print(f"\n" + "="*80)
        print(f"Results for: {label}")
        print(f"="*80)
        
        merged_data_map = {}
        
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARNING] File not found, skipping: {path}")
                continue
                
            try:
                results = load_pkl(path)
                data_map = extract_score_trajectories(results)
                
                for game, seeds_data in data_map.items():
                    if game not in merged_data_map:
                        merged_data_map[game] = {}
                        
                    for seed, runs in seeds_data.items():
                        if seed not in merged_data_map[game]:
                            merged_data_map[game][seed] = []
                        merged_data_map[game][seed].extend(runs)
                        
            except Exception as e:
                print(f"[ERROR] Failed to process {path}: {e}")

        # Now compute metrics per game
        if not merged_data_map:
            print("No valid data found for this configuration.")
            continue
            
        print(f"{'Game':<15} | {'Trajectories':<12} | {'W-AUC (Mean)':<15} | {'Avg Score (Mean)':<18} | {'Best Score':<15}")
        print("-" * 85)
        
        for game, seeds_data in merged_data_map.items():
            max_possible_score = get_max_score(game)
            
            all_runs = []
            for seed, runs in seeds_data.items():
                all_runs.extend(runs)
                
            num_trajectories = len(all_runs)
            
            if num_trajectories == 0:
                continue
                
            # Compute W-AUC for each trajectory and take the mean
            w_aucs = [calculate_w_auc(traj, max_possible_score) for traj in all_runs]
            mean_w_auc = np.mean(w_aucs)

            # Compute unweighted avg score for each trajectory and take the mean
            mean_avg_score = np.mean([np.mean(traj) for traj in all_runs])

            # Find the best overall score across all trajectories and all steps
            best_score = max(max(traj) for traj in all_runs) if all_runs else 0

            print(f"{game:<15} | {num_trajectories:<12} | {mean_w_auc:<15.4f} | {mean_avg_score:<18.4f} | {best_score:<15.1f}")

if __name__ == "__main__":
    # Example usage based on user request
    file_configs = [
        {
            "teacher": "GPT-5",
            "label": "Naive",
            "paths": [
                # ID (Keeping the main 10eps run, omitting the short 5eps one)
                "results/eval/eval_gpt-5_gemini-3-flash-preview_DEFAULT_20260218_125832.pkl",
                "results/eval/eval_gpt-5_gemini-3-flash-preview_DEFAULT_20260204_061506.pkl",
                # OOD
                "results/eval/eval_gpt-5_gemini-3-flash-preview_DEFAULT_20260222_235229.pkl"
            ]
        },
        {
            "teacher": "GPT-5",
            "label": "L-TTL", # Opt13
            "paths": [
                # ID (Keeping main run)
                "results/eval/eval_gpt-5_gemini-3-flash-preview_OPT_13_20260212_065945.pkl",
                "results/eval/eval_gpt-5_gemini-3-flash-preview_OPT_13_20260222_201851.pkl",
                # OOD
                "results/eval/eval_gpt-5_gemini-3-flash-preview_OPT_13_20260223_003607.pkl"
            ]
        }
    ]
    
    process_configs(file_configs)
