import pickle
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local plotting utilities.
from jericho_agent.eval_utils import extract_score_trajectories, plot_multi_run_comparison

def load_pkl(path):
    print(f"[INFO] Loading: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    # ================= Configuration =================
    
    # Use `paths` (a list) so one label can merge multiple pickle files.
    
    file_configs = [
        {
            "label": "GLM-5 Default",
            "linestyle": "--",
            "paths": [
                "results/eval/eval_glm-5_gemini-3-flash-preview_DEFAULT_20260228_082716.pkl",
                "results/eval/eval_glm-5_gemini-3-flash-preview_DEFAULT_20260228_091113.pkl"
            ]
        },
        {
            "label": "GLM-5 Opt8 (Self-Reflect)",
            "linestyle": "-",
            "paths": [
                "results/eval/eval_glm-5_gemini-3-flash-preview_OPT_8_20260228_094956.pkl",
                "results/eval/eval_glm-5_gemini-3-flash-preview_OPT_8_20260228_103926.pkl"
            ]
        },
        {
            "label": "GLM-5 w/ GPT-5 Opt13 (Transfer)",
            "linestyle": "-.",
            "color": "red", # Highlight the transfer result.
            "paths": [
                "results/eval/eval_glm-5_gemini-3-flash-preview_OPT_13_20260228_123357.pkl", # ID
                "results/eval/eval_glm-5_gemini-3-flash-preview_OPT_13_20260228_130347.pkl"  # OOD
            ]
        }
    ]

    # ================= Execution =================
    
    experiment_data_list = []

    for config in file_configs:
        label = config["label"]
        paths = config.get("paths", [])
        
        # Backward compatibility: convert a single `path` entry into a list.
        if "path" in config and config["path"] not in paths:
            paths.append(config["path"])
            
        merged_data_map = {}
        
        print(f"\n[INFO] Gathering data for: {label}")
        
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARNING] File not found, skipping: {path}")
                continue
                
            try:
                # 1. Load data.
                results = load_pkl(path)
                
                # 2. Extract score trajectories with shape:
                #    dict[game] -> dict[seed] -> list_of_runs
                data_map = extract_score_trajectories(results)
                
                # 3. Merge data into `merged_data_map`.
                for game, seeds_data in data_map.items():
                    if game not in merged_data_map:
                        merged_data_map[game] = {}
                        
                    for seed, runs in seeds_data.items():
                        if seed not in merged_data_map[game]:
                            merged_data_map[game][seed] = []
                        # Core merge rule: append runs for matching seeds.
                        merged_data_map[game][seed].extend(runs)
                        
            except Exception as e:
                print(f"[ERROR] Failed to process {path}: {e}")

        # 4. Assemble the merged experiment entry.
        if merged_data_map:
            experiment_data_list.append({
                "data_map": merged_data_map,
                "label": label,
                "linestyle": config.get("linestyle", "-")
            })

    # 5. Plot the merged experiments.
    if experiment_data_list:
        print("\n[INFO] Generating comparison plots...")
        plot_multi_run_comparison(
            experiment_data_list, 
            save_dir="results/plots/Plot_Set_6",
            file_suffix="prompt_benchmark" 
        )
    else:
        print("[ERROR] No valid data loaded.")

if __name__ == "__main__":
    main()
