import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import datetime
import pickle
import copy
import pandas as pd
import argparse
from typing import List, Dict, collections, Any, Optional

# --- Imports (reusing existing project modules) ---
from rllm.engine.mp_simple_runner import SimpleRunnerMP
from rllm.agents.ttl_cum_agent import TTLCumulativeAgent
from rllm.environments.jericho.meta_jericho_env import JerichoMetaEnv
from rllm.environments.jericho.actor_agent import MemoryAgent
from rllm.environments.jericho.openai_helpers import init_global_client

# Data preparation and analysis helpers.
from jericho_agent.prepare_jericho_data import prepare_jericho_data
from jericho_agent.eval_utils import extract_score_trajectories,report_hierarchical_text,plot_learning_curve_internal,str2bool

# Prompt variants.
from jericho_agent.prompts import DEFAULT_META_SYSTEM_PROMPT, OPTIMIZED_PROMPT2,OPTIMIZED_PROMPT13,OPTIMIZED_PROMPT10,OPTIMIZED_PROMPT25,OPTIMIZED_PROMPT11,OPTIMIZED_PROMPT9,OPTIMIZED_PROMPT8,OPTIMIZED_PROMPT14



DEFAULT_API_CONFIG = {
    "base_url": os.environ.get("OPENROUTER_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or "https://openrouter.ai/api/v1",
    "api_key": os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "EMPTY"),
}

# ==============================================================================
# 🚀 Core Evaluation Logic
# ==============================================================================

def run_evaluation_batch(
    tasks: List[Dict],
    meta_llm_config: Dict[str, Any], # Core change: pass the full config dict directly.
    meta_system_prompt: str,
    api_config: Dict[str, str],
    meta_env_config: Dict[str, Any],
    actor_config: Dict[str, Any],
    max_concurrent: int = 32,
    actor_cls = MemoryAgent,
):
    """
    Run a batch of tasks through the multiprocessing runner.
    """
    IS_LOG = False 

    # 2. Initialize the runner.
    # `meta_llm_config` is passed through directly instead of being rebuilt here.
    runner = SimpleRunnerMP(
        agent_class=TTLCumulativeAgent,
        env_class=JerichoMetaEnv,
        agent_args={"system_prompt": meta_system_prompt},
        meta_llm_config=meta_llm_config, # Pass through unchanged.
        env_args={
            "meta_cfg": meta_env_config,
            "actor_cls": actor_cls,
            "actor_args": actor_config,
            "log": IS_LOG,
        },
        log=IS_LOG,
    )

    # 3. Execute in parallel.
    print(f"🚀 Submitting {len(tasks)} tasks to MP Runner (Max Concurrent: {max_concurrent})...")
    results = runner.execute_tasks(tasks, max_concurrent=max_concurrent, api_cfg=api_config)
    results.sort(key=lambda x: x.task['uid'])
    return results

def evaluate(
    meta_llm_config: Dict[str, Any], # Core change: accept the full config dict.
    system_prompt: str,
    api_config: Dict,
    game_list: List[str] = ["detective"],
    num_seeds: int = 3,
    num_repeats: int = 5,
    meta_env_config: Optional[Dict] = None,   
    actor_config: Optional[Dict] = None,
    max_concurrent: int = 32,
    save_dir: str = "results/eval",
    run_name_suffix: str = "",
    prompt_tag: str = "DEFAULT",
    save_and_plot: bool = True,
):
    """
    Main evaluation entry point.
    """
    # 1. Handle default configuration logic.

    # Extract the model name for logging and filenames.
    model_name_for_log = meta_llm_config.get("model", "unknown_model")
    
    print(f"\n🔥 STARTING EVALUATION | Meta LLM: {model_name_for_log}")
    print(f"   Meta Config: {meta_llm_config}") 
    print(f"   Actor Config: {actor_config}")
    print(f"   Env Config: {meta_env_config}")
    print(f"   Games: {game_list} | Seeds: {num_seeds} | Repeats: {num_repeats}")
    
    start_time = time.perf_counter()
    
    # ------------------------------------------------------------------
    # 2. Task generation and expansion.
    # ------------------------------------------------------------------
    all_tasks = []
    
    for game in game_list:
        dataset = prepare_jericho_data(test_size=num_seeds, game_name=game) 
        raw_tasks = dataset.get_data()
        
        if not raw_tasks:
            print(f"⚠️ Warning: No data found for game {game}")
            continue

        seed_tasks = raw_tasks[:num_seeds]
        
        for seed_idx, task in enumerate(seed_tasks):
            for run_i in range(num_repeats):
                new_task = copy.deepcopy(task)
                new_task['game_name'] = game 
                new_task['uid'] = f"{game}_s{seed_idx}_r{run_i}"
                new_task['run_id'] = run_i
                new_task['seed_idx'] = seed_idx
                all_tasks.append(new_task)

    # ------------------------------------------------------------------
    # 3. Execute tasks.
    # ------------------------------------------------------------------
    results = run_evaluation_batch(
        tasks=all_tasks,
        meta_llm_config=meta_llm_config,
        meta_system_prompt=system_prompt,
        api_config=api_config,
        meta_env_config=meta_env_config,
        actor_config=actor_config,
        max_concurrent=max_concurrent
    )

    elapsed_time = time.perf_counter() - start_time

    # ------------------------------------------------------------------
    # 4. Save results.
    # ------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_full = meta_llm_config.get("model", "unknown")
    actor_full = actor_config.get("llm_model", "unknown")

    # Strip the provider prefix, e.g. "openai/gpt-4o" -> "gpt-4o".
    meta_short = meta_full.split("/")[-1]
    actor_short = actor_full.split("/")[-1]

    # Build the filename: eval_gpt-4o_gpt-4o-mini_...
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_short = meta_llm_config.get("model", "unknown").split("/")[-1]
    actor_short = actor_config.get("llm_model", "unknown").split("/")[-1]
    filename = f"eval_{meta_short}_{actor_short}_{prompt_tag}_{timestamp}.pkl"
    
    if save_and_plot:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{filename}"
        
        print(f"\n💾 Saving raw results to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\n💾 Raw pkl saved to {save_dir}/{filename}")

    # ------------------------------------------------------------------
    # 5. Analyze results and report.
    # ------------------------------------------------------------------
    data_map = extract_score_trajectories(results)
    metadata = {
        "meta_model": meta_short,
        "actor_model": actor_short,
        "steps": meta_env_config["env_step_limit"],
        "prompt_tag": prompt_tag,
        "max_episodes": meta_env_config["max_episodes"],
    }
    for game in data_map.keys():
        # Extract the data for one game.
        single_game_map = {game: data_map[game]}
        
        # 1. Print the hierarchical report.
        report_hierarchical_text(single_game_map, metadata)
        
        # 2. Plot and save directly.
        if save_and_plot:
            plot_learning_curve_internal(single_game_map, metadata, save_dir="results/plots1")

    formatted_time = str(datetime.timedelta(seconds=int(elapsed_time)))
    print("-" * 50)
    print(f"✅ Evaluation Complete. Total Time: {formatted_time}")
    print("-" * 50)


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    # 0. Global Init
    init_global_client(base_url=DEFAULT_API_CONFIG['base_url'], api_key=DEFAULT_API_CONFIG["api_key"])

    # 1. Define the argument parser.
    parser = argparse.ArgumentParser(description="Run Jericho Evaluation Experiments")
    
    # A. Model arguments.
    parser.add_argument("--meta_model", type=str, required=True, help="Meta Agent (Teacher) Model Name")
    parser.add_argument("--actor_model", type=str, required=True, help="Actor Agent (Student) Model Name")
    
    # B. Experiment scope.
    parser.add_argument("--games", nargs="+", default=["detective"], help="List of games to run")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds (Test size)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats per seed")
    
    # C. Environment and hyperparameters.
    parser.add_argument("--env_step_limit", type=int, default=30, help="Max steps per episode in Jericho env")
    parser.add_argument("--max_episodes", type=int, default=3, help="Max episodes (trials) of guidance for Meta Agent") 
    parser.add_argument("--disable_actor_thinking", type=str2bool, default="True", help="Whether to disable actor reasoning")
    parser.add_argument("--blocked_providers", nargs="+", default=[], help="List of OpenRouter providers to block (e.g., friendli)")


    parser.add_argument("--prompt_tag", type=str, default="DEFAULT", help="Key for PROMPT_VARIANTS")
    parser.add_argument("--save_and_plot", type=str2bool, default="True")
    
    args = parser.parse_args()

    # 2. Prepare the selected prompt.
    PROMPT_VARIANTS = {
        "DEFAULT": DEFAULT_META_SYSTEM_PROMPT,
        "OPT_V2": OPTIMIZED_PROMPT2,
        "OPT_13": OPTIMIZED_PROMPT13,
        "OPT_10": OPTIMIZED_PROMPT10,
        "OPT_25": OPTIMIZED_PROMPT25,
        "OPT_11": OPTIMIZED_PROMPT11,
        "OPT_9": OPTIMIZED_PROMPT9,
        "OPT_8": OPTIMIZED_PROMPT8,
        "OPT_14": OPTIMIZED_PROMPT14,
    }
    
    if args.prompt_tag not in PROMPT_VARIANTS:
        raise ValueError(f"Prompt tag '{args.prompt_tag}' not found in PROMPT_VARIANTS.")
    
    system_prompt_text = PROMPT_VARIANTS[args.prompt_tag]

    # 3. Build configs directly (pure construction, no copy/patch flow).
    
    # --- Meta LLM Config ---
    meta_llm_config = {
        "model": args.meta_model,
        "temperature": 0.7,
    }
    if "qwen" in args.meta_model.lower():
        meta_llm_config["max_tokens"] = 48000
    
    provider_config = {}
    if args.blocked_providers:
        provider_config = {"provider": {"ignore": args.blocked_providers}}
        meta_llm_config["extra_body"] = provider_config.copy()

    
    # --- Actor Config ---
    # Keep actor configuration explicit here for readability.
    # NO_THINK_PAYLOAD = {"chat_template_kwargs": {"enable_thinking": False}}
    # if "qwen" in args.actor_model.lower() or "glm" in args.actor_model.lower():
    actor_config = {
        "llm_model": args.actor_model,
        "llm_temperature": 0.5,
        "max_memory": 5,
        # "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        # "extra_body":{"reasoning": {
        #     "effort": "none"
        # }}
    }
    if args.disable_actor_thinking:
        print("Actor Thinking Disabled (injecting extra_body)")
        actor_config["extra_body"] = {
            "reasoning": {"effort": "none"}
        }
        if provider_config:
            actor_config["extra_body"].update(provider_config)
    else:
        print("Actor Thinking Enabled (Default Behavior)")
        if provider_config:
            actor_config["extra_body"] = provider_config.copy()


    # --- Meta Environment Config ---
    # Pass `env_step_limit` explicitly.
    meta_env_config = {
        "max_episodes": args.max_episodes,
        "env_step_limit": args.env_step_limit, # Provided from CLI arguments.
        "initial_prompt": "You are a helpful game player. Explore systematically."
    }

    print(f"\n🚀 Launching Experiment Group:")
    print(f"   Meta: {args.meta_model}")
    print(f"   Actor: {args.actor_model}")
    print(f"   Games: {args.games}")
    print(f"   Settings: Seeds={args.seeds}, Repeats={args.repeats}, Steps={args.env_step_limit}, Prompt={args.prompt_tag}, Save&Plot={args.save_and_plot}")

    # 4. Run evaluation.
    evaluate(
        meta_llm_config=meta_llm_config,
        system_prompt=system_prompt_text,
        api_config=DEFAULT_API_CONFIG,
        meta_env_config=meta_env_config,  
        actor_config=actor_config,       
        game_list=args.games,
        num_seeds=args.seeds,
        num_repeats=args.repeats,
        prompt_tag=args.prompt_tag,
        save_and_plot=args.save_and_plot
    )
