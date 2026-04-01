"""
Jericho Baseline Evaluation — Reflexion & Cross-Episode Memory agents.

Runs episodes directly on JerichoEnv (no meta-env, no meta-agent).
The agent itself handles cross-episode learning.

Produces results compatible with extract_score_trajectories / report_hierarchical_text
so that metrics (W-AUC, Best Score, learning curves) are directly comparable with
the meta-TTL evaluation in evaluate.py.

Usage:
    python jericho_agent/evaluate_baselines.py \
        --agent_type reflexion \
        --actor_model google/gemini-3-flash-preview \
        --games detective zork1 temple balances library zork3 \
        --seeds 3 --repeats 5 \
        --num_episodes 6 --env_step_limit 50
"""

import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

import os
import sys

# Add repo root to sys.path so `jericho_agent` and `rllm` are importable
# regardless of how the script is invoked (python path/to/script vs python -m)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import time
import datetime
import pickle
import argparse
import multiprocessing
import concurrent.futures
import traceback
from types import SimpleNamespace
from typing import List, Dict, Any

from rllm.agents.agent import Trajectory, Step
from rllm.environments.jericho.raw_env import JerichoEnv
from rllm.environments.jericho.meta_jericho_env import StateNode
from rllm.environments.jericho.openai_helpers import init_global_client

from jericho_agent.prepare_jericho_data import prepare_jericho_data
from jericho_agent.eval_utils import (
    extract_score_trajectories,
    report_hierarchical_text,
    plot_learning_curve_internal,
)


DEFAULT_API_CONFIG = {
    "base_url": os.environ.get("OPENROUTER_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or "https://openrouter.ai/api/v1",
    "api_key": os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "EMPTY"),
}


def _build_trajectory(task_config: Dict, scores: List[float]) -> Trajectory:
    """Build a Trajectory object from per-episode scores."""
    traj = Trajectory()
    traj.task = task_config
    traj.reward = float(scores[-1]) if scores else 0.0

    for i in range(1, len(scores)):
        step = Step()
        step.reward = float(scores[i])
        step.info = {}
        if i == 1:
            step.info["raw_info"] = {"score": float(scores[0])}
        else:
            step.info["raw_info"] = {"score": float(scores[i - 1])}
        traj.steps.append(step)

    return traj


def _worker_run_baseline(
    task_config: Dict,
    agent_cls_path: str,
    agent_config: Dict,
    num_episodes: int,
    env_step_limit: int,
    api_cfg: Dict,
):
    """Run all episodes for one task and return a Trajectory."""
    try:
        if api_cfg:
            init_global_client(
                base_url=api_cfg.get("base_url"),
                api_key=api_cfg.get("api_key"),
            )

        module_path, cls_name = agent_cls_path.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        agent_cls = getattr(mod, cls_name)

        args_obj = SimpleNamespace(**agent_config)
        agent = agent_cls(args=args_obj)

        scores = []
        uid = task_config.get("uid", "unknown")

        for ep_idx in range(num_episodes):
            agent.start_episode()
            print(f"[Task {uid}] Episode {ep_idx} | guiding_prompt: {agent.guiding_prompt[:200]}...")

            env = JerichoEnv(
                rom_path=task_config["rom_path"],
                seed=task_config["seed"],
                step_limit=env_step_limit + 1,
            )
            ob, info = env.reset()

            episode_trajectory = []

            for _step in range(env_step_limit):
                state_node = StateNode(state=ob)
                action_str, _ = agent.generate_action(state_node)
                ob, reward, done, info = env.step(action_str)
                episode_trajectory.append((ob, action_str, info["score"]))
                if done:
                    break

            final_score = info["score"]
            scores.append(final_score)
            env.close()

            agent.end_episode(
                state=ob,
                score=final_score,
                trajectory=episode_trajectory,
            )

        print(f"[Task {uid}] scores={scores}")
        return _build_trajectory(task_config, scores)

    except Exception as e:
        uid = task_config.get("uid", "unknown")
        print(f"[Task {uid}] FAILED: {e}")
        traceback.print_exc()
        return None


def evaluate_baselines(
    agent_cls_path: str,
    agent_config: Dict,
    api_config: Dict,
    game_list: List[str],
    num_seeds: int = 3,
    num_repeats: int = 5,
    num_episodes: int = 6,
    env_step_limit: int = 50,
    max_concurrent: int = 32,
    save_dir: str = "results/baselines",
    agent_tag: str = "reflexion",
    save_and_plot: bool = True,
):
    model_short = agent_config.get("llm_model", "unknown").split("/")[-1]
    print(f"\n{'='*80}")
    print(f"BASELINE EVALUATION — {agent_tag.upper()}")
    print(f"  Actor model : {agent_config.get('llm_model')}")
    print(f"  Games       : {game_list}")
    print(f"  Episodes    : {num_episodes} x {env_step_limit} steps")
    print(f"  Seeds       : {num_seeds}  Repeats: {num_repeats}")
    print(f"{'='*80}\n")

    start_time = time.perf_counter()

    all_tasks = []
    for game in game_list:
        dataset = prepare_jericho_data(test_size=num_seeds, game_name=game)
        raw_tasks = dataset.get_data()
        if not raw_tasks:
            print(f"Warning: no data for game {game}")
            continue
        seed_tasks = raw_tasks[:num_seeds]
        for seed_idx, task in enumerate(seed_tasks):
            for run_i in range(num_repeats):
                new_task = copy.deepcopy(task)
                new_task["game_name"] = game
                new_task["uid"] = f"{game}_s{seed_idx}_r{run_i}"
                new_task["run_id"] = run_i
                new_task["seed_idx"] = seed_idx
                all_tasks.append(new_task)

    print(f"Total tasks: {len(all_tasks)}")

    ctx = multiprocessing.get_context("spawn")
    results = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_concurrent, mp_context=ctx
    ) as executor:
        futures = [
            executor.submit(
                _worker_run_baseline,
                task,
                agent_cls_path,
                agent_config,
                num_episodes,
                env_step_limit,
                api_config,
            )
            for task in all_tasks
        ]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: x.task["uid"])
    elapsed = time.perf_counter() - start_time

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_{agent_tag}_{model_short}_{timestamp}.pkl"
    save_path = os.path.join(save_dir, filename)
    print(f"\nSaving results to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    data_map = extract_score_trajectories(results)
    metadata = {
        "meta_model": agent_tag,
        "actor_model": model_short,
        "steps": env_step_limit,
        "prompt_tag": agent_tag,
        "max_episodes": num_episodes - 1,
    }
    for game in data_map:
        single = {game: data_map[game]}
        report_hierarchical_text(single, metadata)
        if save_and_plot:
            try:
                plot_learning_curve_internal(
                    single, metadata, save_dir=os.path.join(save_dir, "plots")
                )
            except Exception as e:
                print(f"Warning: plotting failed ({e}). Install matplotlib to enable plots.")

    fmt_time = str(datetime.timedelta(seconds=int(elapsed)))
    print("-" * 50)
    print(f"Evaluation complete. Total time: {fmt_time}")
    print("-" * 50)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    init_global_client(
        base_url=DEFAULT_API_CONFIG["base_url"],
        api_key=DEFAULT_API_CONFIG["api_key"],
    )

    parser = argparse.ArgumentParser(description="Run Jericho Baseline Experiments")

    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        choices=["reflexion", "crossmem", "static"],
        help="Baseline agent type",
    )
    parser.add_argument("--actor_model", type=str, required=True)
    parser.add_argument(
        "--games",
        nargs="+",
        default=["detective"],
        help="List of games to evaluate",
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num_episodes", type=int, default=6)
    parser.add_argument("--env_step_limit", type=int, default=50)
    parser.add_argument("--max_concurrent", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="results/baselines")
    parser.add_argument(
        "--disable_actor_thinking",
        type=str2bool,
        default="True",
        help="Disable actor reasoning via extra_body",
    )
    parser.add_argument(
        "--save_and_plot",
        type=str2bool,
        default="False",
        help="Save results and generate learning curve plots",
    )

    args = parser.parse_args()

    AGENT_MAP = {
        "reflexion": "jericho_agent.agents.reflexion_agent.JerichoReflexionAgent",
        "crossmem": "jericho_agent.agents.crossmem_agent.JerichoCrossMemAgent",
        "static": "jericho_agent.agents.static_agent.JerichoStaticAgent",
    }
    agent_cls_path = AGENT_MAP[args.agent_type]

    agent_config = {
        "llm_model": args.actor_model,
        "llm_temperature": 0.5,
        "max_memory": 5,
    }
    if args.disable_actor_thinking:
        agent_config["extra_body"] = {"reasoning": {"effort": "none"}}

    evaluate_baselines(
        agent_cls_path=agent_cls_path,
        agent_config=agent_config,
        api_config=DEFAULT_API_CONFIG,
        game_list=args.games,
        num_seeds=args.seeds,
        num_repeats=args.repeats,
        num_episodes=args.num_episodes,
        env_step_limit=args.env_step_limit,
        max_concurrent=args.max_concurrent,
        save_dir=args.save_dir,
        agent_tag=args.agent_type,
        save_and_plot=args.save_and_plot,
    )
