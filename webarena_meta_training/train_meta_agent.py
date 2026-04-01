"""
WebArena Evolutionary Meta-Training Entry Point.

Mirrors jericho_meta_training/train_meta_agent.py for WebArena tasks.
Optimizes a meta-agent's strategy prompt via evolutionary search over a task distribution.
"""

import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import litellm
import gepa

from webarena_meta_training.webarena_adapter import (
    WebArenaMetaTrainingAdapter,
    WEBARENA_META_SYSTEM_PROMPT_HEADER,
    WEBARENA_META_SYSTEM_PROMPT_FOOTER,
)
from webarena_meta_training.prepare_webarena_dataset import (
    create_webarena_datasets,
)

# API configuration (OpenRouter)
DEFAULT_BASE_URL = (
    os.environ.get("OPENROUTER_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or "https://openrouter.ai/api/v1"
)
DEFAULT_API_KEY = (
    os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or "EMPTY"
)

os.environ.setdefault("OPENAI_API_BASE", DEFAULT_BASE_URL)
os.environ.setdefault("OPENAI_API_KEY", DEFAULT_API_KEY)

API_CONFIG = {
    "base_url": os.environ["OPENAI_API_BASE"],
    "api_key": os.environ["OPENAI_API_KEY"],
}

# Initial strategy prompt (seed candidate for meta-training)
INITIAL_STRATEGY = """
"""


def reflection_lm_func(prompt):
    """Wrapper around litellm for the reflection model."""
    try:
        response = litellm.completion(
            model="openai/gpt-5.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            drop_params=True,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Reflection LM Error: {e}")
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebArena Evolutionary Meta-Training")
    parser.add_argument("--meta_model", type=str, default="google/gemini-3-flash-preview",
                        help="Model for the meta-agent")
    parser.add_argument("--n_val_per_domain", type=int, default=2,
                        help="Validation tasks per domain")
    parser.add_argument("--n_id_eval_per_domain", type=int, default=10,
                        help="ID eval tasks per domain (held out from training)")
    parser.add_argument("--max_concurrent", type=int, default=4,
                        help="Max parallel WebArena groups")
    parser.add_argument("--max_metric_calls", type=int, default=200,
                        help="Total optimization budget")
    parser.add_argument("--reflection_minibatch_size", type=int, default=1,
                        help="Tasks per reflection iteration")
    parser.add_argument("--max_episodes", type=int, default=2,
                        help="Max meta-agent episodes per task")
    parser.add_argument("--env_step_limit", type=int, default=10,
                        help="Max steps per episode")
    args = parser.parse_args()

    # Create datasets
    trainset, valset, id_evalset = create_webarena_datasets(
        n_val_per_domain=args.n_val_per_domain,
        n_id_eval_per_domain=args.n_id_eval_per_domain,
    )

    # Configure meta-env
    meta_env_config = {
        "max_episodes": args.max_episodes,
        "env_step_limit": args.env_step_limit,
        "initial_prompt": "You are a helpful web agent. Complete the task carefully.",
    }

    # Initialize adapter
    adapter = WebArenaMetaTrainingAdapter(
        meta_model_name=args.meta_model,
        api_config=API_CONFIG,
        max_concurrent=args.max_concurrent,
        meta_env_config=meta_env_config,
    )
    RUN_DIR = "train_results_webarena/meta_training_Mar25"
    print("=" * 60)
    print("WebArena Meta-Training Configuration")
    print("=" * 60)
    print(f"  Meta model:          {args.meta_model}")
    print(f"  Reflection LM:       openai/gpt-5.2")
    print(f"  Actor LM:            {adapter.actor_config.get('llm_model', 'N/A')}")
    print(f"  Actor temperature:   {adapter.actor_config.get('llm_temperature', 'N/A')}")
    print(f"  Actor agent_type:    {adapter.actor_config.get('agent_type', 'N/A')}")
    print(f"  Actor max_memory:    {adapter.actor_config.get('max_memory', 'N/A')}")
    print(f"  Max episodes:        {args.max_episodes}")
    print(f"  Env step limit:      {args.env_step_limit}")
    print(f"  Max concurrent:      {args.max_concurrent}")
    print(f"  Max metric calls:    {args.max_metric_calls}")
    print(f"  Reflection minibatch:{args.reflection_minibatch_size}")
    print(f"  Max reflect tasks:   {adapter.max_reflect_tasks}")
    print(f"  N val per domain:    {args.n_val_per_domain}")
    print(f"  N ID eval per domain:{args.n_id_eval_per_domain}")
    print(f"  API base URL:        {API_CONFIG['base_url']}")
    print(f"  Run dir:             {RUN_DIR}")
    print("-" * 60)
    print(f"  Trainset size:       {len(trainset)}")
    print(f"  Valset size:         {len(valset)}")
    print(f"  ID Eval size:        {len(id_evalset)}")
    print("=" * 60)

    # Run evolutionary meta-training
    result = gepa.optimize(
        seed_candidate={"strategy_section": INITIAL_STRATEGY},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm_func,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        task_lm=None,
        use_wandb=True,
        run_dir=RUN_DIR,
        raise_on_exception=False,
        display_progress_bar=True,
    )

    # Output results
    print("\n" + "=" * 40)
    print("Optimization Complete!")
    try:
        best_index = result.best_idx
        best_score = result.val_aggregate_scores[best_index]
        print(f"Best Score (on Valset): {best_score}")
    except (IndexError, ValueError):
        print("Best Score: N/A (List empty or error)")

    print("Best System Prompt:")
    print("-" * 20)
    if result.best_candidate:
        print(result.best_candidate.get("strategy_section", "Key 'strategy_section' not found!"))
    else:
        print("No candidates found.")
    print("-" * 20)
