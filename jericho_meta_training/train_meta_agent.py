import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import litellm
import gepa
from jericho_meta_training.jericho_adapter import JerichoMetaTrainingAdapter
from jericho_meta_training.prepare_jericho_dataset import create_jericho_datasets,create_jericho_datasets_multigame
from jericho_agent.prompts import FIXED_HEADER,FIXED_FOOTER,INITIAL_STRATEGY


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

# API config dictionary passed into the adapter.
API_CONFIG = {
    "base_url": os.environ["OPENAI_API_BASE"],
    "api_key": os.environ["OPENAI_API_KEY"]
}

# --- Reflection LM ---
# Since the environment variables are already set, the optimizer could use a plain model string.
# We still keep a wrapper so the model name is explicit and easy to adjust.
# That is safer for OpenRouter-style model aliases.
def reflection_lm_func(prompt):
    try:
        response = litellm.completion(
            model="openai/gpt-5.2", # OpenRouter-mapped model identifier.
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            drop_params=True,
            # litellm reads the API key and base URL from the environment automatically.
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Reflection LM Error: {e}")
        return "" # Return an empty string to avoid crashing the run.

if __name__ == "__main__":
    GAME_DATA_CONFIGS = [
        {"name": "detective", "n_train": 1, "n_val": 1},
        {"name": "zork1",     "n_train": 1, "n_val": 1},
        {"name": "temple",    "n_train": 1, "n_val": 1},
    ]
    trainset, valset = create_jericho_datasets_multigame(
        game_configs=GAME_DATA_CONFIGS,
        train_repeat=1, 
        val_repeat=5,
    )
    # breakpoint()

    # 2. Initial prompt used as the seed candidate.
    # This prompt becomes the adapter's `system_prompt`.
    initial_strategy = INITIAL_STRATEGY

    # 3. Initialize the adapter.
    # Pass the API config through explicitly.
    adapter = JerichoMetaTrainingAdapter(
        meta_model_name="z-ai/glm-5", # Base model for the meta agent.
        api_config=API_CONFIG,           # Forward API credentials.
        max_concurrent=32                # Concurrency level.
    )

    print(f"🚀 Starting Evolutionary Meta-Training (Multi-Game Mode)...")
    print(f"   Trainset size: {len(trainset)}")
    print(f"   Valset size:   {len(valset)}")

    # 4. Launch evolutionary meta-training optimization.
    result = gepa.optimize(
        seed_candidate={"strategy_section": initial_strategy},
        
        trainset=trainset,
        valset=valset, 
        
        adapter=adapter,
        
        # Reflection model.
        reflection_lm=reflection_lm_func,
        
        # Enable system-aware merge
        # use_merge=True,
        # max_merge_invocations=5, # default is 5
        
        max_metric_calls=100, 
        
        # Reflect over all three games in each minibatch.
        reflection_minibatch_size=1,
        
        # Key fix: `task_lm` must be `None` when an adapter is provided.
        task_lm=None, 
        
        # Logging.
        use_wandb=True,
        run_dir="train_results/meta_training_Feb25_100",
        # run_dir=None,
        
        # Improve fault tolerance so transient network issues do not kill the whole run.
        raise_on_exception=False,
        display_progress_bar=True,
    )

    # 5. Print the result summary.
    print("\n" + "="*40)
    print("🏆 Optimization Complete!")
    try:
        best_index = result.best_idx
        best_score = result.val_aggregate_scores[best_index]
        print(f"Best Score (on Valset): {best_score}")
    except (IndexError, ValueError):
        print("Best Score: N/A (List empty or error)")

    print("Best System Prompt:")
    print("-" * 20)
    
    # --- Fix 2: read the prompt content from `best_candidate`. ---
    # `result.best_candidate` already resolves `candidates[best_idx]` internally.
    # The key must stay `"strategy_section"` to match the seed candidate passed to `optimize`.
    if result.best_candidate:
        print(result.best_candidate.get("strategy_section", "Key 'strategy_section' not found!"))
    else:
        print("No candidates found.")
    
    print("-" * 20)
