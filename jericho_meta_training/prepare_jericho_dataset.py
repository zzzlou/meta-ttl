from jericho_agent.prepare_jericho_data import prepare_jericho_data
import random
def create_jericho_datasets(n_train,n_val,train_repeat,val_repeat):
    """
    Reuse `prepare_jericho_data` directly,
    generate all tasks in one pass, then split them and inject `n_repeats`.
    """
    
    # 1. Define the desired dataset size.
    # Example: train may use 3 seeds and val may use 4 seeds.
    # Because the generator samples randomly, we create all tasks up front,
    # then take the first slice for train and the rest for validation.

    total_size = n_train + n_val
    
    print(f"🎲 Generating {total_size} tasks using prepare_jericho_data...")
    
    # 2. Call the existing dataset builder.
    # It returns a dataset object containing `total_size` tasks.
    full_dataset = prepare_jericho_data(test_size=total_size, game_name="detective")
    
    # Get the raw task list.
    # Assume the dataset exposes `get_data()`.
    all_tasks = full_dataset.get_data() 
    
    # 3. Split the list manually.
    raw_train = all_tasks[:n_train]
    raw_val = all_tasks[n_train:]
    
    # 4. Inject `n_repeats`.
    
    # --- Build the training set (favor speed). ---
    trainset = []
    for t in raw_train:
        # Copy defensively even if it may not be strictly necessary here.
        task = t.copy()
        task['n_repeats'] = train_repeat      # Inject the key repeat setting.
        task['split'] = 'train'
        trainset.append(task)
        
    # --- Build the validation set (favor stability). ---
    valset = []
    for t in raw_val:
        task = t.copy()
        task['n_repeats'] = val_repeat      # Inject the key repeat setting.
        task['split'] = 'val'
        valset.append(task)
        
    return trainset, valset

def create_jericho_datasets_multigame(
    game_configs: list[dict], 
    train_repeat: int = 1, 
    val_repeat: int = 3
):
    """
    Generate a mixed-game meta-training dataset.
    
    Args:
        game_configs: List of per-game configs in the following format:
            [
                {"name": "detective", "n_train": 15, "n_val": 2},
                {"name": "zork1", "n_train": 15, "n_val": 2},
                {"name": "temple", "n_train": 15, "n_val": 2},
            ]
        train_repeat: Number of repeats for each training task.
        val_repeat: Number of repeats for each validation task.
    """
    
    final_trainset = []
    final_valset = []
    
    print("🎲 Generating Multi-Game Datasets...")
    
    for config in game_configs:
        game_name = config["name"]
        n_train = config["n_train"]
        n_val = config["n_val"]
        total_needed = n_train + n_val
        
        print(f"  - Fetching {total_needed} tasks for game: {game_name.upper()}...")
        
        # 1. Use the original data generator to fetch tasks for this game.
        # If different samples are needed across runs, make sure randomness is
        # handled inside `prepare_jericho_data` or controlled here explicitly.
        dataset = prepare_jericho_data(test_size=total_needed, game_name=game_name)
        all_tasks = dataset.get_data()
        
        # Make sure we received enough tasks.
        if len(all_tasks) < total_needed:
            print(f"⚠️ Warning: Requested {total_needed} tasks for {game_name}, but only got {len(all_tasks)}. Using all available.")
            # If there are not enough tasks, use a simple rule:
            # prioritize validation, then put the remainder into training.
            if len(all_tasks) <= n_val:
                raw_val = all_tasks
                raw_train = []
            else:
                raw_val = all_tasks[:n_val]
                raw_train = all_tasks[n_val:]
        else:
            # 2. Split train and validation.
            # Use the first `n_train` for training and the next `n_val` for validation.
            raw_train = all_tasks[:n_train]
            raw_val = all_tasks[n_train : n_train + n_val]
            
        # 3. Inject config fields and append to the final lists.
        for t in raw_train:
            task = t.copy()
            task['n_repeats'] = train_repeat
            task['split'] = 'train'
            task['game'] = game_name # Store the game name for debugging.
            final_trainset.append(task)
            
        for t in raw_val:
            task = t.copy()
            task['n_repeats'] = val_repeat
            task['split'] = 'val'
            task['game'] = game_name
            final_valset.append(task)

    # 4. Shuffle the training set.
    # This matters for evolutionary search because it samples by batch.
    # Mixing games prevents long stretches of a single game and gives the
    # reflection model more varied failures across nearby iterations.
    random.shuffle(final_trainset)
    
    # Validation usually does not need shuffling; keeping order helps log inspection.
    
    print(f"✅ Multi-Game Dataset Created.")
    print(f"   Total Train: {len(final_trainset)} tasks (Mixed)")
    print(f"   Total Val:   {len(final_valset)} tasks (Mixed)")
    
    return final_trainset, final_valset
