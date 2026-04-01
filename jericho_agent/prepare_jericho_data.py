import os
import numpy as np


class SimpleDataset:
    def __init__(self, data):
        self._data = list(data)

    def get_data(self):
        return list(self._data)


ROMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roms")

GAME_ROMS_MAPPINGS = {
    "detective": os.path.join(ROMS_DIR, "detective.z5"),
    "temple": os.path.join(ROMS_DIR, "temple.z5"),
    "zork1": os.path.join(ROMS_DIR, "zork1.z5"),
    "balances": os.path.join(ROMS_DIR, "balances.z5"),
    "library": os.path.join(ROMS_DIR, "library.z5"),
    "zork3": os.path.join(ROMS_DIR, "zork3.z5"),
}


def prepare_jericho_data(test_size=5,game_name="detective"):
    """
    Only define which game instances should be played.
    This does not specify the actor model or the number of meta-agent rounds.
    """
    np.random.seed(42)
    test_seeds = np.random.randint(0, 100000, size=test_size)
    test_game_rom = GAME_ROMS_MAPPINGS[game_name]
    if not os.path.exists(test_game_rom):
        raise FileNotFoundError(
            f"ROM not found for game '{game_name}': {test_game_rom}"
        )
    tasks = []
    for seed in test_seeds:
        task_info = {
            "rom_path": test_game_rom,
            "seed": int(seed),
            "game_name": game_name,
            "uid": f"{os.path.basename(test_game_rom)}_{seed}"
        }
        tasks.append(task_info)

    print(f"Registered {len(tasks)} Jericho game {game_name} tasks.")
    return SimpleDataset(tasks)

if __name__ == "__main__":
    test_dataset = prepare_jericho_data()
    tasks = test_dataset.get_data()
    tasks = tasks[:1]
    print(tasks)
