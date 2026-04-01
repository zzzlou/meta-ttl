"""
WebArena meta-training dataset preparation.

Dynamically loads task IDs from the webarena package and splits them
into meta-train / meta-test domains for evolutionary meta-training.
"""

import json
import random
import importlib.resources
from typing import Dict, List, Tuple, Optional

import webarena

# Domain splits
META_TRAIN_DOMAINS = ["shopping", "gitlab", "map"]
META_TEST_DOMAINS = ["reddit", "shopping_admin"]


def _load_task_ids_by_domain() -> Dict[str, List[int]]:
    """Load task IDs grouped by domain from webarena's test.raw.json.

    Only includes single-site tasks (len(sites)==1).
    Returns: {"shopping": [4, 10, ...], "shopping_admin": [0, 2, ...], ...}
    """
    raw_text = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
    all_configs = json.loads(raw_text)

    domain_tasks: Dict[str, List[int]] = {}
    for conf in all_configs:
        sites = conf.get("sites", [])
        task_id = conf.get("task_id")
        if task_id is None or len(sites) != 1:
            continue
        domain = sites[0]
        domain_tasks.setdefault(domain, []).append(task_id)

    # Sort task IDs within each domain
    for domain in domain_tasks:
        domain_tasks[domain] = sorted(set(domain_tasks[domain]))

    return domain_tasks


def create_webarena_datasets(
    n_val_per_domain: int = 2,
    n_train_per_domain: Optional[int] = None,
    val_repeats: int = 1,
    n_id_eval_per_domain: int = 10,
    seed: int = 42,
    config_dir: str = "config_files_lite",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create train/val/ID-eval splits from meta-train domains.

    For each meta-train domain:
      1. Shuffle task IDs deterministically (seed)
      2. Reserve n_val_per_domain tasks for validation
      3. Reserve n_id_eval_per_domain tasks for in-distribution evaluation
      4. Remaining tasks go to training (n_repeats=1)

    Returns: (trainset, valset, id_evalset)
    """
    rng = random.Random(seed)
    domain_tasks = _load_task_ids_by_domain()

    trainset = []
    valset = []
    id_evalset = []

    for domain in META_TRAIN_DOMAINS:
        task_ids = domain_tasks.get(domain, [])[:]
        rng.shuffle(task_ids)

        # Split: val -> id_eval -> train
        val_ids = task_ids[:n_val_per_domain]
        id_eval_ids = task_ids[n_val_per_domain:n_val_per_domain + n_id_eval_per_domain]
        train_ids = task_ids[n_val_per_domain + n_id_eval_per_domain:]

        if n_train_per_domain is not None:
            train_ids = train_ids[:n_train_per_domain]

        for tid in train_ids:
            trainset.append(_make_task_dict(tid, domain, "train", 1, seed, config_dir))

        for tid in val_ids:
            valset.append(_make_task_dict(tid, domain, "val", val_repeats, seed, config_dir))

        for tid in id_eval_ids:
            id_evalset.append(_make_task_dict(tid, domain, "id_eval", 1, seed, config_dir))

    # Shuffle training set for mixed-domain batches
    rng.shuffle(trainset)

    print(f"WebArena Meta-Training Dataset Created:")
    print(f"  Train: {len(trainset)} tasks")
    print(f"  Val:   {len(valset)} tasks")
    print(f"  ID Eval: {len(id_evalset)} tasks")

    return trainset, valset, id_evalset


def create_ood_eval_set(
    config_dir: str = "config_files_lite",
    seed: int = 42,
) -> List[Dict]:
    """Create OOD evaluation set from meta-test domains (reddit + shopping_admin)."""
    domain_tasks = _load_task_ids_by_domain()
    ood_set = []

    for domain in META_TEST_DOMAINS:
        task_ids = domain_tasks.get(domain, [])
        for tid in task_ids:
            ood_set.append(_make_task_dict(tid, domain, "ood_eval", 1, seed, config_dir))

    print(f"  OOD Eval: {len(ood_set)} tasks")
    return ood_set


def _make_task_dict(
    task_id: int,
    domain: str,
    split: str,
    n_repeats: int,
    seed: int,
    config_dir: str,
) -> Dict:
    return {
        "task_id": task_id,
        "task_name": f"webarena.{task_id}",
        "game_name": "webarena",
        "uid": f"webarena.{task_id}",
        "domain": domain,
        "split": split,
        "n_repeats": n_repeats,
        "seed": seed,
        "config_dir": config_dir,
    }
