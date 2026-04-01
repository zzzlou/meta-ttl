import re
from typing import List
import numpy as np


def extract_scores_from_webarena_trajectory(trajectory) -> List[float]:
    """Get list of scores [ep0_score, ep1_score, ...] from a WebArena trajectory.

    ep0 comes from trajectory.ep0_score (set by runner after reset),
    falling back to raw_info in the first step for old pkl files.
    ep1+ come from step.reward set by the runner after each env.step().
    """
    scores = []
    # 1. Baseline (Ep0) score
    if hasattr(trajectory, 'ep0_score'):
        scores.append(float(trajectory.ep0_score))
    elif trajectory.steps:
        try:
            scores.append(float(trajectory.steps[0].info['raw_info']['score']))
        except (KeyError, TypeError, IndexError):
            scores.append(0.0)
    else:
        scores.append(0.0)

    # 2. Subsequent episode scores (Ep1..N)
    for step in trajectory.steps:
        scores.append(float(step.reward))
    return scores


def calculate_binary_auc(trajectory) -> float:
    """Weighted AUC for WebArena (max_score=1).

    Formula: sum(w_i * s_i) / sum(w_i) where w_i = i+1.
    Same as _calculate_w_auc() in evaluate_webarena.py.
    """
    scores = extract_scores_from_webarena_trajectory(trajectory)
    if not scores:
        return 0.0
    K = len(scores)
    weights = np.arange(1, K + 1)
    return float(np.sum(weights * np.array(scores)) / np.sum(weights))


def format_meta_trajectory_webarena(trajectory) -> str:
    """Format a WebArena meta-agent trajectory for the reflection LM.

    Mirrors format_meta_trajectory() from jericho_agent/utils.py.
    Iterates trajectory.full_history where:
      - 'user' messages -> [EPISODE N LOG]
      - 'assistant' messages -> [META-AGENT FEEDBACK (After Episode N)]
    """
    history_report = ""
    history_report += "=== HISTORY FOR TASK ===\n"
    episode_idx = 0

    for msg in trajectory.full_history:
        role = msg['role']
        content = msg['content']

        if role == 'user':
            history_report += f"\n[EPISODE {episode_idx} LOG]\n"
            clean_content = content.replace("Result: ", "")
            # # Strip TTL wrapper prefixes
            # clean_content = clean_content.replace("Here is the trajectory of the first task attempt:\n", "")
            # clean_content = re.sub(r"Here is the result of the next task attempt \(Round \d+\):\n", "", clean_content)
            # Strip TTL wrapper suffixes
            clean_content = clean_content.replace("Please reflect on it and give a feedback on how to improve.", "")
            clean_content = clean_content.replace("Please continue to reflect and provide updated feedback.", "")
            clean_content = clean_content.strip()
            history_report += f"{clean_content}\n"

        elif role == 'assistant':
            history_report += f"\n[META-AGENT FEEDBACK (After Episode {episode_idx})]\n"
            history_report += f"{content}\n"
            episode_idx += 1

    history_report += "\n" + "=" * 40 + "\n"

    # Score analysis
    scores = extract_scores_from_webarena_trajectory(trajectory)
    score_trend = " -> ".join([str(int(s)) for s in scores])
    history_report += f"Score Trend: {score_trend}\n"
    history_report += f"Max Possible Score: 1\n"
    history_report += "=" * 40 + "\n"

    return history_report
