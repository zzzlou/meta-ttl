from typing import List
import numpy as np

JERICHO_MAX_SCORES = {
    "905": 1, "acorncourt": 30, "advent": 350, "adventureland": 100, "afflicted": 75,
    "anchor": 100, "awaken": 50, "balances": 51, "deephome": 300, "detective": 360,
    "dragon": 25, "enchanter": 400, "gold": 100, "inhumane": 90, "jewel": 90,
    "karn": 170, "library": 30, "ludicorp": 150, "moonlit": 1, "omniquest": 50,
    "pentari": 70, "reverb": 50, "snacktime": 50, "sorcerer": 400, "spellbrkr": 600,
    "spirit": 250, "temple": 35, "tryst205": 350, "yomomma": 35, "zenon": 20,
    "zork1": 350, "zork3": 7, "ztuu": 100, "webarena":1
}

def get_max_score(game_name: str = "detective") -> float:
    """Return the maximum score for a game, defaulting to Detective (360)."""
    clean_name = game_name.lower().split('.')[0]

    return float(JERICHO_MAX_SCORES.get(clean_name, 360.0))
def calculate_time_weighted_auc(traj):
    scores = extract_scores_from_trajectory(traj)
    K = len(scores)
    game_name = traj.task.get('game_name', 'unknown')
    max_score = get_max_score(game_name)
    norm_scores = [s / max_score for s in scores]
    norm_auc = sum(norm_scores) / K
    weights = np.array([t for t in range(1, K + 1)])
    weighted_sum = np.sum(weights * np.array(scores))
    max_weighted_sum = np.sum(weights * max_score)
    weighted_auc = weighted_sum / max_weighted_sum if max_weighted_sum > 0 else 0.0
    return weighted_auc


def format_meta_trajectory(trajectory) -> str:
    """
    Absolute cleanest version: No summarization, no snippets, no extra prefixes.
    Just raw Game Logs and raw Feedbacks in chronological order.
    """
    history_report = ""
    traj = trajectory
        # Task header.
    history_report += f"=== HISTORY FOR TASK ===\n"
    # Episode counter.
    episode_idx = 0
    
    for msg in traj.full_history:
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            # --- 1. Game Log ---
            history_report += f"\n[EPISODE {episode_idx} LOG]\n"
            # Remove the optional "Result: " prefix while keeping the raw content.
            clean_content = content.replace("Result: ", "").strip()
            history_report += f"{clean_content}\n"
            
        elif role == 'assistant':
            # --- 2. Feedback ---
            history_report += f"\n[META-AGENT FEEDBACK (After Episode {episode_idx})]\n"
            history_report += f"{content}\n"
            # A feedback message marks the end of the current episode.
            episode_idx += 1
            
    history_report += "\n" + "="*40 + "\n"

    # --- Part 2: Score Analysis (Simple & Robust) ---
    scores = extract_scores_from_trajectory(traj)
        
    # 3. Compute the AUC using the trapezoid rule.
    auc = np.trapezoid(scores, dx=1) if len(scores) > 1 else 0.0
    
    # 4. Build the score trend string.
    score_trend = " -> ".join([str(int(s)) for s in scores])
    
    # Append the summary to the end of the report.
    game_name = traj.task.get('game_name', 'unknown')
    game_max_score = get_max_score(game_name)
    history_report += f"Score Trend: {score_trend}\n"
    history_report += f"Max Possible Score: {game_max_score}\n"
    history_report += "="*40 + "\n"

    # breakpoint()
    return history_report

def extract_scores_from_trajectory(trajectory) -> List[float]:
    """Helper to get list of scores [Ep0, Ep1, ...]"""
    scores = []
    # 1. Read the baseline score for episode 0.
    if trajectory.steps:
        try:
            scores.append(float(trajectory.steps[0].info['raw_info']['score']))
        except:
            scores.append(0.0)
    
    # 2. Read scores from later episodes (Ep1..N).
    for step in trajectory.steps:
        scores.append(float(step.reward))
    return scores

def calculate_auc_float(trajectory) -> float:
    """Helper to return AUC as a float for sorting"""
    scores = extract_scores_from_trajectory(trajectory)
    if len(scores) < 2: return 0.0
    return np.trapz(scores, dx=1)
