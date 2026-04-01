import io
import os
import threading
import datetime
import gymnasium as gym
import browsergym.webarena  # Register the environment.
from typing import Dict, Tuple, Any, Type
from types import SimpleNamespace

# Base classes imported to match the rest of the project.
from rllm.environments.base.base_env import BaseEnv
from rllm.misc import colorful_print

class StateNode:
    def __init__(self, state, reward=0.0):
        self.state = state  # Store the processed string observation here.
        self.raw_obs = None # Optionally keep the raw dict as well.
        self.reward = reward
        self.response = ""

class WebArenaMetaEnv(BaseEnv):
    def __init__(self, 
                 meta_cfg: Dict, 
                 actor_cls: Type, 
                 actor_args: Dict, 
                 log=True,
                 **task_cfg):
        """
        Args:
            meta_cfg: Meta-loop config (`max_episodes`, `env_step_limit`, etc.).
            actor_cls: Actor class.
            actor_args: Actor initialization arguments.
            **task_cfg: Concrete task config such as `task_name`, `seed`, and `headless`.
        """
        self.meta_cfg = meta_cfg
        self.task_cfg = task_cfg
        
        # Extract key task parameters.
        self.task_name = task_cfg.get("task_name", "browsergym/webarena.0")
        self.base_seed = task_cfg.get("seed", 42)
        # `headless=True` keeps the browser hidden; set False to watch execution.
        self.headless = task_cfg.get("headless", True) 
        
        self.env_step_limit = meta_cfg.get("env_step_limit", 30) # WebArena episodes are usually short.
        self.max_episodes = meta_cfg.get("max_episodes", 5)
        self.feedback_history = []
        
        # Instantiate the actor.
        args_obj = SimpleNamespace(**actor_args)
        self.actor = actor_cls(args=args_obj, guiding_prompt="Initialize...")
        
        self.current_episode_idx = 0
        self.best_score = -float('inf')
        self.log = log

    def reset(self) -> Tuple[str, Dict]:
        self.current_episode_idx = 0
        self.best_score = -float('inf')
        self.feedback_history = []
        
        initial_prompt = self.meta_cfg.get("initial_prompt", "Complete the web task accurately.")
        
        # Run the first episode.
        trajectory_log, score, info = self._run_full_game_episode(initial_prompt)
        
        self.best_score = max(self.best_score, score)
        return trajectory_log, {"score": score, "raw_info": info, "termination_reason": info.get("termination_reason")}

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        # `action` is the new prompt or feedback from the meta agent.
        new_feedback = str(action).strip()
        if new_feedback:
            self.feedback_history = [new_feedback] # Could also append, depending on the strategy.
        
        self.current_episode_idx += 1
        
        # Build the guiding prompt.
        guiding_feedbacks = ""
        for i, fb in enumerate(self.feedback_history):
            guiding_feedbacks += f"{i+1}. {fb}\n"
            
        # Run the next episode.
        trajectory_log, score, info = self._run_full_game_episode(guiding_feedbacks)
        
        reward = float(score)
        done = self.current_episode_idx >= self.max_episodes
        
        self.best_score = max(self.best_score, score)
        info["best_score"] = self.best_score
        
        return trajectory_log, reward, done, info

    def _format_obs(self, obs: Dict) -> str:
        """
        Convert BrowserGym's dict observation into an actor-readable string.
        In practice we usually need the URL, goal, and accessibility tree.
        """
        # BrowserGym observations typically contain `chat_messages`,
        # `open_pages_urls`, `axtree_txt`, and related fields.
        
        url = obs.get("open_pages_urls", ["Unknown URL"])[0]
        # The accessibility tree is the most important perception signal here.
        ax_tree = obs.get("axtree_txt", "") 
        # Include the last action error if one exists.
        last_action_error = obs.get("last_action_error", "")

        buffer = []
        buffer.append(f"Current URL: {url}")
        if last_action_error:
            buffer.append(f"Last Action Error: {last_action_error}")
        buffer.append("Page Content (Accessibility Tree):")
        buffer.append(ax_tree)
        
        return "\n".join(buffer)

    def _run_full_game_episode(self, guiding_prompt: str) -> Tuple[str, float, Dict]:
        pid = os.getpid()
        tid = threading.get_ident()
        
        colorful_print(f"🚀 [PID: {pid}] [Thread {tid}] START WebArena Episode {self.current_episode_idx} Task: {self.task_name}", "green")
        
        # Update the actor prompt.
        self.actor.guiding_prompt = guiding_prompt
        self.actor.start_episode() 

        # 1. Initialize a fresh WebArena environment.
        # Recreating the env each time is the safest way to ensure a clean state.
        env = gym.make(
            self.task_name, 
            headless=self.headless,
            # wait_for_user_message=False # Non-interactive mode.
        )
        
        # Seed and reset the environment.
        obs, info = env.reset(seed=self.base_seed)
        
        # WebArena goals usually live in chat messages or `task_info`.
        # Include the goal in the prompt or at the top of the log.
        goal_txt = info.get("task_info", {}).get("intent", "Unknown Goal")
        
        traj_buffer = io.StringIO()
        traj_buffer.write(f"Episode: {self.current_episode_idx}\n")
        traj_buffer.write(f"Task Goal: {goal_txt}\n")
        traj_buffer.write(f"Guiding Prompt: {guiding_prompt}\n")
        traj_buffer.write("--- START BROWSER SESSION ---\n")
        
        score = 0.0
        done = False
        terminated = False
        truncated = False
        
        prev_obs_str = None

        # 2. Interaction loop.
        for step in range(self.env_step_limit):
            # Process the observation.
            current_obs_str = self._format_obs(obs)
            
            # Avoid printing duplicate observations verbatim.
            log_obs = "(Same as previous step)" if (prev_obs_str and current_obs_str == prev_obs_str) else current_obs_str
            prev_obs_str = current_obs_str
            
            state_node = StateNode(state=f"Goal: {goal_txt}\n{current_obs_str}")
            
            # Let the actor choose an action.
            # WebArena expects executable Python-style actions such as `click('55')`.
            action_str, _ = self.actor.generate_action(state_node)
            
            traj_buffer.write(f"\n[STEP {step}]\n[OBS]:\n{log_obs}\n")
            traj_buffer.write(f"[ACTION]: {action_str}\n")
            
            # Execute the chosen action.
            try:
                obs, reward, terminated, truncated, info = env.step(action_str)
            except Exception as e:
                # Treat invalid generated actions as a failed attempt instead of crashing.
                traj_buffer.write(f"[EXECUTION ERROR]: {str(e)}\n")
                # For now, terminate the episode directly.
                terminated = True
                reward = 0.0
            
            score = reward # BrowserGym WebArena tasks are often binary success/failure.
            # Some tasks expose stepwise reward; `info` may be safer in those cases.
            # score = info.get("reward", reward) 
            
            traj_buffer.write(f"[REWARD]: {reward}\n")
            
            if terminated or truncated:
                final_reason = "DONE" if terminated else "TRUNCATED"
                traj_buffer.write(f"\nGame finished at step {step}. Reason: {final_reason}. Final Score: {score}\n")
                break
        
        # 3. Finalize after the interaction loop.
        if not (terminated or truncated):
            traj_buffer.write(f"\nStep limit ({self.env_step_limit}) reached. Force Stop.\n")
            info["termination_reason"] = "STEP_LIMIT"
        else:
            info["termination_reason"] = "DONE" if terminated else "TRUNCATED"

        env.close()
        
        traj_log = traj_buffer.getvalue()
        
        if self.log:
            # The log can be very large, so only print the summary.
            colorful_print(f"Episode {self.current_episode_idx} Finished. Score: {score}", "green")
            # colorful_print(f"Log: {traj_log[:500]}...", "yellow") # Useful for debugging.

        return traj_log, score, info

    @staticmethod
    def from_dict(env_args: Dict):
        args = env_args.copy()
        meta_cfg = args.pop("meta_cfg")
        actor_cls = args.pop("actor_cls")
        actor_args = args.pop("actor_args")
        return WebArenaMetaEnv(meta_cfg, actor_cls, actor_args, **args)
