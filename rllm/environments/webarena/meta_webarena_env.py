import os
import io
import datetime
import threading
from typing import Dict, Tuple, Any, Type
from types import SimpleNamespace

from rllm.environments.base.base_env import BaseEnv
from rllm.misc import colorful_print

# Import from webarena package directly
# Assuming we run this with PYTHONPATH including webarena, or run from inside webarena
# We will do dynamic import or standard import if paths are set up.
import sys
# Make sure we can import from webarena if we are not already in it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../webarena')))

from run import run_single_episode
from memory_agents.memory_agent import MemoryAgentArgs

class WebArenaMetaEnv(BaseEnv):
    def __init__(self, 
                 meta_cfg: Dict, 
                 actor_cls: Type, 
                 actor_args: Dict, 
                 log=True,
                 **game_cfg):
        """
        Args:
            meta_cfg: Configuration for the Meta-Loop (e.g. max_episodes, env_step_limit)
            actor_cls: Actor class (e.g. MemoryAgent) - not strictly used here as run.py handles it, but kept for interface compatibility
            actor_args: Arguments to initialize the Actor (e.g. llm_model, temperature)
            **game_cfg: Specific game configuration (e.g. task_name, start_url)
        """
        self.meta_cfg = meta_cfg
        self.game_cfg = game_cfg
        self.actor_args = actor_args
        
        # Meta settings
        self.task_name = game_cfg.get("task_name")
        self.max_episodes = meta_cfg.get("max_episodes", 3)
        self.max_obs_chars = meta_cfg.get("max_obs_chars", None)
        self.log = log
        
        self.current_episode_idx = 0
        self.best_score = -float('inf')
        self.feedback_history = []
        
        # We don't instantiate the actor here directly because run_single_episode 
        # takes care of creating the Env and the Agent internally based on arguments.
        # However, we need to prepare the standard arguments that `run.py` expects.
        
        # Set up default namespace similar to parse_args in run.py
        self.run_args_namespace = self._build_run_args(actor_args, meta_cfg, game_cfg)

    def _build_run_args(self, actor_args: Dict, meta_cfg: Dict, game_cfg: Dict) -> SimpleNamespace:
        args_dict = dict(
            task_name=self.task_name,
            start_url="https://www.google.com",
            slow_mo=30,
            headless=True,
            demo_mode=False,
            use_html=False,
            use_ax_tree=True,
            use_screenshot=True,
            multi_actions=True,
            action_space="bid",
            use_history=True,
            use_thinking=True,
            max_steps=meta_cfg.get("env_step_limit", 25),
            workflow_path=None,
            seed=game_cfg.get("seed", 0),
            llm_model=actor_args.get("llm_model", "gpt-4o"),
            llm_eval=None,
            use_screenshot_eval=False,
            use_screenshot_action=False,
            disable_memory=actor_args.get("disable_memory", False),
            top_actions=3,
            llm_temperature=actor_args.get("llm_temperature", 0.5),
            logit_mode="token",
            max_memory=actor_args.get("max_memory", 30),
            gamma=0.1,
            max_trajectory_window=5,
            debug_info=False,
            track_valid_changes=False,
            agent_type=actor_args.get("agent_type", "memory"),
            eval_runs=1,
            test_case_start=None,
            test_case_end=None,
            repeat_runs=1,
            evol_temperature=0.7,
            summary_temperature=0.8,
            summary_max_tokens=300,
            retrieval_top_k=3,
            retrieval_threshold=0.1,
            embedding_api_key="None",
            rag_temperature=0.4,
            rag_max_tokens=400,
            initial_prompts_file="initial_prompts.json",
            exploration_constant=1.0,
            depth_constant=0.8,
            freeze_on_win=True,
            win_freeze_threshold=0,
            force_best_after_drop=True,
            drop_threshold=50,
            enable_cross_mem=True,
            config_dir=actor_args.get("config_dir", "config_files_lite"),
            use_llm_success_eval=False
        )
        
        # Override with any additional configurations passed
        for k, v in meta_cfg.items():
            if k in args_dict:
                args_dict[k] = v
        for k, v in game_cfg.items():
            if k in args_dict:
                args_dict[k] = v
        for k, v in actor_args.items():
            if k in args_dict:
                args_dict[k] = v
                
        return SimpleNamespace(**args_dict)

    def reset(self) -> Tuple[str, Dict]:
        self.current_episode_idx = 0
        self.best_score = -float('inf')
        self.feedback_history = []
        
        initial_prompt = self.meta_cfg.get("initial_prompt", "Explore systematically and examine objects to make progress.")
        
        trajectory_log, score, info = self._run_full_game_episode(initial_prompt)
        
        self.best_score = max(self.best_score, score)
        return trajectory_log, {"score": score, "raw_info": info, "termination_reason": info.get("termination_reason", "UNKNOWN")}

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        # action is the new feedback/guiding prompt from Meta-Agent
        new_feedback = str(action).strip()
        if new_feedback:
            self.feedback_history = [new_feedback]
            
        self.current_episode_idx += 1
        
        guiding_feedbacks = ""
        for i, fb in enumerate(self.feedback_history):
            guiding_feedbacks += f"{i+1}. {fb}\n"
            
        trajectory_log, score, info = self._run_full_game_episode(guiding_feedbacks)
        
        reward = float(score)
        
        self.best_score = max(self.best_score, score)
        info["best_score"] = self.best_score
        
        # Early stopping: done if max episodes reached OR episode succeeded
        success = info.get("success", False)
        done = (self.current_episode_idx >= self.max_episodes) or success
        if success and self.log:
            from rllm.misc import colorful_print
            colorful_print(f"🎉 Early stopping: Episode {self.current_episode_idx} succeeded!", "green")
        
        return trajectory_log, reward, done, info

    def _run_full_game_episode(self, guiding_prompt: str) -> Tuple[str, int, Dict]:
        pid = os.getpid()
        tid = threading.get_ident()
        if self.log:
            colorful_print(f"🚀 [PID: {pid}] [Thread {tid}] START Episode {self.current_episode_idx} for WebArena", "green")
            
        # We need to inject guiding_prompt into the agent creation. 
        # Modifying `run_single_episode` in run.py is the cleanest way.
        try:
            success, agent = run_single_episode(self.run_args_namespace, self.task_name, guiding_prompt=guiding_prompt)
            score = 1 if success else 0
        except Exception as e:
            print(f"Error during WebArena run: {e}")
            score = 0
            success = False
            agent = None

        traj_buffer = io.StringIO()
        traj_buffer.write(f"Episode: {self.current_episode_idx}\n")
        traj_buffer.write(f"Guiding Prompt: {guiding_prompt}\n")
        traj_buffer.write("--- START GAME ---\n")

        done = False
        if agent and hasattr(agent, 'game_history') and agent.game_history:
            history = agent.game_history

            task_goal = history[0].get('task_goal', '')
            if task_goal:
                traj_buffer.write(f"Task Goal: {task_goal}\n")

            prev_obs = None

            for step, entry in enumerate(history):
                obs = entry.get('state', '')
                action = entry.get('action', '')

                current_obs_str = obs.strip()
                if prev_obs and current_obs_str == prev_obs:
                    display_obs = "(Same as previous step)"
                else:
                    display_obs = current_obs_str
                    if self.max_obs_chars and len(display_obs) > self.max_obs_chars:
                        display_obs = display_obs[:self.max_obs_chars] + "\n... [truncated]"
                prev_obs = current_obs_str

                traj_buffer.write(f"\n[STEP {step}]\n[OBS]: {display_obs}\n[ACTION]: {action}\n")

            # Final observation: post-last-action state (aligned with Jericho)
            final_obs_text = getattr(agent, 'final_obs', '') or ''
            done = getattr(agent, 'final_terminated', False)

            final_obs_str = final_obs_text.strip()
            if self.max_obs_chars and len(final_obs_str) > self.max_obs_chars:
                final_obs_str = final_obs_str[:self.max_obs_chars] + "\n... [truncated]"
            traj_buffer.write(f"\n[FINAL_OBS]: {final_obs_str}\n")
            if done:
                traj_buffer.write(f"\nGame finished at step {len(history) - 1}. Final score: {score}\n")
            else:
                traj_buffer.write(f"\nStep limit reached. Final score: {score}\n")
        else:
            traj_buffer.write("\nNo game history available (Agent failed or not returned).\n")

        traj_log = traj_buffer.getvalue()
        termination_reason = "DONE" if done else "TRUNCATION"

        info = {
            "score": score,
            "success": success,
            "termination_reason": termination_reason
        }
        
        if self.log:
            colorful_print(f"Episode {self.current_episode_idx} termination_reason: {termination_reason} score: {score}", "green")

        return traj_log, score, info

    @staticmethod
    def from_dict(env_args: Dict):
        args = env_args.copy()
        meta_cfg = args.pop("meta_cfg")
        actor_cls = args.pop("actor_cls")
        actor_args = args.pop("actor_args")

        return WebArenaMetaEnv(
            meta_cfg=meta_cfg,
            actor_cls=actor_cls,
            actor_args=actor_args,
            **args 
        )
