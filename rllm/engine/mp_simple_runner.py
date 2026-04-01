# This function must stay at module scope so it can be pickled.
from rllm.engine.simple_runner import SimpleRunner
import multiprocessing
import concurrent.futures
import multiprocessing
import traceback
from typing import Dict, Any, List

from rllm.environments.jericho.openai_helpers import chat_completion_with_retries, init_global_client

def _worker_run_full_episode(
    task_id: int, 
    task_config: Dict, 
    env_cls: Any, 
    agent_cls: Any, 
    env_args: Dict, 
    agent_args: Dict, 
    meta_llm_cfg: Dict, 
    api_cfg: Dict,
    log: bool,
):
    """
    Worker-process entry point that runs a full trajectory in one shot.
    """
    try:
        # Key point: reinitialize the API client inside each child process.
        if api_cfg:
            init_global_client(base_url=api_cfg.get('base_url'), api_key=api_cfg.get('api_key'))
        
        # 1. Instantiate the environment and agent.
        full_env_args = {**task_config, **env_args}
        env = env_cls.from_dict(full_env_args)
        agent = agent_cls(**agent_args)
        
        # 2. Reset
        obs, info = env.reset()
        agent.reset()
        agent.update_from_env(observation=obs, reward=0, done=False, info=info)
        
        # 3. Run the full meta loop.
        max_meta_steps = env.meta_cfg.get("max_episodes", 3)
        
        for step in range(max_meta_steps):
            messages = agent.chat_completions
            
            if log:
                print(f"--- 🔄 Task {task_id} | Step {step+1}/{max_meta_steps} ---")

            # Call the LLM using the worker-local `meta_llm_cfg`.
            response_text = chat_completion_with_retries(
                model=meta_llm_cfg["model"],
                messages=messages,
                temperature=meta_llm_cfg.get("temperature", 0.7),
                extra_body=meta_llm_cfg.get("extra_body", None)
            ).choices[0].message.content
            
            # --- Parse the action. ---
            action = agent.update_from_model(response_text)
            
            if log:
                # Print only the first 100 characters to keep logs manageable.
                print(f"   💡 [Task {task_id}] Feedback: {action.action[:100]}...")
            
            # --- Execute the environment step. ---
            next_obs, reward, done, info = env.step(action.action)
            
            # --- Feed the result back into the loop. ---
            agent.update_from_env(observation=next_obs, reward=reward, done=done, info=info)
            
            if agent.trajectory.steps:
                last_step = agent.trajectory.steps[-1]
                last_step.reward = reward
                last_step.info.update(info)
                
            if done:
                if log:
                    print(f"   ✅ Task {task_id} finished (Max episodes reached).")
                break

        if log:
            print(f"🏁 [Task {task_id}] Complete. Final Reward: {agent.trajectory.reward}")
            
        agent.trajectory.task = task_config
        # agent.trajectory.full_history = agent.messages # Enable this only if needed.
        agent.trajectory.full_history = agent.messages
        return agent.trajectory
        
    except Exception as e:
        print(f"❌ Worker {task_id} failed: {e}")
        traceback.print_exc()
        return None

class SimpleRunnerMP(SimpleRunner):
    def execute_tasks(self, tasks, api_cfg, max_concurrent=32):
        
        
        # Using the spawn start method is safer and avoids some C-library deadlocks.
        ctx = multiprocessing.get_context("spawn")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent, mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    _worker_run_full_episode,
                    i, task, 
                    self.env_class, self.agent_class, 
                    self.env_args, self.agent_args, 
                    self.meta_llm_config,
                    api_cfg,
                    self.log,
                )
                for i, task in enumerate(tasks)
            ]
            
            results = []
            for f in concurrent.futures.as_completed(futures):
                if f.result(): results.append(f.result())
                
        return results
