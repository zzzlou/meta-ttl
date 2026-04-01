import asyncio
import time
from typing import Dict, Any, Type, List

from rllm.agents.agent import BaseAgent
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.jericho.openai_helpers import chat_completion_with_retries

class SimpleRunner:
    """
    A lightweight pipeline runner that replaces `AgentExecutionEngine`.
    """
    def __init__(
        self,
        agent_class: Type[BaseAgent],
        env_class: Type[BaseEnv],
        agent_args: Dict[str, Any],
        env_args: Dict[str, Any],
        meta_llm_config: Dict[str, Any], # LLM config used specifically by the meta agent.
        log=True,
    ):
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args
        self.env_args = env_args
        self.meta_llm_config = meta_llm_config
        self.log = log

    async def execute_tasks(self, tasks: List[Dict],max_concurrent=15) -> List[Any]:
        """Execute all tasks."""
        if self.log:
            print(f"🐌 [Serial Runner] Starting execution of {len(tasks)} tasks sequentially...")

        results = []
        start_time_all = time.time()
        import traceback
        for i, task in enumerate(tasks):
            try:
                # Call the task runner directly without extra orchestration.
                traj = self.run_single_task(i, task)
                if traj:
                    results.append(traj)
            except Exception as e:
                print(f"❌ Task {i} Failed: {e}")
                traceback.print_exc() # Print the traceback for debugging.
        
        total_time = time.time() - start_time_all
        if self.log:
            print(f"🏁 All tasks finished in {total_time:.2f}s")
            
        return results

    async def run_single_task(self, task_id: int, task_config: Dict,):
        if self.log:
            print(f"\n🎬 [SimpleRunner] Start running Task {task_id}...")
        
        # 1. Prepare environment args by combining task config and env args.
        # `Env.from_dict` extracts fields such as `meta_cfg` and `actor_cls`.
        full_env_args = {**task_config, **self.env_args}
        env = self.env_class.from_dict(full_env_args)
        
        # 2. Initialize the meta agent.
        # `agent_args` should include the system prompt.
        agent = self.agent_class(**self.agent_args)
        
        # 3. Initial observation (round 0).
        obs, info = env.reset()
        agent.reset()
        
        # Feed the round-0 result to the meta agent.
        agent.update_from_env(observation=obs, reward=0, done=False, info=info)
        
        # 4. Read the meta-loop length from `env.meta_cfg`.
        max_meta_steps = env.meta_cfg.get("max_episodes", 3)
        
        for step in range(max_meta_steps):
            
            # --- A. Meta-agent reasoning (LLM API call). ---
            # Get the current conversation history.
            messages = agent.chat_completions
            
            # Call the LLM with the runner-provided meta-agent config.
            if self.log:
                print(f"--- 🔄 Task {task_id} | Optimization Step {step+1}/{max_meta_steps} ---")
                print(f"   🤖 Meta-Agent is giving test time guidance")
            response_text = chat_completion_with_retries(
                model=self.meta_llm_config["model"],
                messages=messages,
                temperature=self.meta_llm_config.get("temperature", 0.7),
            ).choices[0].message.content
            
            # --- B. Generate the meta-agent action by parsing XML. ---
            # `update_from_model` extracts `<learn>` and stores it in the trajectory.
            action = agent.update_from_model(response_text)
            
            if self.log:
                print(f"   💡 Feedback Generated: {action.action}...") # Print the first part of the feedback.
            
            # --- C. Execute the environment episode with actor feedback. ---
            # `env.step` internally runs a full game episode.
            next_obs, reward, done, info = env.step(action.action)
            
            # --- D. Close the loop by feeding results back to the meta agent. ---
            agent.update_from_env(observation=next_obs, reward=reward, done=done, info=info)
            
            if agent.trajectory.steps:
                last_step = agent.trajectory.steps[-1]
                last_step.reward = reward
                last_step.info.update(info)
                
            if done:
                if self.log:
                    print(f"   ✅ Task {task_id} finished (Max episodes reached).")
                break
        if self.log:
            print(f"🏁 [SimpleRunner] Task {task_id} Complete. Final Reward: {agent.trajectory.reward}")
        agent.trajectory.task = task_config
        agent.trajectory.full_history = agent.messages
        return agent.trajectory
