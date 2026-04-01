from rllm.agents.agent import BaseAgent, Action, Trajectory, Step
import copy

class SilentAgent(BaseAgent):
    """An agent that always returns an empty action."""
    def __init__(self, **kwargs):
        self.reset()
        
    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [] # Keep it empty, or insert a dummy system prompt if needed.

    def update_from_model(self, response: str, **kwargs) -> Action:
        # Ignore any model response from the engine.
        # Returning an empty string lets the env skip history updates via `if new_feedback:`.
        return Action(action="") 

    def update_from_env(self, observation, reward, done, info, **kwargs):
        # Record a dummy step so engine logging still works.
        self._trajectory.steps.append(Step(
            chat_completions=[], 
            model_response="", 
            action=Action(action=""), 
            info=info
        ))
        # The observation is the previous episode log, which this agent intentionally discards.

    @property
    def chat_completions(self):
        return []

    @property
    def trajectory(self):
        return self._trajectory
