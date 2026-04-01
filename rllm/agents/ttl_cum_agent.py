import copy
import re
from typing import Any, Dict, List, Tuple, Optional
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
import sys


class TTLCumulativeAgent(BaseAgent):
    def __init__(self, system_prompt, **kwargs):
        self.system_prompt = system_prompt
        self.current_obs_info = {}
        self.step_counter = 0
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._trajectory = Trajectory()
        self.current_obs_info = {}

    def update_from_model(self, response: str, **kwargs) -> Action:
        # Parse the `<learn>` tag.
        # This agent does not solve tasks directly, so `thought` and `answer` are ignored.
        # print(response)
        # breakpoint()
        learn_matches = re.findall(r"<learn>(.*?)</learn>", response, flags=re.DOTALL)
        rule = "\n".join([m.strip() for m in learn_matches]) if learn_matches else ""
        
        # Record the step for RL training.
        new_step = Step(
            chat_completions=copy.deepcopy(self.messages),
            model_response=response,
            action=Action(action=rule), # The action is the rule itself.
            info=copy.deepcopy(self.current_obs_info)
        )
        self._trajectory.steps.append(new_step)
        self.messages.append({"role": "assistant", "content": response})
        return Action(action=rule)

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        # `observation` is a string containing the prior attempt and its outcome.
        # That string becomes the meta-agent state.
        self.current_obs_info = info
        if self.step_counter == 0:
            user_prompt = f"Here is the trajectory of the first task attempt:\n{observation}\n\nPlease reflect on it and give a feedback on how to improve."
        else:
            user_prompt = f"Here is the result of the next task attempt (Round {self.step_counter}):\n{observation}\n\nPlease continue to reflect and provide updated feedback."
     
        self.messages.append({"role": "user", "content": user_prompt})
        self.step_counter += 1

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        return copy.deepcopy(self.messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
