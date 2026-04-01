# Deprecated.
import copy
import logging
import re
from typing import Any, List, Dict, Tuple

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


class JerichoAgent(BaseAgent):
    """
    An agent for text-based adventure games, inspired by the Reflexion methodology.

    This agent is responsible for:
    1.  **Managing Information**: Maintaining the full conversational history with the LLM.
    2.  **Driving Interaction**: Constructing detailed prompts from environment observations
        and parsing structured actions from the LLM's responses.
    3.  **Stateful Operation**: Tracking its own trajectory and internal step count.
    """

    SYSTEM_PROMPT: str = """You are a master player of text-based adventure games. Your objective is to explore the world, solve puzzles, and maximize your score by thinking logically and strategically.

You will receive an observation of your current situation. You MUST respond in the following structured format:
1.  First, provide your step-by-step reasoning inside a `<think>` block.
2.  Immediately after, provide your chosen action inside an `<answer>` block.

--- FORMAT RULES ---
- Your response MUST contain both a `<think>...</think>` block AND an `<answer>...</answer>` block.
- The `<think>` block should explain your reasoning for the chosen action based on the current observation and your long-term goal.
- The `<answer>` block must contain ONLY the single action command you want to execute (e.g., "go north", "take lantern").
- Do NOT add any text outside of these two blocks.

--- EXAMPLE ---
<think>The observation mentions a small mailbox. Mailboxes in these games often contain initial items or clues. I should open it to see what's inside.</think><answer>open mailbox</answer>
--- END EXAMPLE ---

Now, the game begins. Adhere strictly to this format.
"""

    def __init__(self, **kwargs: Any):
        self.reset()

    def reset(self) -> None:
        """Resets the agent's state, clearing history and trajectory for a new episode."""
        self._trajectory = Trajectory()
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        self.step: int = 0
        self.current_observation: str | None = None

    def update_from_env(self, observation: str, reward: float, done: bool, info: Dict, **kwargs):
        """
        Receives a new observation from the environment and constructs the next user prompt for the LLM.
        This is where the agent decides WHAT to ask the LLM.
        """
        self.current_observation = observation
        prompt_content = f"--- Step {self.step} ---\n\n{observation}"

        # **Information management**: detect when the agent is stuck.
        # If the previous action changed nothing, treat it as ineffective.
        if self._trajectory.steps and self._trajectory.steps[-1].action is not None:
            last_observation = self._trajectory.steps[-1].observation
            if last_observation == self.current_observation:
                feedback = "\n\n[SYSTEM FEEDBACK]: Your last action had no observable effect. The game state has not changed. Please re-evaluate and choose a different action."
                prompt_content += feedback

        prompt_content += "\n\nProvide your response in the required `<think>...</think><answer>...</answer>` format."

        # **Interaction**: append the constructed prompt to conversation history.
        self.messages.append({"role": "user", "content": prompt_content})

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Receives the raw response from the LLM, parses it, updates the trajectory,
        and returns a structured Action for the environment.
        This is where the agent UNDERSTANDS the LLM's answer.
        """
        # **Interaction**: first record the raw model reply in conversation history.
        self.messages.append({"role": "assistant", "content": response})

        # **Information management**: parse the reasoning and final action.
        thought, action_str = self._parse_model_response(response)

        # Record the full "observation -> thought -> action" step.
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought=thought,
            action=action_str,
            model_response=response,
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)

        self.step += 1
        return Action(action=action_str)

    def _parse_model_response(self, response: str) -> Tuple[str, str]:
        """
        Parses the LLM's raw output to extract the thought and action.
        This mirrors the parsing logic from your `run_trial` function.
        """
        match = re.search(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", response, re.DOTALL)

        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
            # Ensure an empty parse still falls back to a safe default action.
            if not action:
                return thought, "look"
            return thought, action
        else:
            # **Robustness**: if the format is invalid, log the issue and use a safe default.
            error_thought = f"[PARSING ERROR]: LLM response did not match the required format. Raw output: '{response}'"
            safe_action = "look"  # "look" is usually a safe information-gathering action.
            return error_thought, safe_action

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Provides the conversation history for the LLM API call."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Provides the full interaction history for training."""
        return self._trajectory
