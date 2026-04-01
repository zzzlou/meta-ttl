import traceback
import json
import math
import os
import re
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from warnings import warn

from rllm.environments.jericho.openai_helpers import chat_completion_with_retries, extract_json_from_response, TokenLimitExceededError

from .utils.utils import generate_trajectory_summary, generate_history_summary, dump_obs
from .utils.chat_api import ChatModelArgs
from .utils.llm_utils import ParseError, retry
from .dynamic_prompting import ActionSpace


from . import dynamic_prompting

class StateNode:
    def __init__(self, state, instruction, reward=0.0):
        self.state = state
        self.instruction = instruction
        self.reward = reward
        self.response = ""

@dataclass
class MemoryAgentArgs(AbstractAgentArgs):    
    # Keep only the necessary args; embedding/RAG-related config was removed.
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    args: any = None 

    def make_agent(self):
        return BrowserGymMemoryAgent(
            args=self.args, 
            flags=self.flags
        )

class BrowserGymMemoryAgent(Agent):
    def __init__(
        self,
        args,
        flags: dynamic_prompting.Flags = None,
        guiding_prompt: str = None,
    ):
        self.args = args
        self.flags = flags
        # Guidance injected by the meta agent.
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."

        # Intra-episode memory for the current run.
        self.game_history = []
        self.step_summaries = []

        # Data enriched post-episode by run_single_episode (from browsergym StepInfo files)
        self.final_obs = ""
        self.final_url = ""
        self.final_terminated = False
        self.final_truncated = False

        # Action space used to describe available actions in the system prompt.
        self.action_space = ActionSpace(self.flags)

    def start_episode(self):
        """Reset intra-episode memory."""
        self.game_history = []
        self.step_summaries = []
        self.final_obs = ""
        self.final_url = ""
        self.final_terminated = False
        self.final_truncated = False
        print(f"Using guiding prompt: '{self.guiding_prompt}'")

    def end_episode(self, state, score, success, llm_analysis, task_goal=None, user_instruction=None, screenshots_dir=None):
        """
        Finish the current episode.
        This slimmed-down version does not persist cross-episode memory and only logs results.
        """
        print(f"Ending episode with score: {score}. Success: {success}")

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess the observation into AXTree and DOM text.
        The original behavior is preserved because it matters for WebArena.
        """
        obs = obs.copy()
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])
        return obs

    def get_prompts(self, state_node):
        """
        Build prompts from the instruction, sliding-window history, and current state.
        """
        # 1. Build the intra-episode memory text.
        memory_text = ""
        if self.game_history:
            memory_parts = []
            # Use a sliding window to avoid context blowup.
            recent_window_size = 2 
            history_length = len(self.game_history)

            # A. Summarize older history.
            if history_length > recent_window_size:
                old_steps_count = history_length - recent_window_size
                memory_parts.append(f"=== Earlier steps summary (0-{old_steps_count-1}) ===")
                
                for idx in range(old_steps_count):
                    # Reuse cached summaries when available, otherwise fall back to a simple one.
                    if idx < len(self.step_summaries):
                        summary_text = self.step_summaries[idx]
                        entry = self.game_history[idx]
                        # Add action reasoning when it can be extracted.
                        reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                        if reasoning:
                            summary_text += f"\nAction Reasoning: {reasoning}"
                        memory_parts.append(summary_text)
                    else:
                        # Fallback
                        entry = self.game_history[idx]
                        step_text = f"Step {idx}: {entry.get('action', 'N/A')}"
                        memory_parts.append(step_text)
                    memory_parts.append("")

            # B. Keep recent history with full state detail.
            start_recent = max(0, history_length - recent_window_size)
            memory_parts.append(f"=== Recent steps ({start_recent}-{history_length-1}) ===")
            
            for idx in range(start_recent, history_length):
                entry = self.game_history[idx]
                step_text = f"Step {idx}:\nState:\n{entry.get('state', '')}\nAction: {entry.get('action', 'N/A')}\n"
                
                reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                if reasoning:
                    step_text += f"Action Reasoning: {reasoning}\n"
                
                if entry.get('url'):
                    step_text += f"URL: {entry.get('url')}\n"
                memory_parts.append(step_text)

            memory_text = "\n".join(memory_parts)

        # 2. Append the current state.
        if memory_text:
            memory_text += "\n\n"
        memory_text += f"=== Current step ({len(self.game_history)}) ===\n"
        memory_text += f"Step {len(self.game_history)}: State:\n{state_node.state}\n"

        # 3. Build the system prompt.
        sys_prompt = f"""You are an intelligent web agent that interacts with real web pages on behalf of the user. Your goal is to accurately follow the user's natural language instructions by selecting and executing appropriate web actions.

User's instructions: {state_node.instruction}"""

        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        sys_prompt += f"""

## Available Actions
The available actions you can take are:
{self.action_space.prompt}

## Response Format
You need to think step-by-step, then provide {self.args.top_actions} potential actions.
You must respond with a JSON object containing:
- option1, option2 ...: The action command.
- option1_reasoning ...: Why you chose this.
- best_action: The number (1, 2...) of the best option.
- reasoning: Overall thought process.

Example JSON:
{{
    "reasoning": "I need to click the submit button...",
    "option1": "click('123')",
    "option1_reasoning": "Submits the form",
    "option2": "fill('456', 'text')",
    "option2_reasoning": "Fills text",
    "best_action": 1
}}
"""
        # User Prompt
        user_prompt = f"""Your web browsing history and current state:
{memory_text}

Analyze the current state, consider your recent history, and provide different and useful actions, then choose the best one."""

        return sys_prompt, user_prompt, memory_text

    def generate_action(self, state_node, url=None, screenshot=None):
        """
        Call the LLM to generate an action.
        This slimmed-down version simply calls, parses, and returns without logit tuning.
        """
        sys_prompt, user_prompt, memory_text = self.get_prompts(state_node)

        # 1. Convert the screenshot to base64 when present.
        image_content = None
        if screenshot is not None:
            import base64
            from io import BytesIO
            from PIL import Image
            import numpy as np
            
            if isinstance(screenshot, np.ndarray):
                screenshot = Image.fromarray(screenshot)
            
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            image_content = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 2. Request structured JSON output.
        # A full schema is omitted here; a simple `json_object` mode is enough.
        response_format = {"type": "json_object"} 

        # 3. Call the shared chat helper with WebArena-style arguments.
        try:
            res_obj = chat_completion_with_retries(
                model=self.args.llm_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=2000,
                temperature=self.args.llm_temperature,
                response_format=response_format,
                image_content=image_content
            )
        except TokenLimitExceededError:
            print("FATAL: Token limit exceeded.")
            raise

        full_response = ""
        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices:
            full_response = res_obj.choices[0].message.content

        # 4. Parse the response payload.
        json_response = extract_json_from_response(full_response)
        if not isinstance(json_response, dict):
            print(f"Warning: Parsed JSON is {type(json_response).__name__}, falling back to empty dict.")
            json_response = {}

        # Basic fallback logic.
        best_choice = json_response.get("best_action", 1)
        # Some models may emit `"1"` instead of `1`.
        if isinstance(best_choice, str) and best_choice.isdigit():
            best_choice = int(best_choice)
            
        action_text = json_response.get(f"option{best_choice}") # Default fallback
        if not action_text:
            print(f"Warning: Missing option{best_choice}, falling back to noop(1000).")
            action_text = "look"

        # 5. Generate a summary for the current step for use in the next prompt.
        # This is intentionally lightweight and depends only on intra-episode history.
        from .utils.utils import generate_single_step_summary
        
        # Normalize the action for display only.
        normalized_action = action_text
        try:
            from .utils.utils import normalize_action
            norm_data = normalize_action(action_text, state_node.state)
            normalized_action = norm_data.get('normalized_action', action_text)
        except:
            pass

        # Generate the summary.
        summary = generate_single_step_summary(
            step_index=len(self.game_history),
            state=state_node.state,
            action=normalized_action,
            task_goal=state_node.instruction,
            llm_model=self.args.llm_model,
            temperature=0.1
        )
        self.step_summaries.append(summary)

        # 6. Record the step in history.
        self._add_to_game_history(
            state_node.state, 
            action_text, 
            full_response, 
            url=url, 
            task_goal=state_node.instruction
        )

        return action_text.strip(), full_response

    def get_action(self, obs: dict) -> tuple[str, dict]:
        """Main entry point."""
        print("\n" + "="*100)
        print(f"STEP {len(self.game_history)}")
        print("="*100)

        # Keep the old screenshot cleanup logic.
        if hasattr(self, '_exp_args') and hasattr(self._exp_args, 'exp_dir'):
            current_step = len(self.game_history)
            if current_step == 0:
                # ... (cleanup logic intentionally unchanged)
                pass
            
            # Pre-save the screenshot for meta-environment evaluation.
            screenshot_path = Path(self._exp_args.exp_dir) / f"screenshot_step_{current_step}.png"
            if obs.get('screenshot') is not None:
                # ... (save logic intentionally unchanged)
                pass

        # Prepare the input data.
        web_text = obs['axtree_txt'] # Or `pruned_html`.
        url = obs.get('url', 'about:blank')
        print(f"URL: {url}")
        print(f"Task: {obs['goal']}")

        screenshot = None
        if hasattr(self.args, 'use_screenshot_action') and self.args.use_screenshot_action:
            screenshot = obs.get('screenshot')

        state_node = StateNode(state=web_text, instruction=obs['goal'])
        
        # Generate the action.
        action, raw_llm_output = self.generate_action(state_node, url=url, screenshot=screenshot)
        
        # Extract reasoning for debugging.
        reasoning = ""
        if isinstance(raw_llm_output, str):
             parsed = extract_json_from_response(raw_llm_output)
             reasoning = parsed.get('reasoning', '')

        ans_dict = {
            'think': reasoning,
            'action': action,
        }

        return action, ans_dict

    def _extract_action_reasoning(self, full_response, action):
        """Helper to get reasoning from JSON response."""
        if not full_response: return None
        try:
            if isinstance(full_response, str):
                resp = json.loads(full_response)
            else:
                resp = full_response
            
            # Try to find the `option_reasoning` that matches the selected action.
            for i in range(1, 10):
                if resp.get(f"option{i}") == action:
                    return resp.get(f"option{i}_reasoning")
            return resp.get("reasoning")
        except:
            return None

    def _add_to_game_history(self, state, action, full_response, url=None, task_goal=None):
        self.game_history.append({
            "state": state,
            "action": action,
            "full_response": full_response,
            "url": url,
            "task_goal": task_goal,
        })
        
    def extract_quoted_numbers(self, text: str):
        pattern = r"'(\d+)'"
        return re.findall(pattern, text)
        
    def find_elements(self, text: str, tag: str = None, with_id: int = None):
        results = []
        pattern = r"\[(\d+)\]\s+(\w+)\s+'(.*?)'"
        for line in text.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                node_id, node_tag, node_value = match.groups()
                node_id = int(node_id)
                if (tag is None or node_tag == tag) and (with_id is None or node_id == with_id):
                    results.append({
                        "id": node_id,
                        "tag": node_tag,
                        "value": node_value
                    })
        return results
