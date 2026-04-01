import math
import os
import re

from pathlib import Path
from dataclasses import asdict, dataclass, field
from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from .dynamic_prompting import ActionSpace
from .utils.chat_api import ChatModelArgs
from rllm.environments.jericho.openai_helpers import chat_completion_with_retries, extract_json_from_response, TokenLimitExceededError

from . import dynamic_prompting


class StateNode:
    def __init__(self, state, instruction, reward=0.0):
        self.state = state
        self.instruction = instruction
        self.reward = reward
        self.response = ""
        
        
@dataclass
class ReferenceAgentArgs(AbstractAgentArgs):    
    chat_model_args: ChatModelArgs = None
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    args: any = None  # To hold the parsed arguments

    def make_agent(self):
        return ReferenceAgent(
            args=self.args, 
            chat_model_args=self.chat_model_args, 
            flags=self.flags,
            guiding_prompt=None
        )
        
        
class ReferenceAgent(Agent):
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Augment observations with text HTML and AXTree representations, which will be stored in
        the experiment traces.
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
    
    
    def __init__(
        self,
        args,
        chat_model_args: ChatModelArgs = None,
        flags: dynamic_prompting.Flags = None,
        guiding_prompt: str = None,
    ):
        self.args = args
        self.chat_model_args = chat_model_args
        self.flags = flags
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.memory = [] # Used by agent
        self.episodes = [] # Cross-episode memory
        self.game_history = [] # Used by evolutionary LLM
        
        self.action_space = ActionSpace(self.flags)
        
        
    def add_to_memory(self, state, response):
        memory_entry = {"state": state, "response": response}
        self.memory.append(memory_entry)
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)  # Remove oldest entry if exceeding max_memory
    
    
    def _format_memory_for_prompt(self):
        if not self.memory:
            return ""
            
        memory_text = "MEMORY (Recent few states and agent's responses):\n"
        for i, entry in enumerate(self.memory):
            memory_text += f"Memory {i+1}:\n"
            memory_text += f"STATE: {entry['state']}\n"
            if entry['response']:
                memory_text += f"AGENT'S RESPONSE: {entry['response']}\n"
        
        return memory_text


    def _format_episodes_for_prompt(self):
        if not self.episodes:
            return ""
        
        memory_text = "Past episodes (Previous episodes' states and agent's responses):\n"
        for i, episode in enumerate(self.episodes):
            memory_text += f"Episode {i+1}:\n"
            for j, entry in enumerate(episode):
                memory_text += f"Step {j+1}:\n"
                memory_text += f"STATE: {entry['state']}\n"
                if entry['response']:
                    memory_text += f"AGENT'S RESPONSE: {entry['response']}\n"
        
        return memory_text
    
    
    def _extract_current_state(self):
        """
        Extract the current state from the game history using the state extractor code.
        Returns:
            str: The extracted state description or empty string if extraction fails.
        """
        if not hasattr(self, 'code') or not self.code:
            return ""
        
        try:
            namespace = {}
            exec(self.code, namespace)
            history_str = self._format_game_history(self.game_history)
            extracted_state = namespace["extract_state"](history_str)
            return str(extracted_state) if extracted_state else ""
        except Exception as e:
            print(f"Error extracting state: {e}")
            return ""
    
    
    def start_episode(self):
        self.episodes.append(self.memory)
        self.memory = []
        print(f"Using initial prompt: '{self.guiding_prompt}'")


    def end_episode(self, state, score, success=None, llm_analysis=None, task_goal=None, user_instruction=None, screenshots_dir=None):
        """
        End an episode: update the current node's score and game history.
        """
        # Append a final entry with last known URL (to avoid N/A in summaries)
        last_known_url = None
        if self.game_history and isinstance(self.game_history[-1], dict):
            last_known_url = self.game_history[-1].get("url")
        self._add_to_game_history(state, '', '', '', url=last_known_url)
        
        print(f"Ending episode with score: {score}.")
        if success is not None:
            print(f"  Success: {success}")
        if llm_analysis:
            print(f"  Analysis: {llm_analysis[:200]}...")


    def get_prompts(self, state_node):
        memory_text = self._format_memory_for_prompt()
        episodes_text = self._format_episodes_for_prompt()
        
        # Extract current state using the state extractor
        extracted_state = self._extract_current_state()
        state_summary = ""
        if extracted_state:
            state_summary = f"GAME STATE SUMMARY: {extracted_state}"

        # System prompt: Web agent role + allowed actions (no JSON/logit_mode; keep REASONING/ACTION)
        sys_prompt = f"""You are an intelligent web agent that interacts with real web pages on behalf of the user. Your goal is to accurately follow the user's natural language instructions by selecting and executing appropriate web actions. Select promising actions based on the web state and memory of past interactions. You will be provided with the past episodes', please think about them and use them to make your decision. Note that you do not have to follow the episodes' actions exactly, you can use them as a reference to make your decision.

User's instructions: {state_node.instruction}"""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"
        sys_prompt += f"""
        
## Available Actions
The available actions you can take are:
{self.action_space.prompt}

## Guidance
- Prefer actions that historically increased score or advanced progress
- Avoid repeating ineffective actions in the same context
- Stop and message the user once the instruction is fulfilled

## Response Format (IMPORTANT)
You MUST respond with ONLY a JSON object (no extra text, no markdown). Use exactly these fields:
{{
  "reasoning": "1-2 sentences explaining why this action should work now",
  "action": "fill('b534', '06/24/2002')"
}}
"""

        # User prompt: history + current state; model returns PURE JSON per above schema
        user_prompt = f"""
{state_summary}

Past episodes:
{episodes_text}

Your web browsing history and current state:
{memory_text}

Current state content:
{state_node.state}

Remember: Output ONLY the JSON object with fields "reasoning" and "action". No additional commentary or code fences.
"""
        return sys_prompt, user_prompt, extracted_state


    # Generates the next action from the LLM based on its memory and the current state node.
    def generate_action(self, state_node, url=None, screenshot=None):
        sys_prompt, user_prompt, extracted_state = self.get_prompts(state_node)
        # Prepare screenshot as base64 if provided
        image_content = None
        if screenshot is not None:
            import base64
            from io import BytesIO
            from PIL import Image
            import numpy as np

            # Convert numpy array to PIL Image if needed
            if isinstance(screenshot, np.ndarray):
                screenshot = Image.fromarray(screenshot)

            # Convert PIL Image to base64
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_content = image_base64
            print(f"[INFO] Screenshot encoded for action generation (base64 length: {len(image_base64)})")
        
        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=2000,
            temperature=self.args.llm_temperature,
        )

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and res_obj.choices[0].message:
            full_response = res_obj.choices[0].message.content
            # Try to parse JSON first (required format)
            json_obj = extract_json_from_response(full_response) if isinstance(full_response, str) else {}
            if isinstance(json_obj, dict) and 'action' in json_obj:
                action_text = str(json_obj.get('action') or '').strip() or "look"
            else:
                # Fallback: parse from REASONING/ACTION text if model didn't follow JSON
                action_text = self._parse_llm_response(full_response)
        else:
            print(f"Warning: LLM API call might have failed or returned empty. Defaulting action.")
            full_response = ""
            action_text = "look" # Default action
            
        self.add_to_memory(state_node.state, full_response)
        self._add_to_game_history(state_node.state, action_text, full_response, extracted_state, url=url)
        
        return action_text.strip(), full_response
    
    
    def _add_to_game_history(self, state, action, full_response, extracted_state, reward=None, score=None, url=None):
        self.game_history.append({
            "state": state,
            "action": action,
            "full_response": full_response,
            "extracted_state": extracted_state,
            "reward": reward,
            "score": score,
            "url": url
        })


    def _parse_llm_response(self, full_response: str):
        """
        Parses the LLM's full string response to extract action.
        """
        action_text = "look" # Default action

        if not full_response or not isinstance(full_response, str):
            return action_text

        lines = full_response.strip().split('\n')
        try:
            for line in lines:
                if line.upper().startswith("ACTION:"):
                    action_text = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Response was: '{full_response}'")

        return action_text


    def extract_quoted_numbers(self, text: str):
        pattern = r"'(\d+)'"
        return re.findall(pattern, text)
    
    
    def find_elements(self, text: str, tag: str = None, with_id: int = None):
        """Find elements in accessibility tree text matching the given criteria."""
        results = []
        pattern = r"\[(\d+)\]\s+(\w+)\s+'([^']*)'"
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
    
    
    def get_action(self, obs: dict) -> tuple[str, dict]:
        print("\n" + "="*100)
        print(f"STEP {len(self.game_history)}")
        print("="*100)

        # Clean up old screenshots at the start of the episode (step 0)
        if hasattr(self, '_exp_args') and hasattr(self._exp_args, 'exp_dir'):
            current_step = len(self.game_history)

            # Clean up old screenshots only at step 0
            if current_step == 0:
                import glob
                screenshot_pattern = str(Path(self._exp_args.exp_dir) / "screenshot_step_*.png")
                old_screenshots = glob.glob(screenshot_pattern)
                if old_screenshots:
                    print(f"[INFO] Cleaning up {len(old_screenshots)} old screenshots from previous runs")
                    for old_screenshot in old_screenshots:
                        try:
                            os.remove(old_screenshot)
                        except Exception as e:
                            print(f"[Warning] Failed to remove {old_screenshot}: {e}")

            # Pre-save current step's screenshot for trajectory context extraction
            # BrowserGym normally saves screenshots AFTER get_action returns (in save_step_info),
            # but we need it DURING generate_action for extract_effective_trajectory_context
            screenshot_path = Path(self._exp_args.exp_dir) / f"screenshot_step_{current_step}.png"
            current_screenshot = obs.get('screenshot')
            if current_screenshot is not None:
                try:
                    from PIL import Image
                    import numpy as np
                    if isinstance(current_screenshot, np.ndarray):
                        img = Image.fromarray(current_screenshot)
                        img.save(screenshot_path)
                        print(f"[INFO] Pre-saved screenshot for step {current_step}")
                except Exception as e:
                    print(f"[Warning] Failed to pre-save screenshot: {e}")

        # web_text = obs['pruned_html']
        web_text = obs['axtree_txt']

        # Get URL from observation
        url = obs.get('url', 'about:blank')
        print(f"URL: {url}")
        print(f"Task: {obs['goal']}")

        # Save current task goal
        self.current_task_goal = obs['goal']

        # Get screenshot if use_screenshot_action is enabled
        screenshot = None
        if hasattr(self.args, 'use_screenshot_action') and self.args.use_screenshot_action:
            screenshot = obs.get('screenshot')

        state_node = StateNode(state=web_text, instruction=obs['goal'])
        action, raw_llm_output = self.generate_action(state_node, url=url, screenshot=screenshot)
        element_id = self.extract_quoted_numbers(action)
        target_element = self.find_elements(web_text, with_id=int(element_id[0])) if element_id else []
        
        # Extract JSON from raw_llm_output using robust extraction
        raw_llm_output_dict = {}
        if raw_llm_output and isinstance(raw_llm_output, str):
            raw_llm_output_dict = extract_json_from_response(raw_llm_output)

        # Normalize action for display
        from .utils.utils import normalize_action
        try:
            normalized_action_data = normalize_action(action, web_text)
            action_display = normalized_action_data.get('normalized_action', action)
        except:
            action_display = action

        ans_dict = {
            'think': raw_llm_output_dict.get('reasoning', '') if isinstance(raw_llm_output_dict, dict) else '',
            'action': action,
            'target_element': target_element
        }

        return action, ans_dict