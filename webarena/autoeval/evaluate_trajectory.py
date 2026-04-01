import os
import json
import argparse
import traceback
from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT
from autoeval.enhanced_evaluator import (
    evaluator_router, 
    create_trajectory_from_info, 
    create_pseudo_page_from_info
)


def load_blocks(path: str) -> list[list[str]]:
    """Load blank-line separated blocks from the log file."""
    blocks, block = [], []
    for line in open(path, 'r'):
        if line.strip() == "":
            if block:  # Only append non-empty blocks
                blocks.append(block)
            block = []
        else:
            if line.strip():
                block.append(line.strip())
    # Add the last block if it's not empty (file might not end with blank line)
    if block:
        blocks.append(block)
    # Note: blocks may not be even if the task failed before completing
    # assert len(blocks) % 2 == 0
    return blocks

def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            try:
                arg = a[a.index("(")+1: a.index(")")]
                if type(eval(arg)) == str:
                    valid_actions.append(a)
            except (SyntaxError, ValueError, NameError):
                # Skip actions with malformed arguments
                pass
        elif "fill(" in a:
            try:
                arg = a[a.index("(")+1: a.index(",")].strip()
                if type(eval(arg)) == str:
                    valid_actions.append(a)
            except (SyntaxError, ValueError, NameError):
                # Skip actions with malformed arguments
                pass
        else:
            valid_actions.append(a)
    return valid_actions

def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
    """Extract the task trajectory from the log file."""
    blocks = load_blocks(path)
    think_list, action_list = [], []
    processed_blocks = set()

    for i in range(1, len(blocks), 2):
        # action
        b = blocks[i]
        actions = remove_invalid_steps(b[1:])
        if len(actions) == 0: continue
        action_list.append(actions)
        processed_blocks.add(i)
        # think
        b = blocks[i-1]
        # Find the line containing the thinking information
        think_content = ""
        for line in reversed(b):
            if "browsergym.experiments.loop - INFO -" in line:
                idx = line.index("browsergym.experiments.loop - INFO -")
                think_content = line[idx+36:].strip()
                break
        # If no line with INFO found, use the last line as fallback
        if not think_content and len(b) > 0:
            think_content = b[-1].strip()
        think_list.append(think_content)
        processed_blocks.add(i-1)

    # Check for any remaining unprocessed blocks that start with "action:"
    # This handles cases where the last "action" block is empty/invalid
    # but there's a preceding valid action block that got skipped
    for i in range(len(blocks)):
        if i in processed_blocks:
            continue
        block = blocks[i]
        if block and block[0].strip().startswith("action:"):
            actions = remove_invalid_steps(block[1:])
            if len(actions) > 0:
                action_list.append(actions)
                think_list.append("")  # Add empty think for consistency

    assert len(think_list) == len(action_list)

    # TODO: merge same actions
    return think_list, action_list

def extract_response(action: str) -> str:
    """Extract response from send_msg_to_user() call.

    Handles nested parentheses by finding the matching closing parenthesis.
    For example: send_msg_to_user("text (with) parens")
    """
    s = action.index("(") + 1
    # Find the matching closing parenthesis by counting nesting level
    level = 1
    e = s
    while e < len(action) and level > 0:
        if action[e] == '(':
            level += 1
        elif action[e] == ')':
            level -= 1
        e += 1
    # e now points one past the closing paren, so e-1 is the closing paren
    result = action[s:e-1]
    # Remove quotes if present
    if result and result[0] in ('"', "'") and result[-1] in ('"', "'"):
        result = result[1:-1]
    return result


def process_sample(
    idx: str, traj_info: dict, log_save_path,
    model: str, eval_version: str, ground_truth: str, user_instruction: str,
    config_dir: str = "config_files_lite"
) -> list[dict]:
    try:
        # Load task config to determine evaluation type
        config_path = os.path.join(config_dir, f"{idx}.json")
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            # Fallback to old evaluation method
            return process_sample_fallback(idx, traj_info, log_save_path, model, eval_version, ground_truth, user_instruction)

        with open(config_path, 'r') as f:
            config = json.load(f)

        eval_types = config["eval"]["eval_types"]
        print(f"Using evaluation types: {eval_types}")

        # Check if we need the enhanced evaluator
        if any(eval_type in ["url_match", "program_html"] for eval_type in eval_types):
            print(f"Using enhanced evaluator for task {idx}")

            # Create trajectory from info
            trajectory = create_trajectory_from_info(traj_info)

            # For program_html, we need a real browser
            if "program_html" in eval_types:
                print("program_html evaluation requires real browser - launching Playwright...")

                # Run browser in a separate thread to avoid asyncio loop conflict
                import threading
                from playwright.sync_api import sync_playwright

                result_container = {"score": None, "error": None}

                def run_browser_eval():
                    try:
                        with sync_playwright() as p:
                            browser = p.chromium.launch(headless=True)

                            # Load storage state if available
                            storage_state = config.get("storage_state")
                            if storage_state and os.path.exists(storage_state):
                                print(f"Loading storage state from {storage_state}")
                                context = browser.new_context(storage_state=storage_state)
                            else:
                                context = browser.new_context()

                            page = context.new_page()

                            # Create pseudo page with real page object
                            pseudo_page = create_pseudo_page_from_info(traj_info, page)

                            # Get the enhanced evaluator
                            enhanced_evaluator = evaluator_router(config_path)

                            # Run evaluation
                            try:
                                result_container["score"] = enhanced_evaluator(trajectory, config_path, pseudo_page, client=None)
                            finally:
                                context.close()
                                browser.close()
                    except Exception as e:
                        result_container["error"] = e

                # Run in thread
                thread = threading.Thread(target=run_browser_eval)
                thread.start()
                thread.join(timeout=120)  # 2 minute timeout

                if thread.is_alive():
                    raise TimeoutError("Browser evaluation timed out after 120 seconds")

                if result_container["error"]:
                    raise result_container["error"]

                score = result_container["score"]
                eval_result = score > 0.5
                print(f"\n🎯 Browser evaluation completed:")
                print(f"   Score: {score}")
                print(f"   Result: {eval_result} (threshold: 0.5)")
            else:
                # For url_match, we don't need a real browser
                pseudo_page = create_pseudo_page_from_info(traj_info)
                enhanced_evaluator = evaluator_router(config_path)
                score = enhanced_evaluator(trajectory, config_path, pseudo_page, client=None)
                eval_result = score > 0.5
                print(f"\n🎯 URL match evaluation completed:")
                print(f"   Score: {score}")
                print(f"   Result: {eval_result} (threshold: 0.5)")

            result = [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": f"Enhanced evaluation score: {score}",
                "uid": traj_info["traj_name"],
            }]
            print(f"\n✅ Returning evaluation result: {result}\n")
            return result
        else:
            # Use fallback for string_match only
            print(f"Using fallback evaluator for task {idx}")
            return process_sample_fallback(idx, traj_info, log_save_path, model, eval_version, ground_truth, user_instruction)

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR in enhanced evaluation for {idx}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"{'='*80}")
        print("Full traceback:")
        print(traceback.format_exc())
        print(f"{'='*80}")
        print(f"⚠️  Falling back to original evaluator...")
        print(f"{'='*80}\n")
        return process_sample_fallback(idx, traj_info, log_save_path, model, eval_version, ground_truth, user_instruction)


def process_sample_fallback(
    idx: str, traj_info: dict, log_save_path,
    model: str, eval_version: str, ground_truth: str, user_instruction: str
) -> list[dict]:
    """Fallback to original evaluation method"""
    # Use CLIENT_DICT if model is predefined, otherwise use LM_Client directly
    from autoeval.clients import LM_Client
    if model in CLIENT_DICT:
        client_class = CLIENT_DICT[model]
    else:
        client_class = LM_Client
    clients = {model: client_class(model_name=model)}
    evaluator = Evaluator(clients, log_save_path=log_save_path + "/trajs")
    try:
        out, _ = evaluator(traj_info, ground_truth, user_instruction, client=model, version=eval_version)
        eval_result = None
        if out["status"].lower() == "success": eval_result = True
        else: eval_result = False
        return [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out["thoughts"], 
                "uid": traj_info["traj_name"],
        }]
    except Exception as e:
        print(f"Error on {idx}, {e}")
        print(traceback.format_exc())
        return {
            "idx": idx,
            "gt": traj_info["eval"],
            "rm": None,
            "thoughts": None, 
            "uid": traj_info["traj_name"],
        }


def evaluate_trajectory_direct(result_dir: str, llm_eval: str = "gpt-4o", prompt: str = "text", config_dir: str = "config_files_lite"):
    """
    Direct function call version of trajectory evaluation

    Args:
        result_dir: Path to the result directory (e.g., "results/webarena.82" or "results/webarena.82_run1")
        llm_eval: LLM model for evaluation
        prompt: Evaluation prompt type ("text" or "vision")
        config_dir: Configuration directory ("config_files_lite" for WebArena-Lite or "config_files" for full WebArena)

    Returns:
        tuple: (eval_info, success_status)
    """
    # load task config
    # Extract task_id from result_dir (handle both "webarena.68" and "webarena.68_run1" formats)
    task_name_part = result_dir.split('/')[-1].split(".")[1]  # e.g., "68" or "68_run1"
    # Extract just the numeric part before any underscore
    task_id = task_name_part.split("_")[0]  # e.g., "68"
    # Use specified config directory
    # config_files_lite: WebArena-Lite tasks (165 tasks)
    # config_files: Original WebArena tasks (812 tasks)
    config_path = os.path.join(config_dir, f"{task_id}.json")
    config = json.load(open(config_path))
    # reference_answer_raw_annotation is only used for string_match evaluations
    # For url_match and program_html, it may not exist in the config
    reference_answer_raw_annotation = config['eval'].get('reference_answer_raw_annotation', '')

    # Try to load response from states.json first (more reliable)
    states_path = os.path.join(result_dir, "states.json")
    response = ""
    if os.path.exists(states_path):
        try:
            with open(states_path, 'r', encoding='utf-8') as f:
                states_data = json.load(f)
            # Search for send_msg_to_user from the end
            for state in reversed(states_data):
                action = state.get('action', '')
                if 'send_msg_to_user' in action:
                    response = extract_response(action)
                    break
            print(f"Evaluating response (from states.json): {response[:100] if len(response) > 100 else response}")
        except Exception as e:
            print(f"Warning: Failed to load response from states.json: {e}")
            response = ""

    # Fallback to experiment.log if states.json doesn't have response
    if not response:
        log_path = os.path.join(result_dir, "experiment.log")
        if os.path.exists(log_path):
            think_list, action_list = extract_think_and_action(log_path)
            actions = [act for acts in action_list for act in acts]
            # Search for send_msg_to_user in all action blocks, starting from the end
            for action_block in reversed(action_list):
                for action_line in action_block:
                    if "send_msg_to_user" in action_line:
                        response = extract_response(action_line)
                        break
                if response:
                    break
            print(f"Evaluating response (from experiment.log): {response[:100] if len(response) > 100 else response}")
        else:
            print("Warning: Neither states.json nor experiment.log found")

    # Load trajectory log for other info
    log_path = os.path.join(result_dir, "experiment.log")
    if os.path.exists(log_path):
        think_list, action_list = extract_think_and_action(log_path)
        actions = [act for acts in action_list for act in acts]
    else:
        think_list, action_list = [], []
        actions = []
    
    # load summary info
    summary_path = os.path.join(result_dir, "summary_info.json")
    try:
        summary = json.load(open(summary_path, 'r'))
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: summary_info.json not found at {summary_path}, using default cum_reward=0")
        summary = {"cum_reward": 0}

    # load states info if available
    states_path = os.path.join(result_dir, "states.json")
    states_data = []
    if os.path.exists(states_path):
        with open(states_path, 'r', encoding='utf-8') as f:
            states_data = json.load(f)
        print(f"Loaded {len(states_data)} states from {states_path}")

    # collect traj info
    image_paths = [
        os.path.join(result_dir, f) for f in os.listdir(result_dir)
        if f.startswith("screenshot_step_") and (f.endswith(".jpg") or f.endswith(".png"))
    ]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split("_")[-1].split(".")[0]))
    traj_info = {
        "intent": config["intent"],
        "response": response,
        "captions": think_list,
        "actions": actions,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"],
        "states": states_data  # Add states data
    }

    # evaluate trajectory
    log_save_path = os.path.join("autoeval/log", result_dir.split('/')[-1])
    print("Log Save Path:", log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
        os.makedirs(log_save_path + "/trajs")
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path,
        model=llm_eval, eval_version=prompt,
        ground_truth=reference_answer_raw_annotation, user_instruction=config["intent"],
        config_dir=config_dir
    )
    # Replace '/' with '_' in model name for safe filename
    safe_model_name = llm_eval.replace('/', '_')
    output_eval_path = os.path.join(result_dir, f"{safe_model_name}_autoeval.json")
    json.dump(eval_info, open(output_eval_path, 'w'), indent=4)
    
    # Extract success status
    if isinstance(eval_info, list) and len(eval_info) > 0:
        success = eval_info[0].get('rm', False)
    else:
        success = eval_info.get('rm', False)
    
    return eval_info, success


def main():
    """Command line interface - kept for backward compatibility"""
    eval_info, success = evaluate_trajectory_direct(
        result_dir=args.result_dir,
        llm_eval=args.llm_eval,
        prompt=args.prompt
    )
    print(f"Evaluation completed. Success: {success}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to the result directory, e.g., 'webarena.0'.")
    # autoeval
    parser.add_argument("--model", type=str, default="gpt-4o",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o"])
    parser.add_argument("--llm_eval", type=str, default=None,
                        help="LLM model for evaluation (if not specified, uses --model)")
    parser.add_argument("--prompt", type=str, default="text",
                        choices=["text", "vision"])

    args = parser.parse_args()

    # Use llm_eval if specified, otherwise fall back to model
    if args.llm_eval is None:
        args.llm_eval = args.model

    # if args.model == "gpt-4o" and args.prompt != "vision":
    #     print(f"Waring: use vision prompt by default for {args.model}.")
    #     args.prompt = "vision"
    main()
