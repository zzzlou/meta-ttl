import os
import json
import traceback
import ast
from typing import List, Tuple, Dict, Any
from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT

def run_autoeval(
    result_dir: str,
    model: str = "gpt-4o",
    prompt: str = "vision",
    work_path: str = ".",
    ) -> Tuple[Dict[str, Any], str]:
    def load_blocks(path: str) -> List[List[str]]:
        """Load blank-line separated blocks from the log file; robust to trailing/duplicate blanks."""
        blocks, block = [], []
        with open(path, "r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.strip() == "":
                    if block:  # finish a block
                        blocks.append(block)
                        block = []
                else:
                    block.append(line.strip())
        if block:
            blocks.append(block)
        # we expect pairs: [think_block, action_block, think_block, action_block, ...]
        if len(blocks) % 2 != 0:
            # tolerate an extra trailing block by dropping the last orphan if empty/suspicious
            # otherwise raise to surface log formatting error
            if len(blocks[-1]) == 0:
                blocks.pop()
            else:
                raise ValueError(f"Log blocks count should be even, got {len(blocks)} from {path}")
        return blocks

    def _safe_literal(arg: str):
        try:
            return ast.literal_eval(arg)
        except Exception:
            return None

    def remove_invalid_steps(actions: List[str]) -> List[str]:
        """Filter out steps with non-string arguments for click/fill; keep others."""
        valid = []
        for a in actions:
            s = a.strip()
            if s.startswith("click(") and s.endswith(")"):
                arg = s[s.find("(")+1: s.rfind(")")]
                v = _safe_literal(arg)
                if isinstance(v, str):
                    valid.append(s)
            elif s.startswith("fill("):
                # tolerate "fill(selector, text)" variations with spaces
                inside = s[s.find("(")+1: s.rfind(")")]
                first = inside.split(",", 1)[0].strip()
                v = _safe_literal(first)
                if isinstance(v, str):
                    valid.append(s)
            else:
                valid.append(s)
        return valid

    def extract_think_and_action(path: str) -> Tuple[List[str], List[List[str]]]:
        blocks = load_blocks(path)
        think_list, action_list = [], []
        for i in range(1, len(blocks), 2):
            # actions block
            b_actions = blocks[i]
            actions = remove_invalid_steps(b_actions[1:] if len(b_actions) > 1 else b_actions)
            if not actions:
                continue
            action_list.append(actions)
            # think block
            b_think = blocks[i - 1]
            # find the last occurrence to be robust
            marker = "browsergym.experiments.loop - INFO -"
            line = b_think[-1]
            idx = line.rfind(marker)
            if idx == -1:
                # fallback: take the whole last line
                think_list.append(line.strip())
            else:
                think_list.append(line[idx + len(marker):].strip())
        if len(think_list) != len(action_list):
            raise ValueError(f"Mismatch think/actions: {len(think_list)} vs {len(action_list)} in {path}")
        return think_list, action_list

    def extract_response(action: str) -> str:
        # extracts string inside the first (...) pair, without evaluating
        s, e = action.find("(") + 1, action.rfind(")")
        return action[s:e].strip()

    def process_sample(
        idx: str, traj_info: Dict[str, Any], log_save_path: str,
        model_name: str, eval_version: str,
    ) -> Dict[str, Any]:
        clients = {model_name: CLIENT_DICT[model_name](model_name=model_name)}
        evaluator = Evaluator(clients, log_save_path=os.path.join(log_save_path, "trajs"))
        try:
            out, _ = evaluator(traj_info, model_name, eval_version)
            eval_result = out.get("status", "").lower() == "success"
            return {
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out.get("thoughts"),
                "uid": traj_info["traj_name"],
            }
        except Exception as e:
            print(f"[autoeval] Error on {idx}: {e}")
            print(traceback.format_exc())
            return {
                "idx": idx,
                "gt": traj_info.get("eval"),
                "rm": None,
                "thoughts": None,
                "uid": traj_info.get("traj_name"),
                "error": str(e),
            }

    # parameter normalization 
    if model == "gpt-4o" and prompt != "vision":
        prompt = "vision"
    if model != "gpt-4o" and prompt not in ("text", "vision"):
        prompt = "text"

    # load task config 
    task_id = os.path.basename(result_dir).split(".")[1] if "." in os.path.basename(result_dir) else os.path.basename(result_dir)
    config_path = os.path.join(work_path, "config_files", f"{task_id}.json")
    config = json.load(open(config_path, "r"))

    # parse logs -> think/actions/response 
    log_path = os.path.join(work_path, "results", result_dir, "experiment.log")
    think_list, action_list = extract_think_and_action(log_path)
    actions = [a for group in action_list for a in group]
    # response: look at the very last action group, first action if it contains send_msg_to_user
    response = ""
    if action_list and action_list[-1]:
        first_last = action_list[-1][0]
        if "send_msg_to_user" in first_last:
            response = extract_response(first_last)

    # summary
    summary_path = os.path.join(work_path, "results", result_dir, "summary_info.json")
    summary = json.load(open(summary_path, "r"))

    # collect image
    image_paths = [
        os.path.join(result_dir, f)
        for f in os.listdir(result_dir)
        if f.startswith("screenshot_step_") and f.endswith(".jpg")
    ]
    image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
    
    traj_info = {
        "intent": config["intent"],
        "response": response,
        "captions": think_list,
        "actions": actions,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"],
    }

    # evaluate
    log_save_path = os.path.join("autoeval", "log", os.path.basename(result_dir))
    os.makedirs(os.path.join(log_save_path, "trajs"), exist_ok=True)
    
    print('Intent:', traj_info["intent"])
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path, model_name=model, eval_version=prompt,
    )

    # save & return
    output_eval_path = os.path.join(result_dir, f"{model}_autoeval.json")
    with open(output_eval_path, "w") as f:
        json.dump(eval_info, f, ensure_ascii=False, indent=2)

    return eval_info, output_eval_path
