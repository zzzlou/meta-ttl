import re
import sys
import os

def clean_log_file(input_file, output_file):
    # Prevent accidental data loss when input and output paths are the same.
    if os.path.abspath(input_file) == os.path.abspath(output_file):
        print("❌ Error: input and output files cannot be the same path, or the original file would be cleared.")
        print("Please use a different output filename, for example by appending '_new'.")
        return

    # 1. Compile regex patterns.
    pid_pattern = re.compile(r"\[PID:\s*\d+\]\s*\[Thread\s*\d+\]")
    proposed_text_pattern = re.compile(r"Iteration \d+: Proposed new text for ")
    
    # Counters.
    deleted_counts = {
        "openai_init": 0,
        "pid_thread": 0,
        "red_circle": 0,
        "task_context_block": 0,  # Count lines removed by block deletion.
        "retry_noise": 0
    }
    
    total_lines = 0
    kept_lines = 0

    # --- Core state flag ---
    # False: process lines normally.
    # True: drop everything between "Task Context" and "Submitting".
    in_task_context_block = False 

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                total_lines += 1
                stripped_line = line.strip() # Trim surrounding spaces before checking.

                # =================================================
                # Priority 1: handle block-deletion mode.
                # =================================================
                if in_task_context_block:
                    # Exit block-deletion mode once we reach the terminating "Submitting" line.
                    if stripped_line.startswith("Submitting") or stripped_line.startswith("🚀 Submitting"):
                        in_task_context_block = False # Leave block-deletion mode.
                        # We intentionally keep the "🚀 Submitting..." line.
                        # To drop it as well, remove the next two lines and restore the counter in `else`.
                        f_out.write(line) 
                        kept_lines += 1
                    else:
                        # Still inside the block, so keep deleting.
                        deleted_counts["task_context_block"] += 1
                    
                    # Skip the remaining filters while block deletion is active.
                    continue

                # =================================================
                # Priority 2: detect the start of block deletion.
                # =================================================
                if "{'Task Context':" in line:
                    in_task_context_block = True
                    deleted_counts["task_context_block"] += 1
                    continue

                if proposed_text_pattern.search(line):
                    in_task_context_block = True
                    deleted_counts["task_context_block"] += 1
                    continue

                # =================================================
                # Priority 3: apply normal line-level filters.
                # =================================================

                # Rule 1: drop OpenAI initialization logs.
                if "Initializing Global OpenAI Client with Base URL" in line:
                    deleted_counts["openai_init"] += 1
                    continue

                # Rule 2: drop the red-circle end marker.
                if "🔴" in line:
                    deleted_counts["red_circle"] += 1
                    continue

                # Rule 3: drop PID/thread prefixes.
                if pid_pattern.search(line):
                    deleted_counts["pid_thread"] += 1
                    continue
                if ("Hit openai.error exception" in line) or \
                   ("Invalid or empty response received" in line) or \
                   ("Error code:" in line):
                    deleted_counts["retry_noise"] += 1
                    continue

                # Keep the line if it passes every filter above.
                f_out.write(line)
                kept_lines += 1

        print("="*40)
        print("✅ Cleaning complete!")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print("-" * 20)
        print(f"Total lines: {total_lines}")
        print(f"Kept lines: {kept_lines}")
        print("Deletion summary:")
        print(f"  - Task Context block: {deleted_counts['task_context_block']} lines (including contents)")
        print(f"  - OpenAI Init      : {deleted_counts['openai_init']} lines")
        print(f"  - PID/Thread       : {deleted_counts['pid_thread']} lines")
        print(f"  - 🔴 End marker     : {deleted_counts['red_circle']} lines")
        print(f"  - Retry/error noise: {deleted_counts['retry_noise']} lines")
        print("="*40)

    except FileNotFoundError:
        print(f"❌ Error: file not found: {input_file}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    # Usage: edit the filenames below directly or pass them on the command line.
    # By default this script processes the generated training log.

    input_log = "train_combined.log"  # Original log file.
    output_log = "train_combined_clean.log" # Cleaned log file.
    
    # Override the defaults with command-line arguments when provided.
    if len(sys.argv) >= 2:
        input_log = sys.argv[1]
    if len(sys.argv) >= 3:
        output_log = sys.argv[2]
        
    clean_log_file(input_log, output_log)
