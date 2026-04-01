"""
Jericho Cross-Episode Memory Agent — accumulated knowledge baseline.
"""

from rllm.environments.jericho.actor_agent import MemoryAgent
from rllm.environments.jericho.openai_helpers import chat_completion_with_retries


class JerichoCrossMemAgent(MemoryAgent):
    """Cross-episode memory agent that accumulates knowledge across episodes."""

    def __init__(self, args, guiding_prompt: str = None):
        super().__init__(args, guiding_prompt)
        self.episode_history = []
        self.episode_count = 0

    def start_episode(self, evaluation_status=None):
        """Clear intra-episode memory; cross-episode history persists."""
        self.memory = []
        self.episode_count += 1

    def end_episode(self, state, score, trajectory=None):
        """Summarise the episode and update guiding prompt using all history."""
        summary = self._summarise_episode(trajectory, score) if trajectory else ""
        self.episode_history.append({
            "episode": self.episode_count,
            "score": score,
            "summary": summary,
        })
        self._generate_prompt_update()
        print(f"Ending episode with score: {score}.")

    def _generate_prompt_update(self):
        """Call LLM to compare ALL past episodes and produce a new guiding prompt."""
        if not self.episode_history:
            return

        history_text = self._format_episode_history()

        sys_prompt = (
            "You are an expert text-adventure strategist. You are given "
            "summaries of ALL previous episodes the agent has played, "
            "including scores and key events. Analyse the progression "
            "across episodes and produce an improved guiding strategy "
            "for the next attempt. Focus on:\n"
            "- Actions that reliably increased score\n"
            "- Dead-ends or traps to avoid\n"
            "- Unexplored areas or objects\n"
            "- The most efficient path discovered so far\n\n"
            "Output ONLY the new guiding strategy (no preamble)."
        )

        user_prompt = (
            f"Current guiding prompt:\n{self.guiding_prompt}\n\n"
            f"Episode history:\n{history_text}\n\n"
            "Based on all episodes above, produce an improved guiding "
            "strategy for the next episode."
        )

        extra_body_param = getattr(self.args, "extra_body", None)
        try:
            res = chat_completion_with_retries(
                model=self.args.llm_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
                extra_body=extra_body_param,
            )
            if res and hasattr(res, "choices") and res.choices:
                new_prompt = res.choices[0].message.content.strip()
                if new_prompt and new_prompt != self.guiding_prompt:
                    print(f"[CrossMem] Updating guiding prompt ({len(new_prompt)} chars)")
                    self.guiding_prompt = new_prompt
                else:
                    print("[CrossMem] Guiding prompt unchanged.")
            else:
                print("[CrossMem] LLM returned empty response.")
        except Exception as e:
            print(f"[CrossMem] Prompt update error: {e}")

    def _summarise_episode(self, trajectory, score):
        """Call LLM to produce a compact episode summary."""
        traj_text = self._format_trajectory(trajectory)

        sys_prompt = (
            "Summarise this text-adventure episode in 3-5 bullet points. "
            "Focus on: rooms visited, key objects found, scoring actions, "
            "dead-ends encountered, and the final outcome."
        )
        user_prompt = f"Score: {score}\n\nTrajectory:\n{traj_text}"

        extra_body_param = getattr(self.args, "extra_body", None)
        try:
            res = chat_completion_with_retries(
                model=self.args.llm_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.3,
                extra_body=extra_body_param,
            )
            if res and hasattr(res, "choices") and res.choices:
                return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[CrossMem] Summary error: {e}")
        return ""

    def _format_episode_history(self):
        """Format all past episodes for the prompt update call."""
        parts = []
        for entry in self.episode_history:
            parts.append(
                f"--- Episode {entry['episode']} (Score: {entry['score']}) ---\n"
                f"{entry['summary']}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_trajectory(trajectory):
        """Format a list of (observation, action, score) tuples into text."""
        if not trajectory:
            return ""
        lines = []
        for i, entry in enumerate(trajectory):
            ob, action, score = entry[0], entry[1], entry[2]
            ob_short = ob[:300] + "..." if len(ob) > 300 else ob
            lines.append(f"[Step {i}] OBS: {ob_short}")
            lines.append(f"[Step {i}] ACTION: {action}")
            lines.append(f"[Step {i}] SCORE: {score}")
        return "\n".join(lines)
