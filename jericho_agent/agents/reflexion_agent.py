"""
Jericho Reflexion Agent — cross-episode reflection baseline.

After each episode the agent calls the LLM to analyse the most recent
trajectory and produces an updated guiding prompt for the next episode.
Intra-episode action generation is identical to the base MemoryAgent.
"""

from rllm.environments.jericho.actor_agent import MemoryAgent
from rllm.environments.jericho.openai_helpers import chat_completion_with_retries


class JerichoReflexionAgent(MemoryAgent):
    """Cross-episode reflexion agent for Jericho text games."""

    def __init__(self, args, guiding_prompt: str = None):
        super().__init__(args, guiding_prompt)
        self.evaluation_status = []
        self.reflection_content = []

    def start_episode(self, evaluation_status=None):
        """Clear intra-episode memory; reflect if prior episode data exists."""
        self.memory = []

        if evaluation_status:
            for es in evaluation_status:
                if es not in self.evaluation_status:
                    self.evaluation_status.append(es)

        if self.evaluation_status:
            self._reflexion()

    def end_episode(self, state, score, trajectory=None):
        """Record episode outcome for reflection in the next episode."""
        traj_text = self._format_trajectory(trajectory) if trajectory else ""
        self.evaluation_status.append({
            "score": score,
            "trajectory_text": traj_text,
        })
        print(f"Ending episode with score: {score}.")

    def _reflexion(self):
        """Analyse the most recent episode and update self.guiding_prompt."""
        user_prompt, sys_prompt = self._get_reflexion_prompt()
        if not user_prompt:
            return

        latest = self.evaluation_status[-1]
        print(f"[Reflexion] Previous episode score: {latest['score']}")

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
                content = res.choices[0].message.content
                self.reflection_content.append(content)
                self.guiding_prompt = f"Reflection from previous episode: {content}"
                print(f"[Reflexion] Updated guiding prompt ({len(content)} chars)")
            else:
                print("[Reflexion] LLM returned empty response.")
        except Exception as e:
            print(f"[Reflexion] Error: {e}")

    def _get_reflexion_prompt(self):
        """Build prompts for the reflection LLM call."""
        if not self.evaluation_status:
            return None, None

        latest = self.evaluation_status[-1]
        score = latest["score"]
        traj = latest.get("trajectory_text", "No trajectory available.")

        sys_prompt = (
            "You are an expert text-adventure advisor. Analyse the agent's "
            "previous episode trajectory and provide actionable suggestions "
            "for improvement in the next attempt. Be concise and specific."
        )

        user_prompt = (
            f"Episode outcome — Score: {score}\n\n"
            f"Trajectory:\n{traj}\n\n"
            "Based on the trajectory above:\n"
            "1. What went well and what went wrong?\n"
            "2. Which actions were wasteful or counter-productive?\n"
            "3. What concrete strategy should the agent follow in the next episode?\n\n"
            "Provide a concise action plan (max 5 bullet points)."
        )
        return user_prompt, sys_prompt

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
