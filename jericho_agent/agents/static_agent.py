"""
Jericho Static Agent — no cross-episode learning baseline.
"""

from rllm.environments.jericho.actor_agent import MemoryAgent


class JerichoStaticAgent(MemoryAgent):
    """Static agent: intra-episode memory only, no cross-episode learning."""

    def start_episode(self):
        """Clear intra-episode memory. No cross-episode updates."""
        self.memory = []

    def end_episode(self, state, score, trajectory=None):
        """Accept trajectory kwarg for API compatibility but discard it."""
        print(f"Ending episode with score: {score}.")
