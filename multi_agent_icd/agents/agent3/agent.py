from __future__ import annotations

from typing import Any


class Agent3Reviewer:
    def run(self, state: Any) -> dict[str, Any]:
        return {
            "agent_id": "agent3",
            "role": "reviewer",
            "status": "pending",
            "message": "agent3 is reserved for critique, evidence checking, and refinement feedback.",
            "available_inputs": list(getattr(state, "agent_outputs", {}).keys()),
        }
