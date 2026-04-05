from __future__ import annotations

from typing import Any


class Agent2Coder:
    def run(self, state: Any) -> dict[str, Any]:
        return {
            "agent_id": "agent2",
            "role": "coder",
            "status": "pending",
            "message": "agent2 is reserved for ICD code prediction from the structured case summary.",
            "available_inputs": list(getattr(state, "agent_outputs", {}).keys()),
        }
