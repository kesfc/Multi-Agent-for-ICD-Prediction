from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from multi_agent_icd.agents.agent1 import Agent1PrimaryAnalyzer
from multi_agent_icd.agents.agent2 import Agent2Coder
from multi_agent_icd.agents.agent3 import Agent3KnowledgeSynthesizer
from multi_agent_icd.utils.clinical_text import build_evidence_index, normalize_clinical_text


@dataclass
class PipelineTraceEntry:
    agent_name: str
    status: str
    started_at: str
    finished_at: str
    message: str = ""


@dataclass
class PipelineState:
    note_text: str
    patient_context: dict[str, Any] = field(default_factory=dict)
    training_context: dict[str, Any] = field(default_factory=dict)
    requested_agents: list[str] = field(default_factory=list)
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    shared_memory: dict[str, Any] = field(default_factory=dict)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    final_output: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_text": self.note_text,
            "patient_context": self.patient_context,
            "training_context": self.training_context,
            "requested_agents": self.requested_agents,
            "agent_outputs": self.agent_outputs,
            "shared_memory": self.shared_memory,
            "execution_trace": self.execution_trace,
            "final_output": self.final_output,
        }


class MultiAgentController:
    def __init__(
        self,
        agent1: Any | None = None,
        agent2: Any | None = None,
        agent3: Any | None = None,
        agent_models: dict[str, str] | None = None,
        knowledge_base_path: str | None = None,
        knowledge_retrieval_limit: int = 3,
        pipeline_order: list[str] | None = None,
    ) -> None:
        agent_models = agent_models or {}
        self.agents: dict[str, Any] = {
            "agent1": agent1 or Agent1PrimaryAnalyzer(model_name=agent_models.get("agent1")),
            "agent2": agent2
            or Agent2Coder(model_name=agent_models.get("agent2") or agent_models.get("agent1")),
        }
        if agent3 is not None or knowledge_base_path is not None:
            self.agents["agent3"] = agent3 or Agent3KnowledgeSynthesizer(
                model_name=agent_models.get("agent3")
                or agent_models.get("agent2")
                or agent_models.get("agent1"),
                knowledge_base_path=knowledge_base_path,
            )
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_retrieval_limit = knowledge_retrieval_limit
        self.pipeline_order = pipeline_order or ["agent1", "agent2"]

    def register_agent(self, agent_name: str, agent: Any) -> None:
        self.agents[agent_name] = agent
        if agent_name not in self.pipeline_order:
            self.pipeline_order.append(agent_name)

    def run(
        self,
        note_text: str,
        patient_context: dict[str, Any] | None = None,
        training_context: dict[str, Any] | None = None,
        requested_agents: list[str] | None = None,
    ) -> dict[str, Any]:
        patient_context = patient_context or {}
        training_context = training_context or {}
        requested_agents = requested_agents or list(self.pipeline_order)

        state = PipelineState(
            note_text=normalize_clinical_text(note_text),
            patient_context=patient_context,
            training_context=training_context,
            requested_agents=requested_agents,
        )
        state.shared_memory["note_evidence_index"] = build_evidence_index(state.note_text)
        knowledge_base_path = (
            state.training_context.get("knowledge_base_path")
            or state.patient_context.get("knowledge_base_path")
            or self.knowledge_base_path
        )
        if knowledge_base_path:
            state.shared_memory["knowledge_base_path"] = str(knowledge_base_path)
        state.shared_memory["knowledge_base_top_k"] = (
            state.patient_context.get("knowledge_base_top_k") or self.knowledge_retrieval_limit
        )

        execution_order = list(self.pipeline_order)
        for agent_name in requested_agents:
            if agent_name not in execution_order:
                execution_order.append(agent_name)

        for agent_name in execution_order:
            if agent_name not in requested_agents:
                continue
            self._run_single_agent(agent_name, state)

        if state.final_output is None and state.agent_outputs:
            for agent_name in reversed(requested_agents):
                if agent_name in state.agent_outputs:
                    state.final_output = state.agent_outputs.get(agent_name)
                    break

        return state.to_dict()

    def _run_single_agent(self, agent_name: str, state: PipelineState) -> None:
        started_at = datetime.now(timezone.utc).isoformat()
        agent = self.agents.get(agent_name)

        if agent is None:
            finished_at = datetime.now(timezone.utc).isoformat()
            state.execution_trace.append(
                PipelineTraceEntry(
                    agent_name=agent_name,
                    status="skipped",
                    started_at=started_at,
                    finished_at=finished_at,
                    message="Agent is not registered.",
                ).__dict__
            )
            return

        try:
            if agent_name == "agent1":
                result = agent.run(
                    note_text=state.note_text,
                    patient_context=state.patient_context,
                )
                state.shared_memory["structured_case_summary"] = result
            else:
                result = self._run_downstream_agent(agent, state)
                state.shared_memory[f"{agent_name}_output"] = result

            state.agent_outputs[agent_name] = result
            state.final_output = result
            status = "completed"
            message = ""
        except Exception as exc:
            result = {
                "agent_id": agent_name,
                "status": "failed",
                "message": str(exc),
            }
            state.agent_outputs[agent_name] = result
            state.final_output = result
            status = "failed"
            message = str(exc)

        finished_at = datetime.now(timezone.utc).isoformat()
        state.execution_trace.append(
            PipelineTraceEntry(
                agent_name=agent_name,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                message=message,
            ).__dict__
        )

    def _run_downstream_agent(self, agent: Any, state: PipelineState) -> Any:
        if hasattr(agent, "run"):
            try:
                return agent.run(state=state)
            except TypeError:
                return agent.run(state)
        if callable(agent):
            return agent(state)
        raise TypeError("Registered agent must be callable or expose run(...).")
