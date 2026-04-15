from __future__ import annotations

import unittest

from multi_agent_icd import Agent1PrimaryAnalyzer, Agent2Coder, MultiAgentController
from multi_agent_icd.agents.agent1.prompt import build_agent1_prompts
from multi_agent_icd.agents.agent2.prompt import build_agent2_prompts, resolve_agent2_code_systems
from multi_agent_icd.run import PipelineState


class StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_json(self, system_prompt, user_prompt, metadata=None, response_model=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": metadata or {},
                "response_model": response_model,
            }
        )
        return self.payload


class AgentPipelineTests(unittest.TestCase):
    def test_agent1_normalizes_structured_summary(self):
        llm = StubLLM(
            {
                "gender": "female",
                "chief_complaint": "shortness of breath",
                "procedure": ["bronchoscopy"],
                "history_present_illness": "Patient admitted with acute shortness of breath.",
                "past_medical_history": ["copd"],
                "physical_exam_discharge": ["breathing improved"],
                "pertinent_results": ["cxr with bibasilar infiltrates"],
                "hospital_course": ["treated for pneumonia"],
                "discharge_diagnosis": ["pneumonia"],
            }
        )
        agent = Agent1PrimaryAnalyzer(llm=llm)

        result = agent.run(
            note_text="Chief Complaint: shortness of breath\nDischarge Diagnosis: pneumonia",
            patient_context={"sex": "female"},
        )

        self.assertEqual(result["chief_complaint"], "shortness of breath")
        self.assertEqual(result["discharge_diagnosis"], ["pneumonia"])

    def test_agent1_prompt_template_does_not_seed_specific_case(self):
        prompts = build_agent1_prompts(
            note_text="Chief Complaint: right sided pain and nausea",
            patient_context={"sex": "female"},
            evidence_index=[{"id": "E1", "text": "Chief Complaint: right sided pain and nausea"}],
        )

        self.assertNotIn("l2 fracture", prompts["user_prompt"].lower())
        self.assertNotIn("corpectomy", prompts["user_prompt"].lower())
        self.assertNotIn("posterior fusion", prompts["user_prompt"].lower())

    def test_agent2_returns_grounded_code_structure(self):
        llm = StubLLM(
            {
                "principal_diagnosis": {
                    "code": "J18.9",
                    "description": "pneumonia, unspecified organism",
                    "code_system": "ICD-10-CM",
                    "category": "principal_diagnosis",
                    "confidence": "high",
                    "rationale": "Discharge diagnosis states pneumonia.",
                    "evidence_ids": ["E2"],
                    "missing_details": [],
                },
                "secondary_diagnoses": [],
                "procedures": [],
                "coding_queries": [],
                "coding_summary": "Pneumonia is the principal coded condition.",
            }
        )
        agent = Agent2Coder(llm=llm)
        state = PipelineState(
            note_text="Chief Complaint: cough\nDischarge Diagnosis: pneumonia",
            patient_context={},
        )
        state.shared_memory["structured_case_summary"] = {
            "gender": "unknown",
            "chief_complaint": "cough",
            "procedure": [],
            "history_present_illness": "Cough and fever.",
            "past_medical_history": [],
            "physical_exam_discharge": [],
            "pertinent_results": [],
            "hospital_course": ["treated for pneumonia"],
            "discharge_diagnosis": ["pneumonia"],
        }

        result = agent.run(state)

        self.assertEqual(result["principal_diagnosis"]["code"], "J18.9")
        self.assertEqual(result["coding_summary"], "Pneumonia is the principal coded condition.")

    def test_agent2_prompt_template_does_not_seed_specific_codes(self):
        prompts = build_agent2_prompts(
            structured_case_summary={"discharge_diagnosis": ["right sided pain and nausea"]},
            patient_context={"coding_version": "ICD-9"},
            evidence_index=[{"id": "E1", "text": "right sided pain and nausea"}],
        )

        self.assertNotIn("pneumonia", prompts["user_prompt"].lower())
        self.assertNotIn("hypertension", prompts["user_prompt"].lower())
        self.assertNotIn('"486"', prompts["user_prompt"])
        self.assertNotIn('"401.9"', prompts["user_prompt"])

    def test_agent2_prompt_uses_candidate_descriptions_when_present(self):
        prompts = build_agent2_prompts(
            structured_case_summary={"discharge_diagnosis": ["right kidney stone"]},
            patient_context={"coding_version": "ICD-9"},
            evidence_index=[{"id": "E1", "text": "Discharge diagnosis: right kidney stone"}],
            candidate_code_set=["274.9", "585.6"],
            candidate_code_records=[
                {"code": "274.9", "description": "Gout, unspecified"},
                {"code": "585.6", "description": "End stage renal disease"},
            ],
            candidate_output_limit=2,
        )

        self.assertIn("Allowed ICD code candidates with descriptions", prompts["user_prompt"])
        self.assertIn("Gout, unspecified", prompts["user_prompt"])
        self.assertIn("End stage renal disease", prompts["user_prompt"])
        self.assertIn("source of truth", prompts["user_prompt"])

    def test_agent2_uses_icd9_defaults_when_dataset_requests_it(self):
        llm = StubLLM(
            {
                "principal_diagnosis": {
                    "code": "486",
                    "description": "pneumonia, organism unspecified",
                    "category": "principal_diagnosis",
                    "confidence": "high",
                    "rationale": "Discharge diagnosis states pneumonia.",
                    "evidence_ids": ["E2"],
                    "missing_details": [],
                },
                "secondary_diagnoses": [],
                "procedures": [],
                "coding_queries": [],
                "coding_summary": "ICD-9 test output.",
            }
        )
        agent = Agent2Coder(llm=llm)
        state = PipelineState(
            note_text="Chief Complaint: cough\nDischarge Diagnosis: pneumonia",
            patient_context={"coding_version": "ICD-9"},
        )
        state.shared_memory["structured_case_summary"] = {
            "gender": "unknown",
            "chief_complaint": "cough",
            "procedure": [],
            "history_present_illness": "Cough and fever.",
            "past_medical_history": [],
            "physical_exam_discharge": [],
            "pertinent_results": [],
            "hospital_course": ["treated for pneumonia"],
            "discharge_diagnosis": ["pneumonia"],
        }

        result = agent.run(state)
        _, diagnosis_code_system, _ = resolve_agent2_code_systems(state.patient_context)

        self.assertEqual(result["principal_diagnosis"]["code_system"], diagnosis_code_system)

    def test_agent2_respects_candidate_set_and_output_limit(self):
        llm = StubLLM(
            {
                "principal_diagnosis": {
                    "code": "486",
                    "description": "pneumonia, organism unspecified",
                    "category": "principal_diagnosis",
                    "confidence": "high",
                    "rationale": "Discharge diagnosis states pneumonia.",
                    "evidence_ids": ["E2"],
                    "missing_details": [],
                },
                "secondary_diagnoses": [
                    {
                        "code": "401.9",
                        "description": "unspecified essential hypertension",
                        "category": "secondary_diagnosis",
                        "confidence": "medium",
                        "rationale": "History supports hypertension.",
                        "evidence_ids": ["E3"],
                        "missing_details": [],
                    },
                    {
                        "code": "999.99",
                        "description": "not in top candidates",
                        "category": "secondary_diagnosis",
                        "confidence": "low",
                        "rationale": "Should be filtered.",
                        "evidence_ids": ["E4"],
                        "missing_details": [],
                    },
                ],
                "procedures": [
                    {
                        "code": "96.6",
                        "description": "enteral infusion of concentrated nutritional substances",
                        "category": "procedure",
                        "confidence": "low",
                        "rationale": "Procedure support.",
                        "evidence_ids": ["E5"],
                        "missing_details": [],
                    }
                ],
                "coding_queries": [],
                "coding_summary": "Top-code constrained output.",
            }
        )
        agent = Agent2Coder(llm=llm)
        state = PipelineState(
            note_text="Discharge Diagnosis: pneumonia",
            patient_context={
                "coding_version": "ICD-9",
                "candidate_code_set": ["486", "401.9", "96.6"],
                "candidate_output_limit": 2,
            },
        )
        state.shared_memory["structured_case_summary"] = {
            "gender": "unknown",
            "chief_complaint": "cough",
            "procedure": [],
            "history_present_illness": "Cough and fever.",
            "past_medical_history": ["hypertension"],
            "physical_exam_discharge": [],
            "pertinent_results": [],
            "hospital_course": ["treated for pneumonia"],
            "discharge_diagnosis": ["pneumonia"],
        }

        result = agent.run(state)

        self.assertEqual(result["principal_diagnosis"]["code"], "486")
        self.assertEqual([item["code"] for item in result["secondary_diagnoses"]], ["401.9"])
        self.assertEqual(result["procedures"], [])
        self.assertIn("Allowed ICD code candidates", llm.calls[0]["user_prompt"])

    def test_controller_defaults_to_two_agent_pipeline(self):
        class StaticAgent1:
            def run(self, note_text, patient_context=None):
                return {
                    "gender": "unknown",
                    "chief_complaint": "fever",
                    "procedure": [],
                    "history_present_illness": "Fever and cough.",
                    "past_medical_history": [],
                    "physical_exam_discharge": [],
                    "pertinent_results": [],
                    "hospital_course": ["treated empirically"],
                    "discharge_diagnosis": ["pneumonia"],
                }

        class StaticAgent2:
            def run(self, state):
                return {
                    "principal_diagnosis": {
                        "code": "J18.9",
                        "description": "pneumonia, unspecified organism",
                        "code_system": "ICD-10-CM",
                        "category": "principal_diagnosis",
                        "confidence": "medium",
                        "rationale": "Supported by the discharge diagnosis.",
                        "evidence_ids": ["E2"],
                        "missing_details": [],
                    },
                    "secondary_diagnoses": [],
                    "procedures": [],
                    "coding_queries": [],
                    "coding_summary": "Pneumonia is the principal coded condition.",
                }

        controller = MultiAgentController(agent1=StaticAgent1(), agent2=StaticAgent2())
        state = controller.run(note_text="Discharge Diagnosis: pneumonia")

        self.assertEqual(controller.pipeline_order, ["agent1", "agent2"])
        self.assertIn("agent2", state["agent_outputs"])
        self.assertNotIn("agent3", state["agent_outputs"])


if __name__ == "__main__":
    unittest.main()
