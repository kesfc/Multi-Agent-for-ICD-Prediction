from __future__ import annotations

import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from multi_agent_icd import (
    Agent1PrimaryAnalyzer,
    Agent2Coder,
    Agent3KnowledgeSynthesizer,
    KnowledgeBase,
    MultiAgentController,
)
from multi_agent_icd.agents.agent1.prompt import build_agent1_prompts
from multi_agent_icd.agents.agent2.prompt import build_agent2_prompts, resolve_agent2_code_systems
from multi_agent_icd.utils.clinical_text import build_evidence_index, compact_evidence_index_for_prompt
from multi_agent_icd.run import PipelineState


@contextmanager
def workspace_tempdir():
    base = Path.cwd() / ".tmp_test"
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


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

    def test_agent1_prompt_compacts_evidence_fields(self):
        prompts = build_agent1_prompts(
            note_text="Hospital Course: acute blood loss anemia",
            patient_context={},
            evidence_index=[
                {
                    "id": "E37",
                    "section": "hospital_course",
                    "section_label": "Hospital Course",
                    "text": "acute blood loss anemia",
                    "start_char": 5231,
                    "end_char": 5254,
                }
            ],
        )

        self.assertIn('"id": "E37"', prompts["user_prompt"])
        self.assertIn('"text": "acute blood loss anemia"', prompts["user_prompt"])
        self.assertNotIn("section_label", prompts["user_prompt"])
        self.assertNotIn("start_char", prompts["user_prompt"])
        self.assertNotIn("end_char", prompts["user_prompt"])

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

    def test_agent2_prompt_compacts_evidence_fields(self):
        prompts = build_agent2_prompts(
            structured_case_summary={"discharge_diagnosis": ["anemia"]},
            patient_context={"coding_version": "ICD-10"},
            evidence_index=[
                {
                    "id": "E37",
                    "section": "hospital_course",
                    "section_label": "Hospital Course",
                    "text": "acute blood loss anemia",
                    "start_char": 5231,
                    "end_char": 5254,
                }
            ],
        )

        self.assertIn('"id": "E37"', prompts["user_prompt"])
        self.assertIn('"text": "acute blood loss anemia"', prompts["user_prompt"])
        self.assertNotIn("section_label", prompts["user_prompt"])
        self.assertNotIn("start_char", prompts["user_prompt"])
        self.assertNotIn("end_char", prompts["user_prompt"])

    def test_compact_evidence_index_for_prompt_merges_short_adjacent_items(self):
        compacted = compact_evidence_index_for_prompt(
            [
                {"id": "E1", "section": "hospital_course", "text": "Post op pain improved."},
                {"id": "E2", "section": "hospital_course", "text": "Ambulating with assistance."},
                {"id": "E3", "section": "hospital_course", "text": "98.6"},
                {"id": "E4", "section": "discharge_diagnosis", "text": "Pneumonia."},
            ]
        )

        self.assertEqual(
            compacted,
            [
                {"id": "E1", "text": "Post op pain improved. Ambulating with assistance."},
                {"id": "E4", "text": "Pneumonia."},
            ],
        )

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

    def test_agent2_includes_retrieved_knowledge_when_memory_exists(self):
        with workspace_tempdir() as tmpdir:
            db_path = Path(tmpdir) / "knowledge.sqlite3"
            knowledge_base = KnowledgeBase(db_path)
            knowledge_base.insert_memory(
                note_text="Chief Complaint: cough\nDischarge Diagnosis: pneumonia",
                structured_case_summary={
                    "chief_complaint": "cough",
                    "history_present_illness": "Cough and fever.",
                    "procedure": [],
                    "past_medical_history": [],
                    "pertinent_results": [],
                    "hospital_course": ["treated for pneumonia"],
                    "discharge_diagnosis": ["pneumonia"],
                },
                agent3_output={
                    "case_summary": "Pneumonia discharge diagnosis with infectious symptoms.",
                    "salient_clinical_patterns": ["cough and fever with discharge diagnosis of pneumonia"],
                    "correct_prediction_reasons": [],
                    "missed_code_lessons": [],
                    "unsupported_prediction_lessons": [],
                    "coding_lessons": ["When discharge diagnosis clearly states pneumonia, verify it as the principal condition."],
                    "retrieval_queries": ["pneumonia cough fever discharge diagnosis"],
                    "knowledge_summary": "Similar pneumonia cases often center on an explicit discharge diagnosis plus infectious symptoms.",
                },
                gold_codes=["J18.9"],
                predicted_codes=["J18.9"],
                missed_codes=[],
                extra_codes=[],
                source_case_id="2",
                coding_version="ICD-10",
            )

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
                patient_context={
                    "coding_version": "ICD-10",
                    "knowledge_base_path": str(db_path),
                    "knowledge_base_top_k": 1,
                },
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
        self.assertIn("Retrieved knowledge base memories", llm.calls[0]["user_prompt"])
        self.assertIn("When discharge diagnosis clearly states pneumonia", llm.calls[0]["user_prompt"])
        self.assertEqual(len(state.shared_memory["agent2_retrieved_knowledge"]), 1)

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

    def test_agent3_writes_knowledge_entry_to_sqlite(self):
        with workspace_tempdir() as tmpdir:
            db_path = Path(tmpdir) / "knowledge.sqlite3"
            llm = StubLLM(
                {
                    "case_summary": "Pneumonia discharge diagnosis supported by cough and fever.",
                    "salient_clinical_patterns": [
                        "explicit discharge diagnosis of pneumonia",
                        "infectious respiratory symptoms",
                    ],
                    "correct_prediction_reasons": [
                        "The principal diagnosis was correctly supported by the discharge diagnosis."
                    ],
                    "missed_code_lessons": [
                        "Review chronic comorbidities in the summary because hypertension can be easy to miss."
                    ],
                    "unsupported_prediction_lessons": [],
                    "coding_lessons": [
                        "When the note names pneumonia at discharge, prioritize it as the principal condition if no competing reason for admission is documented.",
                    ],
                    "retrieval_queries": [
                        "pneumonia discharge diagnosis cough fever",
                        "pneumonia with hypertension comorbidity",
                    ],
                    "knowledge_summary": "Cases with pneumonia at discharge and infectious respiratory symptoms should keep pneumonia front-of-mind during code selection.",
                }
            )
            agent = Agent3KnowledgeSynthesizer(llm=llm, knowledge_base_path=str(db_path))
            state = PipelineState(
                note_text="Chief Complaint: cough\nDischarge Diagnosis: pneumonia",
                patient_context={"coding_version": "ICD-10", "candidate_output_limit": 3},
                training_context={
                    "gold_labels": ["J18.9", "I10"],
                    "subject_id": "1",
                    "hadm_id": "2",
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
            state.shared_memory["note_evidence_index"] = build_evidence_index(state.note_text)
            state.agent_outputs["agent2"] = {
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

            result = agent.run(state)
            knowledge_base = KnowledgeBase(db_path)
            retrieved = knowledge_base.search(
                note_text=state.note_text,
                structured_case_summary=state.shared_memory["structured_case_summary"],
                coding_version="ICD-10",
                top_k=1,
            )
            entry_count = knowledge_base.count_entries()

            self.assertTrue(result["stored"])
            self.assertEqual(result["missed_codes"], ["I10"])
            self.assertEqual(entry_count, 1)
            self.assertEqual(retrieved[0]["gold_codes"], ["J18.9", "I10"])
            self.assertIn("pneumonia", retrieved[0]["knowledge_summary"].lower())

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

    def test_controller_can_run_agent3_when_requested(self):
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

        class StaticAgent3:
            def run(self, state):
                return {
                    "stored": True,
                    "knowledge_entry_id": 1,
                    "knowledge_summary": "demo memory",
                    "gold_codes": state.training_context.get("gold_labels", []),
                    "predicted_codes": ["J18.9"],
                }

        controller = MultiAgentController(
            agent1=StaticAgent1(),
            agent2=StaticAgent2(),
            agent3=StaticAgent3(),
        )
        state = controller.run(
            note_text="Discharge Diagnosis: pneumonia",
            requested_agents=["agent1", "agent2", "agent3"],
            training_context={"gold_labels": ["J18.9"]},
        )

        self.assertIn("agent3", state["agent_outputs"])
        self.assertTrue(state["agent_outputs"]["agent3"]["stored"])


if __name__ == "__main__":
    unittest.main()
