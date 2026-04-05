# Multi-Agent for ICD Prediction

This repository currently contains the first agent described in the proposal:

- `agent1`: primary analyzer for raw clinical text

## What Agent 1 does

Agent 1 does not predict ICD codes directly. It prepares a coding-ready intermediate representation by:

- extracting coding-relevant evidence from a raw note
- organizing the note into a structured case summary
- surfacing likely primary diagnoses and active conditions
- preserving evidence references for downstream agents
- calling out coding modifiers such as acuity, laterality, and causal language
- flagging missing information that may matter for final coding

## Project structure

```text
multi_agent_icd/
  agents/
    agent1/
      agent.py
      prompt.py
    agent2/
      agent.py
    agent3/
      agent.py
  run.py
  providers/
    openai_responses.py
  utils/clinical_text.py
  utils/schema.py
  cli/run_agent1.py
```

## Multi-agent controller

Besides `agent1`, the package now includes a master controller in `multi_agent_icd/run.py`.

It is responsible for:

- registering available agents
- defining the pipeline order
- passing shared state between agents
- collecting each agent's output
- exposing one final pipeline result

Default order:

```text
agent1 -> agent2 -> agent3
```

Right now:

- `agent1` is implemented
- `agent2` and `agent3` are scaffolded as pending placeholders

## About OpenAI keys

The current `agent1` implementation is now designed to call an OpenAI large model directly.

Required inputs:

- `OPENAI_API_KEY` in your environment
- a model name passed explicitly from the CLI or controller

`agent1` no longer uses the old local heuristic fallback as its default execution path.

Example:

```python
from multi_agent_icd import MultiAgentController

controller = MultiAgentController(agent_models={"agent1": "gpt-5"})
state = controller.run(note_text="...", patient_context={})

print(state["agent_outputs"]["agent1"])
print(state["execution_trace"])
```

## Run Agent 1

```bash
python -m multi_agent_icd.cli.run_agent1 path/to/note.txt --model gpt-5
```

Optional patient context:

```bash
python -m multi_agent_icd.cli.run_agent1 path/to/note.txt --model gpt-5 --context path/to/patient-context.json
```

Example patient context:

```json
{
  "age": "67",
  "sex": "female",
  "encounter_type": "admission",
  "care_setting": "inpatient"
}
```

## Model configuration

`Agent1PrimaryAnalyzer` requires a model name unless you inject a custom `llm` object.

Expected contract:

```python
class ExampleLLM:
    def generate_json(self, system_prompt: str, user_prompt: str, metadata: dict):
        return {
            "patient_snapshot": {},
            "note_summary": {},
            "primary_diagnosis_candidates": []
        }
```

If you use the built-in OpenAI client, set `OPENAI_API_KEY` and pass a model name such as `gpt-5`.

## Suggested next steps

- connect `agent2` to `primary_diagnosis_candidates`, `active_conditions`, and `evidence_index`
- let `agent3` critique unsupported diagnoses, missing modifiers, and evidence gaps
- add model configuration for `agent2` and `agent3` once they are implemented
