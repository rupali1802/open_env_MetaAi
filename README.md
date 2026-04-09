---
title: AI Data Privacy Impact Assessor
emoji: "🛡️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI Data Privacy Impact Assessor

## Project Overview
AI Data Privacy Impact Assessor is a production-ready OpenEnv environment for training AI agents to:
- detect personally identifiable information (PII),
- trace data flows across systems,
- identify GDPR violations,
- apply safe remediation actions.

The environment includes deterministic task graders, a FastAPI runtime, strict stepwise rewards, and a portable Docker deployment path for Hugging Face Spaces.

## OpenEnv Compliance
This project implements the required OpenEnv contract:
- `reset()`
- `step(action)`
- `state()`
- `step()` returns `(observation, reward, done, info)`
- Pydantic models for `Observation`, `Action`, and `Reward`

`openenv.yaml` points to:
- `main: server/app:app`

## Observation Space
Each observation includes:
- 4 databases:
  - `customer_db`
  - `employee_db`
  - `analytics_logs`
  - `marketing_db`
- realistic columns including PII (`email`, `phone`, `ssn`, `ip`, etc.)
- 4 data flows with mixed authorization states
- `current_phase` in:
  - `identify_pii`
  - `trace_flows`
  - `flag_violations`
  - `remediate`
- `violations_found_so_far`
- `risk_score` in range `[0.0, 1.0]`
- `step_count`

## Action Space
Supported actions (strict schema):

1. Identify PII
```json
{"action_type":"identify_pii","pii_fields":["customer_db.email","customer_db.ssn"]}
```

2. Trace Flow
```json
{"action_type":"trace_flow","flow_id":"flow_1"}
```

3. Flag Violation
```json
{"action_type":"flag_violation","violation_id":"flow_1"}
```

4. Remediate
```json
{"action_type":"remediate","remediation_action":"mask_pii"}
```

5. Complete Audit
```json
{"action_type":"complete_audit"}
```

6. Request Hint
```json
{"action_type":"request_hint","field_name":"ssn"}
```

## Tasks and Graders
Three deterministic tasks are implemented:

1. EASY
- Objective: Identify PII fields.
- Scoring: `correct / total`.
- Grader: `easy_grader()`

2. MEDIUM
- Objective: Detect violating data flows.
- Scoring: F1 score from precision and recall.
- Grader: `medium_grader()`

3. HARD
- Objective: Perform full audit (`identify_pii -> trace_flows -> flag_violations -> remediate`).
- Scoring: Weighted composite score (PII + violations + integrity).
- Grader: `hard_grader()`

All graders return scores in `[0.0, 1.0]`.

## Reward Function
Incremental step reward is always returned.

Positive rewards:
- `+0.15` per correct PII
- `+0.20` per correct violation
- `+0.25` per correct remediation
- `+0.50` for complete audit

Penalties:
- `-0.05` false PII
- `-0.15` false violation
- `-0.10` wrong remediation
- `-0.25` destructive action
- `-0.05` repeated actions (loop detection)

## API Endpoints
FastAPI endpoints:
- `GET /` -> `{"status":"ok"}`
- `POST /reset` -> observation
- `POST /step` -> observation, reward, done, info
- `GET /state` -> current state

Default runtime port:
- `7860`

## Local Setup (VS Code)
1. Create environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

3. Configure env vars:
```bash
copy .env.example .env
```

Required variables in `.env`:
- `ENV_BASE_URL` for the OpenEnv runtime (`/reset` and `/step`), default `http://127.0.0.1:7860`
- `API_BASE_URL` for the injected LiteLLM proxy
- `API_KEY` for the injected LiteLLM key
- `MODEL_NAME` (optional, defaults to `gpt-4o-mini`)

4. Run inference benchmark:
```bash
python inference.py
```

## Inference Logging Contract
`inference.py` prints strict logs only:
- `[START] task=... env=... model=...`
- `[STEP] step=X action=Y reward=Z done=true/false error=null`
- `[END] success=true/false steps=N score=0.XXX rewards=a,b,c`

No extra output is emitted.

## Docker Instructions
Build:
```bash
docker build -t privacy-compliance-env .
```

Run:
```bash
docker run --rm -p 7860:7860 privacy-compliance-env
```

Docker guarantees:
- base image `python:3.11-slim`
- non-root runtime user
- exposed port `7860`

## Hugging Face Spaces Deployment
1. Create a new Space with Docker SDK.
2. Push this repository content to the Space.
3. In Space variables, set:
- `ENV_BASE_URL` (your OpenEnv service URL, if not local default)
- `API_BASE_URL` (LiteLLM proxy URL provided by the evaluator)
- `API_KEY` (LiteLLM proxy API key provided by the evaluator)
- `MODEL_NAME`
4. Ensure the service listens on port `7860`.
5. Deploy and verify:
- `GET /` returns `{"status":"ok"}`
- run inference workflow for all three tasks.

## Baseline Scores
Deterministic baseline policy in `inference.py` targets:
- EASY: `1.000`
- MEDIUM: `1.000`
- HARD: `1.000`

## Validation Checklist
- `openenv validate` passes with `openenv.yaml`
- Docker image builds successfully
- `inference.py` runs all 3 tasks without runtime failures
- all graders output values in `[0.0, 1.0]`
- full run completes well under 20 minutes
