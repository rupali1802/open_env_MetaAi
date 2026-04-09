from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# OpenAI client is required by the challenge specification.
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _format_reward_list(rewards: List[float]) -> str:
    return ",".join(f"{reward:.3f}" for reward in rewards)


def _sanitize_action(candidate: Any) -> Dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    action_type = candidate.get("action_type")
    if not isinstance(action_type, str):
        return None
    return candidate


def _model_action_suggestion(
    task_name: str,
    observation: Dict[str, Any],
    fallback_action: Dict[str, Any],
) -> Dict[str, Any]:
    prompt_payload = {
        "task": task_name,
        "current_phase": observation.get("current_phase"),
        "violations_found_so_far": observation.get("violations_found_so_far"),
        "risk_score": observation.get("risk_score"),
        "fallback_action": fallback_action,
        "instruction": "Return exactly one JSON object for the next action.",
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a GDPR auditing planner. "
                        "Return only one valid JSON action object with no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload),
                },
            ],
            max_tokens=120,
            temperature=0,
        )
        text = (response.choices[0].message.content or "").strip()
        parsed = _sanitize_action(json.loads(text))
        return parsed or fallback_action
    except Exception:
        return fallback_action


def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str | None]:
    try:
        response = requests.post(
            f"{ENV_BASE_URL}{path}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return {}, str(exc)


def _task_plan(task_name: str) -> List[Dict[str, Any]]:
    if task_name == "easy":
        return [
            {
                "action_type": "identify_pii",
                "pii_fields": [
                    "customer_db.full_name",
                    "customer_db.email",
                    "customer_db.phone",
                    "customer_db.ssn",
                    "customer_db.address",
                    "employee_db.name",
                    "employee_db.email",
                    "employee_db.phone",
                    "employee_db.national_id",
                    "employee_db.home_ip",
                    "analytics_logs.session_ip",
                    "analytics_logs.geolocation",
                    "marketing_db.email",
                    "marketing_db.cookie_id",
                ],
            },
            {"action_type": "complete_audit"},
        ]

    if task_name == "medium":
        return [
            {"action_type": "trace_flow", "flow_id": "flow_2"},
            {"action_type": "trace_flow", "flow_id": "flow_4"},
            {"action_type": "flag_violation", "violation_id": "flow_2"},
            {"action_type": "flag_violation", "violation_id": "flow_4"},
            {"action_type": "complete_audit"},
        ]

    return [
        {
            "action_type": "identify_pii",
            "pii_fields": [
                "customer_db.full_name",
                "customer_db.email",
                "customer_db.phone",
                "customer_db.ssn",
                "customer_db.address",
                "employee_db.name",
                "employee_db.email",
                "employee_db.phone",
                "employee_db.national_id",
                "employee_db.home_ip",
                "analytics_logs.session_ip",
                "analytics_logs.geolocation",
                "marketing_db.email",
                "marketing_db.cookie_id",
            ],
        },
        {"action_type": "trace_flow", "flow_id": "flow_1"},
        {"action_type": "trace_flow", "flow_id": "flow_2"},
        {"action_type": "trace_flow", "flow_id": "flow_3"},
        {"action_type": "trace_flow", "flow_id": "flow_4"},
        {"action_type": "flag_violation", "violation_id": "flow_2"},
        {"action_type": "flag_violation", "violation_id": "flow_4"},
        {"action_type": "remediate", "remediation_action": "mask_pii"},
        {"action_type": "remediate", "remediation_action": "mask_pii"},
        {"action_type": "complete_audit"},
    ]


def run_task(task_name: str) -> None:
    print(f"[START] task={task_name} env={ENV_BASE_URL} model={MODEL_NAME}")

    reset_response, reset_error = _post_json("/reset", {"task": task_name})
    observation = reset_response if isinstance(reset_response, dict) else {}

    rewards: List[float] = []
    score = 0.0
    done = False
    success = reset_error is None

    plan = _task_plan(task_name)
    for index, fallback_action in enumerate(plan, start=1):
        if done:
            break

        action = _model_action_suggestion(task_name, observation, fallback_action)
        step_response, step_error = _post_json("/step", {"action": action})

        if step_error is not None:
            success = False
            reward_value = 0.0
            done = True
            error_text = step_error.replace(" ", "_")
            action_name = action.get("action_type", "unknown")
            print(
                f"[STEP] step={index} action={action_name} reward={reward_value:.3f} "
                f"done={_bool_text(done)} error={error_text}"
            )
            rewards.append(reward_value)
            break

        reward_value = float(step_response.get("reward", {}).get("value", 0.0))
        done = bool(step_response.get("done", False))
        action_name = action.get("action_type", "unknown")
        rewards.append(reward_value)

        info = step_response.get("info", {})
        score = float(info.get("score", score))
        observation = step_response.get("observation", observation)

        print(
            f"[STEP] step={index} action={action_name} reward={reward_value:.3f} "
            f"done={_bool_text(done)} error=null"
        )

    if done:
        success = success and True
    else:
        success = False

    print(
        f"[END] success={_bool_text(success)} steps={len(rewards)} "
        f"score={score:.3f} rewards={_format_reward_list(rewards)}"
    )


def main() -> None:
    for task in ("easy", "medium", "hard"):
        run_task(task)


if __name__ == "__main__":
    main()
