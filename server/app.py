from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from fastapi import Body, FastAPI
from pydantic import BaseModel

from server.models import (
    Action,
    DataFlow,
    DatabaseSchema,
    Observation,
    Phase,
    Reward,
    TaskType,
)
from tasks.graders import easy_grader, hard_grader, medium_grader


class ResetRequest(BaseModel):
    task: TaskType = "easy"


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class PrivacyComplianceEnv:
    def __init__(self) -> None:
        self.max_steps = 30
        self._databases = {
            "customer_db": DatabaseSchema(
                columns=[
                    "customer_id",
                    "full_name",
                    "email",
                    "phone",
                    "ssn",
                    "address",
                    "consent_flag",
                ],
                pii_columns=["full_name", "email", "phone", "ssn", "address"],
            ),
            "employee_db": DatabaseSchema(
                columns=[
                    "employee_id",
                    "name",
                    "email",
                    "phone",
                    "salary",
                    "national_id",
                    "home_ip",
                ],
                pii_columns=["name", "email", "phone", "national_id", "home_ip"],
            ),
            "analytics_logs": DatabaseSchema(
                columns=[
                    "event_id",
                    "user_id",
                    "session_ip",
                    "device_id",
                    "referrer",
                    "geolocation",
                ],
                pii_columns=["session_ip", "geolocation"],
            ),
            "marketing_db": DatabaseSchema(
                columns=[
                    "lead_id",
                    "email",
                    "campaign_id",
                    "cookie_id",
                    "ad_id",
                    "inferred_income",
                ],
                pii_columns=["email", "cookie_id"],
            ),
        }

        self._flows = [
            DataFlow(
                flow_id="flow_1",
                source_db="customer_db",
                target_system="billing_service",
                purpose="payment_reconciliation",
                authorized=True,
                pii_involved=True,
            ),
            DataFlow(
                flow_id="flow_2",
                source_db="employee_db",
                target_system="ad_partner",
                purpose="behavioral_ad_targeting",
                authorized=False,
                pii_involved=True,
            ),
            DataFlow(
                flow_id="flow_3",
                source_db="analytics_logs",
                target_system="security_team",
                purpose="fraud_monitoring",
                authorized=True,
                pii_involved=True,
            ),
            DataFlow(
                flow_id="flow_4",
                source_db="customer_db",
                target_system="marketing_db",
                purpose="cross_sell_without_consent",
                authorized=False,
                pii_involved=True,
            ),
        ]

        self.expected_pii: Set[str] = {
            self._normalize_field(f"{db_name}.{column}")
            for db_name, schema in self._databases.items()
            for column in schema.pii_columns
        }
        self.flow_ids = {flow.flow_id for flow in self._flows}
        self.violating_flows = {
            flow.flow_id for flow in self._flows if (not flow.authorized and flow.pii_involved)
        }

        self.allowed_remediations = {"mask_pii", "tokenize_sensitive_fields", "block_transfer"}
        self.destructive_actions = {
            "delete_raw_records",
            "drop_table",
            "export_unencrypted_dump",
            "share_with_third_party",
        }

        self.current_task: TaskType = "easy"
        self.current_phase: Phase = "identify_pii"
        self.step_count = 0
        self.risk_score = 1.0
        self.done = False
        self.audit_completed = False

        self.correctly_identified_pii: Set[str] = set()
        self.false_pii: Set[str] = set()
        self.traced_flows: Set[str] = set()
        self.correctly_flagged_violations: Set[str] = set()
        self.false_flagged_violations: Set[str] = set()
        self.remediated_violations: Set[str] = set()
        self.action_history: List[str] = []

        self.reset(task="easy")

    @staticmethod
    def _normalize_field(field_name: str) -> str:
        return field_name.strip().lower()

    def _build_observation(self) -> Observation:
        return Observation(
            databases=self._databases,
            data_flows=self._flows,
            current_phase=self.current_phase,
            violations_found_so_far=len(self.correctly_flagged_violations),
            risk_score=self.risk_score,
            step_count=self.step_count,
            task=self.current_task,
        )

    def _current_score(self) -> float:
        if self.current_task == "easy":
            return easy_grader(self.correctly_identified_pii, self.expected_pii)

        if self.current_task == "medium":
            return medium_grader(self.correctly_flagged_violations, self.violating_flows)

        return hard_grader(
            identified_pii=self.correctly_identified_pii,
            expected_pii=self.expected_pii,
            flagged_violations=self.correctly_flagged_violations,
            expected_violations=self.violating_flows,
            correct_remediations=len(self.remediated_violations),
            required_remediations=len(self.violating_flows),
            audit_completed=self.audit_completed,
        )

    def _compute_risk_score(self) -> float:
        unidentified_ratio = (
            len(self.expected_pii - self.correctly_identified_pii) / len(self.expected_pii)
            if self.expected_pii
            else 0.0
        )
        unflagged_ratio = (
            len(self.violating_flows - self.correctly_flagged_violations)
            / len(self.violating_flows)
            if self.violating_flows
            else 0.0
        )
        unremediated_ratio = (
            len(self.violating_flows - self.remediated_violations) / len(self.violating_flows)
            if self.violating_flows
            else 0.0
        )
        risk = (0.40 * unidentified_ratio) + (0.35 * unflagged_ratio) + (0.25 * unremediated_ratio)
        return round(min(1.0, max(0.0, risk)), 3)

    def _update_phase(self) -> None:
        if self.correctly_identified_pii and self.current_phase == "identify_pii":
            self.current_phase = "trace_flows"
        if self.traced_flows and self.current_phase == "trace_flows":
            self.current_phase = "flag_violations"
        if self.correctly_flagged_violations and self.current_phase == "flag_violations":
            self.current_phase = "remediate"

    def _action_fingerprint(self, action: Action) -> str:
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
            "pii_fields": sorted(action.pii_fields or []),
            "flow_id": action.flow_id,
            "violation_id": action.violation_id,
            "remediation_action": action.remediation_action,
            "field_name": action.field_name,
        }
        return str(payload)

    @staticmethod
    def _add_component(components: Dict[str, float], key: str, delta: float) -> None:
        components[key] = round(components.get(key, 0.0) + delta, 3)

    def _can_complete_audit(self) -> bool:
        if self.current_task == "easy":
            return len(self.correctly_identified_pii) > 0

        if self.current_task == "medium":
            return len(self.correctly_flagged_violations) > 0

        return (
            len(self.correctly_identified_pii) > 0
            and self.violating_flows.issubset(self.correctly_flagged_violations)
            and len(self.remediated_violations) >= len(self.violating_flows)
        )

    def _hint_for(self, field_name: str) -> str:
        normalized = self._normalize_field(field_name)
        if "ssn" in normalized:
            return "SSN is a high-risk direct identifier; mask or tokenize before storage and transfer."
        if "ip" in normalized:
            return "IP addresses are personal data under GDPR; restrict retention and enforce minimization."
        if "email" in normalized:
            return "Email is personally identifiable and should be protected in transit and at rest."
        return "Review lawful basis, data minimization, and purpose limitation for this field."

    def reset(self, task: TaskType = "easy") -> Observation:
        self.current_task = task
        self.current_phase = "identify_pii"
        self.step_count = 0
        self.risk_score = 1.0
        self.done = False
        self.audit_completed = False

        self.correctly_identified_pii.clear()
        self.false_pii.clear()
        self.traced_flows.clear()
        self.correctly_flagged_violations.clear()
        self.false_flagged_violations.clear()
        self.remediated_violations.clear()
        self.action_history.clear()

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            observation = self._build_observation()
            reward = Reward(value=0.0, components={}, reason="episode_already_done")
            info = {
                "task": self.current_task,
                "score": self._current_score(),
                "message": "Episode has ended. Call reset() to start a new task.",
            }
            return observation, reward, True, info

        self.step_count += 1
        total_reward = 0.0
        components: Dict[str, float] = {}
        info: Dict[str, Any] = {"task": self.current_task}

        fingerprint = self._action_fingerprint(action)
        if fingerprint in self.action_history[-4:]:
            total_reward -= 0.05
            self._add_component(components, "loop_penalty", -0.05)
        self.action_history.append(fingerprint)

        if action.action_type == "identify_pii":
            for pii_field in sorted(set(action.pii_fields or [])):
                normalized_field = self._normalize_field(pii_field)
                if normalized_field in self.expected_pii and normalized_field not in self.correctly_identified_pii:
                    self.correctly_identified_pii.add(normalized_field)
                    total_reward += 0.15
                    self._add_component(components, "correct_pii", 0.15)
                elif normalized_field not in self.expected_pii:
                    self.false_pii.add(normalized_field)
                    total_reward -= 0.05
                    self._add_component(components, "false_pii", -0.05)

        elif action.action_type == "trace_flow":
            if action.flow_id in self.flow_ids:
                self.traced_flows.add(action.flow_id)
            else:
                info["trace_warning"] = f"Unknown flow_id: {action.flow_id}"

        elif action.action_type == "flag_violation":
            violation_id = action.violation_id
            if violation_id in self.violating_flows and violation_id not in self.correctly_flagged_violations:
                self.correctly_flagged_violations.add(violation_id)
                total_reward += 0.20
                self._add_component(components, "correct_violation", 0.20)
            else:
                if violation_id:
                    self.false_flagged_violations.add(violation_id)
                total_reward -= 0.15
                self._add_component(components, "false_violation", -0.15)

        elif action.action_type == "remediate":
            remediation_action = (action.remediation_action or "").strip().lower()
            unresolved_flagged = sorted(
                self.correctly_flagged_violations - self.remediated_violations
            )

            if remediation_action in self.destructive_actions or any(
                keyword in remediation_action for keyword in ("delete", "drop", "erase")
            ):
                total_reward -= 0.25
                self._add_component(components, "destructive_action", -0.25)
            elif remediation_action in self.allowed_remediations and unresolved_flagged:
                self.remediated_violations.add(unresolved_flagged[0])
                total_reward += 0.25
                self._add_component(components, "correct_remediation", 0.25)
            else:
                total_reward -= 0.10
                self._add_component(components, "wrong_remediation", -0.10)

        elif action.action_type == "request_hint":
            info["hint"] = self._hint_for(action.field_name or "")

        elif action.action_type == "complete_audit":
            if self._can_complete_audit():
                self.audit_completed = True
                self.done = True
                total_reward += 0.50
                self._add_component(components, "complete_audit", 0.50)
            else:
                info["completion_warning"] = "Audit is incomplete for the current task."

        self._update_phase()
        self.risk_score = self._compute_risk_score()

        if self.step_count >= self.max_steps:
            self.done = True
            info["termination"] = "max_steps_reached"

        score = self._current_score()
        info["score"] = score
        info["phase"] = self.current_phase
        info["violations_found"] = len(self.correctly_flagged_violations)

        observation = self._build_observation()
        reward = Reward(
            value=round(total_reward, 3),
            components=components,
            reason="incremental_step_reward",
        )

        return observation, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        return {
            "observation": self._build_observation().model_dump(),
            "task": self.current_task,
            "done": self.done,
            "audit_completed": self.audit_completed,
            "score": self._current_score(),
            "correctly_identified_pii": sorted(self.correctly_identified_pii),
            "traced_flows": sorted(self.traced_flows),
            "correctly_flagged_violations": sorted(self.correctly_flagged_violations),
            "remediated_violations": sorted(self.remediated_violations),
        }


environment = PrivacyComplianceEnv()


def reset() -> Observation:
    return environment.reset(task=environment.current_task)


def step(action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
    return environment.step(action)


def state() -> Dict[str, Any]:
    return environment.state()


app = FastAPI(title="AI Data Privacy Impact Assessor", version="1.0.0")


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset_endpoint(payload: ResetRequest | None = Body(default=None)) -> Observation:
    task = payload.task if payload else environment.current_task
    return environment.reset(task=task)


@app.post("/step", response_model=StepResponse)
def step_endpoint(payload: StepRequest | Action = Body(...)) -> StepResponse:
    action = payload.action if isinstance(payload, StepRequest) else payload
    observation, reward, done, info = environment.step(action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state")
def state_endpoint() -> Dict[str, Any]:
    return environment.state()


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
