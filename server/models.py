from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Phase = Literal["identify_pii", "trace_flows", "flag_violations", "remediate"]
ActionType = Literal[
    "identify_pii",
    "trace_flow",
    "flag_violation",
    "remediate",
    "complete_audit",
    "request_hint",
]
TaskType = Literal["easy", "medium", "hard"]


class DatabaseSchema(BaseModel):
    columns: List[str]
    pii_columns: List[str]


class DataFlow(BaseModel):
    flow_id: str
    source_db: str
    target_system: str
    purpose: str
    authorized: bool
    pii_involved: bool


class Observation(BaseModel):
    databases: Dict[str, DatabaseSchema]
    data_flows: List[DataFlow]
    current_phase: Phase
    violations_found_so_far: int = Field(ge=0)
    risk_score: float = Field(ge=0.0, le=1.0)
    step_count: int = Field(ge=0)
    task: TaskType


class Action(BaseModel):
    action_type: ActionType
    pii_fields: Optional[List[str]] = None
    flow_id: Optional[str] = None
    violation_id: Optional[str] = None
    remediation_action: Optional[str] = None
    field_name: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "Action":
        required_fields = {
            "identify_pii": ("pii_fields",),
            "trace_flow": ("flow_id",),
            "flag_violation": ("violation_id",),
            "remediate": ("remediation_action",),
            "request_hint": ("field_name",),
        }

        for field_name in required_fields.get(self.action_type, ()): 
            value = getattr(self, field_name)
            if value is None:
                raise ValueError(
                    f"{field_name} is required for action_type={self.action_type}"
                )
            if isinstance(value, list) and not value:
                raise ValueError(
                    f"{field_name} must be non-empty for action_type={self.action_type}"
                )

        return self


class Reward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""
