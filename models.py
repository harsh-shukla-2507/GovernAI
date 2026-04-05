"""GovernAI Pydantic models — OpenEnv-compatible Action, Observation, and State types."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """OpenEnv-compatible base action."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}


class Observation(BaseModel):
    """OpenEnv-compatible base observation."""

    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}


class State(BaseModel):
    """OpenEnv-compatible base state."""

    episode_id: Optional[str] = None
    step_count: int = 0
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}


class PolicyAction(str, Enum):
    """Discrete policy actions available to the governing agent."""

    INVEST_HEALTHCARE = "invest_healthcare"
    INVEST_EDUCATION = "invest_education"
    CUT_TAXES = "cut_taxes"
    RAISE_TAXES = "raise_taxes"
    GREEN_SUBSIDY = "green_subsidy"
    INDUSTRIAL_EXPANSION = "industrial_expansion"
    WELFARE_SUPPORT = "welfare_support"
    INFRASTRUCTURE_PUSH = "infrastructure_push"
    DO_NOTHING = "do_nothing"


class GovernAIAction(Action):
    """Policy action chosen by the governing agent each month."""

    policy: PolicyAction = Field(..., description="The policy action to enact this month")
    reasoning: str = Field(default="", description="Explanation for the policy decision")


class GovernAIObservation(Observation):
    """City status observation returned after each policy action."""

    economy: float = Field(default=50.0, description="Economic health (0-100, higher is better)")
    health: float = Field(default=50.0, description="Public health (0-100, higher is better)")
    education: float = Field(default=50.0, description="Education quality (0-100, higher is better)")
    pollution: float = Field(default=40.0, description="Pollution level (0-100, LOWER is better)")
    happiness: float = Field(default=50.0, description="Citizen happiness (0-100, higher is better)")
    inequality: float = Field(default=40.0, description="Income inequality (0-100, LOWER is better)")
    budget: float = Field(default=50.0, description="Government budget (0-100, higher is better)")
    unemployment: float = Field(default=30.0, description="Unemployment rate (0-100, LOWER is better)")
    month: int = Field(default=0, description="Current month in the governance term")
    max_months: int = Field(default=24, description="Total months in this governance term")
    narrative: str = Field(default="", description="Human-readable city status report")
    active_events: List[str] = Field(default_factory=list, description="Currently active crisis events")
    available_actions: List[str] = Field(default_factory=list, description="Available policy actions")
    task_id: str = Field(default="stable_city", description="Current task identifier")
