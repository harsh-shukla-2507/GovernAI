"""GovernAI Environment Engine — city governance simulation with multi-objective rewards."""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from models import GovernAIAction, GovernAIObservation, PolicyAction, State

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "economy", "health", "education", "pollution",
    "happiness", "inequality", "budget", "unemployment",
]
POSITIVE_METRICS = ["economy", "health", "education", "happiness", "budget"]
NEGATIVE_METRICS = ["pollution", "inequality", "unemployment"]

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "stable_city": {
        "description": "Govern a stable city for 12 months. No crises — focus on balance.",
        "difficulty": "easy",
        "max_months": 12,
        "initial_state": {
            "economy": 50, "health": 50, "education": 50,
            "pollution": 40, "happiness": 50, "inequality": 40,
            "budget": 50, "unemployment": 30,
        },
        "enable_random_events": False,
        "preset_events": {},
    },
    "austerity_challenge": {
        "description": (
            "Recover a city from economic crisis over 18 months. "
            "Budget is critically low and unemployment is high."
        ),
        "difficulty": "medium",
        "max_months": 18,
        "initial_state": {
            "economy": 35, "health": 45, "education": 40,
            "pollution": 50, "happiness": 35, "inequality": 65,
            "budget": 15, "unemployment": 70,
        },
        "enable_random_events": False,
        "preset_events": {6: "energy_crisis", 12: "protests"},
    },
    "crisis_governance": {
        "description": (
            "Navigate 24 months of governance with random crises including "
            "pandemics, floods, and recessions."
        ),
        "difficulty": "hard",
        "max_months": 24,
        "initial_state": {
            "economy": 50, "health": 50, "education": 50,
            "pollution": 45, "happiness": 50, "inequality": 45,
            "budget": 45, "unemployment": 35,
        },
        "enable_random_events": True,
        "event_probability": 0.15,
        "preset_events": {},
    },
}

# Each action's immediate and delayed (turns-later) effects on metrics.
ACTION_EFFECTS: Dict[PolicyAction, Dict[str, Any]] = {
    PolicyAction.INVEST_HEALTHCARE: {
        "immediate": {"health": 8, "happiness": 3, "budget": -8, "economy": -1},
        "delayed": [{"turns": 2, "effects": {"happiness": 2}}],
    },
    PolicyAction.INVEST_EDUCATION: {
        "immediate": {"education": 8, "happiness": 2, "budget": -7, "inequality": -2},
        "delayed": [{"turns": 3, "effects": {"economy": 3, "unemployment": -2}}],
    },
    PolicyAction.CUT_TAXES: {
        "immediate": {"economy": 5, "happiness": 4, "budget": -6},
        "delayed": [{"turns": 3, "effects": {"inequality": 3}}],
    },
    PolicyAction.RAISE_TAXES: {
        "immediate": {"economy": -2, "happiness": -3, "budget": 8, "inequality": -3},
        "delayed": [],
    },
    PolicyAction.GREEN_SUBSIDY: {
        "immediate": {"pollution": -8, "budget": -6, "economy": -2, "happiness": 1},
        "delayed": [{"turns": 2, "effects": {"health": 3}}],
    },
    PolicyAction.INDUSTRIAL_EXPANSION: {
        "immediate": {"economy": 7, "unemployment": -5, "pollution": 6, "happiness": 1},
        "delayed": [{"turns": 2, "effects": {"pollution": 4, "health": -2}}],
    },
    PolicyAction.WELFARE_SUPPORT: {
        "immediate": {"happiness": 4, "inequality": -5, "budget": -5, "unemployment": -2},
        "delayed": [],
    },
    PolicyAction.INFRASTRUCTURE_PUSH: {
        "immediate": {"economy": 3, "unemployment": -4, "budget": -7, "education": 1, "happiness": 1},
        "delayed": [{"turns": 2, "effects": {"economy": 2, "happiness": 1}}],
    },
    PolicyAction.DO_NOTHING: {
        "immediate": {},
        "delayed": [],
    },
}

CRISIS_EVENTS: Dict[str, Dict[str, Any]] = {
    "pandemic": {
        "effects": {"health": -15, "economy": -8, "happiness": -5, "budget": -5},
        "description": "A pandemic has swept through the city. Hospitals are overwhelmed.",
        "duration": 2,
    },
    "flood": {
        "effects": {"economy": -6, "health": -4, "budget": -8, "happiness": -3},
        "description": "Severe flooding has damaged infrastructure and displaced citizens.",
        "duration": 1,
    },
    "recession": {
        "effects": {"economy": -12, "unemployment": 10, "happiness": -6, "budget": -4},
        "description": "A recession has hit. Businesses are closing and unemployment is rising.",
        "duration": 3,
    },
    "energy_crisis": {
        "effects": {"economy": -5, "pollution": 5, "budget": -5, "happiness": -4},
        "description": "An energy crisis has struck. Fuel prices are soaring.",
        "duration": 2,
    },
    "protests": {
        "effects": {"happiness": -8, "economy": -3},
        "description": "Mass protests have erupted due to public dissatisfaction.",
        "duration": 1,
    },
}


class GovernAIEnvironment:
    """City governance simulation environment.

    Implements the OpenEnv interface: reset(), step(), state.
    """

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._metrics: Dict[str, float] = {}
        self._initial_metrics: Dict[str, float] = {}
        self._task_id = "stable_city"
        self._task_config: Dict[str, Any] = TASK_CONFIGS["stable_city"]
        self._delayed_effects: List[Dict[str, Any]] = []
        self._active_events: Dict[str, int] = {}
        self._event_cooldown = 0
        self._rng = random.Random()

        # Tracking for graders
        self._min_metrics: Dict[str, float] = {}
        self._max_metrics: Dict[str, float] = {}
        self._metric_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GovernAIObservation:
        task_id = kwargs.get("task_id")
        if task_id is None and episode_id and episode_id in TASK_CONFIGS:
            task_id = episode_id
        if task_id is None or task_id not in TASK_CONFIGS:
            task_id = "stable_city"

        self._task_id = task_id
        self._task_config = TASK_CONFIGS[task_id]
        self._rng = random.Random(seed) if seed is not None else random.Random()

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        self._metrics = {k: float(v) for k, v in self._task_config["initial_state"].items()}
        self._initial_metrics = dict(self._metrics)
        self._delayed_effects = []
        self._active_events = {}
        self._event_cooldown = 0

        self._min_metrics = dict(self._metrics)
        self._max_metrics = dict(self._metrics)
        self._metric_history = [dict(self._metrics)]

        return self._make_observation()

    def step(
        self,
        action: GovernAIAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GovernAIObservation:
        policy = action.policy
        effects = ACTION_EFFECTS.get(policy, ACTION_EFFECTS[PolicyAction.DO_NOTHING])

        # 1. Immediate effects
        for key, value in effects["immediate"].items():
            self._metrics[key] = self._metrics.get(key, 50) + value

        # 2. Queue delayed effects
        for d in effects["delayed"]:
            self._delayed_effects.append(
                {"turns_remaining": d["turns"], "effects": dict(d["effects"])}
            )

        # 3. Process delayed effects
        still_pending: List[Dict[str, Any]] = []
        for eff in self._delayed_effects:
            eff["turns_remaining"] -= 1
            if eff["turns_remaining"] <= 0:
                for key, value in eff["effects"].items():
                    self._metrics[key] = self._metrics.get(key, 50) + value
            else:
                still_pending.append(eff)
        self._delayed_effects = still_pending

        # 4. Background dynamics
        self._apply_background_dynamics()

        # 5. Ongoing stress from active crises
        for _ in self._active_events:
            self._metrics["happiness"] = max(0, self._metrics["happiness"] - 1)

        # 6. Tick active event durations
        self._tick_events()

        # 7. Preset events
        month = self._state.step_count + 1
        preset = self._task_config.get("preset_events", {})
        if month in preset:
            event_name = preset[month]
            if event_name in CRISIS_EVENTS and event_name not in self._active_events:
                self._trigger_event(event_name)

        # 8. Random events
        if self._task_config.get("enable_random_events", False):
            prob = self._task_config.get("event_probability", 0.15)
            if self._event_cooldown <= 0 and self._rng.random() < prob:
                candidates = [e for e in CRISIS_EVENTS if e not in self._active_events]
                if candidates:
                    self._trigger_event(self._rng.choice(candidates))
                    self._event_cooldown = 3
        if self._event_cooldown > 0:
            self._event_cooldown -= 1

        # 9. Clamp
        self._clamp_metrics()

        # 10. Advance step counter
        self._state.step_count += 1

        # 11. Track history
        self._record_metrics()

        return self._make_observation()

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_id=self._task_id,
            **self._metrics,
        )

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _apply_background_dynamics(self) -> None:
        m = self._metrics

        if m["economy"] > 60:
            m["pollution"] += (m["economy"] - 60) * 0.03

        if m["inequality"] > 50:
            m["happiness"] -= (m["inequality"] - 50) * 0.02

        if m["pollution"] > 50:
            m["health"] -= (m["pollution"] - 50) * 0.02

        # Strong economy recovers budget
        m["budget"] += (m["economy"] - 50) * 0.04

        if m["economy"] < 40:
            m["unemployment"] += (40 - m["economy"]) * 0.03

        if m["unemployment"] > 40:
            m["happiness"] -= (m["unemployment"] - 40) * 0.02

        # Natural decay without active investment
        m["education"] -= 0.3
        m["health"] -= 0.2
        m["budget"] -= 0.5

    def _trigger_event(self, event_name: str) -> None:
        event = CRISIS_EVENTS[event_name]
        for key, value in event["effects"].items():
            self._metrics[key] = self._metrics.get(key, 50) + value
        self._active_events[event_name] = event["duration"]

    def _tick_events(self) -> None:
        expired = []
        for name in self._active_events:
            self._active_events[name] -= 1
            if self._active_events[name] <= 0:
                expired.append(name)
        for name in expired:
            del self._active_events[name]

    def _clamp_metrics(self) -> None:
        for key in METRIC_KEYS:
            self._metrics[key] = max(0.0, min(100.0, self._metrics[key]))

    def _record_metrics(self) -> None:
        snapshot = dict(self._metrics)
        self._metric_history.append(snapshot)
        for key in METRIC_KEYS:
            val = self._metrics[key]
            self._min_metrics[key] = min(self._min_metrics.get(key, val), val)
            self._max_metrics[key] = max(self._max_metrics.get(key, val), val)

    # ------------------------------------------------------------------
    # Reward & grading
    # ------------------------------------------------------------------

    def _compute_step_reward(self) -> float:
        m = self._metrics
        reward = (
            0.20 * m["economy"] / 100
            + 0.20 * m["health"] / 100
            + 0.15 * m["happiness"] / 100
            + 0.10 * m["education"] / 100
            - 0.15 * m["pollution"] / 100
            - 0.10 * m["inequality"] / 100
            - 0.10 * m["unemployment"] / 100
            + 0.05 * m["budget"] / 100
        )
        if m["budget"] < 10:
            reward -= 0.15
        if m["pollution"] > 80:
            reward -= 0.15
        if m["happiness"] < 20:
            reward -= 0.15
        if m["health"] < 15:
            reward -= 0.10
        return round(max(-1.0, min(1.0, reward)), 4)

    def _compute_grader_score(self) -> float:
        dispatch = {
            "stable_city": self._grade_stable_city,
            "austerity_challenge": self._grade_austerity,
            "crisis_governance": self._grade_crisis,
        }
        fn = dispatch.get(self._task_id, self._grade_stable_city)
        raw = round(fn(), 4)
        return max(0.01, min(0.99, raw))

    def _quality_score(self) -> float:
        """Weighted quality score of current metrics — 0.0 to 1.0."""
        m = self._metrics
        score = (
            m["economy"] / 100 * 0.20
            + m["health"] / 100 * 0.20
            + m["happiness"] / 100 * 0.15
            + m["education"] / 100 * 0.10
            + (100 - m["pollution"]) / 100 * 0.15
            + (100 - m["inequality"]) / 100 * 0.10
            + (100 - m["unemployment"]) / 100 * 0.10
        )
        return max(0.0, min(1.0, score))

    def _grade_stable_city(self) -> float:
        return self._quality_score()

    def _grade_austerity(self) -> float:
        final_quality = self._quality_score()

        improvements = 0.0
        count = 0
        for key in POSITIVE_METRICS:
            diff = self._metrics[key] - self._initial_metrics[key]
            improvements += max(0.0, diff) / 100
            count += 1
        for key in NEGATIVE_METRICS:
            diff = self._initial_metrics[key] - self._metrics[key]
            improvements += max(0.0, diff) / 100
            count += 1
        improvement_score = improvements / count if count else 0.0

        survival = 1.0 if self._metrics["budget"] > 5 else 0.3

        score = 0.40 * improvement_score + 0.40 * final_quality + 0.20 * survival
        return max(0.0, min(1.0, score))

    def _grade_crisis(self) -> float:
        final_quality = self._quality_score()

        resilience = 1.0
        for key in POSITIVE_METRICS:
            if self._min_metrics.get(key, 100) < 10:
                resilience -= 0.12
        for key in NEGATIVE_METRICS:
            if self._max_metrics.get(key, 0) > 90:
                resilience -= 0.08
        resilience = max(0.0, resilience)

        if len(self._metric_history) > 1:
            variances = []
            for key in METRIC_KEYS:
                vals = [h[key] for h in self._metric_history]
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                variances.append(var)
            avg_var = sum(variances) / len(variances)
            stability = max(0.0, 1.0 - avg_var / 600)
        else:
            stability = 1.0

        score = 0.30 * resilience + 0.40 * final_quality + 0.30 * stability
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        return self._state.step_count >= self._task_config["max_months"]

    def _make_observation(self) -> GovernAIObservation:
        m = self._metrics
        is_done = self._is_done()
        reward = self._compute_step_reward()

        meta: Dict[str, Any] = {}
        if is_done:
            grader_score = self._compute_grader_score()
            meta["grader_score"] = grader_score
            meta["task_id"] = self._task_id
            meta["final_metrics"] = dict(m)

        return GovernAIObservation(
            economy=round(m["economy"], 1),
            health=round(m["health"], 1),
            education=round(m["education"], 1),
            pollution=round(m["pollution"], 1),
            happiness=round(m["happiness"], 1),
            inequality=round(m["inequality"], 1),
            budget=round(m["budget"], 1),
            unemployment=round(m["unemployment"], 1),
            month=self._state.step_count,
            max_months=self._task_config["max_months"],
            narrative=self._generate_narrative(),
            active_events=list(self._active_events.keys()),
            available_actions=[a.value for a in PolicyAction],
            task_id=self._task_id,
            done=is_done,
            reward=reward,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Narrative generation — rich text for LLM reasoning
    # ------------------------------------------------------------------

    def _generate_narrative(self) -> str:
        m = self._metrics
        month = self._state.step_count
        max_m = self._task_config["max_months"]
        parts: List[str] = []

        parts.append(f"=== CITY STATUS REPORT: Month {month}/{max_m} ===")
        parts.append("")
        parts.append(f"ECONOMY ({m['economy']:.0f}/100): {self._describe_positive(m['economy'])}")
        parts.append(f"PUBLIC HEALTH ({m['health']:.0f}/100): {self._describe_positive(m['health'])}")
        parts.append(f"EDUCATION ({m['education']:.0f}/100): {self._describe_positive(m['education'])}")
        parts.append(f"POLLUTION ({m['pollution']:.0f}/100): {self._describe_negative(m['pollution'])}")
        parts.append(f"HAPPINESS ({m['happiness']:.0f}/100): {self._describe_positive(m['happiness'])}")
        parts.append(f"INEQUALITY ({m['inequality']:.0f}/100): {self._describe_negative(m['inequality'])}")
        parts.append(f"BUDGET ({m['budget']:.0f}/100): {self._describe_positive(m['budget'])}")
        parts.append(f"UNEMPLOYMENT ({m['unemployment']:.0f}/100): {self._describe_negative(m['unemployment'])}")
        parts.append("")

        if self._active_events:
            parts.append("ACTIVE CRISES:")
            for event_name in self._active_events:
                desc = CRISIS_EVENTS.get(event_name, {}).get("description", event_name)
                parts.append(f"  - {desc}")
            parts.append("")

        warnings: List[str] = []
        if m["budget"] < 20:
            warnings.append("Budget is critically low — reduce spending")
        if m["pollution"] > 70:
            warnings.append("Pollution is dangerous — health impacts expected")
        if m["happiness"] < 25:
            warnings.append("Civil unrest is brewing — citizen satisfaction critically low")
        if m["unemployment"] > 60:
            warnings.append("Unemployment crisis — economic intervention needed")
        if m["health"] < 25:
            warnings.append("Healthcare emergency — system near collapse")
        if m["inequality"] > 70:
            warnings.append("Extreme inequality — social cohesion at risk")
        if warnings:
            parts.append("WARNINGS:")
            for w in warnings:
                parts.append(f"  - {w}")
            parts.append("")

        parts.append("AVAILABLE POLICY ACTIONS:")
        for action in PolicyAction:
            parts.append(f"  - {action.value}")

        return "\n".join(parts)

    @staticmethod
    def _describe_positive(value: float) -> str:
        if value > 75:
            return "Excellent"
        if value > 55:
            return "Good — stable"
        if value > 35:
            return "Struggling — needs improvement"
        return "Critical — immediate action required"

    @staticmethod
    def _describe_negative(value: float) -> str:
        if value > 75:
            return "Critical — immediate action required"
        if value > 55:
            return "Concerning — needs attention"
        if value > 35:
            return "Moderate — manageable"
        return "Low — well controlled"
