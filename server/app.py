"""GovernAI FastAPI server — implements the OpenEnv HTTP protocol."""

import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GovernAIAction, GovernAIObservation, PolicyAction
from server.governai_environment import GovernAIEnvironment, TASK_CONFIGS


app = FastAPI(
    title="GovernAI — AI Policy Simulator",
    description="An RL environment where an LLM agent governs a city through policy decisions.",
    version="1.0.0",
)

_env = GovernAIEnvironment()


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    model_config = {"extra": "allow"}


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None
    model_config = {"extra": "allow"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "name": "GovernAI",
        "description": "AI Policy Simulator — govern a city through policy decisions",
        "version": "1.0.0",
        "endpoints": ["/health", "/reset", "/step", "/state", "/schema", "/tasks"],
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    task_id = request.task_id or request.episode_id or "stable_city"
    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=task_id,
    )
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/step")
async def step(request: StepRequest):
    action = GovernAIAction(**request.action)
    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state")
async def get_state():
    return _env.state.model_dump()


@app.get("/schema")
async def get_schema():
    return {
        "action": GovernAIAction.model_json_schema(),
        "observation": GovernAIObservation.model_json_schema(),
    }


@app.get("/tasks")
async def get_tasks():
    return {
        task_id: {
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "max_months": cfg["max_months"],
        }
        for task_id, cfg in TASK_CONFIGS.items()
    }
