"""GovernAI Python client — interact with the environment server via HTTP."""

from typing import Any, Dict, List, Optional

import requests

from models import GovernAIAction, GovernAIObservation, PolicyAction


class GovernAIClient:
    """Synchronous HTTP client for the GovernAI environment server."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def health(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        task_id: str = "stable_city",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        resp = self._session.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def step(self, policy: str, reasoning: str = "") -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/step",
            json={"action": {"policy": policy, "reasoning": reasoning}},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "GovernAIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
