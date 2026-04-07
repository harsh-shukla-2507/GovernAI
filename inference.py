#!/usr/bin/env python3
"""
GovernAI Inference Script
=========================
Runs an LLM agent through all three governance tasks and reports grader scores.

MANDATORY
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.
"""

import json
import os
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

TASKS = ["stable_city", "austerity_challenge", "crisis_governance"]

VALID_ACTIONS = [
    "invest_healthcare",
    "invest_education",
    "cut_taxes",
    "raise_taxes",
    "green_subsidy",
    "industrial_expansion",
    "welfare_support",
    "infrastructure_push",
    "do_nothing",
]

SYSTEM_PROMPT = """\
You are an AI city governor. Each month you make one policy decision.

You receive a status report with city metrics (0-100 scale):
- Economy, Health, Education, Happiness, Budget: higher is better
- Pollution, Inequality, Unemployment: LOWER is better

Goal: maximize overall societal welfare by balancing these competing metrics.

Key considerations:
- Trade-offs: improving one metric often hurts another.
- Budget: every investment costs budget. If budget drops too low you lose capacity.
- Delayed effects: some actions take 2-3 months to fully manifest.
- Crises: random events may require immediate adaptive responses.

Respond with ONLY a JSON object — no markdown, no explanation outside the JSON:
{"policy": "<action_name>", "reasoning": "<one sentence>"}

Valid actions:
  invest_healthcare   — health +8, happiness +3, budget -8
  invest_education    — education +8, happiness +2, budget -7, inequality -2
  cut_taxes           — economy +5, happiness +4, budget -6
  raise_taxes         — economy -2, happiness -3, budget +8, inequality -3
  green_subsidy       — pollution -8, budget -6, economy -2
  industrial_expansion — economy +7, unemployment -5, pollution +6
  welfare_support     — happiness +4, inequality -5, budget -5, unemployment -2
  infrastructure_push — economy +3, unemployment -4, budget -7
  do_nothing          — no immediate changes\
"""


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def get_llm_action(
    client: OpenAI, model: str, observation: dict, max_retries: int = 3
) -> dict:
    """Ask the LLM to choose a policy action given the current city status."""
    narrative = observation.get("narrative", "No status available.")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Here is your city's current status:\n\n"
                            f"{narrative}\n\n"
                            f"What policy do you enact this month?"
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    cleaned = part.strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    if cleaned.startswith("{"):
                        content = cleaned
                        break

            result = json.loads(content)
            if result.get("policy") in VALID_ACTIONS:
                return result

            # LLM returned valid JSON but unknown action — retry
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        except Exception as exc:
            print(f"    LLM call error (attempt {attempt + 1}): {exc}")

    return {"policy": "do_nothing", "reasoning": "Fallback after failed LLM attempts"}


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def _post(url: str, payload: dict, retries: int = 3) -> dict:
    """POST with retry and error handling. Returns parsed JSON response."""
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"    Network error (attempt {attempt + 1}/{retries}): {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to reach {url} after {retries} attempts")


def run_task(env_url: str, task_id: str, llm_client: OpenAI, model: str) -> float:
    """Run a single governance task and return the grader score."""
    print(f"\n{'=' * 60}")
    print(f"  TASK: {task_id}")
    print(f"{'=' * 60}")

    try:
        data = _post(f"{env_url}/reset", {"task_id": task_id, "episode_id": task_id})
        observation = data.get("observation", {})

        step_num = 0
        max_steps = 100
        while not observation.get("done", False) and step_num < max_steps:
            step_num += 1
            month = observation.get("month", step_num)
            max_months = observation.get("max_months", "?")

            action = get_llm_action(llm_client, model, observation)
            policy = action.get("policy", "do_nothing")
            reasoning = action.get("reasoning", "")

            print(f"  Month {month}/{max_months}: {policy} — {reasoning}")

            data = _post(
                f"{env_url}/step",
                {"action": {"policy": policy, "reasoning": reasoning}},
            )
            observation = data.get("observation", {})

        meta = observation.get("metadata", {})
        grader_score = meta.get("grader_score", 0.0)

        print(f"\n  Final city metrics:")
        for key in [
            "economy", "health", "education", "pollution",
            "happiness", "inequality", "budget", "unemployment",
        ]:
            print(f"    {key:>14s}: {observation.get(key, 'N/A')}")
        print(f"\n  GRADER SCORE: {grader_score:.4f}")

        return grader_score

    except Exception as exc:
        print(f"\n  Task {task_id} failed: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_BASE_URL or not MODEL_NAME or not API_KEY:
        print(
            "ERROR: Set API_BASE_URL, MODEL_NAME, and HF_TOKEN (or API_KEY) "
            "environment variables."
        )
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60)
    print("  GovernAI — AI Policy Simulator")
    print("=" * 60)
    print(f"  Environment : {ENV_URL}")
    print(f"  LLM model   : {MODEL_NAME}")
    print()

    # Health check with retries (container may still be starting)
    healthy = False
    for attempt in range(10):
        try:
            resp = requests.get(f"{ENV_URL}/health", timeout=10)
            if resp.status_code == 200:
                healthy = True
                break
        except requests.RequestException:
            pass
        wait = min(2 ** attempt, 30)
        print(f"  Waiting for environment (attempt {attempt + 1}/10, next retry in {wait}s)...")
        time.sleep(wait)

    if not healthy:
        print(f"Cannot connect to environment at {ENV_URL} after 10 attempts")
        sys.exit(1)
    print("  Environment health check: OK")

    scores: dict[str, float] = {}
    start = time.time()

    for task_id in TASKS:
        scores[task_id] = run_task(ENV_URL, task_id, llm_client, MODEL_NAME)

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for task_id, score in scores.items():
        print(f"    {task_id:>25s}: {score:.4f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n    {'Average':>25s}: {avg:.4f}")
    print(f"    {'Total time':>25s}: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
