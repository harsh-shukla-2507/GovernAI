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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL") or os.getenv("ENV_BASE_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "governai"

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


def _log_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


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

        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        except Exception as exc:
            print(f"    LLM call error (attempt {attempt + 1}): {exc}")

    return {"policy": "do_nothing", "reasoning": "Fallback after failed LLM attempts"}


def _post(url: str, payload: dict, retries: int = 3) -> dict:
    """POST with retry and error handling."""
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


def run_task(env_url: str, task_id: str, llm_client: OpenAI, model: str) -> dict:
    """Run a single governance task and return result dict."""
    rewards: list[float] = []
    step_count = 0
    done = False
    success = False

    _log_start(task_id, model)

    try:
        data = _post(f"{env_url}/reset", {"task_id": task_id, "episode_id": task_id})
        observation = data.get("observation", {})

        while not done and step_count < 100:
            action = get_llm_action(llm_client, model, observation)
            policy = action.get("policy", "do_nothing")
            reasoning = action.get("reasoning", "")
            action_str = json.dumps({"policy": policy, "reasoning": reasoning}, separators=(",", ":"))

            try:
                data = _post(
                    f"{env_url}/step",
                    {"action": {"policy": policy, "reasoning": reasoning}},
                )
                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))
                observation = data.get("observation", {})
                rewards.append(reward)
                step_count += 1
                _log_step(step_count, action_str, reward, done, None)
            except Exception as exc:
                _log_step(step_count + 1, action_str, 0.0, False, str(exc))
                break

        meta = observation.get("metadata", {})
        grader_score = meta.get("grader_score", 0.0)
        success = done and grader_score > 0.0

        return {
            "task_id": task_id,
            "steps": step_count,
            "total_reward": round(sum(rewards), 3),
            "average_reward": round(sum(rewards) / step_count, 3) if step_count else 0.0,
            "grader_score": grader_score,
            "success": success,
        }

    except Exception as exc:
        return {
            "task_id": task_id,
            "steps": step_count,
            "total_reward": 0.0,
            "average_reward": 0.0,
            "grader_score": 0.0,
            "success": False,
            "error": str(exc),
        }

    finally:
        _log_end(success=success, steps=step_count, rewards=rewards)


def main() -> int:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is required.")
        return 1

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    base_url = ENV_URL.rstrip("/")

    print("=" * 60)
    print("  GovernAI — AI Policy Simulator")
    print("=" * 60)
    print(f"  Environment : {base_url}")
    print(f"  LLM model   : {MODEL_NAME}")
    print()

    healthy = False
    for attempt in range(10):
        try:
            resp = requests.get(f"{base_url}/health", timeout=10)
            if resp.status_code == 200:
                healthy = True
                break
        except requests.RequestException:
            pass
        wait = min(2 ** attempt, 30)
        print(f"  Waiting for environment (attempt {attempt + 1}/10, next retry in {wait}s)...")
        time.sleep(wait)

    if not healthy:
        print(f"Cannot connect to environment at {base_url} after 10 attempts")
        return 1
    print("  Environment health check: OK\n")

    results = []
    for task_id in TASKS:
        results.append(run_task(base_url, task_id, llm_client, MODEL_NAME))

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        score = r.get("grader_score", 0.0)
        print(f"    {r['task_id']:>25s}: {score:.4f}")
    scores = [r.get("grader_score", 0.0) for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n    {'Average':>25s}: {avg:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
