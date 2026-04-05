---
title: GovernAI
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# GovernAI — AI Policy Simulator

An OpenEnv-compatible RL environment where an LLM agent acts as a **city governor**, making monthly policy decisions to balance competing societal metrics under uncertainty.

## Why This Environment?

Real-world governance is the ultimate multi-objective optimization problem. A policy-maker must balance:

- **Economy** vs **Environment** — industrial growth increases pollution
- **Short-term happiness** vs **Long-term stability** — tax cuts feel good but reduce budget
- **Health** vs **Budget** — healthcare investment costs money
- **Equality** vs **Growth** — welfare reduces inequality but strains finances

GovernAI captures these trade-offs in a clean, tractable simulation that is complex enough to require genuine reasoning but simple enough to evaluate.

## Environment Design

### State Space

Eight normalized metrics (0–100) describe the city each month:

| Metric | Direction | Description |
|--------|-----------|-------------|
| Economy | Higher = better | GDP and economic activity |
| Health | Higher = better | Public health system quality |
| Education | Higher = better | Education system quality |
| Pollution | Lower = better | Environmental pollution level |
| Happiness | Higher = better | Citizen satisfaction |
| Inequality | Lower = better | Income inequality gap |
| Budget | Higher = better | Government spending capacity |
| Unemployment | Lower = better | Jobless rate |

### Action Space

Nine discrete policy actions, each with multi-dimensional effects:

| Action | Primary Effects |
|--------|----------------|
| `invest_healthcare` | health +8, happiness +3, budget -8 |
| `invest_education` | education +8, happiness +2, budget -7, inequality -2 |
| `cut_taxes` | economy +5, happiness +4, budget -6 |
| `raise_taxes` | economy -2, happiness -3, budget +8, inequality -3 |
| `green_subsidy` | pollution -8, budget -6, economy -2 |
| `industrial_expansion` | economy +7, unemployment -5, pollution +6 |
| `welfare_support` | happiness +4, inequality -5, budget -5, unemployment -2 |
| `infrastructure_push` | economy +3, unemployment -4, budget -7 |
| `do_nothing` | No immediate changes |

### Key Dynamics

- **Delayed consequences**: Some actions produce effects 2–3 months later (e.g., education investment eventually boosts economy).
- **Background drift**: Metrics evolve naturally each turn — pollution rises with high industry, happiness falls with inequality, budgets require economic support.
- **Crisis events**: Pandemics, floods, recessions, energy crises, and protests can strike, forcing adaptive responses.
- **No dominant strategy**: Every action involves trade-offs. The agent must learn to sequence decisions for long-term welfare.

### Reward Function

Each step returns a weighted reward combining all metrics:

```
reward = 0.20 * economy + 0.20 * health + 0.15 * happiness
       + 0.10 * education - 0.15 * pollution - 0.10 * inequality
       - 0.10 * unemployment + 0.05 * budget
```

Critical-state penalties apply when budget < 10, pollution > 80, happiness < 20, or health < 15.

## Tasks (Easy / Medium / Hard)

| Task | Difficulty | Months | Description |
|------|-----------|--------|-------------|
| `stable_city` | Easy | 12 | Govern a healthy city with no crises. Focus on balanced optimization. |
| `austerity_challenge` | Medium | 18 | Recover from crisis: budget=15, unemployment=70. Preset events at months 6 and 12. |
| `crisis_governance` | Hard | 24 | Full governance with random crises (15% chance/month). Requires resilience and adaptability. |

Each task has a **grader** that returns a score from 0.0 to 1.0:

- **Easy**: Weighted quality of final metrics.
- **Medium**: 40% improvement from initial state + 40% final quality + 20% budget survival.
- **Hard**: 30% resilience (no metric crashes) + 40% final quality + 30% stability (low variance).

## Setup

### Prerequisites

- Python 3.10+
- Docker (for deployment)

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t governai .
docker run -p 7860:7860 governai
```

### Running the Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"

python inference.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Reset environment (accepts `task_id`, `seed`, `episode_id`) |
| POST | `/step` | Take action (body: `{"action": {"policy": "...", "reasoning": "..."}}`) |
| GET | `/state` | Current environment state |
| GET | `/schema` | Action and observation JSON schemas |
| GET | `/tasks` | Available tasks with descriptions |

## Project Structure

```
GovernAI/
├── models.py                        # Pydantic Action, Observation, State types
├── client.py                        # Python HTTP client
├── inference.py                     # LLM agent inference script
├── openenv.yaml                     # OpenEnv environment config
├── Dockerfile                       # Container for HF Spaces
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata
├── README.md                        # This file
└── server/
    ├── app.py                       # FastAPI server
    └── governai_environment.py      # Core simulation engine
```

## Why RL / LLM Agents Suit This Problem

1. **Delayed effects** — short-term gains can cause long-term damage, requiring temporal reasoning.
2. **Multi-objective trade-offs** — no single metric can be maximized in isolation.
3. **Stochastic events** — crises force adaptive rather than scripted behavior.
4. **Sequential decision-making** — the optimal action depends on the full trajectory, not just the current state.

This makes GovernAI a natural fit for reinforcement learning and demonstrates that LLM agents can reason about complex societal trade-offs.
