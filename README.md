# SVAGA 3.0 — Self-Verifying Agent-Generated Applications



Natural language → small Python app + **specification**, **tests**, **least-privilege policy** → multi-verifier check → **ACCEPT/REJECT** → downloadable bundle and optional live API preview.



## Primary UX (web console)



**Open http://localhost:5303** for the SVAGA 3.0 console. Ports **5173/5174** are often **SVAGA 2.0** or a **generated app** (e.g. leave approval)—not 3.0.



Quick start (Windows):



```powershell

cd "SVAGA 3.0"

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

copy .env.example .env



# Terminal 1 — SVAGA 3.0 API only (8003; 8000=2.0, 8002=often wrong/stale)

$env:PYTHONPATH = "."

$env:SVAGA_SCRIPTED_LLM = "1"

python -m uvicorn svaga_platform.app.main:app --reload --host 127.0.0.1 --port 8003



# Terminal 2 — console UI (proxies /api → 8003)

cd model1_rag_generator\frontend

npm install

$env:VITE_PROXY_TARGET = "http://127.0.0.1:8003"

$env:VITE_DEV_PORT = "5303"

# Do NOT set VITE_API_BASE in dev (causes "Failed to fetch" / CORS). Vite proxies /api → 8003.

npm run dev

```



Or run `scripts\dev.ps1` (starts API in a new window, then UI).



### Console flow



1. **New workflow run** — free-form NL or pick a benchmark; pipeline M1–M4.

2. **Run detail** — retrieved context, spec/policy JSON, code, verifier matrix, release gate.

3. **Ship** — download zip (`run.ps1`, `requirements.txt`, purpose-ready `app.py` with browser UI when applicable) or live preview.

   Every generated app gets a web page at `/` and `/ui`, built from the app's own routes: an "Acting as" bar for user id and role, a form per operation, and a records list that refreshes after each change. Older saved runs get the same page when previewed. Swagger stays at `/docs`.

4. **Experiments** — compare M1–M4 on the same workflow under shared budget.



## Layout



- `shared/` — sandbox, LLM, budget, policy, schemas

- `svaga_platform/` — FastAPI orchestrator

- `model1_rag_generator/frontend/` — **platform console** (React + Monaco)

- `benchmarks/` — 38 workflow YAMLs with executable properties

- `results/` — persisted runs and experiment outputs (gitignored)



## LLM backends (free-friendly)



| Backend | Configuration |

|---------|----------------|

| Scripted (tests, no key) | `SVAGA_SCRIPTED_LLM=1` |

| **Ollama (recommended local)** | `SVAGA_LLM_PROVIDER=ollama`, `SVAGA_SCRIPTED_LLM=0`, `OLLAMA_MODEL=qwen3-coder:30b` (`ollama pull qwen3-coder:30b`, 24 GB GPU). Native `/api/chat` with JSON-schema output and keep-alive; smoke test: `python scripts\test_ollama_provider.py` |

| OpenAI | `SVAGA_LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` |
| Anthropic (Claude) | `SVAGA_LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...`, optional `ANTHROPIC_MODEL` (default `claude-haiku-4-5`) |



## Key API routes



| Method | Path | Purpose |

|--------|------|---------|

| POST | `/api/v1/workflows/run` | Unified run (console) |

| GET | `/api/v1/runs/{id}` | Reload run |

| GET | `/api/v1/runs/{id}/download` | Deployable zip |

| POST | `/api/v1/runs/{id}/preview/start` | Live generated API |

| POST | `/api/v1/experiments` | M1–M4 comparison |



## Tests



```powershell

$env:PYTHONPATH = "."

$env:SVAGA_SCRIPTED_LLM = "1"

pytest -q

```



## Pipelines



| ID | Idea |

|----|------|

| M1 | RAG + generate + full verification + gate |

| M2 | Spec-first synthesis |

| M3 | M2 + Hypothesis property tests |

| M4 | Counterexample-guided repair |



## Deferred



Full-stack Docker apps (parent SVAGA 2.0). CrossHair / full Schemathesis / Postgres run DB.


