# MetaTune

Carbon-aware LLM fine-tuning. Schedules GPU workloads across global data centers to minimise CO₂ emissions using real-time weather forecasts and a MILP solver.

## Quickstart

**Requirements:** Python 3.10+, Node.js 18+

```bash
./setup.sh
```

Then in two terminals:

```bash
# Terminal 1
source .venv/bin/activate && cd backend && uvicorn api:app --reload --port 8000

# Terminal 2
cd metatune-ai && npm run dev
```

Open `http://localhost:8080`.

## API

`GET /api/schedule?job_hours=4&job_power_kw=15&deadline_hours=22`

Returns a carbon forecast for each data center and the optimal start time/location to minimise emissions.
