---
title: Mumbai Local OpenEnv
emoji: 🚂
colorFrom: orange
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---

# 🚂 Mumbai Local — OpenEnv Training Environment  v4.0

> **OpenEnv Hackathon India 2026** · Theme #1 Multi-Agent + #2 Long-Horizon + #3.2 Personal Assistant

> ⚡ **Decision time: ~3 minutes** (sim loop: 150 steps · TRL GRPO: 60 steps · T4 GPU)

---

## 🎯 Problem

Mumbai's local train network carries **8 million passengers daily** across 3 lines (Western, Central, Harbour) and 100+ stations. Yet:

- No LLM has ever been trained to reason about real-time transit decisions under disruption
- Commuters make multi-step decisions (when to leave, which train, which compartment, when to transfer) that require **long-horizon planning**
- Simultaneous agents compete for limited train capacity — classic **multi-agent** problem
- Personal constraints (meetings, rain, strikes) make this a **personal assistant** challenge

---

## 🗺 Environment Design

### Network (GTFS-Grounded — v4 improvement)

Station sequences, coordinates, and transfer walk times are sourced from published Mumbai Railway timetables (WR/CR/HR) via `gtfs_loader.py`. Drop a real GTFS zip at `./gtfs/mumbai_local.zip` to upgrade to live schedule accuracy.

| Line | Stations | Headway (peak) | Avg travel time |
|------|----------|---------------|-----------------|
| Western | 29 (Churchgate → Virar) | 3 min | 2.5 min/station |
| Central | 26 (CSMT → Kalyan) | 3 min | 3.0 min/station |
| Harbour | 26 (CSMT → Panvel) | 5 min | 3.5 min/station |

### 🔀 Transfer Graph (from GTFS transfers.txt)

| Station | Lines | Walk time |
|---------|-------|-----------|
| Dadar | Western ↔ Central | 3 min |
| CSMT | Central ↔ Harbour | 2 min |
| Andheri | Western ↔ Harbour | 4 min |
| Bandra | Western ↔ Harbour | 5 min |
| Kurla | Central ↔ Harbour | 7 min |
| Mahim | Western ↔ Harbour | 3 min |

### 🧠 LLM-Native Decision Loop (v4 NEW)

The LLM is now the actual policy — not just an optional inference step. Three training modes:

| Mode | Command | LLM involved? | Loss signal |
|------|---------|---------------|-------------|
| `simulate` | `python train.py` | No (epsilon-greedy) | Proxy (labelled) |
| `llm` | `TRAINING_MODE=llm python train.py` | Yes (Anthropic API) | Real LLM reward curves |
| `trl` | `TRAINING_MODE=trl python train.py` | Yes (Qwen2.5 GRPO) | Real gradient loss |

---

## 🏆 Reward Design (All Components Active)

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| `arrival_bonus` | +10.0 | Did commuter reach destination? |
| `distance_progress` | ×1.5 | Moving closer each step |
| `time_efficiency` | ±2.5 | Fast vs slow routes |
| `crowd_avoidance` | 0–1.2 | Chose less-crowded train? |
| `disruption_response` | +2.5 / -1.0 | Smart reroute vs wasteful |
| `waiting_penalty` | -0.35 × headway_factor | Penalise standing still (reduced during peak — trains frequent) |
| `personal_task_completion` | 0–5.0 | Schedule items completed |
| `transfer_bonus` | +1.0 | Smart inter-line transfer |
| `repetition_penalty` | -0.3 | Penalise repeating same action 3× |
| `grievance_penalty` | -3.0 | Missing arrival deadline |

---

## 🔧 What Changed v3 → v4

| Issue | v3.0 | v4.0 |
|-------|------|------|
| Station data | ❌ Hardcoded lists | ✅ GTFS-grounded (real timetable sequences + coords) |
| ETA calculation | ❌ Raw station count | ✅ Headway-aware (GTFS peak/off-peak schedule) |
| LLM in training | ❌ Heuristic only | ✅ LLM is the policy (`TRAINING_MODE=llm`) |
| Training loss | ❌ Simulated proxy | ✅ Real reward from LLM policy; real gradient in TRL mode |
| GTFS fallback | ❌ N/A | ✅ Automatic — real zip → parse; no zip → bundled data |
| Prompt context | ❌ No schedule info | ✅ GTFS ETA + headway in every agent prompt |

---

## 🚀 Quick Start

### 1. Run locally
```bash
pip install flask numpy matplotlib
python app.py   # Open http://localhost:5000
```

### 2. Smoke test environment
```bash
python environment.py
# Prints GTFS-sourced station movement + ETA/headway per step
```

### 3. Simulation training (~3 min, no GPU)
```bash
python train.py
# Generates training_results.png + training_log.json
# Reward/arrival from real GTFS-grounded env; proxy loss clearly labelled
```

### 4. LLM-native training (Anthropic API, no GPU)
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
TRAINING_MODE=llm python train.py
# LLM is the actual policy — real reward curves, no simulated loss
```

### 5. Full TRL GRPO training (GPU, ~3 min on Colab T4)
```bash
pip install trl transformers accelerate torch datasets
TRAINING_MODE=trl python train.py
# Real GRPO gradient signal; loss from TRL callbacks
```

### 6. Use a real GTFS feed
```bash
mkdir -p gtfs
# Download from data.gov.in or mmrda.maharashtra.gov.in
cp ~/Downloads/mumbai_local_gtfs.zip ./gtfs/mumbai_local.zip
python environment.py   # Automatically parses real GTFS
```

---

## 📁 Project Structure

```
mumbai-local-env/
├── gtfs_loader.py          # NEW v4: GTFS parser + bundled timetable data
├── gtfs/                   # Drop real GTFS zip here (optional)
│   └── mumbai_local.zip    # → auto-parsed if present
├── Dockerfile
├── pyproject.toml
├── openenv.yaml
├── app.py                  # Flask backend
├── environment.py          # OpenEnv Gym environment (v4 — GTFS-grounded)
├── train.py                # simulate / llm / trl modes (v4)
├── inference.py            # Heuristic & LLM inference
├── requirements.txt
├── README.md
├── training_results.png
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## 🌐 REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Full world snapshot (includes GTFS data_source field) |
| `/api/step` | POST `{"action": "route_optimize"}` | Advance one step |
| `/api/command` | POST `{"text": "avoid crowds"}` | NL → action → step |
| `/api/disrupt` | POST `{"line": "Western", "severity": "High"}` | Inject disruption |
| `/api/agent/<id>` | GET | Rich per-agent state including gtfs_eta_minutes |
| `/api/transfer` | POST `{"agent_id": 0, "to_line": "Central"}` | Force line transfer |
| `/api/tasks` | GET | Live personal task completion |
| `/api/benchmark` | POST | Run 5 heuristic episodes |
| `/api/reset` | POST | Reset + archive to leaderboard |
| `/api/leaderboard` | GET | Episode leaderboard |

---

## 🧠 Why This Stands Out

1. **GTFS-grounded** — Station sequences, headways, and ETAs from real Mumbai Railway timetables
2. **LLM is the policy** — Three training modes; LLM-native mode proves environment usability before GPU commitment
3. **Honest training** — Proxy loss clearly labelled; LLM mode gives real reward signal; TRL mode gives real gradient loss
4. **Novel domain** — No prior RL/LLM env for Indian mass transit
5. **All reward components wired** — Every rubric item fires every step
6. **Inter-line transfers** — 6 junction stations with real walk-time data
7. **Personal task integration** — Commuters have schedules; completing them earns reward

---

*Built for OpenEnv Hackathon India 2026 · MIT License · v4.0.0*
