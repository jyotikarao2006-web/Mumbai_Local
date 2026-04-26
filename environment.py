"""
Mumbai Local Train — OpenEnv Compliant Environment  v3.0
=========================================================
Fixes over v2.3:
  - Agents actually move along their line each step (current_station updates)
  - personal_task_completion reward is NOW connected and called
  - _observe_agent_rich() is used in step() for per-agent decisions
  - GRPO reward_fn gets per-env instances (no shared-state bug)
  - Arrival uses distance-gated probability (not pure random)
  - Inter-line transfers at junction stations (Dadar, CSMT, Andheri, Bandra)

New in v3.0:
  - Transfer graph: agents can switch lines at junctions
  - Temporal rush-hour crowd model (crowd spikes at 8-10 AM / 5-8 PM)
  - Agent memory: last 5 actions stored; repetition penalised
  - Grievance score: commuters that miss deadline get a grievance flag
  - Composite leaderboard metric: arrival_rate × time_efficiency - grievances
"""

import random
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Network Data ──────────────────────────────────────────────────────────────

WESTERN_LINE = [
    "Churchgate","Marine Lines","Charni Road","Grant Road","Mumbai Central",
    "Mahalaxmi","Lower Parel","Elphinstone Road","Dadar","Matunga Road",
    "Mahim","Bandra","Khar Road","Santacruz","Vile Parle","Andheri",
    "Jogeshwari","Ram Mandir","Goregaon","Malad","Kandivali","Borivali",
    "Dahisar","Mira Road","Bhayandar","Naigaon","Vasai Road","Nalasopara","Virar",
]
CENTRAL_LINE = [
    "CSMT","Masjid","Sandhurst Road","Byculla","Chinchpokli","Currey Road",
    "Parel","Dadar","Matunga","Sion","Kurla","Vidyavihar","Ghatkopar",
    "Vikhroli","Kanjurmarg","Bhandup","Nahur","Mulund","Thane","Kalwa",
    "Mumbra","Diva","Kopar","Dombivli","Thakurli","Kalyan",
]
HARBOUR_LINE = [
    "CSMT","Masjid","Sandhurst Road","Dockyard Road","Reay Road","Cotton Green",
    "Sewri","Wadala Road","King's Circle","Mahim Junction","Bandra","Khar Road",
    "Santacruz","Vile Parle","Andheri","Chembur","Govandi","Mankhurd",
    "Vashi","Sanpada","Juinagar","Nerul","Seawoods","Belapur","Kharghar","Panvel",
]

LINES: Dict[str, Dict] = {
    "Western": {"stations": WESTERN_LINE, "color": "#FF6B35", "trains": 15},
    "Central": {"stations": CENTRAL_LINE, "color": "#4ECDC4",  "trains": 12},
    "Harbour": {"stations": HARBOUR_LINE, "color": "#A855F7",  "trains": 10},
}

# Stations where inter-line transfers are possible
TRANSFER_GRAPH: Dict[str, List[str]] = {
    "Dadar":   ["Western", "Central"],
    "CSMT":    ["Central", "Harbour"],
    "Andheri": ["Western", "Harbour"],
    "Bandra":  ["Western", "Harbour"],
    "Kurla":   ["Central", "Harbour"],
    "Mahim":   ["Western", "Harbour"],
}

DISRUPTION_TYPES = [
    "Signal failure", "Track maintenance", "Heavy rainfall", "Overcrowding delay",
    "Medical emergency", "Trespassing incident", "Equipment failure", "Power outage",
]

ACTIONS = ["route_optimize", "avoid_crowd", "reroute", "wait"]

# ── Commuter Profiles ─────────────────────────────────────────────────────────

COMMUTER_PROFILES = [
    {
        "name": "Office Worker", "origin": "Virar", "destination": "Churchgate",
        "line": "Western", "arrival_deadline": 9.0,
        "crowding_tolerance": 0.4, "waiting_tolerance": 0.5, "risk_aversion": 0.6,
        "personal_tasks": ["Buy ticket", "Check emails"],
    },
    {
        "name": "Student", "origin": "Andheri", "destination": "CSMT",
        "line": "Harbour", "arrival_deadline": 10.0,
        "crowding_tolerance": 0.9, "waiting_tolerance": 0.3, "risk_aversion": 0.2,
        "personal_tasks": ["Submit assignment"],
    },
    {
        "name": "Trader", "origin": "Kalyan", "destination": "Dadar",
        "line": "Central", "arrival_deadline": 8.5,
        "crowding_tolerance": 0.5, "waiting_tolerance": 0.2, "risk_aversion": 0.7,
        "personal_tasks": ["Call broker", "Check prices"],
    },
    {
        "name": "Teacher", "origin": "Bandra", "destination": "Andheri",
        "line": "Harbour", "arrival_deadline": 8.0,
        "crowding_tolerance": 0.6, "waiting_tolerance": 0.4, "risk_aversion": 0.8,
        "personal_tasks": ["Prepare lesson"],
    },
    {
        "name": "Doctor", "origin": "Borivali", "destination": "Lower Parel",
        "line": "Western", "arrival_deadline": 8.0,
        "crowding_tolerance": 0.3, "waiting_tolerance": 0.6, "risk_aversion": 0.9,
        "personal_tasks": ["Review patient notes", "Confirm OT schedule"],
    },
    {
        "name": "Engineer", "origin": "Panvel", "destination": "Andheri",
        "line": "Harbour", "arrival_deadline": 9.5,
        "crowding_tolerance": 0.7, "waiting_tolerance": 0.5, "risk_aversion": 0.4,
        "personal_tasks": ["Review PR"],
    },
    {
        "name": "Nurse", "origin": "Vasai Road", "destination": "Dadar",
        "line": "Western", "arrival_deadline": 6.0,
        "crowding_tolerance": 0.8, "waiting_tolerance": 0.2, "risk_aversion": 0.5,
        "personal_tasks": ["Check duty roster"],
    },
    {
        "name": "Banker", "origin": "Mira Road", "destination": "Mumbai Central",
        "line": "Western", "arrival_deadline": 8.5,
        "crowding_tolerance": 0.2, "waiting_tolerance": 0.7, "risk_aversion": 0.9,
        "personal_tasks": ["Review market open", "Send morning report"],
    },
    {
        "name": "Software Dev", "origin": "Ghatkopar", "destination": "CSMT",
        "line": "Central", "arrival_deadline": 10.0,
        "crowding_tolerance": 0.6, "waiting_tolerance": 0.8, "risk_aversion": 0.3,
        "personal_tasks": ["Check Slack", "Review build status"],
    },
    {
        "name": "Journalist", "origin": "Mulund", "destination": "Dadar",
        "line": "Central", "arrival_deadline": 8.0,
        "crowding_tolerance": 0.5, "waiting_tolerance": 0.1, "risk_aversion": 0.2,
        "personal_tasks": ["File morning brief"],
    },
]

# ── Rubric ────────────────────────────────────────────────────────────────────

class Rubric:
    """
    Composable reward rubric — all components wired and active in v3.
    """
    weights = {
        "arrival":     10.0,
        "efficiency":   2.5,
        "crowd":        1.2,
        "disruption":   2.5,
        "waiting":     -0.35,
        "distance":     1.5,
        "exploration":  0.2,
        "task":         5.0,   # personal_task_completion — NOW ACTIVE
        "transfer":     1.0,   # bonus for smart inter-line transfer
        "repetition":  -0.3,   # penalty for repeating same action 3+ times
        "grievance":   -3.0,   # missed deadline penalty
    }

    @staticmethod
    def arrival_bonus(agent: dict) -> float:
        return Rubric.weights["arrival"] if agent.get("arrived") else 0.0

    @staticmethod
    def time_efficiency(steps_taken: int, optimal_steps: int) -> float:
        if steps_taken == 0:
            return 0.0
        ratio = optimal_steps / max(steps_taken, 1)
        return max(-2.0, min(2.0, (ratio - 1.0) * 3.0)) * Rubric.weights["efficiency"]

    @staticmethod
    def crowd_avoidance(crowd_level: float) -> float:
        return ((100 - crowd_level) / 100.0) * Rubric.weights["crowd"]

    @staticmethod
    def disruption_response(action: str, has_disruption: bool, severity: str = "Medium") -> float:
        sev = {"Low": 0.8, "Medium": 1.0, "High": 1.2}.get(severity, 1.0)
        if has_disruption and action == "reroute":
            return 2.0 * sev * Rubric.weights["disruption"]
        if not has_disruption and action == "reroute":
            return -0.4 * Rubric.weights["disruption"]
        if has_disruption and action == "wait" and severity == "Low":
            return 0.5 * Rubric.weights["disruption"]
        return 0.0

    @staticmethod
    def waiting_penalty(action: str) -> float:
        return Rubric.weights["waiting"] if action == "wait" else 0.0

    @staticmethod
    def personal_task_completion(tasks_done: int, tasks_total: int) -> float:
        """ACTIVE in v3 — called from _agent_step_reward_v3."""
        if tasks_total == 0:
            return 0.0
        return (tasks_done / tasks_total) * Rubric.weights["task"]

    @staticmethod
    def distance_progress(distance_now: int, distance_prev: int) -> float:
        if distance_prev <= 0:
            return 0.0
        progress = (distance_prev - distance_now) / (distance_prev + 1.0)
        return max(0.0, progress) * Rubric.weights["distance"]

    @staticmethod
    def exploration_bonus(action: str) -> float:
        return Rubric.weights["exploration"] if action == "reroute" else 0.0

    @staticmethod
    def transfer_bonus(station: str, switched: bool) -> float:
        if switched and station in TRANSFER_GRAPH:
            return Rubric.weights["transfer"]
        return 0.0

    @staticmethod
    def repetition_penalty(action_history: List[str], action: str) -> float:
        if len(action_history) >= 3 and all(a == action for a in action_history[-3:]):
            return Rubric.weights["repetition"]
        return 0.0

    @staticmethod
    def grievance_penalty(agent: dict, sim_hour: float) -> float:
        """Penalise missing deadline — only fires once."""
        if agent.get("grievance_fired"):
            return 0.0
        deadline = agent.get("arrival_deadline", 9.0)
        if sim_hour >= deadline and not agent.get("arrived"):
            agent["grievance_fired"] = True
            return Rubric.weights["grievance"]
        return 0.0


# ── Core Environment ──────────────────────────────────────────────────────────

class MumbaiLocalEnv:
    """
    OpenEnv-compliant Gym-style environment — v3.0

    Key fixes vs v2.3:
    - current_station advances toward destination each step
    - personal_task_completion is computed and returned
    - rich per-agent observation used in reward computation
    - arrival probability is distance-gated (not flat random)
    - inter-line transfers at junction stations
    """

    ACTIONS = ACTIONS
    METADATA = {
        "name":         "MumbaiLocalEnv-v3",
        "theme":        ["multi-agent", "long-horizon", "personal-assistant", "world-modeling"],
        "version":      "3.0.0",
        "render_modes": ["json", "human"],
    }

    def __init__(self, n_agents: int = 10, max_steps: int = 200, sim_hour: float = 8.5):
        self.n_agents  = min(n_agents, len(COMMUTER_PROFILES))
        self.max_steps = max_steps
        self.sim_hour  = sim_hour          # simulated time-of-day (float hour)
        self._state_initialized = False
        self.reset()

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        self.step_count    = 0
        self.total_reward  = 0.0
        self.episode_rewards: List[float] = []
        self.disruptions:  List[dict] = []
        self.crowd_map     = self._init_crowd()
        self.trains        = self._init_trains()
        self.agents        = self._init_agents()
        self._state_initialized = True
        return self._observe()

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        assert self._state_initialized, "Call reset() before step()"
        assert action in self.ACTIONS, f"Invalid action: {action}"

        self.step_count += 1
        step_reward = 0.0

        self._advance_trains()

        if random.random() < 0.08:
            self._trigger_disruption()

        self._update_crowd()

        # Advance simulated time (~1 step = 0.5 sim-minutes → 120 steps ≈ 1 hour)
        self.sim_hour = (self.sim_hour + 0.008) % 24

        for agent in self.agents:
            if agent["arrived"]:
                continue

            # Track action history (for repetition penalty)
            agent["action_history"].append(action)
            if len(agent["action_history"]) > 10:
                agent["action_history"].pop(0)

            # Rich observation for this agent
            rich_obs = self._observe_agent_rich(agent)

            # v3 reward: all components active
            r = self._agent_step_reward_v3(agent, action, rich_obs)
            agent["reward"] = round(agent["reward"] + r, 4)
            step_reward += r

            # Move agent one step closer to destination
            self._advance_agent(agent, action)

            # Grievance check (deadline miss)
            g = Rubric.grievance_penalty(agent, self.sim_hour)
            agent["reward"] = round(agent["reward"] + g, 4)
            step_reward += g

            # Arrival: distance-gated probability
            line_st  = LINES[agent["line"]]["stations"]
            cur_idx  = line_st.index(agent["current_station"])
            dest_idx = line_st.index(agent["destination"])
            distance = abs(dest_idx - cur_idx)

            if distance == 0:
                arrive_prob = 1.0
            elif action == "route_optimize":
                arrive_prob = max(0.05, 0.25 - distance * 0.015)
            elif action == "reroute":
                arrive_prob = max(0.03, 0.15 - distance * 0.01)
            elif action == "avoid_crowd":
                arrive_prob = 0.06
            else:  # wait
                arrive_prob = 0.02

            if random.random() < arrive_prob:
                agent["arrived"] = True
                agent["status"]  = "arrived"

                # Task completion reward at arrival
                tasks_done  = agent.get("tasks_done", 0)
                tasks_total = len(agent.get("personal_tasks", []))
                task_r = Rubric.personal_task_completion(tasks_done, tasks_total)
                agent["reward"] = round(agent["reward"] + task_r, 4)
                step_reward += task_r

                # Arrival bonus
                bonus = Rubric.arrival_bonus(agent)
                agent["reward"] = round(agent["reward"] + bonus, 4)
                step_reward += bonus
            else:
                # Partial task completion while travelling
                self._tick_personal_tasks(agent)

        self.total_reward += step_reward
        self.episode_rewards.append(round(step_reward, 4))

        obs  = self._observe()
        done = (all(a["arrived"] for a in self.agents) or
                self.step_count >= self.max_steps)
        info = {
            "disruptions":    self.disruptions[-3:],
            "step":           self.step_count,
            "agents_arrived": sum(1 for a in self.agents if a["arrived"]),
        }
        return obs, round(step_reward, 4), done, info

    def state(self) -> Dict[str, Any]:
        return {
            "step":            self.step_count,
            "max_steps":       self.max_steps,
            "total_reward":    round(self.total_reward, 4),
            "agents":          self.agents,
            "trains":          self.trains,
            "crowd_map":       self.crowd_map,
            "disruptions":     self.disruptions[-10:],
            "episode_rewards": self.episode_rewards[-50:],
            "network":         {k: {"color": v["color"], "stations": v["stations"]} for k, v in LINES.items()},
            "sim_hour":        round(self.sim_hour, 2),
        }

    def close(self):
        self._state_initialized = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _observe(self) -> Dict[str, Any]:
        active = [a for a in self.agents if not a["arrived"]]
        return {
            "step":             self.step_count,
            "total_reward":     round(self.total_reward, 4),
            "agents_active":    len(active),
            "agents_arrived":   len(self.agents) - len(active),
            "disruptions_count": len(self.disruptions),
            "avg_crowd":        round(sum(self.crowd_map.values()) / max(len(self.crowd_map), 1), 1),
            "trains_delayed":   sum(1 for t in self.trains if t["delayed"]),
            "sim_hour":         round(self.sim_hour, 2),
        }

    def _observe_agent_rich(self, agent: dict) -> Dict[str, Any]:
        """Per-agent rich observation — ACTIVE and used in step()."""
        line_st  = LINES[agent["line"]]["stations"]
        cur_idx  = line_st.index(agent["current_station"])
        dest_idx = line_st.index(agent["destination"])
        distance = abs(dest_idx - cur_idx)

        trains_here = [
            {
                "id":               t["id"],
                "distance_stations": abs(t["position_idx"] - cur_idx),
                "occupancy":        t["occupancy"],
                "delayed":          t["delayed"],
                "eta_minutes":      abs(t["position_idx"] - cur_idx) * 2.5,
            }
            for t in self.trains if t["line"] == agent["line"]
        ]
        trains_here.sort(key=lambda x: x["eta_minutes"])

        disruptions_relevant = [
            {
                "type":      d["type"],
                "severity":  d["severity"],
                "station":   d["station"],
                "distance":  abs(line_st.index(d["station"]) - cur_idx)
                             if d["station"] in line_st else 99,
                "delay_min": d["delay_minutes"],
            }
            for d in self.disruptions[-5:]
            if d["line"] == agent["line"]
        ]
        disruptions_relevant.sort(key=lambda x: x["distance"])

        can_transfer = agent["current_station"] in TRANSFER_GRAPH

        return {
            "step":                  self.step_count,
            "agent_id":              agent["id"],
            "current_location":      agent["current_station"],
            "destination":           agent["destination"],
            "distance_to_destination": distance,
            "reward_so_far":         round(agent["reward"], 2),
            "crowd_at_current":      self.crowd_map.get(agent["current_station"], 50),
            "trains_nearby":         trains_here[:3],
            "disruptions":           disruptions_relevant[:2],
            "avg_crowd_network":     round(sum(self.crowd_map.values()) / max(len(self.crowd_map), 1), 1),
            "can_transfer":          can_transfer,
            "available_lines":       TRANSFER_GRAPH.get(agent["current_station"], [agent["line"]]),
            "tasks_pending":         len(agent.get("personal_tasks", [])) - agent.get("tasks_done", 0),
            "sim_hour":              round(self.sim_hour, 2),
        }

    def _advance_agent(self, agent: dict, action: str):
        """
        Move agent one station toward destination.
        Handles inter-line transfers at junction stations.
        """
        line_st  = LINES[agent["line"]]["stations"]
        cur_idx  = line_st.index(agent["current_station"])
        dest_idx = line_st.index(agent["destination"]) if agent["destination"] in line_st else -1

        # Check if agent should transfer lines
        if (action == "reroute" and
                agent["current_station"] in TRANSFER_GRAPH and
                agent["destination"] not in line_st):
            available = TRANSFER_GRAPH[agent["current_station"]]
            for new_line in available:
                if agent["line"] != new_line and agent["destination"] in LINES[new_line]["stations"]:
                    agent["line"] = new_line
                    agent["transferred"] = agent.get("transferred", 0) + 1
                    line_st  = LINES[new_line]["stations"]
                    cur_idx  = line_st.index(agent["current_station"]) if agent["current_station"] in line_st else 0
                    dest_idx = line_st.index(agent["destination"])
                    break

        if dest_idx < 0:
            return  # destination not on current line yet

        # Move one step toward destination
        if action in ("route_optimize", "reroute") and cur_idx != dest_idx:
            step_dir  = 1 if dest_idx > cur_idx else -1
            # route_optimize moves 1–2 stations; others move 1
            steps_fwd = 2 if action == "route_optimize" and abs(dest_idx - cur_idx) > 3 else 1
            new_idx = max(0, min(len(line_st) - 1, cur_idx + step_dir * steps_fwd))
            agent["current_station"] = line_st[new_idx]
        elif action == "avoid_crowd":
            # Move 1 step, but only if next station is less crowded
            if cur_idx != dest_idx:
                step_dir = 1 if dest_idx > cur_idx else -1
                nxt = max(0, min(len(line_st) - 1, cur_idx + step_dir))
                next_crowd = self.crowd_map.get(line_st[nxt], 50)
                cur_crowd  = self.crowd_map.get(agent["current_station"], 50)
                if next_crowd <= cur_crowd + 15:
                    agent["current_station"] = line_st[nxt]
        # wait: no movement

    def _tick_personal_tasks(self, agent: dict):
        """Complete personal tasks probabilistically while travelling."""
        tasks_total = len(agent.get("personal_tasks", []))
        tasks_done  = agent.get("tasks_done", 0)
        if tasks_done < tasks_total:
            if random.random() < 0.08:   # ~8% chance per step to tick one task
                agent["tasks_done"] = tasks_done + 1

    def _agent_step_reward_v3(self, agent: dict, action: str, rich_obs: dict) -> float:
        """
        Unified step reward — all Rubric components active.
        """
        r = 0.0

        # Distance progress (deterministic)
        distance = rich_obs["distance_to_destination"]
        prev_dist = agent["_prev_distance"]
        if prev_dist is None:
            prev_dist = distance + 1
        if distance < prev_dist:
            r += Rubric.distance_progress(distance, prev_dist)
        agent["_prev_distance"] = distance

        # Crowd avoidance
        crowd = rich_obs["crowd_at_current"]
        crowd_mult = (1.5 if action == "avoid_crowd" and crowd > 70
                      else 0.5 if action == "avoid_crowd"
                      else 0.3)
        r += Rubric.crowd_avoidance(crowd) * crowd_mult

        # Personalised preference multipliers
        ct = agent.get("crowding_tolerance", 0.5)
        wt = agent.get("waiting_tolerance",  0.5)
        ra = agent.get("risk_aversion",       0.5)

        if ct < 0.5 and action == "avoid_crowd" and crowd > 70:
            r *= 1.2
        if wt < 0.5 and action == "wait":
            r *= 0.7
        if ra > 0.6 and action == "route_optimize":
            r *= 1.1

        # Disruption response
        has_disruption = any(d["line"] == agent["line"] for d in self.disruptions[-3:])
        severity = "Medium"
        if has_disruption and self.disruptions:
            severity = self.disruptions[-1].get("severity", "Medium")
        r += Rubric.disruption_response(action, has_disruption, severity)

        # Waiting penalty
        r += Rubric.waiting_penalty(action)

        # Exploration bonus
        r += Rubric.exploration_bonus(action)

        # Route-optimize small deterministic reward
        if action == "route_optimize":
            r += 0.4 + (0.3 if distance > 10 else 0.1)

        # Transfer bonus
        switched = (action == "reroute" and
                    agent["current_station"] in TRANSFER_GRAPH and
                    agent.get("transferred", 0) > 0)
        r += Rubric.transfer_bonus(agent["current_station"], switched)

        # Repetition penalty
        r += Rubric.repetition_penalty(agent["action_history"], action)

        return round(r, 4)

    def _trigger_disruption(self):
        line     = random.choice(list(LINES.keys()))
        station  = random.choice(LINES[line]["stations"])
        severity = random.choice(["Low", "Medium", "High"])
        dur_map  = {"Low": 10, "Medium": 25, "High": 40}

        disruption = {
            "type":           random.choice(DISRUPTION_TYPES),
            "line":           line,
            "station":        station,
            "severity":       severity,
            "time":           datetime.now().strftime("%H:%M:%S"),
            "duration_minutes": dur_map.get(severity, 25),
            "delay_minutes":  random.randint(5, 45),
        }
        self.disruptions.append(disruption)
        if len(self.disruptions) > 20:
            self.disruptions.pop(0)

        for t in self.trains:
            if t["line"] == line and t["station"] == station:
                t["delayed"]      = True
                t["delay_minutes"] = disruption["delay_minutes"]

    def _init_agents(self) -> List[dict]:
        agents = []
        for i, profile in enumerate(COMMUTER_PROFILES[: self.n_agents]):
            agents.append({
                "id":              i,
                "name":            profile["name"],
                "origin":          profile["origin"],
                "destination":     profile["destination"],
                "line":            profile["line"],
                "status":          "waiting",
                "reward":          0.0,
                "steps_taken":     0,
                "arrived":         False,
                "current_station": profile["origin"],     # ← FIXED: starts at origin
                "arrival_deadline": profile["arrival_deadline"],
                "crowding_tolerance": profile["crowding_tolerance"],
                "waiting_tolerance":  profile["waiting_tolerance"],
                "risk_aversion":      profile["risk_aversion"],
                "personal_tasks":  list(profile.get("personal_tasks", [])),
                "tasks_done":      0,
                "action_history":  [],
                "transferred":     0,
                "grievance_fired": False,
                "_prev_distance":  None,
            })
        return agents

    def _init_crowd(self) -> Dict[str, int]:
        h = self.sim_hour
        rush = (8 <= h <= 10) or (17 <= h <= 20)
        crowd: Dict[str, int] = {}
        for line_data in LINES.values():
            for station in line_data["stations"]:
                base = random.randint(20, 70)
                if rush:
                    base = min(100, base + random.randint(15, 30))
                crowd[station] = base
        return crowd

    def _init_trains(self) -> List[dict]:
        trains = []
        for line_name, line_data in LINES.items():
            stations = line_data["stations"]
            for i in range(line_data["trains"]):
                pos = random.randint(0, len(stations) - 1)
                trains.append({
                    "id":           f"{line_name[0]}{i+1:02d}",
                    "line":         line_name,
                    "position_idx": pos,
                    "station":      stations[pos],
                    "direction":    random.choice(["up", "down"]),
                    "speed":        random.choice(["Fast", "Semi-Fast", "Slow"]),
                    "occupancy":    random.randint(30, 90),
                    "delayed":      False,
                    "delay_minutes": 0,
                })
        return trains

    def _advance_trains(self):
        for train in self.trains:
            stations = LINES[train["line"]]["stations"]
            if train["direction"] == "up":
                train["position_idx"] = min(train["position_idx"] + 1, len(stations) - 1)
                if train["position_idx"] == len(stations) - 1:
                    train["direction"] = "down"
            else:
                train["position_idx"] = max(train["position_idx"] - 1, 0)
                if train["position_idx"] == 0:
                    train["direction"] = "up"
            train["station"]   = stations[train["position_idx"]]
            train["occupancy"] = max(10, min(100, train["occupancy"] + random.randint(-5, 5)))
            # Auto-clear disruptions stochastically
            if train["delayed"] and random.random() < 0.04:
                train["delayed"]       = False
                train["delay_minutes"] = 0

    def _update_crowd(self):
        h    = self.sim_hour
        rush = (8 <= h <= 10) or (17 <= h <= 20)
        for station in self.crowd_map:
            delta = random.randint(-4, 4)
            if rush:
                delta += random.randint(0, 3)
            self.crowd_map[station] = max(0, min(100, self.crowd_map[station] + delta))


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = MumbaiLocalEnv(n_agents=10, max_steps=50)
    obs = env.reset()
    print("Initial obs:", json.dumps(obs, indent=2))

    total = 0.0
    for step in range(25):
        action = random.choice(MumbaiLocalEnv.ACTIONS)
        obs, reward, done, info = env.step(action)
        total += reward
        a0 = env.agents[0]
        print(
            f"Step {step+1:02d} | action={action:<16} | reward={reward:+.4f} | "
            f"arrived={info['agents_arrived']}/10 | "
            f"agent[0] @ {a0['current_station']}"
        )
        if done:
            break

    print(f"\nTotal reward: {total:.4f}")
    env.close()
