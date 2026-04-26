"""
Mumbai Local OpenEnv — Flask Dashboard  v3.0
=============================================
All v3 environment features are exposed via REST API:
  /api/agent/<id>          — rich per-agent state
  /api/transfer            — inject a line transfer
  /api/benchmark           — run 5 heuristic episodes and return stats
  /api/tasks               — live personal task completion status
"""

from flask import Flask, render_template, jsonify, request
import random, json, math, threading, time, os, argparse
from datetime import datetime

from environment import (
    MumbaiLocalEnv, LINES, COMMUTER_PROFILES, ACTIONS,
    DISRUPTION_TYPES, TRANSFER_GRAPH
)

app = Flask(__name__)

# ── NL command parser ─────────────────────────────────────────────────────────

NL_PATTERNS = [
    (["avoid crowd","avoid crowds","less crowd","low crowd","uncrowded"],  "avoid_crowd"),
    (["reroute","redirect","alternate","bypass","transfer"],               "reroute"),
    (["fast","quick","optimize","best route","shortest","send all"],       "route_optimize"),
    (["wait","hold","stop","pause"],                                        "wait"),
]

def parse_nl_command(text):
    t = text.lower()
    for patterns, action in NL_PATTERNS:
        for p in patterns:
            if p in t:
                return action, p
    return "route_optimize", None


# ── World (wraps MumbaiLocalEnv for the dashboard) ───────────────────────────

class World:
    def __init__(self):
        self._lock    = threading.Lock()
        self.leaderboard      = []
        self.episode_archive  = {}
        self._env_instance    = None
        self._init_fresh()

    def _init_fresh(self):
        self.step_count   = 0
        self.episode      = 0
        self.epsilon      = 1.0
        self.last_action  = "—"
        self.last_reward  = 0.0
        self.last_loss    = 1.8
        self.is_auto      = False
        self.sim_hour     = float(datetime.now().hour) + datetime.now().minute / 60.0
        self._cum_reward  = 0.0
        self.total_reward = 0.0
        self.reward_history   = []
        self.loss_history     = []
        self.step_reward_hist = []
        self.arrival_history  = []
        self.epsilon_history  = []

        # Create a live env instance
        self._env_instance = MumbaiLocalEnv(
            n_agents=len(COMMUTER_PROFILES),
            max_steps=400,
            sim_hour=self.sim_hour,
        )
        self._env_instance.reset()

    def reset_state(self, save_episode=True):
        with self._lock:
            if save_episode and self.episode > 0:
                self._save_episode_locked()
            self._init_fresh()

    # ── Rush / crowd helpers ──────────────────────────────────────────────────

    def _rush_factor(self):
        h = self.sim_hour % 24
        if 8 <= h <= 10:  return 1.55
        if 17 <= h <= 20: return 1.45
        if 22 <= h or h <= 5: return 0.35
        return 1.0

    def _save_episode_locked(self):
        ep  = self.episode
        env = self._env_instance
        arr = sum(1 for a in env.agents if a["arrived"])
        score = round(self.total_reward + arr * 20 - len(env.disruptions) * 5, 2)
        entry = {
            "episode":       ep,
            "score":         score,
            "total_reward":  round(self.total_reward, 2),
            "arrival_rate":  round(arr / max(len(env.agents), 1) * 100, 1),
            "avg_reward":    round(self.total_reward / max(self.step_count, 1), 4),
            "disruptions":   len(env.disruptions),
            "steps":         self.step_count,
            "timestamp":     datetime.now().strftime("%H:%M:%S"),
        }
        self.episode_archive[ep] = {
            "metrics": entry,
            "history": {
                "rewards":  list(self.reward_history[-80:]),
                "losses":   list(self.loss_history[-80:]),
                "step_r":   list(self.step_reward_hist[-80:]),
                "arrivals": list(self.arrival_history[-80:]),
            },
        }
        existing = [x for x in self.leaderboard if x["episode"] != ep]
        existing.append(entry)
        self.leaderboard = sorted(existing, key=lambda x: x["score"], reverse=True)[:10]

    def tick(self, action=None):
        with self._lock:
            env = self._env_instance
            if action is None:
                action = (random.choice(ACTIONS) if random.random() < self.epsilon
                          else "route_optimize")

            self.step_count  += 1
            self.last_action  = action
            self.sim_hour     = (self.sim_hour + 0.008) % 24

            obs, step_r, done, info = env.step(action)

            if done:
                self._save_episode_locked()
                self.episode += 1
                env.reset()

            self.epsilon  = max(0.05, self.epsilon - 0.002)
            self.last_loss = max(0.04, self.last_loss * 0.995 + random.gauss(0, 0.01))
            self.last_reward   = round(step_r, 4)
            self._cum_reward  += step_r
            self.total_reward  = round(self._cum_reward, 2)

            self.reward_history.append(round(self._cum_reward, 2))
            self.loss_history.append(round(self.last_loss, 4))
            self.step_reward_hist.append(round(step_r, 4))
            self.arrival_history.append(info["agents_arrived"])
            self.epsilon_history.append(round(self.epsilon, 4))

            for lst in (self.reward_history, self.loss_history, self.step_reward_hist,
                        self.arrival_history, self.epsilon_history):
                if len(lst) > 200:
                    lst.pop(0)

    def get_station_info(self, station_name):
        with self._lock:
            env   = self._env_instance
            crowd = env.crowd_map.get(station_name, 0)
            agents_here   = [
                {"name": a["name"], "line": a["line"], "dest": a["destination"]}
                for a in env.agents
                if a["current_station"] == station_name and not a["arrived"]
            ]
            trains_here   = [
                {"id": t["id"], "line": t["line"], "occ": t["occupancy"], "delayed": t["delayed"]}
                for t in env.trains if t["station"] == station_name
            ]
            arriving_soon = []
            for t in env.trains:
                st_list = LINES[t["line"]]["stations"]
                if station_name in st_list:
                    sidx = st_list.index(station_name)
                    dist = abs(sidx - t["position_idx"])
                    if 1 <= dist <= 3:
                        arriving_soon.append({
                            "id": t["id"], "line": t["line"],
                            "eta_min": round(dist * 2.5 + random.uniform(0, 1), 1),
                            "delayed": t["delayed"],
                        })
            arriving_soon.sort(key=lambda x: x["eta_min"])

            can_transfer = station_name in TRANSFER_GRAPH

            return {
                "station":          station_name,
                "crowd":            crowd,
                "agents_here":      agents_here,
                "trains_at_station": trains_here[:5],
                "arriving_soon":    arriving_soon[:4],
                "disruptions":      [d for d in env.disruptions[-5:] if d["station"] == station_name],
                "can_transfer":     can_transfer,
                "transfer_lines":   TRANSFER_GRAPH.get(station_name, []),
            }

    def get_agent_info(self, agent_id: int):
        with self._lock:
            env = self._env_instance
            try:
                agent = env.agents[agent_id]
            except IndexError:
                return None
            rich = env._observe_agent_rich(agent)
            return {
                **agent,
                "rich_obs": rich,
            }

    def get_tasks_status(self):
        with self._lock:
            env = self._env_instance
            return [
                {
                    "name":        a["name"],
                    "tasks":       a["personal_tasks"],
                    "tasks_done":  a["tasks_done"],
                    "arrived":     a["arrived"],
                }
                for a in env.agents
            ]

    def snapshot(self):
        with self._lock:
            env = self._env_instance
            return {
                "step":             self.step_count,
                "episode":          self.episode,
                "total_reward":     self.total_reward,
                "last_reward":      self.last_reward,
                "last_loss":        round(self.last_loss, 4),
                "epsilon":          round(self.epsilon, 4),
                "last_action":      self.last_action,
                "is_auto":          self.is_auto,
                "sim_hour":         round(self.sim_hour, 2),
                "agents_arrived":   sum(1 for a in env.agents if a["arrived"]),
                "agents_active":    sum(1 for a in env.agents if not a["arrived"]),
                "disruptions_count": len(env.disruptions),
                "avg_crowd":        round(sum(env.crowd_map.values()) / max(len(env.crowd_map), 1), 1),
                "trains_delayed":   sum(1 for t in env.trains if t["delayed"]),
                "agents":           list(env.agents),
                "trains":           list(env.trains),
                "crowd":            dict(env.crowd_map),
                "disruptions":      env.disruptions[-10:],
                "leaderboard":      list(self.leaderboard[:5]),
                "transfer_nodes":   list(TRANSFER_GRAPH.keys()),
                "history": {
                    "rewards":  list(self.reward_history),
                    "losses":   list(self.loss_history),
                    "step_r":   list(self.step_reward_hist),
                    "arrivals": list(self.arrival_history),
                    "epsilon":  list(self.epsilon_history),
                },
            }


world = World()

# ── Auto-step background thread ───────────────────────────────────────────────

def auto_loop():
    while True:
        if world.is_auto:
            world.tick()
        time.sleep(0.15)

threading.Thread(target=auto_loop, daemon=True).start()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state")
def api_state():
    return jsonify(world.snapshot())

@app.route("/api/step", methods=["POST"])
def api_step():
    data   = request.get_json() or {}
    action = data.get("action", "route_optimize")
    if action not in ACTIONS:
        return jsonify({"error": f"Invalid action. Choose from: {ACTIONS}"}), 400
    world.tick(action)
    return jsonify(world.snapshot())

@app.route("/api/auto", methods=["POST"])
def api_auto():
    world.is_auto = not world.is_auto
    return jsonify({"is_auto": world.is_auto})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    world.reset_state(save_episode=True)
    return jsonify({"ok": True})

@app.route("/api/disrupt", methods=["POST"])
def api_disrupt():
    data = request.get_json() or {}
    with world._lock:
        env  = world._env_instance
        line = data.get("line") or random.choice(list(LINES.keys()))
        st   = data.get("station") or random.choice(LINES[line]["stations"])
        sev  = data.get("severity") or random.choice(["Low", "Medium", "High"])
        dtype = data.get("type") or random.choice(DISRUPTION_TYPES)
        d    = {
            "type": dtype, "line": line, "station": st, "severity": sev,
            "time": datetime.now().strftime("%H:%M:%S"),
            "duration_minutes": {"Low": 10, "Medium": 25, "High": 40}.get(sev, 25),
            "delay_minutes": random.randint(5, 45),
        }
        env.disruptions.append(d)
        if len(env.disruptions) > 20:
            env.disruptions.pop(0)
    return jsonify({**world.snapshot(), "triggered": d})

@app.route("/api/command", methods=["POST"])
def api_command():
    data   = request.get_json() or {}
    text   = data.get("text", "")
    action, matched = parse_nl_command(text)
    world.tick(action)
    snap   = world.snapshot()
    return jsonify({**snap, "interpreted_action": action, "matched_keyword": matched, "nl_text": text})

@app.route("/api/station/<path:station_name>")
def api_station(station_name):
    return jsonify(world.get_station_info(station_name))

@app.route("/api/agent/<int:agent_id>")
def api_agent(agent_id):
    info = world.get_agent_info(agent_id)
    if info is None:
        return jsonify({"error": "agent not found"}), 404
    return jsonify(info)

@app.route("/api/tasks")
def api_tasks():
    return jsonify(world.get_tasks_status())

@app.route("/api/transfer", methods=["POST"])
def api_transfer():
    """Force an agent to transfer lines at their current station."""
    data     = request.get_json() or {}
    agent_id = int(data.get("agent_id", 0))
    with world._lock:
        env = world._env_instance
        try:
            agent = env.agents[agent_id]
        except IndexError:
            return jsonify({"error": "agent not found"}), 404

        station = agent["current_station"]
        if station not in TRANSFER_GRAPH:
            return jsonify({"error": f"{station} is not a transfer node", "transfer_nodes": list(TRANSFER_GRAPH.keys())}), 400

        available = [l for l in TRANSFER_GRAPH[station] if l != agent["line"]]
        if not available:
            return jsonify({"error": "no alternative lines at this station"}), 400

        new_line = data.get("to_line") or available[0]
        if new_line not in available:
            return jsonify({"error": f"{new_line} not available at {station}"}), 400

        old_line         = agent["line"]
        agent["line"]    = new_line
        agent["transferred"] = agent.get("transferred", 0) + 1

    return jsonify({
        "agent":    agent["name"],
        "from":     old_line,
        "to":       new_line,
        "station":  station,
        **world.snapshot(),
    })

@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """Run 5 heuristic episodes and return benchmark stats."""
    from inference import heuristic_action
    results = []
    for ep in range(5):
        _env  = MumbaiLocalEnv(n_agents=len(COMMUTER_PROFILES), max_steps=60)
        _env.reset()
        total = 0.0
        for _step in range(60):
            actions = []
            for agent in _env.agents:
                if agent["arrived"]:
                    actions.append("wait")
                    continue
                rich  = _env._observe_agent_rich(agent)
                actions.append(heuristic_action(rich, agent, _env.disruptions))
            from collections import Counter
            action = Counter(actions).most_common(1)[0][0]
            _, r, done, info = _env.step(action)
            total += r
            if done:
                break
        arrived = sum(1 for a in _env.agents if a["arrived"])
        results.append({
            "episode": ep + 1,
            "total_reward": round(total, 2),
            "agents_arrived": arrived,
        })
        _env.close()

    avg_r = sum(r["total_reward"]   for r in results) / len(results)
    avg_a = sum(r["agents_arrived"] for r in results) / len(results)
    return jsonify({
        "benchmark_episodes": results,
        "avg_reward":  round(avg_r, 2),
        "avg_arrived": round(avg_a, 1),
        "mode":        "heuristic-baseline-v3",
    })

@app.route("/api/leaderboard")
def api_leaderboard():
    with world._lock:
        return jsonify({
            "leaderboard":   list(world.leaderboard),
            "episode_keys":  list(world.episode_archive.keys()),
        })

@app.route("/api/episode/<int:ep_num>")
def api_episode(ep_num):
    with world._lock:
        data = world.episode_archive.get(ep_num)
    if not data:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)

@app.route("/api/time", methods=["POST"])
def api_time():
    data = request.get_json() or {}
    hour = int(data.get("hour", datetime.now().hour)) % 24
    with world._lock:
        world.sim_hour = float(hour)
        world._env_instance.sim_hour = float(hour)
        rf = world._rush_factor()
        for s in world._env_instance.crowd_map:
            target = int(random.randint(20, 65) * rf)
            world._env_instance.crowd_map[s] = min(100, max(0, target))
    return jsonify(world.snapshot())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    args = parser.parse_args()
    app.run(debug=False, host="0.0.0.0", port=args.port, threaded=True)
