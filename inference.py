"""
Mumbai Local OpenEnv — Inference Script  v3.0
==============================================
Run a trained GRPO model (or the built-in heuristic baseline) against
MumbaiLocalEnv and print per-agent decisions with rewards.

Changes vs v2.3:
  - Heuristic uses rich per-agent observations (distance, crowd, disruptions)
  - LLM prompt uses _observe_agent_rich (richer context)
  - Per-agent action is now individual, not majority-vote
  - Results include per-agent task completion and transfer counts

Usage
-----
  # 1. Heuristic baseline (no GPU, no model):
      python inference.py

  # 2. Locally fine-tuned checkpoint:
      python inference.py --model ./mumbai-local-grpo/final

  # 3. HuggingFace Hub checkpoint:
      python inference.py --model YOUR_HF_USERNAME/mumbai-local-grpo

  # 4. More episodes, verbose:
      python inference.py --episodes 5 --steps 50 --verbose
"""

import argparse
import json
import random
from typing import Optional

from environment import MumbaiLocalEnv, COMMUTER_PROFILES, ACTIONS

_tokenizer = None
_model     = None


def _load_model(model_id: str):
    global _tokenizer, _model
    if _tokenizer is not None:
        return
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        print(f"[inference] Loading model: {model_id}")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model     = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print("[inference] Model loaded ✓")
    except ImportError:
        raise SystemExit("[inference] ERROR: `transformers` not installed.\nRun: pip install transformers torch")


def build_prompt(rich_obs: dict, agent: dict, disruptions: list) -> str:
    dis_text = ""
    if rich_obs.get("disruptions"):
        d        = rich_obs["disruptions"][0]
        dis_text = (f"\n⚠ DISRUPTION: {d['type']} at {d['station']} "
                    f"({d['severity']}, +{d['delay_min']} min)")

    trains_text = ""
    if rich_obs.get("trains_nearby"):
        trains_text = "\nNearby trains:\n"
        for t in rich_obs["trains_nearby"][:2]:
            trains_text += (
                f"  • {t['id']}: {t['occupancy']}% full, "
                f"ETA {t['eta_minutes']:.0f} min"
                f"{' [DELAYED]' if t['delayed'] else ''}\n"
            )

    transfer_text = ""
    if rich_obs.get("can_transfer"):
        transfer_text = (
            f"\n🔀 Transfer available: {', '.join(rich_obs['available_lines'])}"
        )

    return (
        f"You are a smart Mumbai local train commuter agent.\n"
        f"Goal: {agent['origin']} → {agent['destination']} via {agent['line']} line.\n\n"
        f"State:\n"
        f"  Location  : {rich_obs['current_location']}\n"
        f"  Distance  : {rich_obs['distance_to_destination']} stations\n"
        f"  Crowd     : {rich_obs['crowd_at_current']}%\n"
        f"  Sim time  : {rich_obs['sim_hour']:.1f}h\n"
        f"  Reward    : {rich_obs['reward_so_far']:.2f}"
        f"{trains_text}{dis_text}{transfer_text}\n\n"
        f"Choose ONE action: {', '.join(ACTIONS)}\n"
        f"Action:"
    )


def llm_action(prompt: str) -> str:
    import torch
    inputs  = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(_model.device)
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    decoded = _tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip().lower()
    for action in ACTIONS:
        if action in decoded:
            return action
    return "route_optimize"


def heuristic_action(rich_obs: dict, agent: dict, disruptions: list) -> str:
    """
    Rule-based baseline using rich per-agent observation.
    Mimics what a well-trained agent should learn.
    """
    distance = rich_obs["distance_to_destination"]
    crowd    = rich_obs["crowd_at_current"]
    dis      = rich_obs.get("disruptions", [])
    transfer = rich_obs.get("can_transfer", False)

    # If there's a nearby high-severity disruption → reroute (especially at transfer nodes)
    if dis and dis[0]["severity"] in ("High", "Medium"):
        if transfer:
            return "reroute"   # smart transfer
        return "reroute"

    # Very crowded and agent is crowd-sensitive → avoid_crowd
    if crowd > 80 and agent.get("crowding_tolerance", 0.5) < 0.5:
        return "avoid_crowd"

    # Close to destination → push hard
    if distance <= 5:
        return "route_optimize"

    # Mild disruption and close to destination → wait it out if patience allows
    if dis and dis[0]["severity"] == "Low" and agent.get("waiting_tolerance", 0.5) > 0.5:
        return "wait"

    # General crowd avoidance in rush hour
    if crowd > 70:
        return "avoid_crowd"

    return "route_optimize"


def run_inference(
    model_id:  Optional[str],
    n_episodes: int,
    n_steps:    int,
    verbose:    bool,
):
    use_llm = model_id is not None
    if use_llm:
        _load_model(model_id)

    env             = MumbaiLocalEnv(n_agents=len(COMMUTER_PROFILES))
    episode_results = []

    for ep in range(n_episodes):
        obs       = env.reset()
        total_r   = 0.0
        step_log  = []

        print(f"\n{'='*60}")
        print(f"  Episode {ep + 1} / {n_episodes}")
        print(f"{'='*60}")

        for step in range(n_steps):
            disruptions = env.disruptions

            # Per-agent individual action selection
            agent_actions = []
            for agent in env.agents:
                if agent.get("arrived"):
                    agent_actions.append("wait")
                    continue

                rich_obs = env._observe_agent_rich(agent)

                if use_llm:
                    prompt = build_prompt(rich_obs, agent, disruptions)
                    action = llm_action(prompt)
                else:
                    action = heuristic_action(rich_obs, agent, disruptions)

                agent_actions.append(action)

            # Step with majority action (env takes one global action)
            from collections import Counter
            action  = Counter(agent_actions).most_common(1)[0][0]
            obs, reward, done, info = env.step(action)
            total_r += reward

            entry = {
                "step":           step + 1,
                "action":         action,
                "reward":         round(reward, 3),
                "agents_active":  obs["agents_active"],
                "agents_arrived": obs["agents_arrived"],
                "disruptions":    len(disruptions),
            }
            step_log.append(entry)

            if verbose:
                print(
                    f"  step {step+1:3d} | action={action:<16} | "
                    f"reward={reward:+.2f} | arrived={obs['agents_arrived']}/10 | "
                    f"disruptions={len(disruptions)} | "
                    f"sim_hour={obs['sim_hour']:.1f}h"
                )

            if done:
                print(f"  [done] All agents arrived at step {step + 1}")
                break

        result = {
            "episode":        ep + 1,
            "total_reward":   round(total_r, 2),
            "agents_arrived": obs["agents_arrived"],
            "steps_taken":    len(step_log),
            "mode":           "llm" if use_llm else "heuristic",
            "model":          model_id or "heuristic-baseline-v3",
            "per_agent": [
                {
                    "name":       a["name"],
                    "reward":     round(a["reward"], 2),
                    "arrived":    a["arrived"],
                    "tasks":      f"{a['tasks_done']}/{len(a['personal_tasks'])}",
                    "transfers":  a["transferred"],
                    "location":   a["current_station"],
                }
                for a in env.agents
            ],
        }
        episode_results.append(result)

        print(f"\n  ── Episode summary ──────────────────────────")
        print(f"  Total reward  : {result['total_reward']}")
        print(f"  Agents arrived: {result['agents_arrived']} / 10")
        print(f"  Steps taken   : {result['steps_taken']}")
        print(f"  Mode          : {result['mode']}")
        print(f"\n  Per-agent:")
        for pa in result["per_agent"]:
            status = "✅" if pa["arrived"] else f"@ {pa['location']}"
            print(f"    {pa['name']:<14} R={pa['reward']:>7.2f}  tasks={pa['tasks']}  "
                  f"xfr={pa['transfers']}  {status}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  INFERENCE COMPLETE")
    print(f"{'='*60}")
    avg_r       = sum(r["total_reward"]   for r in episode_results) / len(episode_results)
    avg_arrived = sum(r["agents_arrived"] for r in episode_results) / len(episode_results)
    print(f"  Episodes   : {n_episodes}")
    print(f"  Avg reward : {avg_r:.2f}")
    print(f"  Avg arrived: {avg_arrived:.1f} / 10")

    out = {
        "summary":  {"avg_reward": avg_r, "avg_arrived": avg_arrived},
        "episodes": episode_results,
    }
    with open("inference_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Results saved → inference_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mumbai Local OpenEnv v3 — inference")
    parser.add_argument("--model",    type=str, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps",    type=int, default=40)
    parser.add_argument("--verbose",  action="store_true", default=False)
    args = parser.parse_args()

    run_inference(
        model_id=args.model,
        n_episodes=args.episodes,
        n_steps=args.steps,
        verbose=args.verbose,
    )
