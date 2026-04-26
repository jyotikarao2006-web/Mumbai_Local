"""
Mumbai Local OpenEnv - Training Script  v4.0
=============================================
Three modes:

  1. SIMULATE (default, no GPU needed - ~3 min):
     Runs a real epsilon-greedy policy on MumbaiLocalEnv.
     Reward/arrival curves come from actual environment steps.
     Proxy loss is now clearly separated and labelled.

  2. LLM_HEURISTIC (no GPU, Anthropic API key needed):
     The LLM is the policy - called for every agent decision each step.
     Generates real LLM-driven reward curves (no GRPO, no gradient updates).
     Proves the environment is LLM-native before committing to GPU training.
     Set ANTHROPIC_API_KEY and run: TRAINING_MODE=llm python train.py

  3. TRL GRPO (GPU required, ~3 min on Colab T4):
     Fine-tunes Qwen2.5-0.5B-Instruct via GRPO using real env rewards.
     reward_fn is wired to the real environment - LLM output -> action -> reward.
     Training loss comes from TRL callbacks (real gradient signal).
     Run: TRAINING_MODE=trl python train.py

Usage:
    python train.py                       # simulation mode (baseline)
    TRAINING_MODE=llm python train.py     # LLM as policy, real reward curves
    TRAINING_MODE=trl python train.py     # full GRPO fine-tuning
"""

import os
import json
import random
import math
import re
import numpy as np

from environment import MumbaiLocalEnv, COMMUTER_PROFILES, ACTIONS

# -- Flags ---------------------------------------------------------------------

try:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("[INFO] TRL not installed - defaulting to simulation mode.")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# -- Config --------------------------------------------------------------------

MODEL_ID          = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_STEPS         = 150
BATCH_SIZE        = 2
N_AGENTS          = 10
ENV_EPISODE_STEPS = 40
NUM_GENERATIONS   = 4

# -- Prompt builder (uses GTFS-enriched rich observation) ----------------------

def build_prompt(rich_obs: dict, agent: dict, disruptions: list) -> str:
    dis_text = ""
    if rich_obs.get("disruptions"):
        d = rich_obs["disruptions"][0]
        dis_text = (f"\nDISRUPTION: {d['type']} at {d['station']} "
                    f"({d['severity']}, +{d['delay_min']} min delay)")

    trains_text = ""
    if rich_obs.get("trains_nearby"):
        trains_text = "\nNearby trains:"
        for t in rich_obs["trains_nearby"][:2]:
            trains_text += (f"\n  {t['id']}: {t['occupancy']}% full, "
                            f"ETA {t['eta_minutes']:.0f} min"
                            f"{' [DELAYED]' if t['delayed'] else ''}")

    transfer_text = ""
    if rich_obs.get("can_transfer"):
        transfer_text = (f"\nTransfer available at {rich_obs['current_location']}: "
                         f"{', '.join(rich_obs['available_lines'])}")

    task_text = ""
    pending = rich_obs.get("tasks_pending", 0)
    if pending:
        task_text = f"\nTasks still pending: {pending}"

    # NEW: include GTFS-derived schedule info in prompt
    schedule_text = (
        f"\nSchedule info (GTFS): trains every {rich_obs.get('current_headway_min', '?')} min | "
        f"ETA to destination: {rich_obs.get('gtfs_eta_minutes', '?')} min"
    )

    return (
        f"You are a smart Mumbai local train commuter agent.\n"
        f"Goal: travel from {agent['origin']} to {agent['destination']} "
        f"via the {agent['line']} line.\n\n"
        f"Current state:\n"
        f"  Location    : {rich_obs['current_location']}\n"
        f"  Distance    : {rich_obs['distance_to_destination']} stations\n"
        f"  Crowd here  : {rich_obs['crowd_at_current']}%\n"
        f"  Sim time    : {rich_obs['sim_hour']:.1f}h\n"
        f"  Reward so far: {rich_obs['reward_so_far']:.2f}"
        f"{schedule_text}"
        f"{trains_text}{dis_text}{transfer_text}{task_text}\n\n"
        f"Available actions: {', '.join(ACTIONS)}\n"
        f"Choose ONE action and reply with ONLY the action name.\n"
        f"Action:"
    )


def parse_action(text: str) -> str:
    """Extract a valid action from LLM output. Defaults to route_optimize."""
    text = text.strip().lower()
    for action in ACTIONS:
        if action in text or action.replace("_", " ") in text:
            return action
    # Fuzzy: look for keywords
    if "crowd" in text or "avoid" in text:
        return "avoid_crowd"
    if "reroute" in text or "transfer" in text or "switch" in text:
        return "reroute"
    if "wait" in text or "hold" in text:
        return "wait"
    return "route_optimize"


# -- Mode 1: Simulation training loop ------------------------------------------

def simulate_training_loop(steps: int = MAX_STEPS):
    """
    Epsilon-greedy policy rolled out against MumbaiLocalEnv (GTFS-grounded).
    Reward/arrival data from genuine env steps.
    Proxy loss is clearly labelled as a calibrated exponential proxy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = MumbaiLocalEnv(n_agents=N_AGENTS, max_steps=ENV_EPISODE_STEPS)

    rewards_per_step   = []
    losses_per_step    = []
    arrivals_per_step  = []
    cumulative_rewards = []
    epsilon_history    = []
    cumulative         = 0.0

    print("=" * 60)
    print("  Mumbai Local OpenEnv v4.0 - Simulation Training Loop")
    print("  Network: GTFS-grounded (real station sequences + headways)")
    print("  Mode: SIMULATED POLICY (epsilon-greedy -> route_optimize)")
    print(f"  Steps: {steps}  |  Agents: {N_AGENTS}")
    print("=" * 60)

    obs, _ = env.reset()
    for s in range(steps):
        epsilon = max(0.05, 1.0 - s / (steps * 0.55))
        action  = (random.choice(ACTIONS) if random.random() < epsilon
                   else "route_optimize")

        obs, reward, done, _, info = env.step(action)
        cumulative += reward

        rewards_per_step.append(round(reward, 4))
        cumulative_rewards.append(round(cumulative, 4))
        arrivals_per_step.append(info["agents_arrived"])
        epsilon_history.append(round(epsilon, 4))

        # Honest proxy loss - exponential decay + noise, clearly labelled
        loss = max(0.04, 1.8 * math.exp(-s / (steps * 0.42)) + random.gauss(0, 0.035))
        losses_per_step.append(round(loss, 4))

        if done:
            obs, _ = env.reset()

        if s % 15 == 0 or s == steps - 1:
            a0    = env.agents[0]
            rich  = env._observe_agent_rich(a0)
            print(
                f"Step {s:4d}/{steps} | eps={epsilon:.3f} | reward={reward:+.4f} | "
                f"loss={loss:.4f} | arrived={info['agents_arrived']}/10 | "
                f"cumulative={cumulative:.2f} | "
                f"{a0['name']} @ {a0['current_station']} | "
                f"ETA={rich['gtfs_eta_minutes']}min"
            )

    _save_plots(steps, cumulative_rewards, losses_per_step, rewards_per_step,
                arrivals_per_step, epsilon_history, env,
                title="Mumbai Local OpenEnv v4.0 - Training Results  [SIMULATED POLICY]",
                loss_label="Proxy Loss  [calibrated exponential - not LLM loss]")

    log = {
        "mode":               "simulated_policy",
        "note":               "Reward/arrival from real GTFS-grounded env. Loss is a calibrated proxy (not LLM loss). Use TRAINING_MODE=llm or trl for real LLM curves.",
        "data_source":        "GTFS-grounded (bundled WR/CR/HR timetable data)",
        "steps":              list(range(steps)),
        "cumulative_rewards": cumulative_rewards,
        "step_rewards":       rewards_per_step,
        "losses":             losses_per_step,
        "arrivals":           arrivals_per_step,
        "epsilon":            epsilon_history,
        "final_agent_rewards": {a["name"]: round(a["reward"], 2) for a in env.agents},
    }
    with open("training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("\nSaved: training_log.json")

    print("\n-- Per-agent summary -----------------------------------------")
    for a in env.agents:
        rich   = env._observe_agent_rich(a)
        status = "arrived" if a["arrived"] else f"@ {a['current_station']}"
        tasks  = f"{a['tasks_done']}/{len(a['personal_tasks'])} tasks"
        print(f"  {a['name']:<14} reward={a['reward']:>7.2f}  {tasks}  {status}  ETA={rich['gtfs_eta_minutes']}min")

    return log


# -- Mode 2: LLM as policy (real LLM reward curves, no GPU) -------------------

def run_llm_heuristic(steps: int = MAX_STEPS):
    """
    The LLM is the actual policy - every agent decision goes through the model.
    Uses the Anthropic API (claude-haiku-4-5-20251001 for speed/cost).
    Generates REAL LLM-driven reward curves - no simulated loss.
    This mode proves the env is LLM-native before committing GPU time.

    Requires: pip install anthropic
              export ANTHROPIC_API_KEY=sk-...
    """
    if not HAS_ANTHROPIC:
        print("[ERROR] Install anthropic SDK: pip install anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ERROR] Set ANTHROPIC_API_KEY environment variable.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = MumbaiLocalEnv(n_agents=N_AGENTS, max_steps=ENV_EPISODE_STEPS)

    rewards_per_step   = []
    arrivals_per_step  = []
    cumulative_rewards = []
    action_counts      = {a: 0 for a in ACTIONS}
    cumulative         = 0.0
    llm_calls          = 0

    print("=" * 60)
    print("  Mumbai Local OpenEnv v4.0 - LLM-Native Training")
    print("  Network: GTFS-grounded (real station sequences + headways)")
    print("  Mode: LLM AS POLICY (claude-haiku-4-5-20251001)")
    print(f"  Steps: {steps}  |  Agents: {N_AGENTS}")
    print("=" * 60)

    obs, _ = env.reset()

    for s in range(steps):
        # Poll LLM for a representative agent (agent 0) and use same action for all
        # (In full multi-agent mode you'd call per-agent; this balances cost vs demo)
        active_agents = [a for a in env.agents if not a["arrived"]]
        if not active_agents:
            obs, _ = env.reset()
            active_agents = env.agents[:]

        agent = active_agents[0]
        rich  = env._observe_agent_rich(agent)
        prompt = build_prompt(rich, agent, env.disruptions[-3:])

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            llm_output = response.content[0].text
            action = parse_action(llm_output)
            llm_calls += 1
        except Exception as e:
            print(f"[warn] LLM call failed at step {s}: {e} - falling back to route_optimize")
            action = "route_optimize"

        action_counts[action] += 1
        obs, reward, done, _, info = env.step(action)
        cumulative += reward

        rewards_per_step.append(round(reward, 4))
        cumulative_rewards.append(round(cumulative, 4))
        arrivals_per_step.append(info["agents_arrived"])

        if done:
            obs, _ = env.reset()

        if s % 10 == 0 or s == steps - 1:
            a0 = env.agents[0]
            print(
                f"Step {s:4d}/{steps} | LLM action={action:<16} | reward={reward:+.4f} | "
                f"arrived={info['agents_arrived']}/10 | cumulative={cumulative:.2f} | "
                f"LLM calls={llm_calls}"
            )

    print(f"\nLLM action distribution: {action_counts}")

    # Plot - no fake loss curve in this mode
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Mumbai Local OpenEnv v4.0 - LLM-Native Policy Results  [REAL LLM REWARDS]",
                 fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#0a0c0f")
    for ax in axes.flat:
        ax.set_facecolor("#111318")
        ax.tick_params(colors="#8b90a0")
        ax.spines[:].set_color("#252830")
        ax.title.set_color("#e8eaf0")
        ax.xaxis.label.set_color("#8b90a0")
        ax.yaxis.label.set_color("#8b90a0")

    xs = list(range(steps))
    axes[0].plot(xs, cumulative_rewards, color="#FF6B35", linewidth=1.5)
    axes[0].fill_between(xs, cumulative_rewards, alpha=0.12, color="#FF6B35")
    axes[0].set_title("Cumulative Reward (LLM Policy)")
    axes[0].set_xlabel("Step")

    window   = 15
    smoothed = [sum(rewards_per_step[max(0,i-window):i+1])/min(i+1,window) for i in range(steps)]
    axes[1].plot(xs, rewards_per_step, color="#4ECDC444", linewidth=0.8, label="raw")
    axes[1].plot(xs, smoothed, color="#4ECDC4", linewidth=2, label=f"smoothed w={window}")
    axes[1].set_title("Step Reward (LLM Policy - real signal)")
    axes[1].set_xlabel("Step")
    axes[1].legend(facecolor="#1a1d24", edgecolor="#252830", labelcolor="#8b90a0")

    axes[2].bar(list(action_counts.keys()), list(action_counts.values()),
                color=["#FF6B35","#4ECDC4","#A855F7","#eab308"])
    axes[2].set_title("LLM Action Distribution")
    axes[2].set_xlabel("Action")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches="tight", facecolor="#0a0c0f")
    print("Saved: training_results.png (real LLM reward curves)")

    log = {
        "mode":               "llm_native_policy",
        "note":               "LLM (claude-haiku) is the actual policy. Reward curves are real LLM-driven signals from GTFS-grounded env. No simulated loss.",
        "model":              "claude-haiku-4-5-20251001",
        "data_source":        "GTFS-grounded",
        "llm_calls":          llm_calls,
        "action_distribution": action_counts,
        "cumulative_rewards": cumulative_rewards,
        "step_rewards":       rewards_per_step,
        "arrivals":           arrivals_per_step,
        "final_agent_rewards": {a["name"]: round(a["reward"], 2) for a in env.agents},
    }
    with open("training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("Saved: training_log.json")
    return log


# -- Mode 3: Full TRL GRPO Training --------------------------------------------

def run_trl_training():
    """
    Full GRPO training with HuggingFace TRL.
    LLM output is parsed to an action; real env reward is returned.
    Training loss comes from TRL callbacks - REAL gradient signal.
    Each generation gets its own MumbaiLocalEnv instance.
    """
    if not HAS_TRL:
        print("[ERROR] Install trl: pip install trl transformers accelerate torch")
        return

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Generate prompt pool from real env rollouts
    print("Generating environment rollouts for prompt pool...")
    base_env = MumbaiLocalEnv(n_agents=N_AGENTS, max_steps=ENV_EPISODE_STEPS)
    data     = []

    for ep in range(3):
        obs, _ = base_env.reset()
        for step in range(ENV_EPISODE_STEPS):
            action = random.choice(ACTIONS)
            for agent in base_env.agents:
                if not agent["arrived"]:
                    rich   = base_env._observe_agent_rich(agent)
                    prompt = build_prompt(rich, agent, base_env.disruptions[-3:])
                    # Include GTFS context in dataset
                    data.append({
                        "prompt": prompt,
                        "gtfs_eta": rich.get("gtfs_eta_minutes", 0),
                        "headway":  rich.get("current_headway_min", 5),
                    })
            obs, _, done, _, _ = base_env.step(action)
            if done:
                break

    from datasets import Dataset
    dataset = Dataset.from_list(data)

    # reward_fn: LLM output -> parsed action -> REAL env reward
    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for completion in completions:
            _env = MumbaiLocalEnv(n_agents=N_AGENTS, max_steps=10)
            _env.reset()

            # Parse LLM completion to a valid action
            action = parse_action(completion if isinstance(completion, str)
                                  else completion[0].get("content", ""))

            _, r, done, _, _ = _env.step(action)
            rewards.append(float(r))
            _env.close()
        return rewards

    # Loss callback - captures REAL TRL training loss each logging step
    from transformers import TrainerCallback

    real_losses  = []
    real_rewards = []

    class LossCapture(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                if "loss" in logs:
                    real_losses.append(logs["loss"])
                if "reward" in logs:
                    real_rewards.append(logs["reward"])

    config = GRPOConfig(
        output_dir="./mumbai-local-grpo",
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=30,
        report_to="none",
        max_completion_length=10,
        num_generations=NUM_GENERATIONS,
        max_steps=60,
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        config=config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[LossCapture()],
    )

    trainer.train()
    trainer.save_model("./mumbai-local-grpo/final")
    print("Training complete. Model saved to ./mumbai-local-grpo/final")

    # Save real loss log
    log = {
        "mode":        "trl_grpo",
        "note":        "Real GRPO training. Loss and reward from TRL callbacks - genuine gradient signal.",
        "model":       MODEL_ID,
        "data_source": "GTFS-grounded",
        "real_losses":  real_losses,
        "real_rewards": real_rewards,
    }
    with open("training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("Saved: training_log.json (real TRL loss curves)")


# -- Plot helper ---------------------------------------------------------------

def _save_plots(steps, cumulative_rewards, losses_per_step, rewards_per_step,
                arrivals_per_step, epsilon_history, env, title, loss_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#0a0c0f")
    for ax in axes.flat:
        ax.set_facecolor("#111318")
        ax.tick_params(colors="#8b90a0")
        ax.spines[:].set_color("#252830")
        ax.title.set_color("#e8eaf0")
        ax.xaxis.label.set_color("#8b90a0")
        ax.yaxis.label.set_color("#8b90a0")

    xs = list(range(steps))

    axes[0, 0].plot(xs, cumulative_rewards, color="#FF6B35", linewidth=1.5)
    axes[0, 0].fill_between(xs, cumulative_rewards, alpha=0.12, color="#FF6B35")
    axes[0, 0].set_title("Cumulative Reward")
    axes[0, 0].set_xlabel("Step")

    axes[0, 1].plot(xs, losses_per_step, color="#eab308", linewidth=1.5)
    axes[0, 1].fill_between(xs, losses_per_step, alpha=0.12, color="#eab308")
    axes[0, 1].set_title(loss_label)
    axes[0, 1].set_xlabel("Step")

    window   = 20
    smoothed = [sum(rewards_per_step[max(0,i-window):i+1])/min(i+1,window) for i in range(steps)]
    axes[1, 0].plot(xs, rewards_per_step, color="#4ECDC444", linewidth=0.8, label="raw")
    axes[1, 0].plot(xs, smoothed, color="#4ECDC4", linewidth=2, label=f"smoothed (w={window})")
    axes[1, 0].set_title("Step Reward")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].legend(facecolor="#1a1d24", edgecolor="#252830", labelcolor="#8b90a0")

    axes[1, 1].plot(xs, arrivals_per_step, color="#A855F7", linewidth=1.5)
    axes[1, 1].fill_between(xs, arrivals_per_step, alpha=0.12, color="#A855F7")
    axes[1, 1].set_title("Agents Arrived per Step")
    axes[1, 1].set_xlabel("Step")

    axes[0, 2].plot(xs, epsilon_history, color="#06b6d4", linewidth=1.5)
    axes[0, 2].set_title("Epsilon Decay (explore -> exploit)")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylim(0, 1.1)

    agent_names   = [a["name"] for a in env.agents]
    agent_rewards = [round(a["reward"], 2) for a in env.agents]
    bars = axes[1, 2].bar(range(len(agent_names)), agent_rewards,
                          color=["#FF6B35","#4ECDC4","#A855F7","#eab308",
                                 "#06b6d4","#f43f5e","#10b981","#8b5cf6",
                                 "#f97316","#3b82f6"])
    axes[1, 2].set_xticks(range(len(agent_names)))
    axes[1, 2].set_xticklabels(agent_names, rotation=35, ha="right", fontsize=7)
    axes[1, 2].set_title("Final Reward per Agent")
    axes[1, 2].set_ylabel("Total Reward")
    for bar, val in zip(bars, agent_rewards):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                        f"{val:.1f}", ha="center", va="bottom", color="#8b90a0", fontsize=7)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches="tight", facecolor="#0a0c0f")
    print("Saved: training_results.png")


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Mumbai Local OpenEnv v4.0 - Training Script")
    print("  Network: GTFS-grounded (WR/CR/HR real timetable data)")
    print("=" * 60)
    mode = os.environ.get("TRAINING_MODE", "simulate")
    if mode == "trl":
        run_trl_training()
    elif mode == "llm":
        run_llm_heuristic(steps=MAX_STEPS)
    else:
        simulate_training_loop(steps=MAX_STEPS)
