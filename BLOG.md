# 🚂 Teaching an AI to Commute Like a Mumbaikar

*A story about 8 million daily decisions — and the machine learning to make them*

---

## Picture This

It's 8:47 AM on a Tuesday.

Priya is standing on Platform 3 at Andheri station. She's a doctor at KEM Hospital in Dadar, and her first patient is at 9:30 AM. In her bag: a thermos of chai, a patient file she still needs to read, and a phone buzzing with notifications.

Three trains could take her south. One is already pulling in — but she can see through the glass that it's packed wall to wall, the kind of crowd where your feet don't touch the floor. The next one comes in 4 minutes. The one after that, 9 minutes.

She has 43 minutes to get to work.

What does she do?

If you've ever commuted on Mumbai's local trains, you know this feeling. You're not just choosing a train — you're making a calculation. How crowded is it? Is there a signal issue further south? Should she switch to the Harbour Line at Mahim? Will that save time or waste it?

**Priya makes this decision in about 15 seconds. Every single day.**

Now imagine — what if a computer could learn to make that decision just as well as she does?

That's exactly what **Mumbai Local OpenEnv** is trying to build.

---

## The Problem Nobody Has Solved Yet

Mumbai's local train network moves over **8 million people every single day**. That's more than the entire population of Switzerland, packed onto three lines, 100+ stations, across 70 kilometres of track.

And yet — there is no intelligent system trained to help commuters navigate it.

Why not?

Because this is genuinely *hard* for computers to learn. Unlike a chess game or a video game (which is where most AI training happens), real transit decisions are messy:

- The situation changes every minute (a signal fails at Bandra, a platform gets overcrowded, a train gets delayed)
- Your decision today affects what options you have in 10 minutes
- You're not the only one deciding — thousands of other commuters are also making choices that affect *your* journey
- Getting it wrong doesn't just lose you points in a game — it means you're late to your patient's surgery

This project builds a **training ground** — a realistic digital simulation of Mumbai's local trains — where an AI can practice making these decisions thousands of times, learn from its mistakes, and gradually get really, really good at it.

---

## The Training Ground: A Virtual Mumbai

Think of it like a flight simulator for AI.

Before a pilot flies a real plane, they spend hundreds of hours in a simulator that looks and feels exactly like the cockpit of a real aircraft — with realistic weather, engine problems, and emergency scenarios thrown in. They can crash a hundred times and learn from every single one, with zero real-world consequences.

Mumbai Local OpenEnv is that simulator — but for transit decisions.

Here's what's inside:

**The Network** — All three lines are here. Western Line from Churchgate to Virar. Central Line from CSMT to Kalyan. Harbour Line connecting them. All the transfer stations — Dadar, CSMT, Andheri, Kurla — with realistic walking times between platforms.

**The Chaos** — The simulation throws real-world disruptions at the AI. Signal failures. Train delays. Platform accidents. Even the occasional strike. Because an AI that only works when everything runs on time isn't actually useful.

**The Rush Hour** — Morning peak from 8 to 10 AM. Evening peak from 5 to 8 PM. Off-peak nights. The simulation knows that a train at 8:45 AM is a fundamentally different experience from the same train at 11:00 AM.

**The Commuters** — Ten different types of people, each with their own constraints. Priya the doctor who absolutely cannot be late. Ravi the vendor who needs to reach Dadar market by 7 AM. A student with a flexible schedule. A nurse on the early morning shift. The AI has to learn to navigate the system *as each of these different people*.

---

## How the AI Learns: Trial, Error, and Rewards

Imagine teaching a child to navigate Mumbai's trains. You wouldn't hand them a rulebook. You'd take them on the journey, and every time they made a good choice, you'd say "great!" — and every time they made a poor one, you'd explain why it didn't work out.

That's essentially how this AI learns, through a method called **Reinforcement Learning**.

The AI (playing the role of Priya, or Ravi, or any of the 10 commuter profiles) starts by making completely random decisions. Board this train? Wait for the next one? Switch lines at Dadar? It has no idea.

Every decision gets a score. Good things earn points:

- Reaching your destination on time → big reward
- Choosing a less crowded train → small reward
- Making a smart line transfer to avoid a disruption → reward
- Getting closer to your destination with each step → reward

Bad things cost points:

- Just standing on the platform doing nothing → penalty
- Missing your deadline → significant penalty
- Making the same bad decision repeatedly → penalty

The AI plays through thousands of simulated commutes, slowly learning which decisions lead to better scores. At first, it's terrible. After training, it arrives on time 89% of the time — compared to 42% when it was just guessing randomly.

More importantly, it learns *generalizable* skills. When a disruption type it's never seen before shows up, it doesn't freeze — it applies the same reasoning it learned from other disruptions. Just like how an experienced Mumbaikar who's been commuting for 10 years knows how to handle a new kind of signal failure even if they've never seen *that specific* failure before.

---

## Back to Priya on Platform 3

Let's return to our doctor at Andheri, 8:47 AM, 43 minutes to get to work.

Here's what the trained AI would consider, almost instantly:

*The packed train arriving now: occupancy at 91%. Priya's crowding tolerance is low — she needs to be sharp for surgery. Skip it.*

*Next train in 4 minutes: occupancy at 55%. That's manageable. But wait — there's a signal issue flagged at Bandra, 3 stations south. That could add 15 minutes.*

*Alternative: Board the 4-minute train, but switch to the Harbour Line at Mahim. 2-minute walk, but it bypasses Bandra entirely. Estimated arrival at Dadar: 9:18 AM. That's 12 minutes before her patient. She can even grab a chai at the station.*

The AI picks the Harbour Line transfer. Priya gets to work on time. The patient is seen.

This is the kind of decision — nuanced, multi-step, context-aware — that the project is training an AI to make.

---

## Why This Matters Beyond Mumbai

The same framework that works for Mumbai's trains can work for any transit network in the world. Chennai. Kolkata. Delhi Metro. Pune's upcoming metro. The simulation can be reconfigured for any city's layout, any network's disruption patterns.

More broadly, the techniques being developed here — training AI systems to make sequential, multi-step decisions in complex, real-world environments — matter for everything from logistics and supply chain to emergency response and urban planning.

But it starts with one doctor, one platform, and one very crowded 8:47 AM train.

---

## What Was Actually Built

For those who want a slightly deeper peek under the hood, without the full technical deep-dive:

The project created a software simulation of Mumbai's train network, designed a scoring system to reward good transit decisions, and then used it to train an AI model (starting from GPT-2, a language model) using a technique called **GRPO** — essentially a way to make language models learn from experience, not just from reading text.

The result: after 6 hours of training on a single GPU, the AI went from 42% arrival rate (random guessing) to **89% arrival rate**, with significantly better crowd avoidance and smarter responses to disruptions.

All of the code, the environment, and the trained models are open source — freely available for other researchers and developers to build on, extend, or adapt to their own city's transit systems.

---

## The Bigger Picture

Every day, millions of Mumbaikars make thousands of micro-decisions just to get to work. They carry this knowledge in their heads — knowledge built from years of experience, passed along informally, never written down.

This project is an attempt to capture that knowledge, formalize it, and train a machine to understand it.

Not to replace the experienced commuter. But to help the new one. To help the visitor who doesn't know which train to take. To help city planners understand where their system is failing people. To help build the infrastructure that a city of 8 million daily train riders deserves.

One simulated commute at a time.

---

*Mumbai Local OpenEnv is an open-source project submitted to the OpenEnv Hackathon India 2026. The environment, trained models, and training code are freely available on [GitHub](https://github.com/jyotikarao2006-web/Mumbai_Local) and [Hugging Face](https://huggingface.co/spaces/jyotikarao/mumbai_local_meta).*


### Learning Curves

**Episode Reward Over Time:**
<img width="1184" height="582" alt="image" src="https://github.com/user-attachments/assets/1d72b2a9-2d0a-4350-b185-9479badf5e98" />


**Arrival Rate Progression:**
<img width="1184" height="582" alt="image" src="https://github.com/user-attachments/assets/5a20c2c7-5ff5-42e0-98a4-baadf9110173" />

**Reward Distribution:**
<img width="1184" height="582" alt="image" src="https://github.com/user-attachments/assets/5cd1917d-ad32-4c50-bc2a-22ab16578aca" />

**Before and After cummulative rewards:**
<img width="2084" height="1036" alt="image" src="https://github.com/user-attachments/assets/ab570085-4ea6-4863-98b9-0fdedc403a47" />

Training Performance Insights

🔴 Before Training (Random Baseline)
Rewards are predominantly negative (≈ -40 to 0)
High variance with no consistent pattern
Frequent sharp drops indicating suboptimal decisions

Interpretation:
The untrained agent follows a random policy, resulting in unstable and poor performance across steps.

🟢 After Training (Trained Agent)
Rewards shift to a predominantly positive range (≈ 0 to +50)
Variability persists due to stochastic environment dynamics
Frequent positive spikes indicate improved decision-making

Interpretation:
After training, the agent learns an effective policy that consistently yields higher rewards and improved performance.

*Questions or want to collaborate? Reach out at jyotika.rao2006@gmail.com*

Project Links
🚀 Live Demo (Hugging Face Space):
👉 https://huggingface.co/spaces/jyotikarao/mumbai_local_meta

📓 Google Colab (Training Notebook):
👉 https://colab.research.google.com/drive/1pPOItHoKZ-wBsmY5IXTcoOdf1422Quiq?usp=sharing

