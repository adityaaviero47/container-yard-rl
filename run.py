"""Reproduces every number in the README. Run: python run.py  (about 3 minutes on a laptop)."""
import json, time
import numpy as np
from yard.env import YardEnv, random_instance
from yard.policies import random_policy, lowest_stack, min_max
from yard.ppo import Agent, train
from yard.exact import optimal_cost

SEEDS = [0, 1, 2]            # PPO training seeds
TEST_BASE = 10_000           # test instances use their own seeds, never seen in training (training uses seed*1 streams)


def evaluate(policy, insts, noise=0.0):
    """Relocations per instance for a policy (an Agent or a function)."""
    out = []
    for k, inst in enumerate(insts):
        rng = np.random.default_rng(777 + k)
        env = YardEnv(inst, noise, rng)
        while True:
            a = policy.act_greedy(env) if isinstance(policy, Agent) else policy(env, rng)
            done, c = env.step(a)
            if done:
                out.append(c)
                break
    return np.array(out, dtype=float)


def make(n_stacks, height, n, count, offset=0):
    return [random_instance(np.random.default_rng(TEST_BASE + offset + k), n_stacks, height, n) for k in range(count)]


def boot_ci(x, reps=4000, seed=1):
    r = np.random.default_rng(seed)
    m = [r.choice(x, len(x)).mean() for _ in range(reps)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def summary(x):
    return {"mean": round(float(x.mean()), 3), "ci95": boot_ci(x)}


res = {"settings": {"main_yard": "6 stacks x 5 high, 24 containers (80% full)", "ppo_seeds": SEEDS}}
t0 = time.time()

# ---- train PPO (3 seeds) on 6x5 yards with 24 containers
agents, curves = [], []
for s in SEEDS:
    ag, hist = train(6, 5, 24, iters=400, seed=s)
    agents.append(ag); curves.append(hist)
    print(f"trained seed {s} in {time.time()-t0:.0f}s, last-50 training cost {np.mean(hist[-50:]):.2f}", flush=True)
np.save("results/training_curves.npy", np.array(curves))

# ---- main comparison on 1,000 unseen instances
main = make(6, 5, 24, 1000)
base = {"random": random_policy, "lowest_stack": lowest_stack, "min_max_heuristic": min_max}
costs = {k: evaluate(p, main) for k, p in base.items()}
ppo_costs = [evaluate(a, main) for a in agents]
costs["ppo_seed0"] = ppo_costs[0]
res["main_relocations_per_yard"] = {**{k: summary(v) for k, v in costs.items()},
                                    "ppo_all_seeds_mean": [round(float(c.mean()), 3) for c in ppo_costs]}
mm = costs["min_max_heuristic"]
paired = {}
for i, c in enumerate(ppo_costs):
    d = c - mm
    paired[f"seed{i}"] = {"mean_diff_vs_min_max": round(float(d.mean()), 3), "ci95": boot_ci(d),
                          "ppo_better": int((d < 0).sum()), "tie": int((d == 0).sum()), "ppo_worse": int((d > 0).sum())}
res["paired_vs_min_max"] = paired

# ---- gap to the exact optimum on tiny yards (3 stacks x 4, 9 containers), 60 instances
tiny = make(3, 4, 9, 60, offset=50_000)
t1 = time.time()
opt = np.array([optimal_cost(i) for i in tiny], dtype=float)
print(f"exact optimum for 60 tiny instances in {time.time()-t1:.0f}s", flush=True)
tiny_res = {"optimal_mean": round(float(opt.mean()), 3)}
for k, p in base.items():
    c = evaluate(p, tiny)
    tiny_res[k] = {"mean": round(float(c.mean()), 3), "instances_at_optimum": int((c == opt).sum())}
for i, a in enumerate(agents):
    c = evaluate(a, tiny)
    tiny_res[f"ppo_seed{i}"] = {"mean": round(float(c.mean()), 3), "instances_at_optimum": int((c == opt).sum())}
res["tiny_vs_exact_optimum"] = tiny_res

# ---- other yard sizes, no retraining
sizes = {"4x4_12": (4, 4, 12), "8x6_36": (8, 6, 36), "10x6_48": (10, 6, 48)}
gen = {}
for name, (S, H, n) in sizes.items():
    insts = make(S, H, n, 300, offset=80_000)
    row = {k: round(float(evaluate(p, insts).mean()), 3) for k, p in base.items()}
    row["ppo_mean_of_seeds"] = round(float(np.mean([evaluate(a, insts).mean() for a in agents])), 3)
    gen[name] = row
res["other_yard_sizes_no_retraining"] = gen

# ---- noisy departure plans: policies see noisy ranks, true order decides the cost
noisy = {}
test = make(6, 5, 24, 300, offset=120_000)
for noise in [0.0, 0.05, 0.1, 0.2, 0.4]:
    row = {k: round(float(evaluate(p, test, noise).mean()), 3) for k, p in base.items()}
    row["ppo_mean_of_seeds"] = round(float(np.mean([evaluate(a, test, noise).mean() for a in agents])), 3)
    noisy[str(noise)] = row
res["noisy_plan_std_as_fraction_of_N"] = noisy

json.dump(res, open("results/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))

# ---- training curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
k = 10
for i, c in enumerate(curves):
    sm = np.convolve(c, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(len(sm)) + k, sm, lw=1.4, label=f"PPO seed {i}")
ax.axhline(mm.mean(), color="#b45309", ls="--", lw=1.2, label="min-max heuristic")
ax.set_xlabel("PPO iteration (64 yards each)"); ax.set_ylabel("relocations per yard (training)")
ax.set_ylim(0, 10); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/training_curve.png")
print(f"done in {time.time()-t0:.0f}s")
