"""Baseline stacking policies. Each takes (env) and returns a stack index."""
from __future__ import annotations
import numpy as np


def random_policy(env, rng):
    return int(rng.choice(np.flatnonzero(env.valid())))


def lowest_stack(env, rng=None):
    """Put it on the stack with the fewest containers (spreads the load, ignores departure order)."""
    h = [len(s) if ok else 10**9 for s, ok in zip(env.stacks, env.valid())]
    return int(np.argmin(h))


def min_max(env, rng=None):
    """Classic heuristic: put the container on the stack whose top leaves just after it (smallest
    positive gap), so nothing it sits on leaves earlier. If none, take an empty stack, else the stack
    whose top leaves latest (the least damage). Uses only the ranks the policy can see."""
    new = env.obs_ranks[env.t]
    best, best_key = -1, None
    for s, (st, ok) in enumerate(zip(env.seen, env.valid())):
        if not ok:
            continue
        top = min(st) if st else 10**9     # earliest-leaving container in the stack decides
        key = (0, top - new) if top > new else (1, -top)
        if best_key is None or key < best_key:
            best, best_key = s, key
    return best
