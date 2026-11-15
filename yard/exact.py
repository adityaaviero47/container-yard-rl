"""Exact optimum for tiny instances: try every legal placement sequence, keep the fewest relocations.
(Uses the same departure relocation rule and full knowledge of the ranks.)"""
from __future__ import annotations
from .env import Instance, retrieval_cost


def optimal_cost(inst: Instance) -> int:
    S, H, ranks = inst.n_stacks, inst.height, inst.ranks
    best = [10**9]
    stacks = [[] for _ in range(S)]

    def rec(t):
        if t == inst.n:
            best[0] = min(best[0], retrieval_cost([list(s) for s in stacks], H))
            return
        seen_empty = False
        for s in range(S):
            if len(stacks[s]) >= H:
                continue
            if not stacks[s]:                 # empty stacks are interchangeable: try only the first
                if seen_empty:
                    continue
                seen_empty = True
            stacks[s].append(int(ranks[t]))
            rec(t + 1)
            stacks[s].pop()

    rec(0)
    return best[0]
