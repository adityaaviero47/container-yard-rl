"""Container yard stacking environment (the container stacking / relocation problem).

One yard bay has S stacks, each holding at most H containers (last in, first out).
N containers arrive one at a time. Each has a departure rank (0 = leaves first), known in
advance from a loading plan. After all have arrived, containers leave in rank order. To pull a
container out, every container above it must be moved first: each such move is a relocation
(a "rehandle"), the cost terminal operators try to avoid.

The agent chooses the stack for every arriving container. Relocations during departure are done
by a fixed rule (`relocate_target`) that is the same for every policy, so policies are compared
only on their stacking decisions.
"""
from __future__ import annotations
import numpy as np


class Instance:
    """One problem: yard size and the arrival order's departure ranks."""
    def __init__(self, n_stacks: int, height: int, ranks: np.ndarray):
        self.n_stacks, self.height = n_stacks, height
        self.ranks = np.asarray(ranks)          # ranks[i] = true departure rank of i-th arrival
        self.n = len(ranks)
        assert self.n <= n_stacks * height


def random_instance(rng: np.random.Generator, n_stacks: int, height: int, n: int) -> Instance:
    return Instance(n_stacks, height, rng.permutation(n))


def relocate_target(stacks: list[list[int]], height: int, src: int, rank: int) -> int:
    """Where to put a blocker with departure `rank` removed from stack `src`.
    Rule: among the other stacks with room, prefer one whose top leaves LATER than the blocker (so the
    blocker blocks nothing new), taking the smallest such gap. If no stack qualifies, take the stack
    whose top leaves latest. Empty stacks count as leaving infinitely late. Ties go to the lower index."""
    best, best_key = -1, None
    for s, st in enumerate(stacks):
        if s == src or len(st) >= height:
            continue
        top = st[-1] if st else 10**9
        key = (0, top - rank) if top > rank else (1, -top)
        if best_key is None or key < best_key:
            best, best_key = s, key
    if best < 0:                                 # no room elsewhere: put back on the source stack
        best = src
    return best


def retrieval_cost(stacks: list[list[int]], height: int, true_ranks_in_stack: list[list[int]] | None = None) -> int:
    """Total relocations needed to pull every container out in rank order (0, 1, 2, ...).
    `stacks` hold true departure ranks; they are consumed in place (pass a copy)."""
    total = 0
    n = sum(len(s) for s in stacks)
    for r in range(n):
        s = next(i for i, st in enumerate(stacks) if r in st)
        while stacks[s][-1] != r:
            b = stacks[s].pop()
            t = relocate_target(stacks, height, s, b)
            stacks[t].append(b)
            total += 1
        stacks[s].pop()
    return total


class YardEnv:
    """Sequential placement environment. Observation per candidate stack is built by `features`.

    `noise`: the policy sees ranks perturbed by Gaussian noise (std = noise * N) while the true
    ranks still decide the retrieval order. noise = 0 means the loading plan is exact."""
    def __init__(self, inst: Instance, noise: float = 0.0, rng: np.random.Generator | None = None):
        self.inst = inst
        self.stacks: list[list[int]] = [[] for _ in range(inst.n_stacks)]   # true ranks
        self.seen: list[list[float]] = [[] for _ in range(inst.n_stacks)]   # ranks as the policy sees them
        self.t = 0
        if noise > 0:
            g = rng if rng is not None else np.random.default_rng(0)
            self.obs_ranks = inst.ranks + g.normal(0, noise * inst.n, size=inst.n)
        else:
            self.obs_ranks = inst.ranks.astype(float)

    def valid(self) -> np.ndarray:
        return np.array([len(s) < self.inst.height for s in self.stacks])

    def features(self) -> np.ndarray:
        """(S, F) features for placing the next container on each stack.
        0 height fraction | 1 empty | 2 containers below that leave EARLIER than the new one (blocked, scaled)
        3 any blocked | 4 (top rank - new rank)/N, or +1 if empty | 5 new container's rank / N | 6 yard fill fraction"""
        inst, n = self.inst, self.inst.n
        new = self.obs_ranks[self.t]
        rows = []
        fill = self.t / (inst.n_stacks * inst.height)
        for st in self.seen:
            h = len(st)
            blocked = sum(1 for r in st if r < new)
            top = (st[-1] - new) / n if h else 1.0
            rows.append([h / inst.height, float(h == 0), blocked / inst.height, float(blocked > 0),
                         top, new / n, fill])
        return np.array(rows)

    def step(self, a: int) -> tuple[bool, int]:
        """Place next container on stack a. Returns (done, relocations at the end or 0)."""
        inst = self.inst
        assert len(self.stacks[a]) < inst.height
        self.stacks[a].append(int(inst.ranks[self.t]))
        self.seen[a].append(float(self.obs_ranks[self.t]))
        self.t += 1
        if self.t < inst.n:
            return False, 0
        return True, retrieval_cost([list(s) for s in self.stacks], inst.height)


N_FEATURES = 7
