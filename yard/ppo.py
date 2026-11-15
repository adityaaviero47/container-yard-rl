"""PPO in plain NumPy (no deep learning library).

Policy: one small MLP scores every candidate stack from its features; a masked softmax over the
stacks with room gives the action probabilities. Because the same network scores every stack, one
trained policy works for any number of stacks. Value: a second MLP on pooled yard features.
Reward: minus the relocations needed when the containers leave (sparse, arrives at the last step).
"""
from __future__ import annotations
import numpy as np
from .env import YardEnv, random_instance, N_FEATURES


class MLP:
    def __init__(self, sizes, rng, out_scale=0.01):
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = out_scale if i == len(sizes) - 2 else np.sqrt(2.0 / a)
            self.W.append(rng.normal(0, scale, (a, b)))
            self.b.append(np.zeros(b))
        self.mW = [np.zeros_like(w) for w in self.W]; self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(x) for x in self.b]; self.vb = [np.zeros_like(x) for x in self.b]
        self.t = 0

    def forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            acts.append(x)
        return x, acts

    def backward(self, acts, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1 - acts[i] ** 2)
        return gW, gb

    def adam(self, gW, gb, lr, clip=0.5):
        norm = np.sqrt(sum((g ** 2).sum() for g in gW) + sum((g ** 2).sum() for g in gb))
        k = min(1.0, clip / (norm + 1e-8))
        self.t += 1
        b1, b2 = 0.9, 0.999
        for i in range(len(self.W)):
            for P, G, M, V in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                g = G[i] * k
                M[i] = b1 * M[i] + (1 - b1) * g
                V[i] = b2 * V[i] + (1 - b2) * g * g
                P[i] -= lr * (M[i] / (1 - b1 ** self.t)) / (np.sqrt(V[i] / (1 - b2 ** self.t)) + 1e-8)


def pooled(feats, valid):
    """Value-net input: mean and min of each feature over the stacks with room."""
    m = np.where(valid[..., None], feats, np.nan)
    return np.concatenate([np.nanmean(m, 1), np.nanmin(m, 1)], axis=1)


class Agent:
    def __init__(self, seed=0, hidden=32):
        rng = np.random.default_rng(seed)
        self.pi = MLP([N_FEATURES, hidden, hidden, 1], rng)
        self.vf = MLP([2 * N_FEATURES, hidden, 1], rng, out_scale=0.1)

    def logits(self, feats, valid):
        B, S, F = feats.shape
        out, acts = self.pi.forward(feats.reshape(B * S, F))
        z = np.where(valid, out.reshape(B, S), -1e9)
        return z, acts

    def probs(self, feats, valid):
        z, acts = self.logits(feats, valid)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z) * valid
        return e / e.sum(1, keepdims=True), acts

    def act_greedy(self, env):
        f = env.features()[None]
        v = env.valid()[None]
        p, _ = self.probs(f, v)
        return int(np.argmax(p[0]))


def rollout(agent, envs, rng, n_steps):
    """Run all envs in lockstep (same n). Returns per-step arrays and final costs."""
    B = len(envs)
    F = np.zeros((n_steps, B, envs[0].inst.n_stacks, N_FEATURES))
    V = np.zeros((n_steps, B, envs[0].inst.n_stacks), dtype=bool)
    A = np.zeros((n_steps, B), dtype=int); LP = np.zeros((n_steps, B))
    cost = np.zeros(B)
    for t in range(n_steps):
        f = np.stack([e.features() for e in envs]); v = np.stack([e.valid() for e in envs])
        p, _ = agent.probs(f, v)
        a = np.array([rng.choice(len(pi), p=pi) for pi in p])
        F[t], V[t], A[t] = f, v, a
        LP[t] = np.log(p[np.arange(B), a] + 1e-12)
        for i, e in enumerate(envs):
            done, c = e.step(int(a[i]))
            if done:
                cost[i] = c
    return F, V, A, LP, cost


def train(n_stacks=6, height=5, n=24, iters=300, batch=64, seed=0, lr=3e-3, clip=0.2, ent=0.01,
          epochs=4, log=None):
    rng = np.random.default_rng(seed)
    agent = Agent(seed)
    history = []
    for it in range(iters):
        envs = [YardEnv(random_instance(rng, n_stacks, height, n)) for _ in range(batch)]
        F, V, A, LP, cost = rollout(agent, envs, rng, n)
        # Values and advantages: reward -cost only at the last step, gamma = 1.
        Fp = np.stack([pooled(F[t], V[t]) for t in range(n)])           # (n, B, 2F)
        vals = np.array([agent.vf.forward(Fp[t])[0][:, 0] for t in range(n)])
        ret = -cost[None, :] / n * np.ones((n, 1))                       # return-to-go equals final reward
        adv = ret - vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        Fb, Vb, Ab, LPb = F.reshape(-1, n_stacks, N_FEATURES), V.reshape(-1, n_stacks), A.reshape(-1), LP.reshape(-1)
        Pb, Rb, advb = Fp.reshape(-1, 2 * N_FEATURES), ret.reshape(-1), adv.reshape(-1)
        idx = np.arange(len(Ab))
        for _ in range(epochs):
            rng.shuffle(idx)
            for mb in np.array_split(idx, 4):
                f, v, a, lp0, ad = Fb[mb], Vb[mb], Ab[mb], LPb[mb], advb[mb]
                z, acts = agent.logits(f, v)
                zs = z - z.max(1, keepdims=True)
                e = np.exp(zs) * v
                p = e / e.sum(1, keepdims=True)
                lp = np.log(p[np.arange(len(a)), a] + 1e-12)
                ratio = np.exp(lp - lp0)
                active = ~(((ratio > 1 + clip) & (ad > 0)) | ((ratio < 1 - clip) & (ad < 0)))
                # d(loss)/d(logit) for loss = -ratio*adv (clipped) - ent*entropy
                coef = (-ad * ratio * active)[:, None]
                onehot = np.zeros_like(p); onehot[np.arange(len(a)), a] = 1
                dz = coef * (onehot - p)
                H = -(p * np.log(p + 1e-12)).sum(1, keepdims=True)
                dz += -ent * (-p * (np.log(p + 1e-12) + H))
                dz *= v / len(a)
                B, S = dz.shape
                gW, gb = agent.pi.backward(acts, dz.reshape(B * S, 1))
                agent.pi.adam(gW, gb, lr)
                out, vacts = agent.vf.forward(Pb[mb])
                dv = 2 * (out[:, 0] - Rb[mb])[:, None] / len(mb)
                gW, gb = agent.vf.backward(vacts, dv)
                agent.vf.adam(gW, gb, lr)
        history.append(float(cost.mean()))
        if log and (it + 1) % 25 == 0:
            log(it + 1, float(np.mean(history[-25:])))
    return agent, history
