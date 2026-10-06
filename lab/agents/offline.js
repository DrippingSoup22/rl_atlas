/* Learning from data instead of from trying: offline RL and imitation.

   Offline RL. The first unit is the log: a behavior policy (p.behavior, a world's own, made a little random by
   p.epsilon) plays p.logEpisodes episodes, and every transition goes into the log. After that the learner never acts
   again: each unit is one pass over the log, a Q-learning update on every transition once, in a random order.
     offline-q    plain Q-learning on the log. Its max ranges over every action, also those the log never tried in s'.
                  Their estimates never move from where they started (p.q0), so the max, and the policy, can prefer them.
     offline-bcq  batch-constrained Q-learning (Fujimoto, Meger & Precup, 2019, in its tabular form): the max, and the
                  policy, only consider actions the log tried in that state.

   Imitation. An expert (the world's optimal policy, worked out by value iteration with p.gamma) can be asked what it
   would do in any state. Each unit is one episode; every state the expert is asked about is labeled with its action.
     bc           behavior cloning: the expert plays every episode, the learner copies the labels.
     dagger       DAgger (Ross, Gordon & Bagnell, 2011): the expert plays the first episode; after that the learner
                  plays with its own policy, and the expert labels the states the learner reaches.
   The learner's policy is the expert's label where it has one. Where it has none, it is uniform over the actions (a
   table never trained knows nothing), or with p.generalize = "nearest" the label of the nearest labeled tile (a
   classifier that generalizes, confidently, from what it was shown). With p.calmExpert the expert never slips, as a
   skilled driver keeps to the middle of the road: its demonstrations show no recoveries. Both show the policy (P) and
   fog over the states without a label. */
(function (RL) {
  "use strict";
  const lab = RL.lab;

  // ---- offline RL ----
  const logSize = (p) => Math.max(1, Math.round(p.logEpisodes ?? 10)) * Math.max(1, Math.round(p.logCap ?? 100));
  const offlineMemory = (env, p) => {
    const n = env.nS * env.nA, L = logSize(p);
    // N: how often the log tried each pair · LS, LA, LR, LS2: the log's transitions · K: how many
    return { Q: n, N: n, LS: L, LA: L, LR: L, LS2: L, K: 1 };
  };
  const offlineInit = (m, env, p) => m.Q.fill(p.q0 || 0);

  const behaviorOf = (env, p) => (p.behaviorTable ||= env.policy ? env.policy(p.behavior || "careful") : lab.randomPolicy(env));

  // The actions the log tried in s (all of them where it tried none: nothing to constrain to).
  function logged(m, env, s) {
    const acts = env.acts(s).filter((a) => m.N[s * env.nA + a] > 0);
    return acts.length ? acts : env.acts(s);
  }
  function bestOf(Q, s, env, acts) {
    let best = acts[0];
    for (const a of acts) if (Q[s * env.nA + a] > Q[s * env.nA + best]) best = a;
    return best;
  }

  function* collect({ env, m, rng, p }) {
    const B = behaviorOf(env, p), cap = Math.max(1, Math.round(p.logCap ?? 100)), E = Math.max(1, Math.round(p.logEpisodes ?? 10));
    for (let e = 0; e < E; e++) {
      let s = env.reset(rng);
      yield { line: "log", type: "start", s, episode: e };
      for (let t = 0; !env.terminal(s) && t < cap; t++) {
        const a = rng.next() < (p.epsilon ?? 0) ? env.acts(s)[rng.int(env.acts(s).length)] : lab.pickRow(B, s, env, rng);
        const o = env.step(s, a, rng), k = m.K[0]++;
        m.LS[k] = s; m.LA[k] = a; m.LR[k] = o.r; m.LS2[k] = o.s2;
        m.N[s * env.nA + a] += 1;
        yield { line: "log", type: "move", s, a, ...o };
        s = o.s2;
      }
    }
    return { logged: m.K[0] };
  }

  function offline(constrained) {
    return function* ({ env, m, rng, p, t }) {
      if (t === 0) return yield* collect({ env, m, rng, p });
      // one pass: every transition of the log once, in a new random order (a Fisher–Yates shuffle)
      const nA = env.nA, n = m.K[0], order = Array.from({ length: n }, (_, k) => k), list = [];
      for (let k = n - 1; k > 0; k--) { const j = rng.int(k + 1); [order[k], order[j]] = [order[j], order[k]]; }
      let size = 0;
      for (let k = 0; k < n; k++) {
        const j = order[k], s = m.LS[j], a = m.LA[j], r = m.LR[j], s2 = m.LS2[j];
        const next = env.terminal(s2) ? 0 : constrained ? m.Q[s2 * nA + bestOf(m.Q, s2, env, logged(m, env, s2))] : lab.maxQ(m.Q, s2, env);
        const i = s * nA + a, old = m.Q[i], target = r + p.gamma * next;
        m.Q[i] = old + p.alpha * (target - old);
        size += Math.abs(target - old);
        if (k < 60) list.push({ s, a });
      }
      yield { line: "update", type: "update", list, s: list[0]?.s ?? env.start };
      return { change: size / Math.max(1, n) };
    };
  }

  // The policy each learner would deploy: greedy on Q, over every action or over the logged ones.
  function deployed(m, env, constrained) {
    const P = new Float64Array(env.nS * env.nA);
    for (let s = 0; s < env.nS; s++) {
      if (env.terminal(s) || env.blocked?.(s)) continue;
      P[s * env.nA + bestOf(m.Q, s, env, constrained ? logged(m, env, s) : env.acts(s))] = 1;
    }
    return P;
  }
  // What each learner believes a state is worth: its best value, over every action or over the logged ones.
  function worth(m, env, constrained) {
    const V = new Float64Array(env.nS);
    for (let s = 0; s < env.nS; s++) if (!env.terminal(s) && !env.blocked?.(s)) V[s] = m.Q[s * env.nA + bestOf(m.Q, s, env, constrained ? logged(m, env, s) : env.acts(s))];
    return V;
  }
  // Fog over the states the log never visited; model[s·nA + a] ≥ 0 where the log tried a in s.
  const coverage = (m) => Float64Array.from(m.N, (c) => (c > 0 ? 1 : -1));

  const offlineAlgorithm = (constrained) => ({
    unit: "pass", memory: offlineMemory, init: offlineInit, run: offline(constrained),
    show: (m, env) => ({ Q: m.Q, V: worth(m, env, constrained), P: deployed(m, env, constrained), model: m.K[0] ? coverage(m) : null }),
    knobs: {
      q0: { sym: "Q₀", name: "first estimate", min: -50, max: 10, step: 1, sweep: [-50, -30, -20, -15, -10, -5, 0, 5] },
      epsilon: { sym: "ε", name: "the logger's randomness", min: 0, max: 1, step: 0.05, sweep: [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1] },
    },
  });

  // ---- imitation ----
  const expertOf = (env, p) => (p.expertTable ||= lab.greedyPolicy(env, lab.valueIteration(env, p.gamma), p.gamma, 1e-9));
  const expertAction = (E, s, env) => { for (const a of env.acts(s)) if (E[s * env.nA + a] > 0) return a; return env.acts(s)[0]; };
  const imitationMemory = (env) => ({ L: env.nS, LAB: 1 }); // L: the expert's label in each state (−1: none) · LAB: labels so far
  const imitationInit = (m) => m.L.fill(-1);

  // The learner's action in s: the label there; elsewhere, with p.generalize = "nearest", the label of the nearest
  // labeled tile (a classifier that generalizes), else any action at random (a table that knows nothing).
  function nearest(m, s, env) {
    if (!env.rc) return -1;
    const [r, c] = env.rc(s);
    let best = -1, dist = Infinity;
    for (let u = 0; u < env.nS; u++) {
      if (m.L[u] < 0) continue;
      const [ru, cu] = env.rc(u), d = Math.abs(ru - r) + Math.abs(cu - c);
      if (d < dist) { dist = d; best = m.L[u]; }
    }
    return best;
  }
  function learnerAct(m, s, env, rng, p) {
    const acts = env.acts(s);
    if (m.L[s] >= 0) return m.L[s];
    const guess = p.generalize === "nearest" ? nearest(m, s, env) : -1;
    return guess >= 0 ? guess : acts[rng.int(acts.length)];
  }

  function imitation(dagger) {
    return function* ({ env, m, rng, p, t }) {
      const E = expertOf(env, p), expertPlays = !dagger || t === 0, cap = p.maxSteps;
      // with p.calmExpert, the expert never slips: its moves go where it means them to (the learner's still slip)
      const world = expertPlays && p.calmExpert && env.slip ? (p.calmWorld ||= lab.grid({ title: env.title, map: env.map, reward: env.rewards })) : env;
      let s = env.reset(rng);
      yield { line: expertPlays ? "demo" : "play", type: "start", s };
      for (let k = 0; !env.terminal(s) && k < cap; k++) {
        // the move comes first, from what the learner knew before this state was labeled; then the expert's label
        const label = expertAction(E, s, env), a = expertPlays ? label : learnerAct(m, s, env, rng, p);
        yield { line: expertPlays ? "demo" : "play", type: "choose", s, a };
        const fresh = m.L[s] < 0;
        if (fresh) { m.L[s] = label; m.LAB[0] += 1; }
        yield { line: "label", type: "label", s, a: label, fresh };
        const o = world.step(s, a, rng);
        yield { line: expertPlays ? "demo" : "play", type: "move", s, a, ...o };
        s = o.s2;
      }
      return { labels: m.LAB[0] };
    };
  }

  // The learner's policy: the label where there is one, uniform elsewhere.
  function learnerPolicy(m, env, p) {
    const P = new Float64Array(env.nS * env.nA);
    for (let s = 0; s < env.nS; s++) {
      if (env.terminal(s) || env.blocked?.(s)) continue;
      const acts = env.acts(s), guess = m.L[s] >= 0 ? m.L[s] : p.generalize === "nearest" ? nearest(m, s, env) : -1;
      if (guess >= 0) P[s * env.nA + guess] = 1;
      else for (const a of acts) P[s * env.nA + a] = 1 / acts.length;
    }
    return P;
  }
  const labeled = (m, env) => {
    const M = new Float64Array(env.nS * env.nA).fill(-1);
    for (let s = 0; s < env.nS; s++) if (m.L[s] >= 0) M[s * env.nA + m.L[s]] = 1;
    return M;
  };

  const imitationAlgorithm = (dagger) => ({
    unit: "episode", memory: imitationMemory, init: imitationInit, run: imitation(dagger),
    show: (m, env, p) => ({ P: learnerPolicy(m, env, p), model: labeled(m, env) }),
  });

  lab.algorithms = Object.assign(lab.algorithms || {}, {
    "offline-q": {
      id: "offline-q", station: "offline-rl", title: "Q-learning on a log", ...offlineAlgorithm(false),
      rule: "\\val{Q(s,a)} \\leftarrow \\val{Q(s,a)} + \\alp\\,\\big[\\rew{r} + \\gam \\max_{a'} \\val{Q(s',a')} - \\val{Q(s,a)}\\big],\\ (s, a, \\rew r, s') \\sim \\mathcal D",
    },
    "offline-bcq": {
      id: "offline-bcq", station: "offline-rl", title: "Batch-constrained Q-learning", ...offlineAlgorithm(true),
      rule: "\\val{Q(s,a)} \\leftarrow \\val{Q(s,a)} + \\alp\\,\\big[\\rew{r} + \\gam \\max_{a' : (s', a') \\in \\mathcal D} \\val{Q(s',a')} - \\val{Q(s,a)}\\big]",
    },
    bc: {
      id: "bc", station: "imitation", title: "Behavior cloning", ...imitationAlgorithm(false),
      rule: "\\pol{\\pi(s)} \\leftarrow \\pol{\\pi_E(s)} \\ \\text{for the states the expert visits}",
    },
    dagger: {
      id: "dagger", station: "imitation", title: "DAgger", ...imitationAlgorithm(true),
      rule: "\\pol{\\pi(s)} \\leftarrow \\pol{\\pi_E(s)} \\ \\text{for the states the learner visits}",
    },
  });
})(globalThis.RL = globalThis.RL || {});
