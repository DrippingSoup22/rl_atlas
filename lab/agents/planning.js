/* Planning with a learned model: Dyna-Q, Dyna-Q+ and prioritized sweeping. After every real step the agent writes
   down what happened (in this state, this action led there and paid that: a deterministic model), learns from it
   like Q-learning, and then replays remembered transitions to itself: n planning updates per real step.
   Everything the agent keeps, the model, the clock and the queue included, lives in its memory, so a run can be
   replayed exactly from any snapshot. */
(function (RL) {
  "use strict";
  const lab = RL.lab;

  const memory = (env) => {
    const n = env.nS * env.nA;
    // MS: where each pair led (−1: never tried) · MR: what it paid · LT: when it was last tried (Dyna-Q+) ·
    // PQ: its priority (prioritized sweeping; 0: not queued) · OL: the pairs tried so far, in order · K: how many · C: the clock
    return { Q: n, MS: n, MR: n, LT: n, PQ: n, OL: n, K: 1, C: 1 };
  };
  const init = (m, env, p) => { m.Q.fill(p.q0 || 0); m.MS.fill(-1); };
  const planningSteps = (p) => Math.max(0, Math.round(p.planning ?? 5));

  // One Q-learning update of pair i toward r + γ max Q(s2); returns the change.
  function learn(m, env, p, s, a, r, s2) {
    const i = s * env.nA + a, old = m.Q[i], next = env.terminal(s2) ? 0 : lab.maxQ(m.Q, s2, env), target = r + p.gamma * next;
    m.Q[i] = old + p.alpha * (target - old);
    return { old, next, target, delta: target - old, value: m.Q[i] };
  }

  // Write a real transition into the model. Dyna-Q+ also lets planning try the actions never taken from a visited
  // state, assuming they lead back to it and pay nothing (Sutton & Barto, §8.3).
  function remember(m, env, s, a, r, s2, untried) {
    const i = s * env.nA + a;
    const add = (j) => { m.OL[m.K[0]++] = j; };
    if (untried && !env.acts(s).some((b) => m.MS[s * env.nA + b] >= 0)) {
      for (const b of env.acts(s)) if (b !== a) { const j = s * env.nA + b; m.MS[j] = s; m.MR[j] = 0; m.LT[j] = 0; add(j); }
    }
    if (m.MS[i] < 0) add(i);
    m.MS[i] = s2;
    m.MR[i] = r;
    m.LT[i] = m.C[0];
  }

  function* dyna({ env, m, rng, p }, plus) {
    const nA = env.nA, n = planningSteps(p), kappa = p.kappa ?? 0.001;
    let s = env.reset(rng);
    env.setTime?.(m.C[0]);
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = lab.epsGreedy(m.Q, s, env, p.epsilon, rng);
      yield { line: "choose", type: "choose", s, a };
      const o = env.step(s, a, rng);
      m.C[0] += 1;
      yield { line: "act", type: "move", s, a, ...o };
      if (env.setTime?.(m.C[0])) yield { line: "act", type: "world", s: o.s2, time: m.C[0] };
      const u = learn(m, env, p, s, a, o.r, o.s2);
      yield { line: "update", type: "update", s, a, r: o.r, s2: o.s2, ...u };
      remember(m, env, s, a, o.r, o.s2, plus);
      yield { line: "model", type: "model", s, a, r: o.r, s2: o.s2, known: m.K[0] };
      if (n) {
        const list = [];
        for (let k = 0; k < n; k++) {
          const i = m.OL[rng.int(m.K[0])], ps = Math.floor(i / nA), pa = i % nA, bonus = plus ? kappa * Math.sqrt(m.C[0] - m.LT[i]) : 0;
          const v = learn(m, env, p, ps, pa, m.MR[i] + bonus, m.MS[i]);
          list.push({ s: ps, a: pa, s2: m.MS[i], delta: v.delta, bonus });
        }
        yield { line: "plan", type: "plan", s: o.s2, list };
      }
      s = o.s2;
      yield { line: "next", type: "next", s };
    }
  }

  // Prioritized sweeping: planning works first on the pairs whose values would change the most, and when one changes,
  // the pairs predicted to lead into its state are queued in turn, so the news sweeps backward from where it began.
  function* sweeping({ env, m, rng, p }) {
    const nA = env.nA, n = planningSteps(p), theta = p.theta ?? 1e-4, { Q, MS, MR, PQ, OL, K } = m;
    const priority = (i) => { const s2 = MS[i]; return Math.abs(MR[i] + p.gamma * (env.terminal(s2) ? 0 : lab.maxQ(Q, s2, env)) - Q[i]); };
    const queue = (i, P) => { if (P > theta && P > PQ[i]) PQ[i] = P; };
    let s = env.reset(rng);
    env.setTime?.(m.C[0]);
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = lab.epsGreedy(Q, s, env, p.epsilon, rng);
      yield { line: "choose", type: "choose", s, a };
      const o = env.step(s, a, rng);
      m.C[0] += 1;
      yield { line: "act", type: "move", s, a, ...o };
      if (env.setTime?.(m.C[0])) yield { line: "act", type: "world", s: o.s2, time: m.C[0] };
      remember(m, env, s, a, o.r, o.s2, false);
      yield { line: "model", type: "model", s, a, r: o.r, s2: o.s2, known: K[0] };
      const i0 = s * nA + a, P0 = priority(i0);
      queue(i0, P0);
      yield { line: "priority", type: "priority", s, a, P: P0, queued: P0 > theta };
      const list = [];
      for (let k = 0; k < n; k++) {
        let i = -1;
        for (let j = 0; j < PQ.length; j++) if (PQ[j] > 0 && (i < 0 || PQ[j] > PQ[i])) i = j;
        if (i < 0) break;
        PQ[i] = 0;
        const ps = Math.floor(i / nA), pa = i % nA, v = learn(m, env, p, ps, pa, MR[i], MS[i]);
        list.push({ s: ps, a: pa, s2: MS[i], delta: v.delta });
        for (let k2 = 0; k2 < K[0]; k2++) { const j = OL[k2]; if (MS[j] === ps) queue(j, priority(j)); }
      }
      let left = 0;
      for (let j = 0; j < PQ.length; j++) if (PQ[j] > 0) left++;
      yield { line: "plan", type: "plan", s: o.s2, list, left, ordered: true };
      s = o.s2;
      yield { line: "next", type: "next", s };
    }
  }

  const tex = lab.texNum;
  const qNumbers = (ev, p) =>
    `\\val{${ev.old.toFixed(2)}} + ${p.alpha}\\,\\big[\\rew{${+ev.r.toFixed(2)}} + ${p.gamma}\\cdot \\val{${tex(ev.next)}} - \\val{${tex(ev.old)}}\\big] = \\val{${ev.value.toFixed(2)}}`;
  const shared = {
    unit: "episode", memory, init,
    show: (m) => ({ Q: m.Q, model: m.MS, t: m.C[0] }),
    numbers: qNumbers,
    // What the live note says about the steps that are not updates of a real transition.
    note: null,
  };
  const qRule = "\\val{Q(S,A)} \\leftarrow \\val{Q(S,A)} + \\alp\\,\\big[\\rew{R} + \\gam \\max_a \\val{Q(S',a)} - \\val{Q(S,A)}\\big]";
  lab.algorithms = Object.assign(lab.algorithms || {}, {
    "dyna-q": { id: "dyna-q", title: "Dyna-Q", ...shared, run: (ctx) => dyna(ctx, false), rule: qRule },
    "dyna-q-plus": {
      id: "dyna-q-plus", title: "Dyna-Q+", ...shared, run: (ctx) => dyna(ctx, true),
      rule: "\\text{planning reward: } \\rew{R} + \\knob{\\kappa}\\sqrt{\\tau}, \\quad \\tau = \\text{steps since } (S,A) \\text{ was last tried}",
    },
    "prioritized-sweeping": { id: "prioritized-sweeping", title: "Prioritized sweeping", ...shared, run: sweeping, rule: qRule },
  });
})(globalThis.RL = globalThis.RL || {});
