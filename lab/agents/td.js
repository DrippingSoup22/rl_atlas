/* Temporal-difference methods as generators: each one plays one episode and pauses at every line of its
   pseudocode (the `line` ids match the {#ids} in the entries; `type` says what happened, for the views).
   The Lab steps through them; a run just drains them. */
(function (RL) {
  "use strict";
  const lab = RL.lab;

  // ---- shared by the algorithm files ----
  // A number inside a TeX formula, in brackets when negative so "+ (−1.00)" reads right.
  lab.texNum = (v, digits = 2) => (v < 0 ? `(${v.toFixed(digits)})` : v.toFixed(digits));
  // An action drawn from a policy table (nS × nA probabilities).
  lab.pickRow = function (P, s, env, rng) {
    const acts = env.acts(s);
    let u = rng.next();
    for (const a of acts) { u -= P[s * env.nA + a]; if (u < 0) return a; }
    for (let i = acts.length - 1; i > 0; i--) if (P[s * env.nA + acts[i]] > 0) return acts[i];
    return acts[0];
  };
  // The policy a prediction method evaluates, built once per run and kept with its settings.
  lab.policyOf = (env, p) => (p.policyTable ||= lab.policyTable(env, p.policy));
  // A table of action values, all starting at q0 (0 unless a preset says otherwise).
  lab.qTable = { memory: (env) => ({ Q: env.nS * env.nA }), init: (m, env, p) => m.Q.fill(p.q0 || 0) };

  const numbers = (ev, p, next) =>
    `\\val{${ev.old.toFixed(2)}} + ${p.alpha}\\,\\big[\\rew{${+ev.r.toFixed(2)}} + ${p.gamma}\\cdot ${next} - \\val{${lab.texNum(ev.old)}}\\big] = \\val{${ev.value.toFixed(2)}}`;

  // ---- TD(0): the value of a fixed policy, one step at a time ----
  function* td0({ env, m, rng, p }) {
    const V = m.V, P = lab.policyOf(env, p);
    let s = env.reset(rng);
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = lab.pickRow(P, s, env, rng);
      yield { line: "choose", type: "choose", s, a };
      const o = env.step(s, a, rng);
      yield { line: "act", type: "move", s, a, ...o };
      const old = V[s], next = env.terminal(o.s2) ? 0 : V[o.s2], target = o.r + p.gamma * next;
      V[s] = old + p.alpha * (target - old);
      yield { line: "update", type: "update", s, a: -1, r: o.r, s2: o.s2, old, next, target, delta: target - old, value: V[s] };
      s = o.s2;
      yield { line: "next", type: "next", s };
    }
  }

  // ---- SARSA: the next action is chosen first, and its value is the target ----
  function* sarsa({ env, m, rng, p }) {
    const Q = m.Q, nA = env.nA;
    let s = env.reset(rng);
    yield { line: "start", type: "start", s };
    let a = lab.epsGreedy(Q, s, env, p.epsilon, rng);
    yield { line: "choose", type: "choose", s, a };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const o = env.step(s, a, rng);
      yield { line: "act", type: "move", s, a, ...o };
      const end = env.terminal(o.s2);
      const a2 = end ? -1 : lab.epsGreedy(Q, o.s2, env, p.epsilon, rng);
      yield { line: "choose-next", type: "choose", s: o.s2, a: a2 };
      const i = s * nA + a, old = Q[i], next = end ? 0 : Q[o.s2 * nA + a2], target = o.r + p.gamma * next;
      Q[i] = old + p.alpha * (target - old);
      yield { line: "update", type: "update", s, a, r: o.r, s2: o.s2, old, next, target, delta: target - old, value: Q[i] };
      s = o.s2;
      a = a2;
      yield { line: "next", type: "next", s, a };
    }
  }

  // ---- Q-learning: the target uses the best next action, whatever the agent does next ----
  function* qLearning({ env, m, rng, p }) {
    const Q = m.Q, nA = env.nA;
    let s = env.reset(rng);
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = lab.epsGreedy(Q, s, env, p.epsilon, rng);
      yield { line: "choose", type: "choose", s, a };
      const o = env.step(s, a, rng);
      yield { line: "act", type: "move", s, a, ...o };
      const i = s * nA + a, old = Q[i], next = env.terminal(o.s2) ? 0 : lab.maxQ(Q, o.s2, env), target = o.r + p.gamma * next;
      Q[i] = old + p.alpha * (target - old);
      yield { line: "update", type: "update", s, a, r: o.r, s2: o.s2, old, next, target, delta: target - old, value: Q[i] };
      s = o.s2;
      yield { line: "next", type: "next", s };
    }
  }

  // ---- Expected SARSA: the target averages the next action values over the ε-greedy policy ----
  function* expectedSarsa({ env, m, rng, p }) {
    const Q = m.Q, nA = env.nA, probs = new Array(nA);
    let s = env.reset(rng);
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = lab.epsGreedy(Q, s, env, p.epsilon, rng);
      yield { line: "choose", type: "choose", s, a };
      const o = env.step(s, a, rng);
      yield { line: "act", type: "move", s, a, ...o };
      let next = 0;
      if (!env.terminal(o.s2)) {
        lab.epsGreedyProbs(Q, o.s2, env, p.epsilon, probs);
        for (const b of env.acts(o.s2)) next += probs[b] * Q[o.s2 * nA + b];
      }
      const i = s * nA + a, old = Q[i], target = o.r + p.gamma * next;
      Q[i] = old + p.alpha * (target - old);
      yield { line: "update", type: "update", s, a, r: o.r, s2: o.s2, old, next, target, delta: target - old, value: Q[i] };
      s = o.s2;
      yield { line: "next", type: "next", s };
    }
  }

  // ---- Double Q-learning: two tables; one picks the best next action, the other says what it is worth ----
  // Q holds their average: the agent acts ε-greedily on it, and the Lab shows it.
  function* doubleQ({ env, m, rng, p }) {
    const { Q1, Q2, Q } = m, nA = env.nA;
    let s = env.reset(rng);
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = lab.epsGreedy(Q, s, env, p.epsilon, rng);
      yield { line: "choose", type: "choose", s, a };
      const o = env.step(s, a, rng);
      yield { line: "act", type: "move", s, a, ...o };
      const first = rng.next() < 0.5, [X, Y] = first ? [Q1, Q2] : [Q2, Q1];
      const i = s * nA + a, old = X[i];
      let next = 0, best = -1;
      if (!env.terminal(o.s2)) { best = lab.greedy(X, o.s2, env, rng); next = Y[o.s2 * nA + best]; }
      const target = o.r + p.gamma * next;
      X[i] = old + p.alpha * (target - old);
      Q[i] = (Q1[i] + Q2[i]) / 2;
      yield { line: first ? "update-1" : "update-2", type: "update", s, a, r: o.r, s2: o.s2, old, next, target, delta: target - old, value: X[i], table: first ? 1 : 2, best, shown: Q[i] };
      s = o.s2;
      yield { line: "next", type: "next", s };
    }
  }

  const control = { unit: "episode", ...lab.qTable, show: (m) => ({ Q: m.Q }) };
  lab.algorithms = Object.assign(lab.algorithms || {}, {
    td0: {
      id: "td0", title: "TD(0)", unit: "episode", run: td0,
      memory: (env) => ({ V: env.nS }),
      init: (m, env, p) => { for (let s = 0; s < env.nS; s++) m.V[s] = env.terminal(s) ? 0 : p.v0 || 0; },
      show: (m, env, p) => ({ V: m.V, P: lab.policyOf(env, p) }), // P: the policy being evaluated, for the arrows
      rule: "\\val{V(S)} \\leftarrow \\val{V(S)} + \\alp\\,\\big[\\rew{R} + \\gam\\,\\val{V(S')} - \\val{V(S)}\\big]",
      numbers: (ev, p) => numbers(ev, p, `\\val{${lab.texNum(ev.next)}}`),
    },
    sarsa: {
      id: "sarsa", title: "SARSA", ...control, run: sarsa,
      rule: "\\val{Q(S,A)} \\leftarrow \\val{Q(S,A)} + \\alp\\,\\big[\\rew{R} + \\gam\\,\\val{Q(S',A')} - \\val{Q(S,A)}\\big]",
      numbers: (ev, p) => numbers(ev, p, `\\val{${lab.texNum(ev.next)}}`),
    },
    "q-learning": {
      id: "q-learning", title: "Q-learning", ...control, run: qLearning,
      rule: "\\val{Q(S,A)} \\leftarrow \\val{Q(S,A)} + \\alp\\,\\big[\\rew{R} + \\gam \\max_a \\val{Q(S',a)} - \\val{Q(S,A)}\\big]",
      numbers: (ev, p) => numbers(ev, p, `\\val{${lab.texNum(ev.next)}}`),
    },
    "expected-sarsa": {
      id: "expected-sarsa", title: "Expected SARSA", ...control, run: expectedSarsa,
      rule: "\\val{Q(S,A)} \\leftarrow \\val{Q(S,A)} + \\alp\\,\\big[\\rew{R} + \\gam \\textstyle\\sum_a \\pol{\\pi(a \\mid S')}\\,\\val{Q(S',a)} - \\val{Q(S,A)}\\big]",
      numbers: (ev, p) => numbers(ev, p, `\\val{${lab.texNum(ev.next)}}`),
    },
    "double-q": {
      id: "double-q", title: "Double Q-learning", unit: "episode", run: doubleQ,
      memory: (env) => ({ Q1: env.nS * env.nA, Q2: env.nS * env.nA, Q: env.nS * env.nA }),
      init: (m, env, p) => { m.Q1.fill(p.q0 || 0); m.Q2.fill(p.q0 || 0); m.Q.fill(p.q0 || 0); },
      show: (m) => ({ Q: m.Q, Q1: m.Q1, Q2: m.Q2 }),
      rule: "\\val{Q_1(S,A)} \\leftarrow \\val{Q_1(S,A)} + \\alp\\,\\big[\\rew{R} + \\gam\\,\\val{Q_2\\big(S', \\operatorname{arg\\,max}_a Q_1(S',a)\\big)} - \\val{Q_1(S,A)}\\big]",
      numbers: (ev, p) => numbers(ev, p, `\\val{${lab.texNum(ev.next)}}`),
    },
  });
})(globalThis.RL = globalThis.RL || {});
