/* Bandit algorithms as generators: each unit is one pull of one arm (choose, pull, update). */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const stepOf = (p, n) => (p.alpha > 0 ? p.alpha : 1 / n);

  // ---- ε-greedy action values: sample averages (α = 0) or a constant step size, starting from Q₁ = q0 ----
  function* epsilonGreedy({ env, m, rng, p }) {
    const { Q, N } = m, acts = env.acts(0);
    const explore = rng.next() < p.epsilon, a = explore ? acts[rng.int(acts.length)] : lab.greedy(Q, 0, env, rng);
    yield { line: "choose", type: "choose", s: 0, a, explore };
    const o = env.step(0, a, rng);
    yield { line: "act", type: "move", s: 0, a, ...o };
    N[a] += 1;
    const old = Q[a], step = stepOf(p, N[a]);
    Q[a] = old + step * (o.r - old);
    yield { line: "update", type: "update", s: 0, a, old, target: o.r, delta: o.r - old, value: Q[a], n: N[a], step };
  }

  // ---- upper confidence bound: the estimate plus a bonus that shrinks as an arm is tried more ----
  const bounds = (Q, N, c, time, out = new Float64Array(Q.length)) => {
    for (let a = 0; a < Q.length; a++) out[a] = N[a] ? Q[a] + c * Math.sqrt(Math.log(time) / N[a]) : Infinity;
    return out;
  };
  function* ucb({ env, m, rng, p, t }) {
    const { Q, N } = m, U = bounds(Q, N, p.c, t + 1, (p.bounds ||= new Float64Array(env.k)));
    let best = -Infinity, ties = 0, a = 0;
    for (let b = 0; b < env.k; b++) {
      if (U[b] > best) { best = U[b]; ties = 1; a = b; } else if (U[b] === best) ties++;
    }
    if (ties > 1) { // break the tie at random among the arms that share the best bound
      let pick = rng.int(ties);
      for (let b = 0; b < env.k; b++) if (U[b] === best && pick-- === 0) { a = b; break; }
    }
    yield { line: "choose", type: "choose", s: 0, a, bound: best };
    const o = env.step(0, a, rng);
    yield { line: "act", type: "move", s: 0, a, ...o };
    N[a] += 1;
    const old = Q[a], step = stepOf(p, N[a]);
    Q[a] = old + step * (o.r - old);
    yield { line: "update", type: "update", s: 0, a, old, target: o.r, delta: o.r - old, value: Q[a], n: N[a], step };
  }

  // ---- gradient bandit: preferences H, the policy is their softmax; a reward above the baseline makes
  // the chosen arm more likely and every other arm less likely ----
  function* gradient({ env, m, rng, p, t }) {
    const { H, base } = m, k = env.k, pi = lab.softmax(H, 0, k, (p.pi ||= new Float64Array(k)));
    const a = lab.pick(pi, rng);
    yield { line: "choose", type: "choose", s: 0, a, pa: pi[a] };
    const o = env.step(0, a, rng);
    yield { line: "act", type: "move", s: 0, a, ...o };
    // The baseline is the average of the rewards before this one (the first reward is its own baseline).
    const baseline = p.baseline === false ? 0 : t === 0 ? o.r : base[0], delta = o.r - baseline, old = H[a];
    for (let b = 0; b < k; b++) H[b] += p.alpha * delta * ((b === a ? 1 : 0) - pi[b]);
    base[0] += (o.r - base[0]) / (t + 1);
    yield { line: "update", type: "update", s: 0, a, r: o.r, old, baseline, delta, value: H[a], pa: pi[a] };
  }

  const averages = {
    unit: "step",
    memory: (env) => ({ Q: env.k, N: env.k }),
    init: (m, env, p) => m.Q.fill(p.q0 || 0),
    numbers: (ev, p) => `\\val{${ev.old.toFixed(2)}} + ${p.alpha > 0 ? p.alpha : `\\tfrac{1}{${ev.n}}`}\\,\\big[\\rew{${ev.target.toFixed(2)}} - \\val{${lab.texNum(ev.old)}}\\big] = \\val{${ev.value.toFixed(2)}}`,
    rule: (p) => `\\val{Q(A)} \\leftarrow \\val{Q(A)} + ${p.alpha > 0 ? "\\alp" : "\\tfrac{1}{N(A)}"}\\,\\big[\\rew{R} - \\val{Q(A)}\\big]`,
  };

  Object.assign(lab.algorithms, {
    "epsilon-greedy": {
      id: "epsilon-greedy", title: "ε-greedy", ...averages, run: epsilonGreedy,
      show: (m) => ({ Q: m.Q, N: m.N }),
    },
    // The same learner started optimistic: Q₁ = q0 (5 in Sutton & Barto's Figure 2.3).
    "optimistic-init": {
      id: "optimistic-init", title: "Optimistic greedy", ...averages, run: epsilonGreedy,
      show: (m) => ({ Q: m.Q, N: m.N }),
    },
    ucb: {
      id: "ucb", title: "UCB", ...averages, run: ucb,
      show: (m, env, p, t) => ({ Q: m.Q, N: m.N, U: bounds(m.Q, m.N, p.c, t + 1) }),
    },
    "gradient-bandit": {
      id: "gradient-bandit", title: "Gradient bandit", unit: "step", run: gradient,
      memory: (env) => ({ H: env.k, base: 1 }),
      show: (m) => ({ H: m.H, pi: lab.softmax(m.H), baseline: m.base[0] }),
      rule: (p) => `\\pol{H(a)} \\leftarrow \\pol{H(a)} + \\alp\\,\\big(\\rew{R} - ${p.baseline === false ? "0" : "\\bar{R}"}\\big)\\big(\\mathbb{1}_{a = A} - \\pol{\\pi(a)}\\big)`,
      numbers: (ev, p) => `\\pol{H(A)}: ${ev.old.toFixed(2)} + ${p.alpha}\\,(\\rew{${ev.r.toFixed(2)}} - ${lab.texNum(ev.baseline)})(1 - ${ev.pa.toFixed(2)}) = \\pol{${ev.value.toFixed(2)}}`,
    },
  });
})(globalThis.RL = globalThis.RL || {});
