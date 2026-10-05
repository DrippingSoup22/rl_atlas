/* Dynamic programming, step by step, for the Lab: each unit is one sweep through the states (policy iteration's
   improvement step counts as a sweep too). The backups themselves are lab.backup and lab.policyBackup (dp.js). */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const TIE = 1e-9; // actions within this of the best count as tied

  // The outcomes behind a backup, for the live formula: [{ a, pa, outs: [{ p, r, v }] }].
  function terms(env, V, s, P, p) {
    return env.acts(s).filter((a) => !P || P[s * env.nA + a] > 0).map((a) => ({
      a, pa: P ? P[s * env.nA + a] : 1,
      outs: env.model(s, a).map(({ p: q, s2, r }) => ({ p: q, r, v: env.terminal(s2) ? 0 : V[s2] })),
    }));
  }

  // One sweep of the Bellman equation for the policy P. Returns the largest change. In place by default: later
  // states already see the new values of earlier ones. With p.sync, every backup uses the values from before the
  // sweep (the "two-array" version, which Sutton & Barto's Figure 4.1 shows).
  function* sweep({ env, m, p }, P) {
    const V = m.V, from = p.sync ? Float64Array.from(V) : V;
    let delta = 0;
    for (const s of lab.dpStates(env)) {
      yield { line: "sweep", type: "sweep", s };
      const old = V[s], parts = terms(env, from, s, P, p), v = lab.policyBackup(env, from, s, P, p.gamma);
      V[s] = v;
      delta = Math.max(delta, Math.abs(v - old));
      yield { line: "update", type: "update", s, a: -1, old, value: v, target: v, delta: v - old, terms: parts };
    }
    yield { line: "delta", type: "info", delta };
    return delta;
  }

  function* evaluation(ctx) {
    const delta = yield* sweep(ctx, lab.policyOf(ctx.env, ctx.p));
    return { delta, converged: +(delta < ctx.p.theta) };
  }

  // Policy iteration: evaluation sweeps until the values settle (largest change below θ), then one improvement
  // step that makes the policy greedy; it stops when an improvement step changes nothing.
  function* policyIteration(ctx) {
    const { env, m, p } = ctx, { V, P, phase } = m, nA = env.nA;
    if (phase[0] === 2) { yield { line: "stable", type: "info" }; return { delta: 0, converged: 1 }; }
    if (phase[0] === 0) {
      const delta = yield* sweep(ctx, P);
      if (delta < p.theta) phase[0] = 1;
      return { delta, improved: 0, changed: 0, converged: 0 }; // every unit reports the same numbers, so short runs chart them too
    }
    let changed = 0;
    const row = new Float64Array(nA);
    for (const s of lab.dpStates(env)) {
      yield { line: "improve", type: "sweep", s };
      const acts = env.acts(s), q = acts.map((a) => lab.backup(env, V, s, a, p.gamma)), best = Math.max(...q);
      const winners = acts.filter((_, i) => q[i] >= best - TIE);
      row.fill(0);
      for (const a of winners) row[a] = 1 / winners.length;
      const before = P.subarray(s * nA, s * nA + nA), same = row.every((x, a) => Math.abs(x - before[a]) < 1e-12);
      if (!same) { changed += 1; before.set(row); }
      yield { line: "improve", type: "improve", s, changed: !same, q: Object.fromEntries(acts.map((a, i) => [a, q[i]])) };
    }
    phase[0] = changed ? 0 : 2;
    yield { line: changed ? "unstable" : "stable", type: "info", changed };
    return { delta: 0, improved: 1, changed, converged: +!changed };
  }

  // Value iteration: every sweep takes the best action's backup.
  function* valueIteration({ env, m, p }) {
    const V = m.V, from = p.sync ? Float64Array.from(V) : V;
    let delta = 0;
    for (const s of lab.dpStates(env)) {
      yield { line: "sweep", type: "sweep", s };
      const old = V[s], parts = terms(env, from, s, null, p);
      let v = -Infinity;
      for (const a of env.acts(s)) v = Math.max(v, lab.backup(env, from, s, a, p.gamma));
      V[s] = v;
      delta = Math.max(delta, Math.abs(v - old));
      yield { line: "update", type: "update", s, a: -1, old, value: v, target: v, delta: v - old, terms: parts, max: true };
    }
    yield { line: "delta", type: "info", delta };
    return { delta, converged: +(delta < p.theta) };
  }

  // The backup with its numbers: ¼·(−1 + 1·(−1.00)) + … for a policy, or max(…) for value iteration.
  function numbers(ev, p) {
    const one = ({ p: q, r, v }) => `${q === 1 ? "" : `${+q.toFixed(3)}\\,`}(\\rew{${+r.toFixed(2)}} + ${p.gamma}\\cdot\\val{${lab.texNum(v)}})`;
    const action = (t) => t.outs.map(one).join(" + ");
    const shown = ev.terms.slice(0, 4), more = ev.terms.length > 4 ? " + \\cdots" : "";
    const body = ev.max
      ? `\\max\\big\\{${shown.map(action).join(",\\ ")}${more ? ",\\ \\dots" : ""}\\big\\}`
      : shown.map((t) => `${t.pa === 1 ? "" : `${+t.pa.toFixed(3)}`}\\big[${action(t)}\\big]`).join(" + ") + more;
    return `${body} = \\val{${ev.value.toFixed(2)}}`;
  }

  const values = { memory: (env) => ({ V: env.nS }) };
  Object.assign(lab.algorithms, {
    "policy-evaluation": {
      id: "policy-evaluation", title: "Policy evaluation", unit: "sweep", ...values, run: evaluation,
      show: (m, env, p) => ({ V: m.V, P: lab.policyOf(env, p) }),
      rule: "\\val{V(s)} \\leftarrow \\sum_a \\pol{\\pi(a \\mid s)} \\sum_{s',r} p(s',r \\mid s,a)\\,\\big[\\rew{r} + \\gam\\,\\val{V(s')}\\big]",
      numbers,
    },
    "policy-iteration": {
      id: "policy-iteration", title: "Policy iteration", unit: "sweep", run: policyIteration,
      memory: (env) => ({ V: env.nS, P: env.nS * env.nA, phase: 1 }),
      init: (m, env) => m.P.set(lab.randomPolicy(env)),
      show: (m) => ({ V: m.V, P: m.P }),
      rule: "\\val{V(s)} \\leftarrow \\sum_a \\pol{\\pi(a \\mid s)} \\sum_{s',r} p(s',r \\mid s,a)\\,\\big[\\rew{r} + \\gam\\,\\val{V(s')}\\big]",
      numbers,
    },
    "value-iteration": {
      id: "value-iteration", title: "Value iteration", unit: "sweep", ...values, run: valueIteration,
      show: (m, env, p) => ({ V: m.V, P: lab.greedyPolicy(env, m.V, p.gamma) }),
      rule: "\\val{V(s)} \\leftarrow \\max_a \\sum_{s',r} p(s',r \\mid s,a)\\,\\big[\\rew{r} + \\gam\\,\\val{V(s')}\\big]",
      numbers,
    },
  });
})(globalThis.RL = globalThis.RL || {});
