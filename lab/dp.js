/* Dynamic programming on a known model: the values the Bellman equations define, computed exactly.
   env.model(s, a) lists the outcomes { p, s2, r } of an action; a policy is an array of nS × nA probabilities.
   (The step-by-step versions the Lab plays are in agents/dp.js; they share lab.backup.) */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const states = (env) => Array.from({ length: env.nS }, (_, s) => s).filter((s) => !env.terminal(s) && !env.blocked?.(s));

  // Every available action equally likely.
  lab.randomPolicy = function (env) {
    const P = new Float64Array(env.nS * env.nA);
    for (let s = 0; s < env.nS; s++) {
      const acts = env.acts(s);
      for (const a of acts) P[s * env.nA + a] = 1 / acts.length;
    }
    return P;
  };

  // A policy by name: "random", or one the world defines (like Blackjack's "stick-20").
  lab.policyTable = (env, name = "random") => (name === "random" ? lab.randomPolicy(env) : env.policy(name));

  // The value of taking a in s and then being worth V: expected reward plus discounted value of where it leads.
  lab.backup = function (env, V, s, a, gamma) {
    let q = 0;
    for (const { p, s2, r } of env.model(s, a)) q += p * (r + gamma * (env.terminal(s2) ? 0 : V[s2]));
    return q;
  };

  // The expected backup of s under a policy: one line of the Bellman equation for v_π.
  lab.policyBackup = function (env, V, s, policy, gamma) {
    let v = 0;
    for (const a of env.acts(s)) {
      const pa = policy[s * env.nA + a];
      if (pa) v += pa * lab.backup(env, V, s, a, gamma);
    }
    return v;
  };

  lab.qFromV = function (env, V, gamma) {
    const Q = new Float64Array(env.nS * env.nA).fill(NaN);
    for (let s = 0; s < env.nS; s++) {
      for (const a of env.acts(s)) Q[s * env.nA + a] = env.terminal(s) ? 0 : lab.backup(env, V, s, a, gamma);
    }
    return Q;
  };

  // Iterative policy evaluation: sweep the states, in place, until no value changes by more than theta.
  // start: values to begin from (a good guess, like the values of a similar policy, saves many sweeps).
  lab.evaluate = function (env, policy, gamma, { theta = 1e-10, sweeps = 100000, start = null } = {}) {
    const V = start ? Float64Array.from(start) : new Float64Array(env.nS), todo = states(env);
    for (let k = 0; k < sweeps; k++) {
      let change = 0;
      for (const s of todo) {
        const v = lab.policyBackup(env, V, s, policy, gamma);
        change = Math.max(change, Math.abs(v - V[s]));
        V[s] = v;
      }
      if (change < theta) break;
    }
    return V;
  };

  // Value iteration: the optimal state values v*.
  lab.valueIteration = function (env, gamma, { theta = 1e-10, sweeps = 100000 } = {}) {
    const V = new Float64Array(env.nS), todo = states(env);
    for (let k = 0; k < sweeps; k++) {
      let change = 0;
      for (const s of todo) {
        let best = -Infinity;
        for (const a of env.acts(s)) best = Math.max(best, lab.backup(env, V, s, a, gamma));
        change = Math.max(change, Math.abs(best - V[s]));
        V[s] = best;
      }
      if (change < theta) break;
    }
    return V;
  };

  // The greedy policy with respect to V; tied best actions share the probability.
  lab.greedyPolicy = function (env, V, gamma, tolerance = 1e-9, P = new Float64Array(env.nS * env.nA)) {
    P.fill(0);
    for (let s = 0; s < env.nS; s++) {
      if (env.terminal(s) || env.blocked?.(s)) continue;
      const acts = env.acts(s), q = acts.map((a) => lab.backup(env, V, s, a, gamma));
      const best = Math.max(...q), winners = acts.filter((_, i) => q[i] >= best - tolerance);
      for (const a of winners) P[s * env.nA + a] = 1 / winners.length;
    }
    return P;
  };

  // The greedy policy with respect to action values Q (ties shared), as a table.
  lab.greedyFromQ = function (env, Q, P = new Float64Array(env.nS * env.nA)) {
    P.fill(0);
    for (let s = 0; s < env.nS; s++) {
      if (env.terminal(s)) continue;
      const m = lab.maxQ(Q, s, env), acts = env.acts(s).filter((a) => Q[s * env.nA + a] === m);
      for (const a of acts) P[s * env.nA + a] = 1 / acts.length;
    }
    return P;
  };

  lab.dpStates = states;
})(globalThis.RL = globalThis.RL || {});
