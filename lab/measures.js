/* Measures: numbers computed after each unit of a run from what the algorithm knows, compared with the truth.
   The truth comes from the rules of the world (dynamic programming, or Blackjack's exact solution), never from
   another learning run, so a chart of these numbers shows how far the learner really is from the answer. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const cache = new Map(); // the truth of each world, worked out once (env.key tells a drawn world from another)
  const once = (key, make) => (cache.has(key) ? cache.get(key) : cache.set(key, make()).get(key));

  // The true values of the policy a prediction method evaluates.
  lab.truth = (env, p) => once(`v:${env.key || env.name}:${p.policy || "random"}:${p.gamma}`, () => {
    if (env.truth) return env.truth();
    if (env.kind === "blackjack") return lab.blackjackExact(lab.policyTable(env, p.policy)).V;
    return lab.evaluate(env, lab.policyTable(env, p.policy), p.gamma);
  });

  lab.optimalValues = (env, p) => once(`o:${env.key || env.name}:${p.gamma}`, () =>
    env.kind === "blackjack" ? lab.blackjackExact().V : lab.valueIteration(env, p.gamma));

  // The optimal actions of every state: optimal[s] is a set of actions (empty where nothing is decided).
  lab.optimalActions = (env, p) => once(`a:${env.key || env.name}:${p.gamma}`, () => {
    const P = env.kind === "blackjack" ? lab.blackjackExact().policy : lab.greedyPolicy(env, lab.optimalValues(env, p), p.gamma, 1e-6);
    return Array.from({ length: env.nS }, (_, s) => new Set(env.acts(s).filter((a) => P[s * env.nA + a] > 0)));
  });

  // Each measure makes a function (display, stats, t) that adds its number to the unit's stats.
  // `display` is what the algorithm shows (algorithm.show): its V, or its Q.
  const MEASURES = {
    // Root-mean-square error of the state values, over the states that count.
    error(env, p) {
      const truth = lab.truth(env, p);
      return (d, stats) => { stats.error = lab.rms(d.V, truth, env); };
    },
    // √VE: the root of the value error, each state's squared error weighted by the share of time spent there (μ), the
    // objective linear methods minimize. (Without μ, every state counts the same.)
    ve(env, p) {
      const truth = lab.truth(env, p), mu = env.mu ? env.mu() : null;
      return (d, stats) => {
        if (!mu) { stats.ve = lab.rms(d.V, truth, env); return; }
        let sum = 0;
        for (let s = 0; s < env.nS; s++) if (mu[s]) sum += mu[s] * (d.V[s] - truth[s]) ** 2;
        stats.ve = Math.sqrt(sum);
      };
    },
    // Root-mean-square distance of the state values from the optimal values v*.
    "optimal-error"(env, p) {
      const best = lab.optimalValues(env, p);
      return (d, stats) => { stats["optimal-error"] = lab.rms(d.V, best, env); };
    },
    // The share of decision states where every best-looking action is an optimal one.
    match(env, p) {
      const optimal = lab.optimalActions(env, p), states = optimal.map((set, s) => s).filter((s) => optimal[s].size && optimal[s].size < env.acts(s).length);
      return (d, stats) => {
        let ok = 0;
        for (const s of states) {
          const best = lab.maxQ(d.Q, s, env);
          if (env.acts(s).every((a) => d.Q[s * env.nA + a] < best || optimal[s].has(a))) ok++;
        }
        stats.match = ok / states.length;
      };
    },
    // The expected return of the greedy policy from the start, computed exactly; with p.judge = 1 (no discount) on
    // Frozen Lake, its chance of reaching the gem. Each evaluation starts from the last one's values, which are
    // nearly right already, and long runs measure only every few units.
    greedy(env, p, units) {
      const every = Math.max(1, Math.round(units / 400)), P = new Float64Array(env.nS * env.nA), gamma = p.judge ?? p.gamma;
      let V = null;
      return (d, stats, t) => {
        if (!V || t % every === 0 || t === units - 1) V = lab.evaluate(env, lab.greedyFromQ(env, d.Q, P), gamma, { theta: 1e-6, sweeps: 5000, start: V });
        stats.greedy = V[env.start];
      };
    },
    // The exact value of the start under the learner's current policy, J(θ) = v_π(start), the objective of a
    // policy-gradient method, worked out from the rules instead of estimated from noisy episodes. As for `greedy`,
    // each evaluation starts from the last one's values, and long runs measure only every few units.
    "policy-value"(env, p, units) {
      const every = Math.max(1, Math.round(units / 400)), gamma = p.judge ?? p.gamma;
      let V = null;
      return (d, stats, t) => {
        if (!V || t % every === 0 || t === units - 1) V = lab.evaluate(env, d.P, gamma, { theta: 1e-6, sweeps: 5000, start: V });
        stats["policy-value"] = V[env.start];
      };
    },
    // The return of the policy the learner would deploy (its P, or greedy on its Q), played once from the start in a
    // world without slips, until it reaches an end, falls or gives up after p.deployCap steps (100): what using it
    // would cost. A fall ends the episode here, so a policy that walks off the cliff scores about as badly as one that
    // walks into a wall for 100 steps.
    deployed(env, p) {
      const cap = p.deployCap ?? 100, rng = lab.rng(1);
      return (d, stats) => {
        let s = env.reset(rng), ret = 0;
        for (let k = 0; k < cap && !env.terminal(s); k++) {
          const a = d.P ? lab.argmax(d.P.subarray(s * env.nA, (s + 1) * env.nA)) : lab.greedy(d.Q, s, env);
          const o = env.step(s, a, rng);
          ret += o.r;
          if (o.fell !== undefined) break;
          s = o.s2;
        }
        stats.deployed = ret;
      };
    },
    // The short corridor: the chance of stepping right, the same in every cell.
    right(env) {
      return (d, stats) => { stats.right = d.P[env.start * env.nA + 1]; };
    },
    // The throw: where the Gaussian policy aims, its mean angle.
    aim() {
      return (d, stats) => { stats.aim = d.mu; };
    },
  };
  lab.measureNames = Object.keys(MEASURES);

  // The function lab.simulate calls after each unit, for the named measures.
  lab.measurer = function (names, env, p, algorithm, units) {
    if (!names?.length) return null;
    const fns = names.map((n) => {
      if (!MEASURES[n]) throw new Error(`unknown measure '${n}'`);
      return MEASURES[n](env, p, units);
    });
    return (m, stats, t) => {
      const d = algorithm.show(m, env, p, t);
      for (const f of fns) f(d, stats, t);
    };
  };
})(globalThis.RL = globalThis.RL || {});
