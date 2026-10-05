/* The k-armed testbed of Sutton & Barto (§2.3): k slot machines in a single state. Each run draws new machines:
   arm a pays a normal reward around its own mean q*(a), and the means are drawn from a normal distribution.
   In the drifting version every mean takes a small random step after each pull. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  // k arms; the means q*(a) ~ N(mean, 1); rewards ~ N(q*(a), 1). drift: the standard deviation of each mean's step.
  lab.bandit = function ({ k = 10, mean = 0, drift = 0, title = "10-armed testbed" } = {}) {
    const q = new Float64Array(k), arms = Array.from({ length: k }, (_, a) => a);
    const env = {
      name: "bandit", kind: "bandit", title, k, nS: 1, nA: k, start: 0, mean, drift, unitName: "pull",
      q, // the true means: part of the world's state, saved with every snapshot
      state: q,
      best: 0,
      valueRange: 3,
      acts: () => arms,
      terminal: () => false,
      describe: (s, a) => (a >= 0 ? `arm ${a + 1}` : ""),
      // A new set of machines. Drifting ones all start level, as in Sutton & Barto's Exercise 2.5.
      init(rng) {
        for (let a = 0; a < k; a++) q[a] = drift ? mean : mean + rng.normal();
        env.best = lab.argmax(q);
      },
      reset: () => 0,
      // `optimal` says whether the pulled arm was a best one when it was chosen, before the means drift.
      step(s, a, rng) {
        const optimal = q[a] >= q[env.best], r = q[a] + rng.normal();
        if (drift) {
          for (let b = 0; b < k; b++) q[b] += drift * rng.normal();
          env.best = lab.argmax(q);
        }
        return { s2: 0, r, optimal };
      },
      track(ev, stats) { stats.optimal = ev.optimal ? 1 : 0; },
      // After a snapshot of the means is put back.
      restore() { env.best = lab.argmax(q); },
    };
    return env;
  };

  lab.worlds.testbed = () => lab.bandit();
  lab.worlds["testbed-4"] = () => lab.bandit({ mean: 4, title: "10-armed testbed, means around +4" });
  lab.worlds.drifting = () => lab.bandit({ drift: 0.01, title: "Drifting 10-armed testbed" });
})(globalThis.RL = globalThis.RL || {});
