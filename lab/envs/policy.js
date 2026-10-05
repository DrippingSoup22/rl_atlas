/* Worlds for policy-gradient methods:
   - the short corridor of Sutton & Barto (Example 13.1): three cells and a goal, where the middle cell swaps left and
     right, and the agent cannot tell the cells apart. The best policy is random: no deterministic one ever arrives;
   - the throw: one throw per episode, the action an angle (a real number) and the reward how far the ball flies.
     It is the simplest world with a continuous action, for a Gaussian policy. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});
  const LEFT = 0, RIGHT = 1;

  // ---- the short corridor ----
  // Cells 0, 1, 2 and the goal 3. Left in cell 0 bumps into the wall; in cell 1 both actions are swapped. Each step
  // costs 1. The policy sees the same single feature in every cell, so it can only choose one chance p of stepping
  // right, the same everywhere. The value of the start is then J(p) = −2(2 − p) / (p(1 − p)): best at p = 2 − √2.
  lab.corridor = function () {
    const both = [LEFT, RIGHT], none = [], best = 2 - Math.SQRT2;
    const J = (p) => (p <= 0 || p >= 1 ? -Infinity : (-2 * (2 - p)) / (p * (1 - p)));
    const env = {
      name: "corridor", key: "corridor", kind: "corridor", title: "Short corridor", nS: 4, nA: 2, start: 0, valueRange: 20,
      reversed: [false, true, false, false], actionNames: ["left", "right"],
      acts: (s) => (s === 3 ? none : both),
      terminal: (s) => s === 3,
      describe: (s, a) => `${s === 3 ? "the goal" : `cell ${s + 1}${s === 1 ? " (swapped)" : ""}`}${a >= 0 ? `, stepping ${a === RIGHT ? "right" : "left"}` : ""}`,
      reset: (rng, s0) => (s0 === undefined ? 0 : s0),
      step(s, a) {
        const dir = (a === RIGHT ? 1 : -1) * (s === 1 ? -1 : 1);
        return { s2: Math.max(0, s + dir), r: -1 };
      },
      model: (s, a) => [{ p: 1, ...env.step(s, a) }],
      // one feature, the same in every cell: the cells look alike, to the policy and to a learned baseline
      features: { n: 1, k: 1, at(s, x) { x.idx[0] = 0; x.val[0] = 1; x.k = 1; } },
      // The first policy steps right 5% of the time, as ε-greedy with ε = 0.1 preferring left does, unless a run says
      // otherwise (right0): the two preferences sit symmetrically around 0.
      policyInit(theta, p) { const q = p.right0 ?? 0.05, h = 0.5 * Math.log(q / (1 - q)); theta[RIGHT] = h; theta[LEFT] = -h; },
      J, best, bestValue: J(best),
      // the values of the three cells and the goal under p, for the views
      values(p) {
        if (p <= 0 || p >= 1) return [-Infinity, -Infinity, -Infinity, 0];
        const v1 = (p - 3) / (p * (1 - p));
        return [v1 - 1 / p, v1, -1 + (1 - p) * v1, 0];
      },
    };
    return env;
  };
  lab.worlds.corridor = () => lab.corridor();

  // ---- the throw ----
  // One state, one throw: the action is the angle in degrees, clipped to 0°–90° (outside, the ball is just dropped).
  // The ball flies 40 sin(2 × angle) meters, 40 m at 45°, and the wind adds or takes away a few (normal, sd 2 m).
  // The reward is the distance. The policy measures angles in steps of 10°.
  lab.throwWorld = function () {
    const env = {
      name: "throw", key: "throw", kind: "throw", title: "The throw", nS: 2, nA: 1, start: 0, unitName: "throw", valueRange: 40,
      continuousActions: { unit: 10, lo: 0, hi: 90, best: 45 },
      wind: 2, reach: 40,
      acts: (s) => (s ? [] : [0]),
      terminal: (s) => s === 1,
      describe: (s, a) => (a === undefined || a < 0 ? "the throw" : `a throw at ${a.toFixed(1)}°`),
      reset: () => 0,
      angle: (a) => Math.max(0, Math.min(90, a)),
      distance: (a) => env.reach * Math.sin((2 * env.angle(a) * Math.PI) / 180),
      step(s, a, rng) {
        const d = env.distance(a), wind = env.wind * rng.normal();
        return { s2: 1, r: d + wind, d, wind, angle: env.angle(a) };
      },
      // the angle actually thrown, for the Lab's captions
      track(ev, stats) { stats.angle = ev.angle; },
      features: { n: 1, k: 1, at(s, x) { x.idx[0] = 0; x.val[0] = 1; x.k = 1; } },
      // The first aim is 20° with a spread of 10°, unless a run says otherwise (mu0, sd0), in the policy's unit of 10°.
      policyInit(theta, p) { theta[0] = (p.mu0 ?? 20) / 10; theta[1] = Math.log((p.sd0 ?? 10) / 10); },
    };
    return env;
  };
  lab.worlds.throw = () => lab.throwWorld();
})(globalThis.RL = globalThis.RL || {});
