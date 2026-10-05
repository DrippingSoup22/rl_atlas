/* Worlds for function approximation, too big or too continuous for a table:
   - the 1000-state random walk of Sutton & Barto (Example 9.1): each step jumps up to 100 states left or right;
   - Mountain Car (Example 10.1): an underpowered car in a valley, whose state is a position and a speed;
   - Baird's counterexample (Example 11.1): seven states whose shared features make off-policy TD diverge.
   Each one gives its coordinates scaled to the unit interval (env.coords), from which lab.features builds features. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  // ---- the 1000-state random walk ----
  // States 1 … 1000 between two exits (0 pays −1, 1001 pays +1); the walk starts at 500. Each step goes left or right
  // at random, by 1 to 100 states, all equally likely; a jump past the end leaves by that exit.
  lab.longWalk = function ({ n = 1000, reach = 100 } = {}) {
    const nS = n + 2, both = [0, 1], none = [], start = n / 2;
    // The value of walking at random and the share of time spent in each state, both worked out from the rules.
    // Each is a fixed point of "average over the 2·reach states a jump can reach", computed with running sums.
    let truth = null, mu = null;
    const window = (V, s, ends) => {
      // the sum of V over the states within reach of s (s itself left out), with the exits' worth beyond each end
      const lo = s - reach, hi = s + reach;
      let sum = (V.sums[Math.min(n, hi)] - V.sums[s]) + (V.sums[s - 1] - V.sums[Math.max(1, lo) - 1]);
      if (ends) sum += Math.max(0, hi - n) * ends[1] + Math.max(0, 1 - lo) * ends[0];
      return sum;
    };
    function fixedPoint(base, ends) {
      let V = new Float64Array(nS);
      for (let k = 0; k < 100000; k++) {
        const sums = new Float64Array(nS);
        for (let s = 1; s <= n; s++) sums[s] = sums[s - 1] + V[s];
        V.sums = sums;
        const next = new Float64Array(nS);
        let change = 0;
        for (let s = 1; s <= n; s++) {
          next[s] = base(s) + window(V, s, ends) / (2 * reach);
          change = Math.max(change, Math.abs(next[s] - V[s]));
        }
        V = next;
        if (change < 1e-13) break;
      }
      return V;
    }
    const env = {
      name: "walk-1000", key: `walk-${n}`, kind: "line", title: `${n}-state random walk`, n, nS, nA: 2, start, reach, valueRange: 1, dims: 1,
      acts: (s) => (s === 0 || s === n + 1 ? none : both),
      terminal: (s) => s === 0 || s === n + 1,
      describe: (s, a) => `state ${s}${a >= 0 ? `, jumping ${a ? "right" : "left"}` : ""}`,
      reset: (rng, s0) => (s0 === undefined ? start : s0),
      step(s, a, rng) {
        const k = 1 + rng.int(reach), s2 = a ? s + k : s - k;
        if (s2 <= 0) return { s2: 0, r: -1, k };
        if (s2 >= n + 1) return { s2: n + 1, r: 1, k };
        return { s2, r: 0, k };
      },
      coords: (s) => [(s - 0.5) / n],
      // The true values (undiscounted): the expected exit payment from each state.
      truth: () => (truth ||= fixedPoint(() => 0, [-1, 1])),
      // μ(s): the share of all steps spent in s. Visits to s are the starts there plus the arrivals from each state within
      // reach, each with chance 1 / (2·reach); dividing by their total (the average length of an episode) gives shares.
      mu() {
        if (mu) return mu;
        const eta = fixedPoint((s) => (s === start ? 1 : 0), null);
        let total = 0;
        for (let s = 1; s <= n; s++) total += eta[s];
        mu = eta.map((v) => v / total);
        mu.steps = total; // the average number of steps in an episode
        return mu;
      },
    };
    return env;
  };
  lab.worlds["walk-1000"] = () => lab.longWalk();

  // ---- Mountain Car ----
  // Position x in [−1.2, 0.5], speed v in [−0.07, 0.07]; the goal is the top of the hill on the right (x ≥ 0.5).
  // Actions: full throttle backward, none, forward. Gravity pulls with −0.0025 cos(3x), stronger than the engine's 0.001,
  // so the car must first back up the left slope. Every step costs −1; the left wall stops the car dead.
  const MC = { xMin: -1.2, xMax: 0.5, vMax: 0.07 };
  lab.mountainCar = function () {
    const all = [0, 1, 2];
    const env = {
      name: "mountain-car", key: "mountain-car", kind: "car", title: "Mountain Car", nA: 3, dims: 2, continuous: true, valueRange: 100, ...MC,
      actionNames: ["reverse", "coast", "forward"],
      acts: () => all,
      terminal: (s) => s[0] >= MC.xMax,
      height: (x) => Math.sin(3 * x), // the hill, for drawing
      describe: (s, a) => `x = ${s[0].toFixed(2)}, v = ${s[1].toFixed(3)}${a >= 0 ? `, ${env.actionNames[a]}` : ""}`,
      // Each episode starts at rest somewhere near the bottom of the valley.
      reset: (rng, s0) => s0 || [-0.6 + 0.2 * rng.next(), 0],
      step(s, a) {
        let v = s[1] + 0.001 * (a - 1) - 0.0025 * Math.cos(3 * s[0]);
        v = Math.max(-MC.vMax, Math.min(MC.vMax, v));
        let x = s[0] + v;
        if (x <= MC.xMin) { x = MC.xMin; v = 0; }
        return { s2: [Math.min(x, MC.xMax), v], r: -1 };
      },
      coords: (s) => [(s[0] - MC.xMin) / (MC.xMax - MC.xMin), (s[1] + MC.vMax) / (2 * MC.vMax)],
    };
    return env;
  };
  lab.worlds["mountain-car"] = () => lab.mountainCar();

  // ---- Baird's counterexample ----
  // Six upper states and one lower state. The dashed action jumps to one of the six upper states at random, the solid
  // action to the lower state. Nothing pays anything, so every true value is 0, and γ = 0.99. The behavior policy takes
  // the dashed action 6 times out of 7, the target policy always the solid one. The features: upper state i has
  // v̂ = 2wᵢ + w₈, the lower state v̂ = w₇ + 2w₈; with eight weights for seven states, any values can be written.
  // It never ends: a run is cut into units of one step each, and the world remembers where the agent is (env.state).
  lab.baird = function () {
    const DASHED = 0, SOLID = 1, both = [DASHED, SOLID], LOW = 6;
    const names = ["1", "2", "3", "4", "5", "6", "7"];
    const env = {
      name: "baird", key: "baird", kind: "star", title: "Baird's counterexample", nS: 7, nA: 2, names, unitName: "step", stretch: 1, valueRange: 10,
      state: new Float64Array(1), DASHED, SOLID, LOW,
      actionNames: ["dashed", "solid"],
      acts: () => both,
      terminal: () => false,
      describe: (s, a) => `state ${names[s]}${a >= 0 ? `, the ${env.actionNames[a]} action` : ""}`,
      init(rng) { env.state[0] = rng.int(7); },
      reset: () => env.state[0],
      step(s, a, rng) {
        const s2 = a === SOLID ? LOW : rng.int(6);
        env.state[0] = s2;
        return { s2, r: 0 };
      },
      model: (s, a) => (a === SOLID ? [{ p: 1, s2: LOW, r: 0 }] : names.slice(0, 6).map((_, i) => ({ p: 1 / 6, s2: i, r: 0 }))),
      policy(name) {
        const P = new Float64Array(14);
        for (let s = 0; s < 7; s++) {
          P[s * 2 + DASHED] = name === "behavior" ? 6 / 7 : 0;
          P[s * 2 + SOLID] = name === "behavior" ? 1 / 7 : 1;
        }
        return P;
      },
      truth: () => new Float64Array(7),
      features: {
        n: 8, k: 2,
        at(s, x) {
          if (s === LOW) { x.idx[0] = 6; x.val[0] = 1; x.idx[1] = 7; x.val[1] = 2; }
          else { x.idx[0] = s; x.val[0] = 2; x.idx[1] = 7; x.val[1] = 1; }
          x.k = 2;
        },
        init: (w) => w.set([1, 1, 1, 1, 1, 1, 10, 1]),
      },
      // The values those first weights give, for a table that starts from the same place.
      initV: Float64Array.from([3, 3, 3, 3, 3, 3, 12]),
    };
    return env;
  };
  lab.worlds.baird = () => lab.baird();
})(globalThis.RL = globalThis.RL || {});
