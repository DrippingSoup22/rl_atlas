/* The random walk of Sutton & Barto (Example 6.2): a row of states between two exits. The agent starts in the
   middle and steps left or right; leaving on the right pays +1, leaving on the left pays `left` (0 by default).
   Walking at random, the value of the i-th of n states is (i / (n + 1)) when the left exit pays 0. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});
  const LEFT = 0, RIGHT = 1;

  lab.chain = function ({ n = 5, left = 0, title = "Random walk" } = {}) {
    const nS = n + 2, both = [LEFT, RIGHT], none = [];
    const names = Array.from({ length: nS }, (_, s) => (s === 0 || s === n + 1 ? "exit" : n <= 26 ? String.fromCharCode(64 + s) : String(s)));
    const env = {
      name: "chain", kind: "chain", title, n, nS, nA: 2, start: (n + 1) >> 1, names, valueRange: 1,
      exits: { left, right: 1 },
      acts: (s) => (s === 0 || s === n + 1 ? none : both),
      terminal: (s) => s === 0 || s === n + 1,
      describe: (s, a) => `state ${names[s]}${a >= 0 ? `, stepping ${a === LEFT ? "left" : "right"}` : ""}`,
      reset: (rng, s0) => (s0 === undefined ? env.start : s0),
      step(s, a) {
        const s2 = a === LEFT ? s - 1 : s + 1;
        return { s2, r: s2 === n + 1 ? 1 : s2 === 0 ? left : 0 };
      },
      model: (s, a) => [{ p: 1, ...env.step(s, a) }],
      // The true values of walking at random, for comparison (undiscounted).
      truth() {
        const V = new Float64Array(nS);
        for (let s = 1; s <= n; s++) V[s] = left + ((1 - left) * s) / (n + 1);
        return V;
      },
    };
    return env;
  };

  lab.worlds["random-walk"] = () => lab.chain();
  lab.worlds["random-walk-19"] = () => lab.chain({ n: 19, left: -1, title: "19-state random walk" });
})(globalThis.RL = globalThis.RL || {});
