/* Lab core: seeded randomness and the helpers every algorithm shares. No DOM: also runs in Node for the tests.
   Every world lists the actions it allows in each state (env.acts(s)); the helpers only ever look at those. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  // mulberry32: a tiny seeded generator whose whole state is one integer,
  // so any moment of a run can be saved and replayed exactly.
  lab.rng = function (seed = 1) {
    let s = seed >>> 0;
    const next = () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return {
      next,
      int: (n) => Math.floor(next() * n),
      // A standard normal sample (Box–Muller). It keeps no spare value, so the state stays one integer.
      normal() {
        const u = 1 - next(), v = next();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      },
      get state() { return s; },
      set state(v) { s = v >>> 0; },
    };
  };

  // Worlds register a factory under their name; lab.make(name) builds a fresh one.
  lab.worlds = lab.worlds || {};
  lab.make = function (name) {
    const make = lab.worlds[name];
    if (!make) throw new Error(`unknown world '${name}'`);
    return make();
  };

  // The available actions of every state, computed once: all of them unless the world says otherwise.
  lab.actionLists = function (nS, nA, allowed = null) {
    const all = Array.from({ length: nA }, (_, a) => a);
    return Array.from({ length: nS }, (_, s) => (allowed ? allowed(s) : all));
  };

  lab.maxQ = function (Q, s, env) {
    let m = -Infinity;
    for (const a of env.acts(s)) m = Math.max(m, Q[s * env.nA + a]);
    return m;
  };

  // A best action in s. Ties are broken at random when an rng is given, else the first one wins.
  lab.greedy = function (Q, s, env, rng) {
    const m = lab.maxQ(Q, s, env), best = [];
    for (const a of env.acts(s)) if (Q[s * env.nA + a] === m) best.push(a);
    return best.length > 1 && rng ? best[rng.int(best.length)] : best[0];
  };

  // ε-greedy: with probability ε any available action at random, otherwise a best one.
  lab.epsGreedy = function (Q, s, env, eps, rng) {
    const acts = env.acts(s);
    return rng.next() < eps ? acts[rng.int(acts.length)] : lab.greedy(Q, s, env, rng);
  };

  // The probability ε-greedy gives each action in s (tied best actions share the greedy part).
  lab.epsGreedyProbs = function (Q, s, env, eps, out = new Array(env.nA)) {
    const acts = env.acts(s), m = lab.maxQ(Q, s, env);
    let ties = 0;
    for (const a of acts) if (Q[s * env.nA + a] === m) ties++;
    out.fill(0);
    for (const a of acts) out[a] = eps / acts.length + (Q[s * env.nA + a] === m ? (1 - eps) / ties : 0);
    return out;
  };

  // Softmax of n preferences starting at H[from]; shifted by the largest one so it never overflows.
  lab.softmax = function (H, from = 0, n = H.length, out = new Float64Array(n)) {
    let top = -Infinity, sum = 0;
    for (let i = 0; i < n; i++) top = Math.max(top, H[from + i]);
    for (let i = 0; i < n; i++) sum += out[i] = Math.exp(H[from + i] - top);
    for (let i = 0; i < n; i++) out[i] /= sum;
    return out;
  };

  // Draw an index with the given probabilities.
  lab.pick = function (probs, rng) {
    let u = rng.next();
    for (let i = 0; i < probs.length; i++) { u -= probs[i]; if (u < 0) return i; }
    for (let i = probs.length - 1; i > 0; i--) if (probs[i] > 0) return i;
    return 0;
  };

  lab.argmax = function (values) {
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
  };

  // Root-mean-square difference between estimates and true values over the states that count (non-terminal).
  lab.rms = function (V, truth, env) {
    let sum = 0, n = 0;
    for (let s = 0; s < env.nS; s++) {
      if (env.terminal(s) || env.blocked?.(s)) continue;
      sum += (V[s] - truth[s]) ** 2;
      n++;
    }
    return Math.sqrt(sum / Math.max(1, n));
  };

  lab.mean = (values, from = 0, to = values.length) => {
    let sum = 0;
    for (let i = from; i < to; i++) sum += values[i];
    return sum / Math.max(1, to - from);
  };
})(globalThis.RL = globalThis.RL || {});
