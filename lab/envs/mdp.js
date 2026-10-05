/* Small MDPs written as tables of transitions, like the recycling robot of Sutton & Barto (Example 3.3).
   A row is: state, action, next state, probability, reward; a reward [mean, sd] is drawn from a normal distribution. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  const B_ACTIONS = 10; // the many actions of state B in the maximization-bias example
  const MDPS = {
    // Our robot uses numbers of its own: searching finds 2 cans on average, waiting finds 1.
    robot: {
      title: "Recycling robot",
      states: ["high", "low"],
      actions: ["search", "wait", "recharge"],
      dynamics: [
        ["high", "search", "high", 0.7, 2],
        ["high", "search", "low", 0.3, 2],
        ["high", "wait", "high", 1, 1],
        ["low", "search", "low", 0.6, 2],
        ["low", "search", "high", 0.4, -3], // the battery ran flat: rescued and recharged
        ["low", "wait", "low", 1, 1],
        ["low", "recharge", "high", 1, 0],
      ],
    },
    // Sutton & Barto, Example 6.7. From A, right ends the episode at once; left leads to B, whose many actions all end
    // it with a reward drawn from N(−0.1, 1). Going left is worse on average, but some of B's estimates will look good.
    "max-bias": {
      title: "Maximization bias",
      states: ["A", "B", "end"],
      actions: ["left", "right", ...Array.from({ length: B_ACTIONS }, (_, i) => `b${i + 1}`)],
      start: "A",
      terminals: ["end"],
      dynamics: [
        ["A", "left", "B", 1, 0],
        ["A", "right", "end", 1, 0],
        ...Array.from({ length: B_ACTIONS }, (_, i) => ["B", `b${i + 1}`, "end", 1, [-0.1, 1]]),
      ],
      // What matters here is how often the agent goes left from A (state 0, action 0).
      track(ev, stats) { if (ev.s === 0 && stats.left === undefined) stats.left = ev.a === 0 ? 1 : 0; },
    },
  };

  lab.mdp = function (name) {
    const m = MDPS[name];
    if (!m) throw new Error(`unknown MDP '${name}'`);
    const nS = m.states.length, nA = m.actions.length;
    const S = (x) => m.states.indexOf(x), A = (x) => m.actions.indexOf(x);
    const outcomes = Array.from({ length: nS * nA }, () => []);
    for (const [s, a, s2, p, r] of m.dynamics) outcomes[S(s) * nA + A(a)].push({ p, s2: S(s2), r: Array.isArray(r) ? r[0] : r, sd: Array.isArray(r) ? r[1] : 0 });
    const terminals = new Set((m.terminals || []).map(S));
    const acts = lab.actionLists(nS, nA, (s) => m.actions.map((_, a) => a).filter((a) => outcomes[s * nA + a].length));
    const start = m.start ? S(m.start) : 0;
    return {
      name, kind: "graph", title: m.title, states: m.states, actions: m.actions, dynamics: m.dynamics, nS, nA, start, valueRange: 1,
      terminal: (s) => terminals.has(s),
      acts: (s) => acts[s],
      available: (s) => acts[s],
      describe: (s, a) => `state ${m.states[s]}${a >= 0 ? `, action ${m.actions[a]}` : ""}`,
      reset: (rng, s0) => (s0 === undefined ? start : s0),
      // The expected rewards, for dynamic programming.
      model: (s, a) => outcomes[s * nA + a].map(({ p, s2, r }) => ({ p, s2, r })),
      step(s, a, rng) {
        const outs = outcomes[s * nA + a];
        let o = outs[0];
        if (outs.length > 1) {
          let u = rng.next();
          o = outs.find((x) => (u -= x.p) < 0) || outs[outs.length - 1];
        }
        return { s2: o.s2, r: o.sd ? o.r + o.sd * rng.normal() : o.r };
      },
      track: m.track,
    };
  };

  lab.worlds["max-bias"] = () => lab.mdp("max-bias");
})(globalThis.RL = globalThis.RL || {});
