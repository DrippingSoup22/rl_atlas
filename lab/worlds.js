/* The shared worlds: the set every algorithm can be tried on in the Lab, where its kind allows. Each world has a
   profile, the settings a run there starts from: how many units (episodes, rounds of workers, sweeps), the discount,
   what to chart, and what counts as ending well. A world that has no units for an algorithm's kind is not offered to
   it. Who may run where:
   - tabular methods need discrete states and actions; dynamic programming also needs the world's model;
   - methods that predict the values of a fixed policy run on the walks (and Blackjack, tables only);
   - value methods with approximation need discrete actions, and features for continuous states;
   - policy-gradient methods run on any world they have units for;
   - bandit methods run on bandits. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  const GROUPS = {
    tables: "Grids and tables",
    approx: "Continuous states",
    walks: "Walks to predict",
    bandits: "Bandits",
  };

  const lose = (n, what = "episode") => ({ metric: "return", min: -n, text: `end up losing less than ${n} per ${what} on average, while still exploring` });
  const WORLDS = [
    { id: "cliff", group: "tables", blurb: "A cliff along the bottom edge; each step costs 1",
      profile: { episode: 500, round: 60, sweep: 100, gamma: 1, charts: ["return"], domain: [-100, 0], success: lose(40), film: [1, 10, 100, 500] } },
    { id: "windy", group: "tables", blurb: "A wind pushes every move up; each step costs 1",
      profile: { episode: 170, round: 60, sweep: 100, gamma: 1, charts: ["steps"], success: { metric: "steps", max: 25, text: "end with episodes under 25 steps on average, while still exploring" }, film: [1, 10, 50, 170] } },
    { id: "dyna-maze", group: "tables", blurb: "Only the gem pays",
      profile: { episode: 50, round: 60, sweep: 100, gamma: 0.95, charts: ["steps"], success: { metric: "steps", max: 40, text: "end with episodes under 40 steps on average" }, film: [1, 2, 10, 50] } },
    { id: "frozen-lake", group: "tables", blurb: "Slippery ice and holes; only the gem pays",
      profile: { episode: 3000, round: 200, sweep: 100, gamma: 0.99, charts: ["greedy"], measures: ["greedy"], success: { metric: "greedy", min: 0.5, text: "end with a greedy policy that reaches the gem at least half the time" }, film: [1, 100, 1000, 3000] } },
    { id: "frozen-lake-8", group: "tables", blurb: "A larger lake with ten holes; gems are rare",
      profile: { episode: 10000, round: 400, sweep: 200, gamma: 0.99, charts: ["greedy"], measures: ["greedy"], success: { metric: "greedy", min: 0.5, text: "end with a greedy policy that reaches the gem at least half the time" }, film: [1, 1000, 5000, 10000] } },
    { id: "taxi", group: "tables", blurb: "Pick a passenger up, drop them off; 500 states",
      profile: { episode: 2000, round: 300, sweep: 100, gamma: 0.99, maxSteps: 200, charts: ["return"], success: lose(5, "trip"), film: [1, 100, 500, 2000] } },
    { id: "blackjack", group: "tables", blurb: "Hit or stick against the dealer",
      profile: { episode: 100000, gamma: 1, charts: ["return"], film: [1, 1000, 10000, 100000] } },
    { id: "mountain-car", group: "approx", blurb: "Rock an underpowered car out of a valley",
      profile: { episode: 500, gamma: 1, charts: ["steps"], success: { metric: "steps", max: 150, text: "end with episodes under 150 steps on average" }, film: [1, 10, 100, 500],
        params: { features: "tiles", tilings: 8, cells: 8 } } },
    { id: "random-walk", group: "walks", blurb: "Five states, a coin flip each step",
      profile: { episode: 100, runs: 100, gamma: 1, charts: ["error"], measures: ["error"], film: [1, 10, 100] } },
    { id: "random-walk-19", group: "walks", blurb: "Nineteen states; −1 on the left, +1 on the right",
      profile: { episode: 20, runs: 100, gamma: 1, charts: ["error"], measures: ["error"], film: [1, 5, 20] } },
    { id: "walk-1000", group: "walks", blurb: "A thousand states, jumps of up to a hundred",
      profile: { episode: 2000, gamma: 1, charts: ["ve"], measures: ["ve"], film: [1, 100, 2000] } },
    { id: "testbed", group: "bandits", blurb: "Ten arms, means drawn around 0",
      profile: { step: 1000, runs: 500, charts: ["return", "optimal"], success: { metric: "optimal", min: 0.5, text: "end up pulling the best arm most of the time" } } },
    { id: "drifting", group: "bandits", blurb: "Ten arms whose means wander",
      profile: { step: 10000, runs: 200, charts: ["return", "optimal"], success: { metric: "optimal", min: 0.5, text: "end up pulling the current best arm most of the time" } } },
  ];

  // The kind of each algorithm, for the rules above. Algorithms not listed stay in their own lab's world.
  const FAMILY = {};
  const kinds = {
    bandit: ["epsilon-greedy", "optimistic-init", "ucb", "gradient-bandit"],
    dp: ["policy-evaluation", "policy-iteration", "value-iteration"],
    predict: ["td0", "mc-prediction", "n-step-td", "td-lambda", "offline-lambda"],
    "predict-linear": ["gradient-mc", "semi-gradient-td"],
    tabular: ["sarsa", "q-learning", "expected-sarsa", "double-q", "mc-control", "exploring-starts", "off-policy-mc", "n-step-sarsa", "sarsa-lambda", "dyna-q", "dyna-q-plus", "prioritized-sweeping"],
    linear: ["semi-gradient-sarsa"],
    policy: ["reinforce", "baseline", "actor-critic", "a2c", "trpo", "ppo"],
  };
  for (const [kind, ids] of Object.entries(kinds)) for (const id of ids) FAMILY[id] = kind;
  const MODEL = new Set(["cliff", "windy", "dyna-maze", "frozen-lake", "frozen-lake-8", "taxi"]); // worlds whose rules are written out
  const PREDICT = new Set(["random-walk", "random-walk-19", "blackjack"]);

  function allows(algorithm, w) {
    const kind = FAMILY[algorithm.id], unit = algorithm.unit;
    if (!kind || w.profile[unit] === undefined) return false;
    switch (kind) {
      case "bandit": return w.group === "bandits";
      case "dp": return MODEL.has(w.id);
      case "predict": return PREDICT.has(w.id);
      case "predict-linear": return w.id === "walk-1000";
      case "tabular": return w.group === "tables";
      case "linear": return w.group === "approx";
      case "policy": return w.group === "tables" && w.id !== "blackjack";
      default: return false;
    }
  }

  // The shared worlds every one of these algorithms may run, grouped: [{ group, title, worlds: [{ id, title, blurb }] }].
  lab.worldsFor = function (algorithms) {
    const ok = WORLDS.filter((w) => algorithms.every((a) => allows(a, w)));
    return Object.entries(GROUPS).map(([group, title]) => ({ group, title, worlds: ok.filter((w) => w.group === group).map((w) => ({ id: w.id, title: lab.make(w.id).title, blurb: w.blurb })) })).filter((g) => g.worlds.length);
  };
  // The settings a run of these algorithms on world id starts from: { units, runs (how many runs the charts average;
  // none: the 20-seed bench), gamma, maxSteps, charts, measures, success,
  // film, domain, params }. Dynamic programming charts its sweeps' largest change; prediction charts its error.
  lab.worldProfile = function (id, algorithms) {
    const w = WORLDS.find((x) => x.id === id);
    if (!w) return null;
    const p = w.profile, unit = algorithms[0].unit, kinds = new Set(algorithms.map((a) => FAMILY[a.id]));
    const out = { units: p[unit], runs: p.runs, gamma: p.gamma, maxSteps: p.maxSteps, charts: p.charts, measures: p.measures || null, success: p.success || null, film: p.film, domain: p.domain || null, params: p.params || {} };
    if (kinds.has("dp")) Object.assign(out, { charts: ["delta"], measures: null, success: null, film: [1, 10, out.units] });
    if (kinds.has("predict") && w.group !== "walks") Object.assign(out, { charts: ["error"], measures: ["error"], success: null });
    if (out.film) out.film = out.film.filter((u) => u <= out.units);
    return out;
  };
  lab.worldBlurb = (id) => WORLDS.find((w) => w.id === id)?.blurb || "";
  lab.WORLD_GROUPS = GROUPS;
})(globalThis.RL = globalThis.RL || {});
