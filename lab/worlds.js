/* The shared worlds: the set every algorithm can be tried on in the Lab, where its kind allows. Each world has a
   profile, the settings a run there starts from: how many units (episodes, rounds of workers, sweeps), the discount,
   what to chart, and what counts as ending well. A world that has no units for an algorithm's kind is not offered to
   it. Who may run where:
   - tabular methods need discrete states and actions; dynamic programming also needs the world's model;
   - methods that predict the values of a fixed policy run on every world whose true values can be worked out, to
     judge them by: the walks, Blackjack, and the grids and Taxi, where the policy is the shortest way nine times in
     ten;
   - value methods with approximation need discrete actions, and features for continuous states (a table otherwise);
   - policy-gradient methods run on any world they have units for (the continuous states with tile features);
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
      profile: { episode: 500, round: 60, sweep: 100, gamma: 1, maxSteps: 1000, charts: ["return"], domain: [-100, 0], success: lose(40), film: [1, 10, 100, 500] } },
    { id: "windy", group: "tables", blurb: "A wind pushes every move up; each step costs 1",
      profile: { episode: 170, round: 60, sweep: 100, gamma: 1, maxSteps: 1000, charts: ["steps"], success: { metric: "steps", max: 25, text: "end with episodes under 25 steps on average, while still exploring" }, film: [1, 10, 50, 170] } },
    { id: "dyna-maze", group: "tables", blurb: "Only the gem pays",
      profile: { episode: 50, round: 60, sweep: 100, gamma: 0.95, maxSteps: 1000, charts: ["steps"], success: { metric: "steps", max: 40, text: "end with episodes under 40 steps on average" }, film: [1, 2, 10, 50] } },
    { id: "frozen-lake", group: "tables", blurb: "Slippery ice and holes; only the gem pays",
      profile: { episode: 3000, round: 200, sweep: 100, gamma: 0.99, maxSteps: 100, charts: ["greedy"], measures: ["greedy"], success: { metric: "greedy", min: 0.6, text: "end with a greedy policy that reaches the gem at least 60% of the time (82% is the best possible)" }, film: [1, 100, 1000, 3000],
        params: { judge: 1 } } },
    { id: "frozen-lake-8", group: "tables", blurb: "A larger lake with ten holes; gems are rare",
      profile: { episode: 10000, round: 400, sweep: 200, gamma: 0.99, maxSteps: 200, charts: ["greedy"], measures: ["greedy"], success: { metric: "greedy", min: 0.5, text: "end with a greedy policy that reaches the gem at least half the time (89% is the best possible)" }, film: [1, 1000, 5000, 10000],
        params: { judge: 1 } } },
    { id: "taxi", group: "tables", blurb: "Pick a passenger up, drop them off; 500 states",
      profile: { episode: 2000, round: 300, sweep: 100, gamma: 0.99, maxSteps: 200, charts: ["return"], success: lose(5, "trip"), film: [1, 100, 500, 2000] } },
    { id: "catch", group: "tables", blurb: "Move a paddle under a falling ball: +1 caught, −1 missed",
      profile: { episode: 1000, round: 200, sweep: 40, gamma: 1, charts: ["return"], domain: [-1, 1], film: [1, 10, 100, 1000],
        success: { metric: "return", min: 0.6, text: "end up catching the ball at least four times in five on average, while still exploring" } } },
    { id: "blackjack", group: "tables", blurb: "Hit or stick against the dealer",
      profile: { episode: 100000, round: 5000, gamma: 1, charts: ["return"], film: [1, 1000, 10000, 100000], params: { policy: "stick-20" } } },
    { id: "mountain-car", group: "approx", blurb: "Rock an underpowered car out of a valley",
      profile: { episode: 500, round: 100, gamma: 1, maxSteps: 2000, charts: ["steps"], success: { metric: "steps", max: 150, text: "end with episodes under 150 steps on average" }, film: [1, 10, 100, 500],
        params: { features: "tiles", tilings: 8, cells: 8, alpha: 0.06, alphaW: 0.05, epsilon: 0 } } },
    { id: "cartpole", group: "approx", blurb: "Push a cart left or right to keep a pole up, for up to 500 steps",
      profile: { episode: 500, round: 150, gamma: 0.99, maxSteps: 500, charts: ["steps"], film: [1, 10, 100, 500],
        success: { metric: "steps", min: 300, text: "end up keeping the pole up for more than 300 steps an episode on average" },
        params: { features: "tiles", tilings: 8, cells: 6, alpha: 0.06, alphaW: 0.05, epsilon: 0.05 } } },
    { id: "acrobot", group: "approx", blurb: "Swing two links, with a motor only at the joint, until the tip clears a line",
      profile: { episode: 300, round: 100, gamma: 1, maxSteps: 500, charts: ["steps"], film: [1, 10, 50, 300],
        success: { metric: "steps", max: 150, text: "end up getting the tip over the line in under 150 steps on average" },
        params: { features: "tiles", tilings: 8, cells: 6, alpha: 0.06, alphaW: 0.05, epsilon: 0 } } },
    { id: "random-walk", group: "walks", blurb: "Five states, a coin flip each step",
      profile: { episode: 100, runs: 100, gamma: 1, charts: ["error"], measures: ["error"], film: [1, 10, 100], params: { policy: "random" } } },
    { id: "random-walk-19", group: "walks", blurb: "Nineteen states; −1 on the left, +1 on the right",
      profile: { episode: 20, runs: 100, gamma: 1, charts: ["error"], measures: ["error"], film: [1, 5, 20], params: { policy: "random" } } },
    { id: "walk-1000", group: "walks", blurb: "A thousand states, jumps of up to a hundred",
      profile: { episode: 2000, gamma: 1, charts: ["ve"], measures: ["ve"], film: [1, 100, 2000], params: { policy: "random" } } },
    { id: "testbed", group: "bandits", blurb: "Ten arms, means drawn around 0",
      profile: { step: 1000, runs: 500, charts: ["return", "optimal"], success: { metric: "optimal", min: 0.5, text: "end up pulling the best arm most of the time" } } },
    { id: "drifting", group: "bandits", blurb: "Ten arms whose means wander",
      profile: { step: 10000, runs: 200, charts: ["return", "optimal"], success: { metric: "optimal", min: 0.5, text: "end up pulling the current best arm most of the time" } } },
  ];

  // Settings found to work in each world, by the world study (recorder/WORLD-STUDY.md): for a family (tabular, policy,
  // predict, dp, linear) or one algorithm. Step sizes are per weight: with tile coding, already divided by the tilings.
  // hard: algorithms that rarely ended well there in the study, with these settings (the World panel marks them).
  const MC = ["mc-control", "exploring-starts", "off-policy-mc"], PG = ["reinforce", "baseline"];
  const TUNED = {
    cliff: { tabular: { alpha: 0.1 }, policy: { alpha: 0.1, alphaW: 0.1 }, reinforce: { alpha: 0.001 }, baseline: { alpha: 0.003, alphaW: 0.1 },
      "mc-control": { epsilon: 0.2 }, "off-policy-mc": { epsilon: 0.1 }, predict: { alpha: 0.1 }, td0: { alpha: 0.3 }, dp: { gamma: 0.9 },
      // Q-learning and the Dyna family learn the edge path and keep falling while they explore: the cliff's own lesson
      hard: [...PG, ...MC, "q-learning", "dyna-q", "dyna-q-plus", "prioritized-sweeping"] },
    windy: { tabular: { alpha: 0.5 }, "n-step-sarsa": { alpha: 0.25 }, "sarsa-lambda": { alpha: 0.25 }, policy: { alpha: 0.3, alphaW: 0.5 }, trpo: { alphaW: 0.1 },
      reinforce: { alpha: 0.001 }, baseline: { alpha: 0.003, alphaW: 0.1 }, "mc-control": { epsilon: 0.2 }, "off-policy-mc": { epsilon: 0.1 }, predict: { alpha: 0.1 },
      dp: { gamma: 0.9 }, hard: [...PG, ...MC] },
    "dyna-maze": { tabular: { alpha: 0.1 }, policy: { alpha: 0.3, alphaW: 0.5 }, a2c: { alpha: 3, alphaW: 0.5 }, reinforce: { alpha: 0.03 }, baseline: { alpha: 0.03, alphaW: 0.1 },
      "off-policy-mc": { epsilon: 0.1 }, predict: { alpha: 0.1 }, hard: [...PG, "actor-critic", "double-q", "exploring-starts"] },
    "frozen-lake": { tabular: { alpha: 0.1 }, "n-step-sarsa": { alpha: 0.05 }, "sarsa-lambda": { alpha: 0.05 }, policy: { alpha: 1, alphaW: 0.1 }, trpo: { alphaW: 0.1 },
      reinforce: { alpha: 0.1 }, baseline: { alpha: 0.1, alphaW: 0.5 }, predict: { alpha: 0.03 }, hard: [...PG, "a2c", "ppo", ...MC, "sarsa-lambda", "double-q", "dyna-q", "dyna-q-plus", "prioritized-sweeping"] },
    "frozen-lake-8": { tabular: { alpha: 0.1 }, "n-step-sarsa": { alpha: 0.05 }, "sarsa-lambda": { alpha: 0.05 }, policy: { alpha: 1, alphaW: 0.1 }, reinforce: { alpha: 0.3 },
      baseline: { alpha: 0.3, alphaW: 0.1 }, predict: { alpha: 0.03 },
      hard: ["baseline", "a2c", "ppo", ...MC, "sarsa-lambda", "dyna-q-plus", "prioritized-sweeping"] },
    taxi: { tabular: { alpha: 0.25 }, "double-q": { alpha: 0.5 }, "sarsa-lambda": { alpha: 0.1 }, "n-step-sarsa": { alpha: 0.05 }, policy: { alpha: 0.1, alphaW: 0.5 },
      reinforce: { alpha: 0.01 }, baseline: { alpha: 0.01, alphaW: 0.1 }, predict: { alpha: 0.1 }, dp: { gamma: 0.9 }, hard: [...PG, "a2c", "ppo", "trpo", ...MC, "n-step-sarsa"] },
    catch: { tabular: { alpha: 0.25 }, policy: { alpha: 1, alphaW: 0.5 }, ppo: { alpha: 0.3, alphaW: 0.1 }, reinforce: { alpha: 0.1 }, baseline: { alpha: 0.1, alphaW: 0.1 },
      "mc-control": { epsilon: 0.05 }, "off-policy-mc": { epsilon: 0.1 }, predict: { alpha: 0.1 }, hard: PG },
    blackjack: { tabular: { alpha: 0.1 }, "mc-control": { epsilon: 0.05 }, "off-policy-mc": { epsilon: 0.1 }, policy: { alpha: 0.1, alphaW: 0.1 }, a2c: { alpha: 0.3, alphaW: 0.5 },
      predict: { alpha: 0.03 } },
    "mountain-car": { policy: { alpha: 0.01, alphaW: 0.06, maxSteps: 1000 }, hard: [...PG, "ppo"] },
    cartpole: { policy: { alpha: 0.03, alphaW: 0.06 }, baseline: { alpha: 0.01, alphaW: 0.06 }, reinforce: { alpha: 0.01 }, ppo: { alpha: 0.03, alphaW: 0.05 }, hard: ["reinforce"] },
    acrobot: { policy: { alpha: 0.01, alphaW: 0.02 }, hard: PG },
    "random-walk": { predict: { alpha: 0.1 }, "predict-linear": { alpha: 0.02 } },
    "random-walk-19": { predict: { alpha: 0.1 }, "predict-linear": { alpha: 0.0005, units: 200 } }, // every-visit updates, about 90 steps an episode
    "walk-1000": { predict: { alpha: 0.1 } },
    drifting: { hard: ["ucb", "gradient-bandit"] }, // sample averages and long memories lose track of drifting arms
  };

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
  const MODEL = new Set(["cliff", "windy", "dyna-maze", "frozen-lake", "frozen-lake-8", "taxi", "catch"]); // worlds whose rules are written out
  const PREDICT = new Set(["random-walk", "random-walk-19", "walk-1000", "blackjack", ...MODEL]);

  function allows(algorithm, w) {
    const kind = FAMILY[algorithm.id], unit = algorithm.unit;
    if (!kind || w.profile[unit] === undefined) return false;
    switch (kind) {
      case "bandit": return w.group === "bandits";
      case "dp": return MODEL.has(w.id);
      case "predict": return PREDICT.has(w.id);
      case "predict-linear": return PREDICT.has(w.id);
      case "tabular": return w.group === "tables";
      // with discrete states, features are a table (one weight per state and action): the tabular method again
      case "linear": return w.group === "tables" || w.group === "approx";
      // TRPO's natural gradient is worked out exactly for one preference per state and action: tables only
      case "policy": return w.group === "tables" || (w.group === "approx" && algorithm.id !== "trpo");
      default: return false;
    }
  }

  // The shared worlds every one of these algorithms may run, grouped: [{ group, title, worlds: [{ id, title, blurb, hard }] }],
  // hard listing those of the algorithms that rarely ended well there in the world study.
  lab.worldsFor = function (algorithms) {
    const ok = WORLDS.filter((w) => algorithms.every((a) => allows(a, w)));
    const hard = (w) => [...new Set(algorithms.filter((a) => TUNED[w.id]?.hard?.includes(a.id)).map((a) => a.title))];
    return Object.entries(GROUPS).map(([group, title]) => ({ group, title, worlds: ok.filter((w) => w.group === group).map((w) => ({ id: w.id, title: lab.make(w.id).title, blurb: w.blurb, hard: hard(w) })) })).filter((g) => g.worlds.length);
  };
  // The settings a run of these algorithms on world id starts from: { units, runs (how many runs the charts average;
  // none: the 20-seed bench), gamma, maxSteps, charts, measures, success,
  // film, domain, params }. Dynamic programming charts its sweeps' largest change; prediction charts its error.
  lab.worldProfile = function (id, algorithms) {
    const w = WORLDS.find((x) => x.id === id);
    if (!w) return null;
    const p = w.profile, unit = algorithms[0].unit, kinds = new Set(algorithms.map((a) => FAMILY[a.id]));
    // the settings found to work here (TUNED): the lead racer's family's, then its own algorithm's
    const lead = algorithms[0], t = TUNED[id] || {}, fam = FAMILY[lead.id];
    // a linear method learning a table takes the tabular settings
    const famT = t[fam] || (w.group !== "approx" && t[{ linear: "tabular", "predict-linear": "predict" }[fam]]) || {};
    // a linear method learning a table explores as the tabular methods do (its home lab counts on optimistic starting
    // values, which a world of zero rewards does not give)
    const explore = fam === "linear" && w.group === "tables" ? { epsilon: 0.1 } : {};
    const { gamma: tg, units: tu, maxSteps: tm, ...tuned } = { ...explore, ...famT, ...t[lead.id] };
    const out = { units: p[unit], runs: p.runs, gamma: p.gamma, maxSteps: p.maxSteps, charts: p.charts, measures: p.measures || null, success: p.success || null, film: p.film, domain: p.domain || null, params: p.params || {} };
    if (kinds.has("dp")) Object.assign(out, { charts: ["delta"], measures: null, success: null, film: [1, 10, out.units] });
    // policy evaluation evaluates the same policy prediction does (iteration starts from random play, as in the book)
    if (algorithms.some((a) => a.id === "policy-evaluation")) out.params = { ...out.params, policy: "mostly-best" };
    if ((kinds.has("predict") || kinds.has("predict-linear")) && w.group !== "walks") {
      Object.assign(out, { charts: ["error"], measures: ["error"], success: null });
      // the shortest way, nine times in ten; episodes start anywhere, so that every state's value gets learned
      if (MODEL.has(id)) out.params = { ...out.params, policy: "mostly-best", anyStart: 1 };
    }
    out.params = { ...out.params, ...tuned };
    if (tg !== undefined) out.gamma = tg;
    if (tu !== undefined) out.units = tu;
    if (tm !== undefined) out.maxSteps = tm;
    if (out.film) out.film = out.film.filter((u) => u <= out.units);
    // a world without features of its own is learned with a table (a lab's own features, like Baird's, stay home);
    // on a walk, a lab's features (its groups, say) work as well as at home, and stay
    if (w.group !== "approx" && w.group !== "walks" && !out.params.features) out.params = { features: "table", ...out.params };
    return out;
  };

  // A racer's own settings, carried to another world. Settings tied to its world go: features the new world can't
  // compute (its own features, or coordinates it lacks), and a behavior policy named by the old world. Its step sizes
  // keep their ratio to the lab's reference (the lab's setting, or else the racers' middle one), applied to the step
  // size the knobs hold now: a racer with double the step size still has double, in the new world's units.
  const SCALED = ["alpha", "alphaW"];
  lab.carryRacer = function (own = {}, { env, base = {}, racers = [], knobs = {} }) {
    const out = { ...own };
    delete out.behavior;
    if (out.features) {
      const f = out.features, ok = f === "own" ? !!env.features : f === "table" ? !!env.nS && !env.continuous : !!env.coords;
      if (!ok) for (const k of ["features", "cells", "tilings", "order"]) delete out[k];
    }
    for (const k of SCALED) {
      if (!(k in out) || out[k] === 0 || knobs[k] === undefined) continue; // 0: a 1/n step size, kept as it is
      const vals = racers.map((r) => r?.[k]).filter((v) => v > 0).sort((a, b) => a - b);
      const ref = base[k] > 0 ? base[k] : vals[(vals.length - 1) >> 1];
      out[k] = +(knobs[k] * (out[k] / ref)).toPrecision(3);
    }
    return out;
  };
  lab.worldBlurb = (id) => WORLDS.find((w) => w.id === id)?.blurb || "";
  lab.WORLD_GROUPS = GROUPS;
})(globalThis.RL = globalThis.RL || {});
