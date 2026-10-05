// The lab must reproduce what the guide teaches, and replays must show exactly what happened.
// Run from rl_atlas/ with: node --test   (after python build.py, which writes the presets the last test runs)
const test = require("node:test");
const assert = require("node:assert/strict");

const FILES = ["core", "envs/grid", "envs/bandit", "envs/chain", "envs/blackjack", "envs/mdp", "envs/approx", "dp", "run", "measures", "features",
  "agents/td", "agents/mc", "agents/dp", "agents/bandit", "agents/traces", "agents/planning", "agents/linear"];
for (const file of FILES) require(`../lab/${file}.js`);
const { lab } = globalThis.RL;
const A = lab.algorithms;
const seeds = (n) => Array.from({ length: n }, (_, i) => i + 1);
const run = (world, id, params, units, seed = 1, measures) => lab.simulate({ world, algorithm: A[id], params, units, seed, measures });
const tenths = (V) => Array.from(V, (v) => Math.round(v * 10) / 10 + 0);

test("gridworld: the random policy's values and the optimal values are those of Sutton & Barto, Figures 3.2 and 3.5", () => {
  const g = lab.make("gridworld");
  assert.deepEqual(tenths(lab.evaluate(g, lab.randomPolicy(g), 0.9)), [
    3.3, 8.8, 4.4, 5.3, 1.5, 1.5, 3.0, 2.3, 1.9, 0.5, 0.1, 0.7, 0.7, 0.4, -0.4,
    -1.0, -0.4, -0.4, -0.6, -1.2, -1.9, -1.3, -1.2, -1.4, -2.0,
  ]);
  assert.deepEqual(tenths(lab.valueIteration(g, 0.9)), [
    22.0, 24.4, 22.0, 19.4, 17.5, 19.8, 22.0, 19.8, 17.8, 16.0, 17.8, 19.8, 17.8, 16.0, 14.4,
    16.0, 17.8, 16.0, 14.4, 13.0, 14.4, 16.0, 14.4, 13.0, 11.7,
  ]);
});

test("4 × 4 gridworld: sweeps of policy evaluation give the values of Sutton & Barto, Figure 4.1", () => {
  const r = run("small-gridworld", "policy-evaluation", { gamma: 1, theta: 1e-6, sync: true }, 400);
  assert.deepEqual(tenths(r.at(3).V), [0, -2.4, -2.9, -3, -2.4, -2.9, -3, -2.9, -2.9, -3, -2.9, -2.4, -3, -2.9, -2.4, 0]);
  assert.deepEqual(tenths(r.at(10).V), [0, -6.1, -8.4, -9, -6.1, -7.7, -8.4, -8.4, -8.4, -8.4, -7.7, -6.1, -9, -8.4, -6.1, 0]);
  assert.deepEqual(Array.from(r.at(400).V, Math.round), [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0]);
});

test("Blackjack: the exact values match Sutton & Barto's Example 5.4 and the optimal policy of Figure 5.2", () => {
  const bj = lab.make("blackjack");
  const v = lab.blackjackExact(bj.policy("stick-20")).V[bj.encode(13, 2, true)];
  assert.ok(Math.abs(v - -0.27726) < 2e-4, `v(13, 2, usable ace) = ${v}`);
  const { policy } = lab.blackjackExact(), sticks = (sum, d, usable) => policy[bj.encode(sum, d, usable) * 2] > 0.5;
  const threshold = (d, usable) => { let sum = 12; while (sum < 21 && !sticks(sum, d, usable)) sum++; return sum; };
  // the smallest sum to stick on, against a dealer showing ace, 2, …, 10
  assert.deepEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((d) => threshold(d, false)), [17, 13, 13, 12, 12, 12, 17, 17, 17, 17]);
  assert.deepEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((d) => threshold(d, true)), [19, 18, 18, 18, 18, 18, 18, 18, 19, 19]);
});

test("Monte Carlo: averaging returns closes in on the true values (Blackjack, the random walk)", () => {
  const bj = run("blackjack", "mc-prediction", { gamma: 1, policy: "stick-20" }, 100000, 5, ["error"]);
  assert.ok(bj.metrics.error[99999] < 0.08, `Blackjack RMS error ${bj.metrics.error[99999]}`);
  const rw = run("random-walk", "mc-prediction", { gamma: 1 }, 3000, 3);
  assert.ok(lab.rms(rw.at(3000).V, lab.make("random-walk").truth(), lab.make("random-walk")) < 0.03);
});

test("random walk: TD(0) learns the values 1/6 … 5/6, faster than Monte Carlo (Sutton & Barto, Figure 6.2)", () => {
  const env = lab.make("random-walk"), truth = env.truth();
  assert.ok(lab.rms(run("random-walk", "td0", { alpha: 0.02, gamma: 1, v0: 0.5 }, 3000, 3).at(3000).V, truth, env) < 0.03);
  const avg = (id, alpha) => lab.mean(lab.average({ world: "random-walk", algorithm: A[id], params: { alpha, gamma: 1, v0: 0.5 }, units: 100, seeds: seeds(100), measures: ["error"] }).mean.error);
  assert.ok(avg("td0", 0.1) < avg("mc-prediction", 0.02));
});

test("bandits: exploring beats greed, and UCB beats ε-greedy (Sutton & Barto, Figures 2.2 and 2.4)", () => {
  const avg = (id, params, key) => lab.mean(lab.average({ world: "testbed", algorithm: A[id], params, units: 1000, seeds: seeds(300) }).mean[key], 500);
  assert.ok(avg("epsilon-greedy", { epsilon: 0.1 }, "optimal") > avg("epsilon-greedy", { epsilon: 0 }, "optimal") + 0.25);
  assert.ok(avg("ucb", { c: 2 }, "return") > avg("epsilon-greedy", { epsilon: 0.1 }, "return"));
});

test("maximization bias: Double Q-learning goes left from A far less often than Q-learning (Figure 6.5)", () => {
  const left = (id) => lab.mean(lab.average({ world: "max-bias", algorithm: A[id], params: { alpha: 0.1, epsilon: 0.1, gamma: 1 }, units: 100, seeds: seeds(500) }).mean.left);
  assert.ok(left("q-learning") > 2 * left("double-q"));
});

const cliff = { alpha: 0.5, epsilon: 0.1, gamma: 1 };
const greedySteps = (r) => {
  const { path, reached } = lab.greedyPath(r.env, r.at(r.units).Q);
  return reached ? path.length - 1 : Infinity;
};

test("cliff walking: Q-learning learns the 13-step path along the edge, SARSA a longer, safer one", () => {
  for (const seed of [1, 2, 3]) {
    assert.equal(greedySteps(run("cliff", "q-learning", cliff, 500, seed)), 13);
    const sarsa = greedySteps(run("cliff", "sarsa", cliff, 500, seed));
    assert.ok(sarsa > 13 && sarsa < Infinity, `SARSA path: ${sarsa} steps`);
  }
});

test("cliff walking: while learning, SARSA earns more reward per episode than Q-learning", () => {
  for (const seed of [1, 2, 3]) {
    const last100 = (id) => lab.mean(run("cliff", id, cliff, 500, seed).metrics.return, 400);
    assert.ok(last100("sarsa") > last100("q-learning"));
  }
});

// The 19-state random walk: the RMS error over the first 10 episodes, averaged over 100 runs, at the best of a few step sizes.
const walkError = (id, params) => Math.min(...[0.1, 0.2, 0.4, 0.6, 0.8].map((alpha) =>
  lab.mean(lab.average({ world: "random-walk-19", algorithm: A[id], params: { gamma: 1, alpha, ...params }, units: 10, seeds: seeds(100), measures: ["error"] }).mean.error)));

test("n-step TD: an intermediate n learns fastest on the 19-state random walk (Sutton & Barto, Figure 7.2)", () => {
  const [one, four, many] = [1, 4, 64].map((n) => walkError("n-step-td", { n }));
  assert.ok(four < one - 0.05 && four < many - 0.05, `best errors: n = 1 ${one}, n = 4 ${four}, n = 64 ${many}`);
  assert.ok(four < 0.28, `n = 4: ${four}`);
});

test("TD(λ) and the λ-return: an intermediate λ is best, and TD(λ) with large α and λ blows up (Figures 12.3 and 12.6)", () => {
  const offline = (lambda) => walkError("offline-lambda", { lambda }), online = (lambda) => walkError("td-lambda", { lambda });
  assert.ok(offline(0.8) < offline(0) - 0.05 && offline(0.8) < offline(1) - 0.05);
  assert.ok(online(0.8) < online(0) - 0.05);
  assert.ok(Math.abs(online(0) - walkError("td0", {})) < 1e-9, "TD(0) is TD(λ) with λ = 0");
  const wild = run("random-walk-19", "td-lambda", { gamma: 1, alpha: 0.8, lambda: 0.95 }, 10, 1, ["error"]);
  assert.ok(!(wild.metrics.error[9] < 10), `error after 10 episodes: ${wild.metrics.error[9]}`);
});

test("Dyna-Q: planning steps cut the episodes needed to find the way (Sutton & Barto, Figure 8.2)", () => {
  const steps = (n) => lab.average({ world: "dyna-maze", algorithm: A["dyna-q"], params: { alpha: 0.1, epsilon: 0.1, gamma: 0.95, planning: n }, units: 10, seeds: seeds(30) }).mean.steps;
  const none = steps(0), fifty = steps(50);
  assert.ok(fifty[2] < 25, `n = 50, third episode: ${fifty[2]} steps`);
  assert.ok(none[2] > 200, `n = 0, third episode: ${none[2]} steps`);
  const ps = lab.mean(lab.average({ world: "dyna-maze", algorithm: A["prioritized-sweeping"], params: { alpha: 0.5, epsilon: 0.1, gamma: 0.95, planning: 5 }, units: 10, seeds: seeds(30) }).mean.steps, 1);
  const dyna = lab.mean(lab.average({ world: "dyna-maze", algorithm: A["dyna-q"], params: { alpha: 0.5, epsilon: 0.1, gamma: 0.95, planning: 5 }, units: 10, seeds: seeds(30) }).mean.steps, 1);
  assert.ok(ps < dyna, `episodes 2-10: prioritized sweeping ${ps} steps, Dyna-Q ${dyna}`);
});

test("Dyna-Q+ finds the shortcut that Dyna-Q never takes (Figure 8.5)", () => {
  const params = { alpha: 1, epsilon: 0.1, gamma: 0.95, planning: 50, kappa: 1e-3 };
  for (const seed of [1, 2, 3]) {
    const last = (id) => lab.mean(run("shortcut-maze", id, params, 450, seed).metrics.steps, 420);
    assert.ok(last("dyna-q-plus") < 14, `Dyna-Q+: ${last("dyna-q-plus")} steps`);
    assert.ok(last("dyna-q") > 15, `Dyna-Q: ${last("dyna-q")} steps`);
  }
});

// The 1000-state walk (Sutton & Barto, Example 9.1): gradient Monte Carlo from seed s, √VE after each episode.
const longWalk = (id, features, params, units, seed) =>
  lab.simulate({ world: "walk-1000", algorithm: A[id], params: { gamma: 1, ...features, ...params }, units, seed, measures: ["ve"] });
// The weights averaged over the second half of a run, every 100 episodes: what they hover around once settled.
const settled = (r) => {
  const w = new Float64Array(r.at(0).w.length), marks = [];
  for (let t = r.units / 2; t <= r.units; t += 100) marks.push(t);
  for (const t of marks) r.at(t).w.forEach((v, i) => { w[i] += v / marks.length; });
  return w;
};

test("1000-state walk: the true values, and gradient Monte Carlo fits each group's average (Sutton & Barto, Figure 9.1)", () => {
  const env = lab.make("walk-1000"), V = env.truth(), mu = env.mu();
  assert.ok(Math.abs(V[1] + 0.922) < 0.002 && Math.abs(V[1000] - 0.922) < 0.002, `v(1) = ${V[1]}, v(1000) = ${V[1000]}`);
  for (let s = 1; s <= 1000; s++) assert.ok(Math.abs(V[s] + V[1001 - s]) < 1e-9, "the values are symmetric");
  const w = settled(longWalk("gradient-mc", { features: "groups", cells: 10 }, { alpha: 5e-4 }, 10000, 1));
  for (let g = 0; g < 10; g++) {
    let num = 0, den = 0;
    for (let s = 100 * g + 1; s <= 100 * g + 100; s++) { num += mu[s] * V[s]; den += mu[s]; }
    assert.ok(Math.abs(w[g] - num / den) < 0.04, `group ${g + 1}: weight ${w[g]}, μ-weighted average of its true values ${num / den}`);
  }
});

test("1000-state walk: semi-gradient TD settles nearer the middle than Monte Carlo (Figure 9.2), and better features learn faster (Figures 9.5 and 9.10)", () => {
  const ends = (id, alpha) => { const w = settled(longWalk(id, { features: "groups", cells: 10 }, { alpha }, 10000, 2)); return [w[0], w[9]]; };
  const [mcLeft, mcRight] = ends("gradient-mc", 5e-4), [tdLeft, tdRight] = ends("semi-gradient-td", 2e-3);
  assert.ok(tdLeft > mcLeft + 0.05 && tdRight < mcRight - 0.05, `end groups: MC ${mcLeft}, ${mcRight}; TD ${tdLeft}, ${tdRight}`);
  const curve = (features, alpha) => lab.mean([1, 2, 3].map((seed) => lab.mean(longWalk("gradient-mc", features, { alpha }, 2000, seed).metrics.ve)));
  const fourier = curve({ features: "fourier", order: 5 }, 5e-5), poly = curve({ features: "poly", order: 5 }, 1e-4);
  assert.ok(fourier < poly, `average √VE: Fourier ${fourier}, polynomials ${poly}`);
  const tiles = curve({ features: "tiles", tilings: 50, cells: 5 }, 1e-4 / 50), one = curve({ features: "groups", cells: 5 }, 1e-4);
  assert.ok(tiles < one, `average √VE: 50 tilings ${tiles}, one tiling ${one}`);
});

test("Mountain Car: semi-gradient SARSA with tile coding learns to get up the hill (Sutton & Barto, Figure 10.2)", () => {
  for (const seed of [1, 2, 3]) {
    const r = lab.simulate({ world: "mountain-car", algorithm: A["semi-gradient-sarsa"], params: { gamma: 1, epsilon: 0, alpha: 0.5 / 8, features: "tiles", tilings: 8, cells: 8 }, units: 300, seed });
    const first = r.metrics.steps[0], last = lab.mean(r.metrics.steps, 250);
    assert.ok(first > 400 && last < 200, `seed ${seed}: first episode ${first} steps, last 50 ${last}`);
  }
});

test("Baird's counterexample: off-policy semi-gradient TD diverges; without approximation or off-policy it does not (Figure 11.2)", () => {
  const size = (params) => lab.simulate({ world: "baird", algorithm: A["semi-gradient-td"], params: { gamma: 0.99, alpha: 0.01, policy: "target", ...params }, units: 1000, seed: 1 }).metrics.weights[999];
  assert.ok(size({ features: "own", behavior: "behavior" }) > 100);
  assert.ok(size({ features: "table", behavior: "behavior" }) < 30);
  assert.ok(size({ features: "own" }) < 11);
});

test("replaying a unit reproduces the run exactly, for every algorithm", () => {
  const cases = [
    ["cliff", "sarsa", cliff, 60], ["cliff", "q-learning", cliff, 60], ["cliff", "expected-sarsa", cliff, 60],
    ["max-bias", "double-q", { alpha: 0.1, epsilon: 0.1, gamma: 1 }, 100], ["random-walk", "td0", { alpha: 0.1, gamma: 1 }, 50],
    ["random-walk", "mc-prediction", { alpha: 0.1, gamma: 1 }, 50], ["blackjack", "exploring-starts", { gamma: 1 }, 30000],
    ["blackjack", "off-policy-mc", { gamma: 1, epsilon: 1 }, 30000], ["frozen-lake", "mc-control", { gamma: 0.99, epsilon: 0.1 }, 300],
    ["drifting", "epsilon-greedy", { epsilon: 0.1, alpha: 0.1 }, 3000], ["testbed", "ucb", { c: 2 }, 500],
    ["testbed-4", "gradient-bandit", { alpha: 0.1 }, 500], ["small-gridworld", "policy-iteration", { gamma: 1, theta: 1e-3 }, 120],
    ["gridworld", "value-iteration", { gamma: 0.9, theta: 1e-3 }, 30],
    ["random-walk-19", "n-step-td", { alpha: 0.4, gamma: 1, n: 4 }, 30], ["cliff", "n-step-sarsa", { ...cliff, n: 3 }, 40],
    ["random-walk-19", "td-lambda", { alpha: 0.4, gamma: 1, lambda: 0.8 }, 30], ["cliff", "sarsa-lambda", { ...cliff, lambda: 0.5, trace: "replacing" }, 40],
    ["dyna-maze", "dyna-q", { alpha: 0.1, epsilon: 0.1, gamma: 0.95, planning: 5 }, 20],
    ["shortcut-maze", "dyna-q-plus", { alpha: 1, epsilon: 0.1, gamma: 0.95, planning: 10, kappa: 1e-3 }, 300],
    ["dyna-maze", "prioritized-sweeping", { alpha: 0.5, epsilon: 0.1, gamma: 0.95, planning: 5 }, 20],
    ["walk-1000", "gradient-mc", { alpha: 2e-4, gamma: 1, features: "tiles", tilings: 5, cells: 5 }, 30],
    ["walk-1000", "semi-gradient-td", { alpha: 2e-3, gamma: 1, features: "groups", cells: 20, n: 4 }, 30],
    ["mountain-car", "semi-gradient-sarsa", { alpha: 0.06, epsilon: 0, gamma: 1, features: "tiles", tilings: 8, cells: 8 }, 20],
    ["baird", "semi-gradient-td", { alpha: 0.01, gamma: 0.99, features: "own", policy: "target", behavior: "behavior" }, 200],
  ];
  const flat = (m) => Object.values(m).flatMap((x) => Array.from(x));
  for (const [world, id, params, units] of cases) {
    const r = run(world, id, params, units, 4);
    for (const t of [0, Math.floor(units / 3), units - 1]) {
      const { m, events } = r.replay(t);
      let G = 0;
      for (const ev of events) if (ev.type === "move") G += ev.r;
      assert.equal(G, r.metrics.return[t], `${id} on ${world}, unit ${t}: return`);
      assert.deepEqual(flat(m), flat(r.at(t + 1)), `${id} on ${world}, unit ${t}: what it learned`);
    }
  }
});

test("every Lab preset runs: its world, algorithms and measures exist", () => {
  require("../app/content.js");
  for (const [id, preset] of Object.entries(globalThis.RL.content.presets)) {
    for (const racer of preset.racers) {
      const algorithm = A[racer.algorithm];
      assert.ok(algorithm, `${id}: no algorithm '${racer.algorithm}' in the lab`);
      const r = lab.simulate({ world: preset.env, algorithm, params: { ...preset.params, ...racer.params }, units: 3, seed: 1, measures: preset.measures });
      assert.equal(r.metrics.return.length, 3, id);
    }
  }
});
