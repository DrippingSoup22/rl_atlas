// The lab must reproduce what the guide teaches, and replays must show exactly what happened.
// Run from rl_atlas/ with: node --test   (after python build.py, which writes the presets the last test runs)
const test = require("node:test");
const assert = require("node:assert/strict");

const FILES = ["core", "envs/grid", "envs/bandit", "envs/chain", "envs/blackjack", "envs/mdp", "dp", "run", "measures",
  "agents/td", "agents/mc", "agents/dp", "agents/bandit"];
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

test("replaying a unit reproduces the run exactly, for every algorithm", () => {
  const cases = [
    ["cliff", "sarsa", cliff, 60], ["cliff", "q-learning", cliff, 60], ["cliff", "expected-sarsa", cliff, 60],
    ["max-bias", "double-q", { alpha: 0.1, epsilon: 0.1, gamma: 1 }, 100], ["random-walk", "td0", { alpha: 0.1, gamma: 1 }, 50],
    ["random-walk", "mc-prediction", { alpha: 0.1, gamma: 1 }, 50], ["blackjack", "exploring-starts", { gamma: 1 }, 30000],
    ["blackjack", "off-policy-mc", { gamma: 1, epsilon: 1 }, 30000], ["frozen-lake", "mc-control", { gamma: 0.99, epsilon: 0.1 }, 300],
    ["drifting", "epsilon-greedy", { epsilon: 0.1, alpha: 0.1 }, 3000], ["testbed", "ucb", { c: 2 }, 500],
    ["testbed-4", "gradient-bandit", { alpha: 0.1 }, 500], ["small-gridworld", "policy-iteration", { gamma: 1, theta: 1e-3 }, 120],
    ["gridworld", "value-iteration", { gamma: 0.9, theta: 1e-3 }, 30],
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
