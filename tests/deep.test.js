// The deep trainer (lab/deep/): its math, its gradients, and runs that a seed reproduces exactly.
const test = require("node:test");
const assert = require("node:assert/strict");
const deep = require("../lab/deep/node.js");
const D = globalThis.RL.dmath;

test("deterministic math: within a few ulps of Math's", () => {
  const r = deep.rng(1), rel = (a, b) => (a === b ? 0 : Math.abs(a - b) / Math.abs(b));
  for (let i = 0; i < 20000; i++) {
    const x = -30 + 60 * r.next(), y = 1e-6 + 50 * r.next();
    assert.ok(rel(D.exp(x), Math.exp(x)) < 2e-15, `exp(${x})`);
    assert.ok(rel(D.log(y), Math.log(y)) < 2e-15 || Math.abs(D.log(y) - Math.log(y)) < 1e-16, `log(${y})`);
    assert.ok(rel(D.tanh(x / 10), Math.tanh(x / 10)) < 2e-15, `tanh(${x / 10})`);
    assert.ok(Math.abs(D.sin(x) - Math.sin(x)) < 1e-15 && Math.abs(D.cos(x) - Math.cos(x)) < 1e-15, `sin/cos(${x})`);
  }
  assert.equal(D.exp(0), 1); assert.equal(D.log(1), 0); assert.equal(D.sin(0), 0); assert.equal(D.cos(0), 1);
});

test("networks: gradients match finite differences; a state's values are the same alone or in a batch", () => {
  for (const act of ["relu", "tanh"]) {
    const r = deep.rng(3), net = new deep.MLP([3, 6, 5, 2], r, act), B = 3, x = Float64Array.from({ length: 9 }, () => r.normal());
    const loss = () => { const y = net.forward(x, B); let s = 0; for (let i = 0; i < y.length; i++) s += 0.5 * y[i] * y[i] * (i + 1); return s; };
    const y = net.forward(x, B), gx = net.backward(Float64Array.from(y, (v, i) => v * (i + 1))).slice();
    const numeric = (arr, k) => { const o = arr[k]; arr[k] = o + 1e-6; const a = loss(); arr[k] = o - 1e-6; const b = loss(); arr[k] = o; return (a - b) / 2e-6; };
    net.params.forEach((p, i) => { for (let k = 0; k < p.length; k++) assert.ok(Math.abs(numeric(p, k) - net.grads[i][k]) < 1e-7, `${act} param ${i}[${k}]`); });
    for (let k = 0; k < x.length; k++) assert.ok(Math.abs(numeric(x, k) - gx[k]) < 1e-7, `${act} input ${k}`);
  }
  const r = deep.rng(5), net = new deep.MLP([4, 64, 64, 2], r), B = 7, x = Float64Array.from({ length: 28 }, () => r.normal());
  const batch = net.forward(x, B).slice();
  for (let i = 0; i < B; i++) assert.deepEqual(Array.from(net.forward(x.subarray(i * 4, i * 4 + 4), 1)), Array.from(batch.subarray(i * 2, i * 2 + 2)));
});

test("DQN on CartPole: a seed is the same run every time, and the network learns", () => {
  const spec = { world: "cartpole", learner: "dqn", steps: 10000, block: 2500, seeds: [4], cfg: { lr: 5e-4, buffer: 10000, batch: 128, train_every: 4, target_every: 500, eps: [1, 0.05, 20000], huber: false } };
  const a = deep.run(spec, 4, { snapshots: true }), b = deep.run(spec, 4, { snapshots: true });
  assert.deepEqual(a, b);
  assert.ok(a.test.at(-1) > 100, `test returns ${a.test}`);
  const rec = deep.recording("t", spec, [a]);
  assert.equal(rec.snapshots.length, 5);
  assert.equal(rec.snapshots[4].test.steps, a.test.at(-1));
  for (const v of [{ double: true }, { dueling: true }, { prioritized: true }, { target_every: 0 }, { buffer: 128 }]) {
    const out = deep.run({ ...spec, steps: 2500, cfg: { ...spec.cfg, ...v } }, 1);
    assert.equal(out.test.length, 1, JSON.stringify(v));
  }
});

test("policy gradients and actor-critics: every algorithm runs, and a seed is the same run every time", () => {
  const pg = (algo, extra) => ({ world: "cartpole", learner: "pg", steps: 2000, block: 1000, seeds: [1], cfg: { algo, workers: 4, steps: 50, minibatch: 50, ...extra } });
  const ac = (algo) => ({ world: "pendulum", learner: "ac", steps: 1300, block: 650, seeds: [1], cfg: { algo, start: 1000, batch: 32, reward_scale: 0.1 } });
  for (const spec of [pg("a2c"), pg("ppo", { epochs: 2 }), pg("trpo", { v_epochs: 2 }), { ...pg("ppo", { epochs: 2 }), world: "pendulum", block: 800, steps: 1600 }, ac("ddpg"), ac("td3"), ac("sac")]) {
    const a = deep.run(spec, 2, { snapshots: true }), b = deep.run(spec, 2, { snapshots: true }), label = `${spec.cfg.algo} on ${spec.world}`;
    assert.deepEqual(a, b, label);
    assert.equal(a.test.length, 2, label);
    assert.equal(a.snapshots.length, 3, label);
    assert.ok(a.test.every(Number.isFinite), label);
  }
});
