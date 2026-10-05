/* Recorded runs: networks trained offline by recorder/record.py on Gymnasium's worlds, played back by the Lab's player.
   A recording keeps, for one seed, a snapshot after every block of training steps (what the network thinks of a grid of
   states, and one test episode played with it, always from the same start) and, for every seed, the average return of
   the training episodes in each block. A recorded run has the shape of a live one (lab.simulate): its units are the
   blocks, at(t) is the snapshot after t blocks, and replay(t) plays that snapshot's test episode, event by event. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const unpacked = new WeakMap();

  // Numbers the recorder packed: quantized to 8 or 16 bits between lo and hi, in base64.
  lab.unpack = function (p) {
    if (unpacked.has(p)) return unpacked.get(p);
    const bin = atob(p.data), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const q = p.bits === 16 ? new Uint16Array(bytes.buffer) : bytes, top = 2 ** p.bits - 1, out = new Float64Array(q.length);
    for (let i = 0; i < q.length; i++) out[i] = p.lo + ((p.hi - p.lo) * q[i]) / top;
    unpacked.set(p, out);
    return out;
  };

  // What one step paid, which the recordings leave out: Gymnasium's rewards, from the states and the action.
  const REWARD = {
    cartpole: () => 1,
    "mountain-car": () => -1,
    pendulum: (s, a) => -(s[0] ** 2 + 0.1 * s[1] ** 2 + 0.001 * a ** 2),
  };

  // A snapshot, unpacked: the grid's values and actions, and the test episode as states, actions and numbers per step.
  function decode(rec, snap) {
    const n = rec.grid.x[3] * rec.grid.y[3], t = snap.test, dims = t.s.map(lab.unpack), steps = t.steps;
    const width = lab.unpack(t.x).length / steps;
    return {
      t: snap, kind: rec.learner === "dqn" ? "dqn" : "pg", grid: rec.grid, v: lab.unpack(snap.v), act: snap.act ? lab.unpack(snap.act) : null, mean: snap.mean ? lab.unpack(snap.mean) : null,
      n, eps: snap.eps, stats: snap.stats || {}, steps, ret: t.return,
      state: (k) => dims.map((d) => d[k]), action: (k) => lab.unpack(t.a)[k], numbers: (k) => lab.unpack(t.x).subarray(k * width, (k + 1) * width),
    };
  }

  // How each kind of learner explains a step of its test episode in the Lab's live panel.
  const fmt = (v, d = 1) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;
  const LEARNERS = {
    dqn: {
      rule: "\\mathbf w \\leftarrow \\mathbf w - \\alp\\,\\nabla_{\\mathbf w}\\big(\\rew{R} + \\gam \\max_{a'} \\val{\\hat q(S', a', \\mathbf w^-)} - \\val{\\hat q(S, A, \\mathbf w)}\\big)^2",
      numbers: (ev, env) => `${Array.from(ev.x, (q, a) => `\\val{\\hat q(s, \\text{${env.actionNames[a]}})} = ${fmt(q)}`).join(" \\qquad ")}`,
      note: (ev, env) => `Step ${ev.k + 1} of the test episode · the larger value wins: <b>${env.actionNames[ev.a]}</b>`,
    },
    pg: {
      rule: "\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\alp\\,\\hat{\\mathbb E}\\big[\\err{\\hat A_t}\\,\\nabla \\ln \\pol{\\pi(A_t \\mid S_t, \\boldsymbol\\theta)}\\big]",
      numbers: (ev, env) => (env.actionNames
        ? `${[1 - ev.x[0], ev.x[0]].map((p, a) => `\\pol{\\pi(\\text{${env.actionNames[a]}} \\mid s)} = ${(100 * p).toFixed(0)}\\%`).join(" \\qquad ")} \\qquad \\val{\\hat v(s)} = ${fmt(ev.x[1])}`
        : `\\pol{\\mu(s)} = ${fmt(ev.x[0], 2)} \\qquad \\pol{\\sigma} = ${fmt(ev.x[1], 2)} \\qquad \\val{\\hat v(s)} = ${fmt(ev.x[2])}`),
      note: (ev, env) => `Step ${ev.k + 1} of the test episode · the policy drew <b>${env.actionNames ? env.actionNames[ev.a] : `a torque of ${fmt(ev.a, 2)}`}</b>`,
    },
  };

  // The learner of a recording, in the shape of a Lab algorithm: the station whose pseudocode it follows, how it shows
  // a snapshot, and its test episode as events.
  lab.recordedLearner = function (rec) {
    const kind = rec.learner === "dqn" ? "dqn" : "pg", L = LEARNERS[kind];
    const reward = REWARD[rec.world];
    return {
      id: rec.station, title: rec.title, unit: "block", recorded: true, kind,
      rule: L.rule,
      numbers: (ev) => L.numbers(ev, lab.make(rec.world)),
      note: (ev) => L.note(ev, lab.make(rec.world)),
      show: (m) => m,
      *run({ m }) {
        let s = m.state(0);
        yield { type: "start", s };
        for (let k = 0; k < m.steps; k++) {
          const a = m.action(k), x = m.numbers(k), s2 = m.state(k + 1);
          yield { type: "choose", s, a, x, k };
          yield { type: "move", s, a, r: reward(s2, a), s2, k, x, end: k === m.steps - 1 };
          s = s2;
        }
      },
    };
  };

  // A recorded run, shaped like lab.simulate's.
  lab.recordedRun = function (rec) {
    const algorithm = lab.recordedLearner(rec), units = rec.snapshots.length - 1, env = lab.make(rec.world);
    const decoded = new Map(), at = (t) => {
      const k = Math.max(0, Math.min(units, t));
      if (!decoded.has(k)) decoded.set(k, decode(rec, rec.snapshots[k]));
      return decoded.get(k);
    };
    const shown = rec.curves.find((c) => c.seed === rec.shown), per = (f) => Float64Array.from({ length: units }, (_, t) => f(t));
    const stat = (k) => per((t) => rec.snapshots[t + 1].stats?.[k] ?? NaN);
    const metrics = {
      return: per((t) => shown.train[t] ?? NaN), // the training episodes of block t (exploring, and learning as they go)
      test: per((t) => rec.snapshots[t].test.return), // the test episode played by the network after t blocks
      steps: per((t) => rec.snapshots[t].test.steps),
      loss: stat("loss"), td: stat("td"), q: stat("q"), kl: stat("kl"), clipped: stat("clipped"),
      eps: per((t) => rec.snapshots[t].eps ?? NaN),
    };
    return {
      env, algorithm, params: { ...rec.config, block: rec.block }, units, seed: rec.shown, metrics, recorded: true, rec,
      at,
      replay(t) { const m = at(t); return { m, events: algorithm.run({ m }) }; },
    };
  };

  // The training curves of every seed, per block: { seeds: [Float64Array, …], mean: Float64Array }.
  lab.recordedCurves = function (rec, key = "train") {
    const units = rec.snapshots.length - 1;
    const seeds = rec.curves.map((c) => Float64Array.from({ length: units }, (_, t) => c[key][t] ?? NaN));
    const mean = Float64Array.from({ length: units }, (_, t) => {
      const vals = seeds.map((s) => s[t]).filter(Number.isFinite);
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN;
    });
    return { seeds, mean };
  };
})(globalThis.RL = globalThis.RL || {});
