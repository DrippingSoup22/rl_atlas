/* Recorded runs: networks trained offline by recorder/record.py on Gymnasium's worlds, played back by the Lab's player.
   A recording keeps, for one seed, a snapshot after every block of training steps (what the network thinks of a grid of
   states, and one test episode played with it, always from the same start) and, for every seed, the average return of
   the training episodes in each block. A recorded run has the shape of a live one (lab.simulate): its units are the
   blocks, at(t) is the snapshot after t blocks, and replay(t) plays the test episode of the network at the end of
   block t (the snapshot after t + 1 blocks), event by event. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const unpacked = new WeakMap();

  // Numbers the recorder packed: quantized to 8 or 16 bits between lo and hi, in base64.
  // Runs trained in the page (lab/deep/) keep their numbers as arrays: those pass through.
  lab.unpack = function (p) {
    if (ArrayBuffer.isView(p) || Array.isArray(p)) return p;
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
    // the numbers of each step: one packed column each (older recordings packed them all together, step by step)
    const cols = Array.isArray(t.x) ? t.x.map(lab.unpack) : null, width = cols ? cols.length : lab.unpack(t.x).length / steps;
    // A critic trained on rewards times reward_scale learns values in those units: shown in the reward's own. The value
    // is DQN's every column, and the last column of the other learners' numbers.
    const unscale = 1 / (rec.config?.reward_scale || 1), first = rec.learner === "dqn" ? 0 : width - 1;
    const v = unscale === 1 ? lab.unpack(snap.v) : Float64Array.from(lab.unpack(snap.v), (x) => x * unscale);
    const numbers = (k) => {
      const x = cols ? Float64Array.from(cols, (c) => c[k]) : lab.unpack(t.x).slice(k * width, (k + 1) * width);
      if (unscale !== 1) for (let j = first; j < width; j++) x[j] *= unscale;
      return x;
    };
    return {
      t: snap, kind: rec.learner === "pg" ? "pg" : rec.learner, grid: rec.grid, v, act: snap.act ? lab.unpack(snap.act) : null, mean: snap.mean ? lab.unpack(snap.mean) : null,
      n, eps: snap.eps, stats: snap.stats || {}, steps, ret: t.return,
      state: (k) => dims.map((d) => d[k]), action: (k) => lab.unpack(t.a)[k], numbers,
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

  // Off-policy actor-critics for continuous actions (DDPG, TD3, SAC): the actor climbs the critic.
  LEARNERS.ac = {
    rule: (rec) => (rec.station === "sac"
      ? "\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\alp\\,\\nabla_{\\boldsymbol\\theta}\\Big(\\min_i \\val{\\hat q_i(s, a_{\\boldsymbol\\theta})} - \\alpha \\ln \\pol{\\pi(a_{\\boldsymbol\\theta} \\mid s)}\\Big)"
      : "\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\alp\\,\\nabla_a \\val{\\hat q(s, a)}\\big|_{a = \\pol{\\mu(s)}}\\,\\nabla_{\\boldsymbol\\theta}\\pol{\\mu(s, \\boldsymbol\\theta)}"),
    numbers: (ev) => `\\pol{\\mu(s)} = ${fmt(ev.x[0], 2)} \\qquad \\text{spread } ${fmt(ev.x[1], 2)} \\qquad \\val{\\hat q(s, \\mu(s))} = ${fmt(ev.x[2])}`,
    note: (ev) => `Step ${ev.k + 1} of the test episode · the actor chose <b>a torque of ${fmt(ev.a, 2)}</b> (no exploration in a test)`,
  };

  // The learner of a recording, in the shape of a Lab algorithm: the station whose pseudocode it follows, how it shows
  // a snapshot, and its test episode as events.
  lab.recordedLearner = function (rec) {
    const kind = rec.learner, L = LEARNERS[kind];
    const reward = REWARD[rec.world];
    return {
      id: rec.station, title: rec.title, unit: "block", recorded: true, kind,
      rule: typeof L.rule === "function" ? L.rule(rec) : L.rule,
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
      test: per((t) => rec.snapshots[t + 1].test.return), // block t's test episode, played by the network at its end
      steps: per((t) => rec.snapshots[t + 1].test.steps),
      loss: stat("loss"), td: stat("td"), q: stat("q"), kl: stat("kl"), clipped: stat("clipped"),
      eps: per((t) => rec.snapshots[t + 1].eps ?? NaN),
    };
    return {
      env, algorithm, params: { ...rec.config, block: rec.block }, units, seed: rec.shown, metrics, recorded: true, rec,
      at,
      replay(t) { const m = at(t + 1); return { m, events: algorithm.run({ m }) }; }, // unit t: the network at its end
    };
  };

  // ---- runs trained in the page ----
  // A recording made by the JavaScript trainer (lab/deep/, recorder/deep.js) can be extended in the page: the same
  // trainer, in a Web Worker, trains any seed or setting, and a seed of the recording comes out exactly as recorded.
  lab.trainable = (rec) => rec?.trainer === "lab/deep" && typeof Worker !== "undefined" && !!RL.deepWorker;
  let workerUrl = null;
  const newWorker = () => new Worker((workerUrl ||= URL.createObjectURL(new Blob([RL.deepWorker], { type: "text/javascript" }))));
  const specOf = (rec, config) => ({ world: rec.world, learner: rec.learner, steps: rec.steps, block: rec.block, cfg: config });
  const keyOf = (rec, config, seed, kind) => JSON.stringify([kind, rec.world, rec.learner, rec.steps, rec.block, config, seed]);
  const finished = new Map(); // what this visit already trained: runs, and bench seeds' curves

  // One run, with its snapshots, as a job: { done (0 to 1), secs (so far), result (a recorded run, once trained),
  // cancel() }. onBlock(job) after each block.
  lab.trainRun = function (rec, config, seed, onBlock) {
    const key = keyOf(rec, config, seed, "run"), blocks = Math.floor(rec.steps / rec.block);
    if (finished.has(key)) return { done: 1, secs: 0, result: finished.get(key), cancel() {} };
    const worker = newWorker(), job = { done: 0, secs: 0, result: null, cancel() { worker.terminate(); } };
    worker.onmessage = ({ data }) => {
      if (!data.run) { job.done = data.block / blocks; job.secs = data.secs; onBlock?.(job); return; }
      const r = data.run;
      job.result = lab.recordedRun({ ...rec, config, shown: seed, curves: [{ seed, train: r.train, test: r.test, episodes: r.episodes, ...(r.q ? { q: r.q } : {}) }], snapshots: r.snapshots });
      job.done = 1;
      finished.set(key, job.result);
      worker.terminate();
      onBlock?.(job);
    };
    worker.onerror = (e) => { job.error = e.message || "the trainer stopped"; onBlock?.(job); };
    worker.postMessage({ id: 0, spec: specOf(rec, config), seed });
    return job;
  };

  // A bench: the training and test curves of many seeds (no snapshots), over a pool of Workers, one per spare core.
  // A job: { curves (by seed, as they finish), done, cancel() }; onSeed(job) after each seed.
  lab.trainBench = function (rec, config, seeds, onSeed) {
    const job = { curves: new Map(), done: 0, cancel() { pool.forEach((w) => w.terminate()); } }, todo = [];
    for (const seed of seeds) {
      const c = finished.get(keyOf(rec, config, seed, "seed"));
      if (c) job.curves.set(seed, c); else todo.push(seed);
    }
    job.done = job.curves.size / seeds.length;
    const n = Math.min(todo.length, Math.max(1, (navigator.hardwareConcurrency || 2) - 1)), pool = [];
    for (let w = 0; w < n; w++) {
      const worker = newWorker(), next = () => { const seed = todo.shift(); if (seed === undefined) worker.terminate(); else worker.postMessage({ id: seed, spec: specOf(rec, config), seed, snapshots: false }); };
      worker.onmessage = ({ data }) => {
        if (!data.run) return;
        const r = data.run, c = { seed: data.id, train: r.train, test: r.test, episodes: r.episodes, ...(r.q ? { q: r.q } : {}) };
        finished.set(keyOf(rec, config, data.id, "seed"), c);
        job.curves.set(data.id, c);
        job.done = job.curves.size / seeds.length;
        onSeed?.(job);
        next();
      };
      pool.push(worker);
      next();
    }
    return job;
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
