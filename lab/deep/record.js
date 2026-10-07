/* Training runs as recordings: what the Lab and the stories play back (lab/recorded.js), in the format
   recorder/record.py wrote. For every seed, the average return of the training episodes in each block of steps and
   the return of one test episode after each block (always from the same start); for the seed shown, a snapshot after
   every block: what the network thinks of a grid of states (values, and the action it prefers), one test episode
   played with it, and the block's statistics.

   The same code trains offline (recorder/deep.js, in Node) and in the page (a Worker): same seed, same settings,
   same run. Offline, numbers are packed to 8 bits in base64; in the page (raw), they stay arrays. */
(function (RL) {
  "use strict";
  const deep = (RL.deep = RL.deep || {});
  const TEST_SEED = 2024; // every test episode starts from the same state, so snapshots compare like with like
  const round = (x, d = 2) => Math.round(x * 10 ** d) / 10 ** d;
  const mean = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null);

  // Numbers quantized to 8 or 16 bits between lo and hi, in base64 (the inverse of lab.unpack); raw: kept as they are.
  deep.pack = function (values, lo, hi, bits = 8, raw = false) {
    const v = Float64Array.from(values);
    if (raw) return v;
    if (lo === undefined) { lo = Infinity; hi = -Infinity; for (const x of v) { lo = Math.min(lo, x); hi = Math.max(hi, x); } }
    if (hi - lo < 1e-9) hi = lo + 1;
    const top = 2 ** bits - 1, q = bits === 16 ? new Uint16Array(v.length) : new Uint8Array(v.length);
    for (let i = 0; i < v.length; i++) q[i] = Math.round(((Math.min(hi, Math.max(lo, v[i])) - lo) / (hi - lo)) * top);
    const bytes = new Uint8Array(q.buffer);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return { lo: round(lo, 6), hi: round(hi, 6), bits, data: btoa(bin) };
  };

  // One episode from the test start; choose(obs) gives the action and the numbers kept for that step.
  function testEpisode(world, choose, raw) {
    const r = deep.rng(TEST_SEED);
    let s = world.reset(r), ret = 0, done = false;
    const states = [], actions = [], extra = [];
    while (!done) {
      const [a, numbers] = choose(world.observe(s));
      states.push(world.state(s)); actions.push(a); extra.push(numbers);
      const o = world.step(s, a);
      s = o.s2; ret += o.r;
      done = o.term || actions.length >= world.maxSteps;
    }
    states.push(world.state(s));
    const cols = extra[0].map((_, j) => extra.map((x) => x[j]));
    return {
      steps: actions.length, return: round(ret),
      s: world.ranges.map(([lo, hi], j) => deep.pack(states.map((x) => x[j]), lo, hi, 8, raw)),
      a: deep.pack(actions, undefined, undefined, 8, raw),
      x: cols.map((c) => deep.pack(c, undefined, undefined, 8, raw)), // each column on its own scale
    };
  }

  // ---- DQN ----
  function snapshotDQN(agent, world, raw) {
    const q = agent.valuesOf(world.grid()), nA = agent.nA, n = q.length / nA, v = new Float64Array(n), act = new Float64Array(n);
    for (let i = 0; i < n; i++) { let b = 0; for (let a = 1; a < nA; a++) if (q[i * nA + a] > q[i * nA + b]) b = a; v[i] = q[i * nA + b]; act[i] = b; }
    const test = testEpisode(world, (o) => { const x = agent.values(o); let b = 0; for (let a = 1; a < nA; a++) if (x[a] > x[b]) b = a; return [b, Array.from(x)]; }, raw);
    return { v: deep.pack(v, undefined, undefined, 8, raw), act: deep.pack(act, 0, nA - 1, 8, raw), test, eps: round(agent.epsilon(), 4) };
  }

  function runDQN(spec, seed, { snapshots = false, raw = false, onBlock } = {}) {
    const world = deep.worlds[spec.world], rng = deep.rng(seed), envRng = deep.rng(seed + 7777);
    const agent = new deep.DQN(world.obs, world.nA, spec.cfg, rng, spec.steps);
    const block = spec.block, blocks = Math.floor(spec.steps / block);
    const perBlock = Array.from({ length: blocks }, () => []), qs = [], tests = [], shots = snapshots ? [snapshotDQN(agent, world, raw)] : [];
    let s = world.reset(envRng), len = 0, ret = 0;
    for (let t = 0; t < spec.steps; t++) {
      const o0 = world.observe(s), a = agent.act(o0), o = world.step(s, a);
      agent.observe(o0, a, o.r, world.observe(o.s2), o.term);
      s = o.s2; ret += o.r; len++;
      if (o.term || len >= world.maxSteps) { perBlock[Math.floor(t / block)].push(ret); s = world.reset(envRng); len = 0; ret = 0; }
      if ((t + 1) % block === 0) {
        const L = agent.log, k = (t + 1) / block;
        qs.push(L.q.length ? round(mean(L.q)) : null); // every seed: overestimation shows here
        let test;
        if (snapshots) {
          const shot = snapshotDQN(agent, world, raw);
          shot.stats = { loss: round(mean(L.loss) ?? 0, 4), td: round(mean(L.td) ?? 0, 4), q: round(mean(L.q) ?? 0, 4) };
          shots.push(shot); test = shot.test.return;
        } else test = testEpisode(world, (ob) => [agent.act(ob, true), []], true).return;
        tests.push(test);
        agent.log = { loss: [], td: [], q: [] };
        onBlock?.(k, { train: perBlock[k - 1].length ? round(mean(perBlock[k - 1])) : null, test, shot: shots[shots.length - 1] });
      }
    }
    return { seed, train: perBlock.map((b) => (b.length ? round(mean(b)) : null)), episodes: perBlock.map((b) => b.length), q: qs, test: tests, snapshots: shots };
  }

  // ---- policy gradients (A2C, PPO, TRPO) ----
  // A test episode with actions drawn from the policy, keeping per step its probability of the last action (two
  // actions: the other is its complement) or its mean and spread, and the critic's value.
  function testPG(agent, world, rng, raw) {
    const pi = agent.pi, one = new Float64Array(world.obs);
    return testEpisode(world, (o) => {
      one.set(o);
      const a = pi.sample(one, 1, rng)[0], v = agent.value(one, 1)[0], d = pi.dist(one, 1);
      return pi.discrete ? [a, [d.p[d.p.length - 1], v]] : [pi.forEnv(a)[0], [d.mu[0], d.sd[0], v]];
    }, raw);
  }
  // The critic's values over the grid, the policy's choice there (the probability of the last action, or the mean
  // action), and a test episode.
  function snapshotPG(agent, world, rng, raw) {
    const g = world.grid(), n = g.length, x = new Float64Array(n * world.obs);
    g.forEach((s, i) => x.set(s, i * world.obs));
    const v = agent.value(x, n), d = agent.pi.dist(x, n), k = agent.pi.out;
    const choice = agent.pi.discrete ? Float64Array.from({ length: n }, (_, i) => d.p[i * k + k - 1]) : Float64Array.from({ length: n }, (_, i) => d.mu[i * k]);
    return { v: deep.pack(v, undefined, undefined, 8, raw), mean: deep.pack(choice, undefined, undefined, 8, raw), test: testPG(agent, world, rng, raw) };
  }

  function runPG(spec, seed, { snapshots = false, raw = false, onBlock } = {}) {
    const world = deep.worlds[spec.world], rng = deep.rng(seed), envRng = deep.rng(seed + 7777), dim = world.obs;
    const agent = new deep.PG(dim, world, spec.cfg, rng), cfg = agent.cfg, N = cfg.workers, T = cfg.steps;
    const block = spec.block, blocks = Math.floor(spec.steps / block);
    const perBlock = Array.from({ length: blocks }, () => []), tests = [], shots = snapshots ? [snapshotPG(agent, world, deep.rng(seed + 10000), raw)] : [];
    let stats = [];
    const states = Array.from({ length: N }, () => world.reset(envRng)), lens = new Int32Array(N), ep = new Float64Array(N);
    let t = 0;
    while (t < spec.steps) {
      const S = new Float64Array(T * N * dim), A = [], R = new Float64Array(T * N), V = new Float64Array(T * N), VN = new Float64Array(T * N), Dn = new Float64Array(T * N), C = new Float64Array(T * N);
      for (let k = 0; k < T; k++) {
        const obs = new Float64Array(N * dim);
        states.forEach((st, i) => obs.set(world.observe(st), i * dim));
        const a = agent.pi.sample(obs, N, rng);
        S.set(obs, k * N * dim); V.set(agent.value(obs, N), k * N); A.push(...a);
        const nxt = new Float64Array(N * dim);
        for (let i = 0; i < N; i++) {
          const o = world.step(states[i], agent.pi.discrete ? a[i] : agent.pi.forEnv(a[i])[0]);
          lens[i]++;
          const cut = !o.term && lens[i] >= world.maxSteps;
          R[k * N + i] = o.r * cfg.reward_scale; Dn[k * N + i] = o.term ? 1 : 0; C[k * N + i] = cut ? 1 : 0;
          states[i] = o.s2; nxt.set(world.observe(o.s2), i * dim); ep[i] += o.r;
          if (o.term || cut) { if (Math.floor(t / block) < blocks) perBlock[Math.floor(t / block)].push(ep[i]); ep[i] = 0; }
        }
        VN.set(agent.value(nxt, N), k * N); // the value of the state each step led to, before any reset: for cut episodes
        for (let i = 0; i < N; i++) if (Dn[k * N + i] || C[k * N + i]) { states[i] = world.reset(envRng); lens[i] = 0; }
        t += N;
        const b = Math.floor(t / block);
        if (t % block < N && b <= blocks && snapshots && shots.length <= b) {
          const shot = snapshotPG(agent, world, deep.rng(seed + 10000 + b), raw);
          shot.stats = stats.length ? Object.fromEntries(Object.keys(stats[0]).map((key) => [key, round(mean(stats.map((x) => x[key])), 4)])) : {};
          shots.push(shot); stats = [];
        }
        if (t % block < N && b <= blocks && tests.length < b) {
          tests.push(snapshots ? shots[shots.length - 1].test.return : testPG(agent, world, deep.rng(seed + 10000 + b), true).return);
          onBlock?.(b, { train: perBlock[b - 1].length ? round(mean(perBlock[b - 1])) : null, test: tests[tests.length - 1], shot: shots[shots.length - 1] });
        }
      }
      const { adv, ret } = deep.gae(R, V, VN, Dn, C, T, N, cfg.gamma, cfg.lam);
      agent.update(S, A, adv, ret);
      stats.push({ ...agent.stats });
    }
    return { seed, train: perBlock.map((b) => (b.length ? round(mean(b)) : null)), episodes: perBlock.map((b) => b.length), test: tests, snapshots: shots };
  }

  // ---- off-policy actor-critics (DDPG, TD3, SAC) ----
  // A test episode played greedily, keeping per step the action, the spread the policy explores with, and the value.
  function testAC(agent, world, raw) {
    const one = new Float64Array(world.obs);
    return testEpisode(world, (o) => { one.set(o); const a = agent.policy(one, 1, true)[0]; return [a, [a, agent.spread(one, 1)[0], agent.value(one, 1)[0]]]; }, raw);
  }
  // What the critic thinks of the grid (Q of the greedy action), the actor's greedy action there, and a test episode.
  function snapshotAC(agent, world, raw) {
    const g = world.grid(), n = g.length, x = new Float64Array(n * world.obs);
    g.forEach((s, i) => x.set(s, i * world.obs));
    const v = agent.value(x, n), a = agent.policy(x, n, true), ad = agent.ad;
    return { v: deep.pack(v, undefined, undefined, 8, raw), mean: deep.pack(Float64Array.from({ length: n }, (_, i) => a[i * ad]), undefined, undefined, 8, raw), test: testAC(agent, world, raw) };
  }

  function runAC(spec, seed, { snapshots = false, raw = false, onBlock } = {}) {
    const world = deep.worlds[spec.world], rng = deep.rng(seed), envRng = deep.rng(seed + 7777);
    const agent = new deep.AC(world.obs, world, spec.cfg, rng, spec.steps), cfg = agent.cfg;
    const block = spec.block, blocks = Math.floor(spec.steps / block);
    const perBlock = Array.from({ length: blocks }, () => []), qs = [], tests = [], shots = snapshots ? [snapshotAC(agent, world, raw)] : [];
    let s = world.reset(envRng), len = 0, ret = 0;
    for (let t = 0; t < spec.steps; t++) {
      const o0 = world.observe(s), a = agent.act(o0), o = world.step(s, a[0]);
      agent.observe(o0, a, o.r, world.observe(o.s2), o.term);
      s = o.s2; ret += o.r; len++;
      if (o.term || len >= world.maxSteps) { perBlock[Math.floor(t / block)].push(ret); s = world.reset(envRng); len = 0; ret = 0; }
      if ((t + 1) % block === 0) {
        const L = agent.log, k = (t + 1) / block;
        qs.push(L.q.length ? round(mean(L.q) / cfg.reward_scale) : null); // in the reward's own units
        let test;
        if (snapshots) {
          const shot = snapshotAC(agent, world, raw);
          shot.stats = { q: qs[qs.length - 1] ?? 0, loss: round(mean(L.loss) ?? 0, 4), ...(L.alpha.length ? { alpha: round(L.alpha[L.alpha.length - 1], 4) } : {}) };
          shots.push(shot); test = shot.test.return;
        } else test = testAC(agent, world, true).return;
        tests.push(test);
        agent.log = { q: [], loss: [], alpha: [] };
        onBlock?.(k, { train: perBlock[k - 1].length ? round(mean(perBlock[k - 1])) : null, test, shot: shots[shots.length - 1] });
      }
    }
    return { seed, train: perBlock.map((b) => (b.length ? round(mean(b)) : null)), episodes: perBlock.map((b) => b.length), q: qs, test: tests, snapshots: shots };
  }

  deep.runners = { dqn: runDQN, pg: runPG, ac: runAC };
  deep.run = (spec, seed, opts) => deep.runners[spec.learner](spec, seed, opts);

  // A recording from its runs (the shown seed's carrying the snapshots).
  deep.recording = function (name, spec, results) {
    const world = deep.worlds[spec.world], shown = spec.shown ?? spec.seeds[0], main = results.find((r) => r.seed === shown);
    return {
      name, title: spec.title, station: spec.station, world: spec.world, learner: spec.learner, trainer: "lab/deep",
      config: { ...spec.cfg }, steps: spec.steps, block: spec.block, seeds: spec.seeds, shown,
      grid: { x: [...world.x, deep.GRID], y: [...world.y, deep.GRID] },
      curves: results.map((r) => ({ seed: r.seed, train: r.train, test: r.test, episodes: r.episodes, ...(r.q ? { q: r.q } : {}) })),
      snapshots: main ? main.snapshots : [],
    };
  };
})(globalThis.RL = globalThis.RL || {});
