/* A run: one algorithm played from one seed for a number of units (episodes, sweeps of the states, or single steps).
   It keeps snapshots of everything the algorithm has learned, the world's own state and the random state, so the
   Lab can show any moment and replay any unit line by line, exactly as it happened. Long runs keep a snapshot every
   few units and recompute the units in between when asked. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const BUDGET = 2e6; // numbers a run may keep in snapshots (16 MB)

  // Named arrays sharing one buffer: memory({ Q: 8, N: 8 }) → { buf, m: { Q, N } }, so a snapshot is one copy.
  lab.memory = function (layout, buf) {
    buf ||= new Float64Array(Object.values(layout).reduce((a, b) => a + b, 0));
    const m = {};
    let at = 0;
    for (const [k, n] of Object.entries(layout)) { m[k] = buf.subarray(at, at + n); at += n; }
    return { buf, m };
  };

  // Play one unit and return its numbers: the reward collected, the steps taken, the falls, and whatever the
  // algorithm returns (a sweep's largest change, say) or the world tracks (did a bandit pull a best arm?).
  lab.play = function (algorithm, ctx, onEvent) {
    const stats = { return: 0, steps: 0, falls: 0 };
    const it = algorithm.run(ctx);
    for (;;) {
      const { value: ev, done } = it.next();
      if (done) { if (ev) Object.assign(stats, ev); return stats; }
      if (ev.type === "move") {
        stats.return += ev.r;
        stats.steps += 1;
        if (ev.fell !== undefined) stats.falls += 1;
        ctx.env.track?.(ev, stats);
      }
      onEvent?.(ev);
    }
  };

  // world: a world's name or a function that makes one. measures: names of numbers to add after each unit (measures.js).
  lab.simulate = function ({ world, algorithm, params, units, seed, snapshots = true, measures = null }) {
    const make = typeof world === "function" ? world : () => lab.make(world);
    const p = { maxSteps: 5000, ...params };
    const env = make(), rng = lab.rng(seed);
    env.init?.(rng);
    const layout = algorithm.memory(env, p);
    const { buf, m } = lab.memory(layout);
    algorithm.init?.(m, env, p);
    const measure = lab.measurer(measures, env, p, algorithm, units);
    const bufLen = buf.length, worldLen = env.state ? env.state.length : 0, size = bufLen + worldLen;
    const every = snapshots ? Math.max(1, Math.ceil((units * size) / BUDGET)) : Infinity;
    const marks = snapshots ? Math.ceil(units / every) : 0;
    const saved = new Float64Array(marks * size), rngAt = new Uint32Array(marks);
    const metrics = {};

    for (let t = 0; t < units; t++) {
      if (snapshots && t % every === 0) {
        const k = t / every;
        saved.set(buf, k * size);
        if (worldLen) saved.set(env.state, k * size + bufLen);
        rngAt[k] = rng.state;
      }
      const stats = lab.play(algorithm, { env, m, rng, p, t });
      measure?.(m, stats, t);
      for (const key in stats) (metrics[key] ||= new Float64Array(units))[t] = stats[key];
    }
    const end = { buf: Float64Array.from(buf), world: worldLen ? Float64Array.from(env.state) : null, rng: rng.state };

    // A working copy that can be moved to the start of any unit, and the world the views look at.
    const work = { env: make(), ...lab.memory(layout), rng: lab.rng(), t: -1 };
    const shown = make();
    function goTo(t) {
      if (t >= units) {
        work.buf.set(end.buf);
        if (worldLen) work.env.state.set(end.world);
        work.rng.state = end.rng;
        work.t = units;
      } else {
        const k = Math.floor(t / every);
        if (!(work.t >= k * every && work.t <= t)) {
          work.buf.set(saved.subarray(k * size, k * size + bufLen));
          if (worldLen) work.env.state.set(saved.subarray(k * size + bufLen, (k + 1) * size));
          work.rng.state = rngAt[k];
          work.t = k * every;
        }
        work.env.restore?.();
        while (work.t < t) lab.play(algorithm, { env: work.env, m: work.m, rng: work.rng, p, t: work.t++ });
      }
      if (worldLen) { shown.state.set(work.env.state); shown.restore?.(); }
    }

    return {
      env: shown, algorithm, params: p, units, seed, metrics, every,
      // What the algorithm knew at the start of unit t (t = units: at the end), as a fresh copy.
      at(t) {
        if (!snapshots) throw new Error("this run kept no snapshots");
        goTo(t);
        return lab.memory(layout, Float64Array.from(work.buf)).m;
      },
      // Unit t again, from its own starting point: { m, events }, where events is the algorithm's generator.
      replay(t) {
        goTo(t);
        const copy = lab.memory(layout, Float64Array.from(work.buf)), r = lab.rng();
        r.state = work.rng.state;
        return { m: copy.m, events: algorithm.run({ env: shown, m: copy.m, rng: r, p, t }) };
      },
      // The tiles unit t visited, in order; a fall adds the cliff tile, a −1 break, then where it landed.
      trail(t) {
        const { events } = this.replay(t), path = [];
        for (const ev of events) {
          if (ev.type === "start") path.push(ev.s);
          else if (ev.type === "move") { if (ev.fell !== undefined) path.push(ev.fell, -1, ev.s2); else path.push(ev.s2); }
        }
        return path;
      },
    };
  };

  // Run many seeds without snapshots and average their numbers: { mean: { metric: Float64Array }, runs }.
  lab.average = function ({ world, algorithm, params, units, seeds, measures = null }) {
    const mean = {};
    for (const seed of seeds) {
      const { metrics } = lab.simulate({ world, algorithm, params, units, seed, snapshots: false, measures });
      for (const k in metrics) {
        const sum = (mean[k] ||= new Float64Array(units));
        for (let t = 0; t < units; t++) sum[t] += metrics[k][t] / seeds.length;
      }
    }
    return { mean, runs: seeds.length };
  };

  // Follow the first best action from the start. `reached` is false if it falls or goes round in circles.
  // (On slippery ice there is no single path, so there is none to follow.)
  lab.greedyPath = function (env, Q) {
    if (env.slip) return { path: [env.start], reached: false };
    const path = [env.start], seen = new Set(path);
    let s = env.start;
    while (!env.terminal(s)) {
      const { s2, fell } = env.step(s, lab.greedy(Q, s, env));
      if (fell !== undefined || seen.has(s2)) return { path, reached: false };
      path.push(s2);
      seen.add(s2);
      s = s2;
    }
    return { path, reached: true };
  };

  // How a run ended, by a preset's rule: { metric, min or max, window: [first, last] unit (1-based; the last tenth by
  // default) }. score is the metric's average over the window; ok whether it reached min or stayed within max.
  lab.success = function (rule, metrics) {
    const v = metrics[rule.metric], n = v.length;
    const [from, to] = rule.window || [n - Math.max(1, Math.round(n / 10)) + 1, n];
    let sum = 0, count = 0;
    for (let t = Math.max(1, from) - 1; t < Math.min(n, to); t++) if (Number.isFinite(v[t])) { sum += v[t]; count++; }
    const score = count ? sum / count : NaN;
    return { score, ok: rule.min !== undefined ? score >= rule.min : rule.max !== undefined ? score <= rule.max : undefined };
  };
})(globalThis.RL = globalThis.RL || {});
