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
  lab.play = (algorithm, ctx, onEvent) => unit(algorithm, ctx, onEvent)();
  // The same unit, played in pieces: each call plays its events until the unit ends (and returns its numbers) or the
  // time runs out (null: the next call goes on from there). A round of many workers then never holds the page long.
  function unit(algorithm, ctx, onEvent) {
    const stats = { return: 0, steps: 0, falls: 0 };
    const it = algorithm.run(ctx);
    return (until = Infinity) => {
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
        if (until !== Infinity && performance.now() >= until) return null; // looked at after every event: some take ms
      }
    };
  }

  // world: a world's name or a function that makes one. measures: names of numbers to add after each unit (measures.js).
  // giveUp (runs without snapshots: the odds): stop a run once that many episodes in a row ran to the step cap, and take
  // its remaining units as more of the same. `stopped` then says after how many units it stopped; 0 if it never did.
  lab.simulate = (opts) => {
    const job = lab.simulateJob(opts);
    job.step(Infinity);
    return job.result;
  };

  // The same run as a job the page can spread over many short slices, so that a long one never freezes it:
  // job.step(ms) runs units for about that long and says whether the run is done; then job.result is what
  // lab.simulate returns. job.done is the share of units run so far (0 to 1).
  // reads: a Set that collects the name of every setting the run looks at (lab.reads).
  lab.simulateJob = function ({ world, algorithm, params, units, seed, snapshots = true, measures = null, giveUp = 0, reads = null }) {
    // params.anyStart: episodes start from any state (prediction on a world whose policy would visit only its own
    // path; the values of a policy do not depend on where its episodes start)
    const fresh = typeof world === "function" ? world : () => lab.make(world);
    const make = !params.anyStart ? fresh : () => {
      const e = fresh(), from = e.kind === "grid" ? e.starts : lab.dpStates(e);
      e.reset = (r, s0) => (s0 !== undefined ? s0 : from[Math.floor(r.next() * from.length)]);
      return e;
    };
    const env = make(), rng = lab.rng(seed);
    const settings = { maxSteps: env.maxSteps || 5000, ...params }; // a world may cap its own episodes (Taxi: 200 steps)
    const p = reads ? new Proxy(settings, { get: (o, k) => (reads.add(k), o[k]) }) : settings;
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
    let capped = 0, stopped = 0, t = 0, result = null, playing = null;

    const job = {
      get done() { return result ? 1 : t / units; },
      get result() { return result; },
      step(ms) {
        const until = performance.now() + ms;
        while (!result && t < units) {
          if (advance(until)) break;
          if (performance.now() >= until) break;
        }
        if (!result && (t >= units || stopped)) result = finish();
        return !!result;
      },
    };
    // One unit, or as much of it as the time allows (the rest comes at the next step, the same as if it had not
    // stopped): returns true when the run gives up (giveUp) or the time ran out in the middle of the unit.
    function advance(until) {
      if (!playing) {
        if (snapshots && t % every === 0) {
          const k = t / every;
          saved.set(buf, k * size);
          if (worldLen) saved.set(env.state, k * size + bufLen);
          rngAt[k] = rng.state;
        }
        playing = unit(algorithm, { env, m, rng, p, t });
      }
      const stats = playing(until);
      if (!stats) return true;
      playing = null;
      measure?.(m, stats, t);
      for (const key in stats) (metrics[key] ||= new Float64Array(units))[t] = stats[key];
      if (giveUp && !snapshots && t < units - 1) {
        capped = stats.steps >= p.maxSteps ? capped + 1 : 0;
        if (capped >= giveUp) {
          for (const key in metrics) metrics[key].fill(metrics[key][t], t + 1);
          stopped = t + 1;
          return true;
        }
      }
      t++;
      return false;
    }
    return job;

    function finish() {
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
        env: shown, world, algorithm, params: p, units, seed, metrics, every, stopped,
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
    }
  };

  // A story chart's averaged runs (app/js/scenes/common.js: curves): run `name` of the story config cfg, the k-th time,
  // on seed 1000 + k and without snapshots. Its key names a chart (its runs, settings and metric) among the averages
  // worked out offline (tools/story-curves.js), which the page then shows at once instead of averaging them itself.
  lab.curveRun = (cfg, name, k) => {
    const { algorithm, units, seed: _seed, world, measures, name: _label, ...params } = cfg.runs[name];
    return { world: world || cfg.world || cfg.env, algorithm: lab.algorithms[algorithm], params: { ...cfg.params, ...params }, units: units || cfg.units, seed: 1000 + k, snapshots: false, measures };
  };
  lab.curveKey = (cfg, names, metric) => JSON.stringify([names.map((n) => cfg.runs[n]), cfg.world || cfg.env || null, cfg.params || null, cfg.units || null, metric, cfg.average || 200]);

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

  // Follow the first best action from the start, or a policy P's likeliest action. `reached` is false if it falls or
  // goes round in circles; with showFall, a fall ends the path on the cliff tile it stepped into.
  // (On slippery ice there is no single path, so there is none to follow.)
  lab.greedyPath = function (env, Q, P = null, { showFall = false } = {}) {
    if (env.slip) return { path: [env.start], reached: false };
    const path = [env.start], seen = new Set(path);
    let s = env.start;
    while (!env.terminal(s)) {
      const a = P ? lab.argmax(P.subarray(s * env.nA, (s + 1) * env.nA)) : lab.greedy(Q, s, env);
      const { s2, fell } = env.step(s, a);
      if (fell !== undefined && showFall) path.push(fell);
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
