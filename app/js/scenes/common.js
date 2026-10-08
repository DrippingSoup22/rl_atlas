/* Story scenes: the pictures that follow a story's text. A scene turns each step's small state into a picture;
   states are complete descriptions, so scrolling back shows the same picture again. This file holds what the
   scenes share; each scene lives in its own file (grid, loop, timeline, mdp, bandit, chain, cards, graph). */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const SUB = "₀₁₂₃₄₅₆₇₈₉";
  const SHOWN = new Set(["update", "improve", "trace", "plan", "world", "critic"]); // events after which a view redraws what the algorithm knows
  RL.scenes = RL.scenes || {};

  // Timers a scene cancels whenever its step changes.
  function timers() {
    let list = [];
    return { later: (fn, ms) => { list.push(setTimeout(fn, ms)); }, stop: () => { list.forEach(clearTimeout); list = []; } };
  }

  // A story formula whose \step{n}{..} pieces appear one step at a time, and named lines of numbers under it.
  function formula(card, cfg) {
    const sym = card.querySelector(".f-sym"), num = card.querySelector(".f-num");
    sym.innerHTML = cfg.formula ? RL.math.tex(cfg.formula, true) : "";
    const pieces = Array.from(sym.querySelectorAll("[data-step]"));
    pieces.forEach((p) => p.style.setProperty("--k", p.dataset.step));
    const lines = typeof cfg.numbers === "string" ? { main: cfg.numbers } : cfg.numbers || {};
    // a story without lines of numbers keeps room for the formula alone
    card.querySelector(".scene-formula")?.classList.toggle("bare", !Object.keys(lines).length);
    // too wide for the card: shrink it, down to 70% (the pieces stay side by side, as the steps reveal them)
    const fit = () => {
      sym.style.fontSize = "";
      const room = sym.clientWidth, need = sym.scrollWidth;
      if (room && need > room + 1) sym.style.fontSize = `${Math.max(0.7, (room - 4) / need).toFixed(3)}em`;
    };
    return (st) => {
      const n = st.formula || 0;
      pieces.forEach((p) => p.classList.toggle("shown", +p.dataset.step <= n));
      sym.classList.toggle("on", n > 0);
      if (n > 0) fit();
      const line = st.numbers === true ? lines.main : st.numbers ? lines[st.numbers] : null;
      if (line) num.innerHTML = RL.math.tex(line, true);
      num.classList.toggle("on", !!line);
    };
  }

  // Move an SVG element along a path, sampled into keyframes.
  function along(el, path, ms) {
    const len = path.getTotalLength(), frames = [];
    for (let i = 0; i <= 30; i++) {
      const p = path.getPointAtLength((len * i) / 30);
      frames.push({ transform: `translate(${p.x}px, ${p.y}px)` });
    }
    return el.animate(frames, { duration: RL.reducedMotion() ? 1 : ms, easing: "ease-in-out", fill: "forwards" });
  }

  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    parent?.appendChild(e);
    return e;
  }

  // A unit of a run replayed: its events, kept. A step drawn at rest replays the unit before its moment (a round of
  // PPO, with its ten passes over the batch, takes tens of milliseconds), and steps often come back to the same moment.
  const rests = new WeakMap();
  const restsKept = (r) => rests.get(r) || rests.set(r, new Map()).get(r);
  function restOf(r, u) {
    const kept = restsKept(r), k = kept.get(u);
    if (k?.done) return k.events;
    const events = [...r.replay(u).events]; // (one half worked out ahead is dropped: the replay starts over)
    kept.set(u, { done: true, events });
    return events;
  }
  // The same, worked out ahead a few milliseconds at a time; true once kept. Not for a world with a state of its own
  // (a bandit's drifting arms): replays share the run's world, and two at once would mix it up; its units are short.
  function restAhead(r, u, ms) {
    if (r.env.state) return true;
    const kept = restsKept(r);
    let k = kept.get(u);
    if (k?.done) return true;
    if (!k) kept.set(u, (k = { done: false, events: [], it: r.replay(u).events }));
    const until = performance.now() + ms;
    for (;;) {
      const { value, done } = k.it.next();
      if (done) { k.done = true; k.it = null; return true; }
      k.events.push(value);
      if (performance.now() >= until) return false;
    }
  }
  const restReady = (r, u) => u < 0 || !!r.env.state || !!rests.get(r)?.get(u)?.done;

  // The unit a step draws at rest: the one before the moment it shows (-1: none, or a step of checkpoints, which shows
  // several in turn). name: the step's run.
  function restUnit(cfg, st, name) {
    if (st.checkpoints || !cfg.runs?.[name] || cfg.runs[name].recording) return -1;
    return Math.min(st.at ?? 0, cfg.runs[name].units || cfg.units || 100) - 1;
  }
  // Every unit the steps will draw at rest, checkpoints included: { name, u }. fallback: the run of a step naming none.
  function restsOf(cfg, states, fallback) {
    const out = new Map();
    for (const st of states) {
      const name = st.run || fallback;
      if (!name || !cfg.runs?.[name] || cfg.runs[name].recording) continue;
      const units = cfg.runs[name].units || cfg.units || 100, n = st.checkpoints;
      const ts = !n ? [Math.min(st.at ?? 0, units)] : Array.isArray(n) ? n.map((t) => Math.min(t, units)) : Array.from({ length: n }, (_, k) => Math.round((units * (k + 1)) / n));
      for (const t of ts) if (t > 0) out.set(`${name} ${t - 1}`, { name, u: t - 1 });
    }
    return [...out.values()];
  }

  // Named runs a story shows and replays, from its config: [story.runs] name = { algorithm, units, seed, <knobs> }, or
  // name = { recording } for a run trained offline (recorder/record.py). The world, the shared knobs and the seed come
  // from the story config unless a run says otherwise.
  // With host, the runs are also worked out in the background, a few milliseconds at a time, while the reader is on
  // the first steps (and the host on the page), and then the units the steps draw at rest (ahead: restsOf): a step
  // then finds them ready instead of making the page wait. A step that comes to a run, or a unit, not ready yet asks
  // for it (runOf.when): it goes first, still in slices, and the page keeps scrolling meanwhile.
  function runs(cfg, host = null, ahead = []) {
    const made = {}, jobs = {};
    const spec = (name) => {
      const s = cfg.runs?.[name];
      if (!s) throw new Error(`story: no run named '${name}'`);
      return s;
    };
    const setup = (name) => {
      const { algorithm, units, seed, world, measures, name: _label, ...params } = spec(name);
      return { world: world || cfg.world || cfg.env, algorithm: RL.lab.algorithms[algorithm], params: { ...cfg.params, ...params }, units: units || cfg.units || 100, seed: seed ?? cfg.seed ?? 1, measures };
    };
    const job = (name) => (jobs[name] ||= RL.lab.simulateJob(setup(name)));
    const runOf = (name) => {
      if (made[name]) return made[name];
      const s = spec(name);
      if (s.recording) return (made[name] = RL.lab.recordedRun(RL.recordings[s.recording]));
      job(name).step(Infinity); // the same run, finished now (sliced or not, a run comes out the same)
      return (made[name] = job(name).result);
    };
    const made_ = (name) => !!(made[name] || spec(name).recording || jobs[name]?.result);
    // the run, and the unit it draws at rest (u ≥ 0), ready
    runOf.ready = (name, u = -1) => made_(name) && (spec(name).recording || restReady(runOf(name), u));
    // what a run is (its world, its algorithm, how many units), known without running it
    runOf.about = (name) => {
      if (made_(name)) return runOf(name);
      const { world, algorithm, units } = setup(name);
      return { env: RL.lab.make(world), algorithm, units };
    };
    let urgent = null, timer = 0;
    runOf.when = (name, u, then) => {
      if (runOf.ready(name, u)) return then(runOf(name));
      urgent = { name, u, then }; // the step asking last is the one shown: an earlier request is dropped
      clearTimeout(timer);
      timer = setTimeout(work, 0);
    };
    const todo = Object.keys(cfg.runs || {}).filter((n) => !cfg.runs[n].recording);
    // a slice of 10 ms on a run, or (once it is made, in the next slice) on one of its units: true when both are done
    const slice = (name, u) => {
      if (!made_(name)) { if (!job(name).step(10)) return false; made[name] = job(name).result; return u < 0; }
      return u < 0 || !!spec(name).recording || restAhead(runOf(name), u, 10);
    };
    function work() {
      timer = 0;
      if (host && !host.isConnected) return;
      if (urgent) { // a step waits for it: slices of 10 ms, scrolling or not
        const w = urgent;
        if (slice(w.name, w.u)) urgent = null;
        timer = setTimeout(work, urgent ? 0 : 10);
        if (!urgent) setTimeout(() => w.then(runOf(w.name)), 0); // drawn in a task of its own
        return;
      }
      if (!host) return;
      // slices of 10 ms (a run stops between two events of a unit, lab.simulateJob), a frame drawn between them
      const name = todo.find((n) => !made[n]), next = name ? null : ahead.find((a) => !a.done);
      if (!name && !next) return;
      if (!RL.scrolling()) {
        if (name) { if (job(name).step(10)) made[name] = job(name).result; }
        else if (slice(next.name, next.u)) next.done = true;
      }
      timer = setTimeout(work, RL.scrolling() ? 120 : 4);
    }
    if (host) timer = setTimeout(work, 300);
    return runOf;
  }

  // Replay units of a run on a view, event by event, the way the Lab walks: the agent moves, values update.
  // Human pace: one move takes at least MIN_PACE ms, a choice half of that, so the eye can follow every step.
  // With updates = n, only the first n updates of the unit are played, and the replay stops there. With instant, the
  // events are played without animation: the view jumps to where they lead (a moment inside a unit, say). With lead = n,
  // a long episode shows its first n moves, then jumps to its last move and shows what follows at pace (the updates
  // made at the end of an episode, say).
  const MIN_PACE = 300;
  function player(later) {
    return function play(view, run, from, count, { pace = 400, fine = false, after, updates = 0, instant = false, lead = 0 } = {}) {
      const P = Math.max(MIN_PACE, pace);
      let t = from, w = null, seen = 0, moves = 0, total = 0;
      const end = Math.min(run.units, from + count);
      // one event; returns how long to wait before the next, or -1 when the replay is over
      const one = () => {
        if (!w) {
          if (t >= end) { after?.(t); return -1; }
          w = run.replay(t);
          moves = 0;
          total = 0;
          if (lead) for (const ev of run.replay(t).events) if (ev.type === "move") total++;
        }
        const { value: ev, done } = w.events.next();
        if (done) { w = null; t++; after?.(t, true); return 700; }
        const skipping = lead > 0 && moves >= lead && moves < total;
        if (ev.type === "move") moves++;
        if (SHOWN.has(ev.type)) view.show(run.algorithm.show(w.m, run.env, run.params, t), run.params);
        const wait = view.event(ev, { line: fine, p: run.params }) || 0;
        if (updates && ev.type === "update" && ++seen >= updates) return -1;
        // a team of workers moves together: their moves land at once, and each tick of the clock takes one beat
        if (skipping || ev.w !== undefined) return 0;
        const quiet = ev.type === "info" || ev.type === "next" || ev.type === "skip";
        return quiet ? 40 : ev.type === "choose" ? P / 2 : ev.type === "plan" ? P / 2 : P + wait;
      };
      // instant: as fast as the page allows, in slices of a few milliseconds (hundreds of events, each drawn, would hold
      // the page a while); the picture runs ahead over a few frames to where they lead
      if (instant) {
        const burst = () => { const t0 = performance.now(); while (one() >= 0) if (performance.now() - t0 > 8) return later(burst, 0); };
        return burst();
      }
      // waits of 0 run on at once, in one go, so the picture jumps instead of flickering through them
      const next = () => { let ms = one(); while (ms === 0) ms = one(); if (ms >= 0) later(next, ms); };
      next();
    };
  }

  // A chart under a story's picture: named runs, each averaged over many seeds, filled in as the seeds come in.
  // Averages are kept, so scrolling back and forth does not compute them again.
  const METRIC = {
    optimal: { label: "How often the best arm is pulled", percent: true },
    return: { label: (noun) => (noun[0] === "step" ? "Reward per step" : noun[0] === "round" ? "Total reward per episode (average of each round)" : noun[0] === "block" ? "Training episodes: their average return in each block" : `Total reward per ${noun[0]}`) },
    left: { label: "How often the agent goes left from A", percent: true },
    error: { label: "Error: distance from the true values (RMS)", zero: true },
    match: { label: "States where the greedy action is optimal", percent: true },
    greedy: { label: "Chance the greedy policy reaches the gem", percent: true },
    steps: { label: "Steps per episode", log: true },
    ve: { label: "Value error √VE", zero: true },
    weights: { label: "Size of the weights ‖w‖", log: true },
    "policy-value": { label: "Value of the policy from the start, J(θ)" },
    right: { label: "Chance of stepping right, π(right)", percent: true },
    aim: { label: "Where the policy aims, in degrees" },
    test: { label: "Test episode after each block: its return" },
    deployed: { label: "Return of the learned policy, played once from the start" },
    q: { label: (noun, learner) => (learner === "ac" ? "The critics' targets in each batch, on average" : "The largest Q-value in each batch, on average") },
  };
  function curves(host, cfg, runOf) {
    const cache = new Map();
    let chart = null, shown = "", timer = 0;
    return function show(st) {
      const names = st.curves || [], metric = st.metric || "optimal", key = `${names}|${metric}|${st.domain || ""}|${st.log || ""}|${st.ref || ""}|${st.title || ""}`;
      host.hidden = !names.length;
      if (!names.length || key === shown) return;
      shown = key;
      clearTimeout(timer);
      chart?.destroy();
      const first = runOf.about(names[0]), unit = first.env.unitName || first.algorithm.unit;
      const noun = unit === "pull" || unit === "step" ? ["step", "steps"] : unit === "hand" ? ["hand", "hands"] : unit === "round" ? ["round", "rounds"] : unit === "throw" ? ["throw", "throws"] : unit === "block" ? ["block", "blocks"] : unit === "pass" ? ["pass", "passes"] : ["episode", "episodes"];
      const learner = RL.recordings?.[cfg.runs[names[0]].recording]?.learner;
      const label = st.title || (typeof METRIC[metric].label === "function" ? METRIC[metric].label(noun, learner) : METRIC[metric].label);
      const legend = names.length > 1 ? `<div class="scene-chart-legend">${names.map((n, i) => `<span><i class="key" style="--k: var(--s${i + 1})"></i>${RL.esc(cfg.runs[n].name || n)}</span>`).join("")}</div>` : "";
      host.innerHTML = `<div class="scene-chart-title">${RL.asIs(label)}<span class="faint"></span></div>${legend}<div class="scene-chart-host"></div>`;
      chart = new RL.LineChart(host.querySelector(".scene-chart-host"), { height: 150, percent: METRIC[metric].percent, zero: METRIC[metric].zero, log: st.log ?? METRIC[metric].log, domain: st.domain || null, noun });
      // st.ref = [value, "label"]: a dashed line across the chart, such as the most a state can be worth
      const refs = st.ref ? [{ value: st.ref[0], label: st.ref[1] || "" }] : [];
      const total = cfg.average || 200, status = host.querySelector(".faint");
      const acc = cache.get(key) || { done: 0, sums: names.map(() => new Float64Array(first.units)) };
      cache.set(key, acc);
      const draw = () => {
        chart.set(names.map((n, i) => ({ name: cfg.runs[n].name || n, color: `--s${i + 1}`, values: acc.sums[i].map((v) => v / Math.max(1, acc.done)) })), refs);
        chart.playhead(first.units);
        status.textContent = acc.done < total ? ` · averaging ${acc.done} of ${total} runs…` : total === 1 ? " · one run" : ` · average of ${total} runs`;
      };
      // averaged offline (tools/story-curves.js): the same runs, the same seeds, shown at once
      const kept = RL.storyCurves?.[RL.lab.curveKey(cfg, names, metric)];
      if (kept) { // (a long curve keeps the mean of each stretch of units: the chart spreads its points over the run)
        chart.set(names.map((n, i) => ({ name: cfg.runs[n].name || n, color: `--s${i + 1}`, values: kept.curves[n].map((v) => v ?? NaN), units: first.units })), refs);
        chart.playhead(first.units);
        status.textContent = total === 1 ? " · one run" : ` · average of ${total} runs`;
        return;
      }
      if (names.every((n) => cfg.runs[n].recording)) { // trained offline: every seed's curve is in the recording
        const seeds = names.map((n) => RL.lab.recordedCurves(RL.recordings[cfg.runs[n].recording]));
        // every seed kept its training returns (and in newer recordings its test returns), DQN's and the actor-critics'
        // their Q-values; anything else comes from the seed played back
        const key = metric === "return" ? "train" : metric, all = names.every((n) => RL.recordings[cfg.runs[n].recording].curves.every((c) => c[key]));
        chart.set(names.map((n, i) => ({ name: cfg.runs[n].name || n, color: `--s${i + 1}`, values: all ? RL.lab.recordedCurves(RL.recordings[cfg.runs[n].recording], key).mean : runOf(n).metrics[metric] })), refs);
        chart.playhead(first.units);
        status.textContent = all ? ` · average of ${seeds[0].seeds.length} seeds` : " · the seed played back";
        return;
      }
      // Slices of a few milliseconds, so the page scrolls smoothly meanwhile: a run is cut where it stands and goes on
      // in the next slice (acc keeps it, with the averages, for a reader who scrolls away and back).
      let drawn = 0;
      const more = () => {
        if (!host.isConnected) return;
        if (RL.scrolling()) { timer = setTimeout(more, 120); return; } // the reader is scrolling: later
        const t0 = performance.now();
        while (acc.done < total && performance.now() - t0 < 8) {
          const c = (acc.cur ||= { i: 0, job: null });
          c.job ||= RL.lab.simulateJob(RL.lab.curveRun(cfg, names[c.i], acc.done));
          if (!c.job.step(Math.max(2, 8 - (performance.now() - t0)))) break;
          const s = acc.sums[c.i], v = c.job.result.metrics[metric];
          for (let t = 0; t < s.length; t++) s[t] += v[t];
          c.job = null;
          if (++c.i === names.length) { acc.cur = null; acc.done++; }
        }
        // drawn once a run is in (an average of none would be a flat line at zero), then at most five times a second
        if (acc.done >= total || (acc.done && performance.now() - drawn > 200)) { draw(); drawn = performance.now(); }
        if (acc.done < total) timer = setTimeout(more, 8);
      };
      more();
    };
  }

  // A run at one moment on a Lab view: what it knew at the start of unit t, and the unit before it at rest.
  function showRun(view, r, t) {
    view.env = r.env;
    view.show(r.algorithm.show(r.at(t), r.env, r.params, t), r.params);
    view.rest(t > 0 ? restOf(r, t - 1) : []);
  }

  // Checkpoints: a run at a few moments of its training, each held in turn, instead of a fast replay of every event
  // between them. spec: a list of units, or n for n moments evenly spaced up to the run's end (n = 5: every 20%).
  function checkpoints(view, r, spec, { later, note, noun, text, hold = 1800, each }) {
    const list = Array.isArray(spec) ? spec.map((t) => Math.min(t, r.units)) : Array.from({ length: spec }, (_, k) => Math.round((r.units * (k + 1)) / spec));
    let k = 0;
    const next = () => {
      const t = list[k];
      showRun(view, r, t);
      each?.(t);
      note.textContent = t === 0 ? "at the start" : text ? text(t) : `after ${t.toLocaleString("en")} ${noun(t)}`;
      if (++k < list.length) later(next, hold);
    };
    next();
  }

  // A scene built on a Lab view. A step names a run (st.run, else the first) and a moment (st.at, in units); it can
  // replay some units from there (st.play, at st.pace ms an event; st.updates stops after that many updates), step
  // through moments of the run (st.checkpoints, held st.hold ms each) and chart averaged runs (st.curves, st.metric; st.log
  // for a log axis, st.ref for a reference line, st.title for a title of its own).
  // options(st): the view's options for the step; more(st, view, run, t): anything else the scene adds.
  // test: runs recorded offline, whose units are blocks of training each followed by a test episode; a step that
  // plays nothing else plays the test episode of the network it shows, in a loop, instead of a still.
  function runScene(View, { options = () => ({}), more = null, test = false } = {}) {
    return {
      create(card, cfg, states = []) {
        card.innerHTML = `<div class="scene-view"></div>${RL.sceneKit.FORMULA}<div class="scene-chart" hidden></div><div class="scene-foot"><span class="scene-note"></span></div>`;
        const names = Object.keys(cfg.runs || {}), runOf = runs(cfg, card, restsOf(cfg, states, names[0]));
        const view = new View(card.querySelector(".scene-view"), runOf.about(names[0]).env, options({}));
        const note = card.querySelector(".scene-note"), showFormula = formula(card, cfg);
        const { later, stop } = timers(), play = player(later), chart = curves(card.querySelector(".scene-chart"), cfg, runOf);
        let wanted = null;
        const stage = {
          apply(st) {
            stop();
            wanted = st;
            const u = restUnit(cfg, st, st.run || names[0]);
            if (!runOf.ready(st.run || names[0], u)) { // still being worked out: the step comes as soon as its run does
              note.textContent = "Working out this run…";
              chart(st);
              showFormula(st);
              return runOf.when(st.run || names[0], u, () => { if (wanted === st) stage.apply(st); });
            }
            const r = runOf(st.run || names[0]), t = Math.min(st.at ?? 0, r.units);
            view.setOptions(options(st));
            showRun(view, r, t);
            note.textContent = st.note || "";
            more?.(st, view, r, t);
            const noun = r.env.unitName || (r.algorithm.unit === "sweep" ? "sweep" : "episode");
            // a recorded run's units are blocks of training steps: its checkpoints say how many steps
            const block = r.recorded ? RL.recordings?.[cfg.runs[st.run || names[0]].recording]?.block : 0;
            if (st.checkpoints) checkpoints(view, r, st.checkpoints, { later, note, hold: st.hold, noun: (n) => `${noun}${n === 1 ? "" : "s"}`, text: block ? (u) => `after ${(u * block).toLocaleString("en")} steps of training` : null, each: (u) => more?.(st, view, r, u) });
            else if (st.play) {
              play(view, r, t, st.play, { pace: st.pace, fine: !!st.fine, updates: st.updates, instant: !!st.instant, lead: st.lead, after: (u) => { if (!st.note) note.textContent = `${u.toLocaleString("en")} ${noun}${u === 1 ? "" : "s"} played`; } });
            } else if (test && t > 0 && !RL.reducedMotion()) {
              // the test episode the network shown played after block t (the still drawn above), again and again, at the
              // world's own pace (a story's human pace would make 500 steps last minutes)
              const rec = RL.recordings?.[cfg.runs[st.run || names[0]].recording], steps = rec ? t * rec.block : 0;
              if (!st.note) note.textContent = `Its test episode${steps ? ` after ${steps.toLocaleString("en")} steps of training` : ""}, played back`;
              const every = st.pace || 30, loop = () => {
                const events = r.replay(t - 1).events;
                const tick = () => {
                  for (;;) {
                    const { value: ev, done } = events.next();
                    if (done) { later(() => { showRun(view, r, t); later(loop, 500); }, 1400); return; }
                    const wait = view.event(ev, { p: r.params }) || 0;
                    if (ev.type === "move") { later(tick, every + wait); return; }
                  }
                };
                tick();
              };
              later(loop, 700);
            }
            chart(st);
            showFormula(st);
          },
          destroy() { wanted = null; stop(); view.destroy(); },
        };
        return stage;
      },
    };
  }

  RL.sceneKit = {
    timers, formula, along, svgEl, runs, restsOf, restUnit, player, curves, showRun, checkpoints, runScene,
    FORMULA: '<div class="scene-formula"><div class="f-sym"></div><div class="f-num"></div></div>',
    ACTION: { up: 0, right: 1, down: 2, left: 3 },
    sub: (n) => String(n).replace(/\d/g, (d) => SUB[d]),
    signed: (v, digits = 0) => `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(v).toFixed(digits)}`,
  };
})(globalThis.RL = globalThis.RL || {});
