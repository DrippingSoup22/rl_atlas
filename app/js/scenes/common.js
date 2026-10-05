/* Story scenes: the pictures that follow a story's text. A scene turns each step's small state into a picture;
   states are complete descriptions, so scrolling back shows the same picture again. This file holds what the
   scenes share; each scene lives in its own file (grid, loop, timeline, mdp, bandit, chain, cards, graph). */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const SUB = "₀₁₂₃₄₅₆₇₈₉";
  const SHOWN = new Set(["update", "improve", "trace", "plan", "world"]); // events after which a view redraws what the algorithm knows
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
    return (st) => {
      const n = st.formula || 0;
      pieces.forEach((p) => p.classList.toggle("shown", +p.dataset.step <= n));
      sym.classList.toggle("on", n > 0);
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

  // Named runs a story shows and replays, from its config: [story.runs] name = { algorithm, units, seed, <knobs> }, or
  // name = { recording } for a run trained offline (recorder/record.py). The world, the shared knobs and the seed come
  // from the story config unless a run says otherwise.
  function runs(cfg) {
    const made = {};
    return (name) => {
      if (made[name]) return made[name];
      const spec = cfg.runs?.[name];
      if (!spec) throw new Error(`story: no run named '${name}'`);
      if (spec.recording) return (made[name] = RL.lab.recordedRun(RL.recordings[spec.recording]));
      const { algorithm, units, seed, world, measures, name: _label, ...params } = spec;
      return (made[name] = RL.lab.simulate({
        world: world || cfg.world || cfg.env, algorithm: RL.lab.algorithms[algorithm], params: { ...cfg.params, ...params },
        units: units || cfg.units || 100, seed: seed ?? cfg.seed ?? 1, measures,
      }));
    };
  }

  // Replay units of a run on a view, event by event, the way the Lab walks: the agent moves, values update.
  // With updates = n, only the first n updates of the unit are played, and the replay stops there.
  function player(later) {
    return function play(view, run, from, count, { pace = 240, fine = false, after, updates = 0 } = {}) {
      let t = from, w = null, seen = 0;
      const end = Math.min(run.units, from + count);
      const next = () => {
        if (!w) {
          if (t >= end) { after?.(t); return; }
          w = run.replay(t);
        }
        const { value: ev, done } = w.events.next();
        if (done) { w = null; t++; after?.(t, true); later(next, 700); return; }
        if (SHOWN.has(ev.type)) view.show(run.algorithm.show(w.m, run.env, run.params, t), run.params);
        const wait = view.event(ev, { line: fine, p: run.params }) || 0;
        if (updates && ev.type === "update" && ++seen >= updates) return;
        const quiet = ev.type === "info" || ev.type === "next" || ev.type === "skip";
        later(next, quiet ? 40 : ev.type === "choose" ? pace / 2 : pace + wait);
      };
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
    aim: { label: "Where the policy aims: its mean angle (degrees)" },
    test: { label: "Test episode after each block: its return" },
    q: { label: "The largest Q-value in each batch, on average" },
  };
  function curves(host, cfg, runOf) {
    const cache = new Map();
    let chart = null, shown = "", timer = 0;
    return function show(st) {
      const names = st.curves || [], metric = st.metric || "optimal", key = `${names}|${metric}|${st.domain || ""}`;
      host.hidden = !names.length;
      if (!names.length || key === shown) return;
      shown = key;
      clearTimeout(timer);
      chart?.destroy();
      const first = runOf(names[0]), unit = first.env.unitName || first.algorithm.unit;
      const noun = unit === "pull" || unit === "step" ? ["step", "steps"] : unit === "hand" ? ["hand", "hands"] : unit === "round" ? ["round", "rounds"] : unit === "throw" ? ["throw", "throws"] : unit === "block" ? ["block", "blocks"] : ["episode", "episodes"];
      const label = typeof METRIC[metric].label === "function" ? METRIC[metric].label(noun) : METRIC[metric].label;
      const legend = names.length > 1 ? `<div class="scene-chart-legend">${names.map((n, i) => `<span><i class="key" style="--k: var(--s${i + 1})"></i>${RL.esc(cfg.runs[n].name || n)}</span>`).join("")}</div>` : "";
      host.innerHTML = `<div class="scene-chart-title">${label}<span class="faint"></span></div>${legend}<div class="scene-chart-host"></div>`;
      chart = new RL.LineChart(host.querySelector(".scene-chart-host"), { height: 150, percent: METRIC[metric].percent, zero: METRIC[metric].zero, log: METRIC[metric].log, domain: st.domain || null, noun });
      const total = cfg.average || 200, status = host.querySelector(".faint");
      const acc = cache.get(key) || { done: 0, sums: names.map(() => new Float64Array(first.units)) };
      cache.set(key, acc);
      const draw = () => {
        chart.set(names.map((n, i) => ({ name: cfg.runs[n].name || n, color: `--s${i + 1}`, values: acc.sums[i].map((v) => v / Math.max(1, acc.done)) })));
        chart.playhead(first.units);
        status.textContent = acc.done < total ? ` · averaging ${acc.done} of ${total} runs…` : ` · average of ${total} runs`;
      };
      if (names.every((n) => cfg.runs[n].recording)) { // trained offline: every seed's curve is in the recording
        const seeds = names.map((n) => RL.lab.recordedCurves(RL.recordings[cfg.runs[n].recording]));
        // every seed kept its training returns (and in newer recordings its test returns), DQN's and the actor-critics'
        // their Q-values; anything else comes from the seed played back
        const key = metric === "return" ? "train" : metric, all = names.every((n) => RL.recordings[cfg.runs[n].recording].curves.every((c) => c[key]));
        chart.set(names.map((n, i) => ({ name: cfg.runs[n].name || n, color: `--s${i + 1}`, values: all ? RL.lab.recordedCurves(RL.recordings[cfg.runs[n].recording], key).mean : runOf(n).metrics[metric] })));
        chart.playhead(first.units);
        status.textContent = all ? ` · average of ${seeds[0].seeds.length} seeds` : " · the seed played back";
        return;
      }
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (acc.done < total && performance.now() - t0 < 25) {
          names.forEach((n, i) => {
            const spec = cfg.runs[n], { algorithm, units, seed, world, measures, name: _label, ...params } = spec;
            const { metrics } = RL.lab.simulate({ world: world || cfg.world || cfg.env, algorithm: RL.lab.algorithms[algorithm], params: { ...cfg.params, ...params }, units: units || cfg.units, seed: 1000 + acc.done, snapshots: false, measures });
            const s = acc.sums[i], v = metrics[metric];
            for (let t = 0; t < s.length; t++) s[t] += v[t];
          });
          acc.done++;
        }
        draw();
        if (acc.done < total) timer = setTimeout(more, 16);
      };
      more();
    };
  }

  // A run at one moment on a Lab view: what it knew at the start of unit t, and the unit before it at rest.
  function showRun(view, r, t) {
    view.env = r.env;
    view.show(r.algorithm.show(r.at(t), r.env, r.params, t), r.params);
    view.rest(t > 0 ? [...r.replay(t - 1).events] : []);
  }

  // A scene built on a Lab view. A step names a run (st.run, else the first) and a moment (st.at, in units); it can
  // replay some units from there (st.play, at st.pace ms an event; st.updates stops after that many updates) and chart
  // averaged runs (st.curves, st.metric).
  // options(st): the view's options for the step; more(st, view, run, t): anything else the scene adds.
  function runScene(View, { options = () => ({}), more = null } = {}) {
    return {
      create(card, cfg) {
        card.innerHTML = `<div class="scene-view"></div>${RL.sceneKit.FORMULA}<div class="scene-chart" hidden></div><div class="scene-foot"><span class="scene-note"></span></div>`;
        const runOf = runs(cfg), names = Object.keys(cfg.runs || {});
        const view = new View(card.querySelector(".scene-view"), runOf(names[0]).env, options({}));
        const note = card.querySelector(".scene-note"), showFormula = formula(card, cfg);
        const { later, stop } = timers(), play = player(later), chart = curves(card.querySelector(".scene-chart"), cfg, runOf);
        return {
          apply(st) {
            stop();
            const r = runOf(st.run || names[0]), t = Math.min(st.at ?? 0, r.units);
            view.setOptions(options(st));
            showRun(view, r, t);
            note.textContent = st.note || "";
            more?.(st, view, r, t);
            if (st.play) {
              const noun = r.env.unitName || (r.algorithm.unit === "sweep" ? "sweep" : "episode");
              play(view, r, t, st.play, { pace: st.pace, fine: !!st.fine, updates: st.updates, after: (u) => { if (!st.note) note.textContent = `${u.toLocaleString("en")} ${noun}${u === 1 ? "" : "s"} played`; } });
            }
            chart(st);
            showFormula(st);
          },
          destroy() { stop(); view.destroy(); },
        };
      },
    };
  }

  RL.sceneKit = {
    timers, formula, along, svgEl, runs, player, curves, showRun, runScene,
    FORMULA: '<div class="scene-formula"><div class="f-sym"></div><div class="f-num"></div></div>',
    ACTION: { up: 0, right: 1, down: 2, left: 3 },
    sub: (n) => String(n).replace(/\d/g, (d) => SUB[d]),
    signed: (v, digits = 0) => `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(v).toFixed(digits)}`,
  };
})(globalThis.RL = globalThis.RL || {});
