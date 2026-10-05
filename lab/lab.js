/* The Lab: one or more algorithms race on a world. Each run is computed at once and then played like a video:
   walking speeds replay the current unit (an episode, a sweep or a pull) line by line, faster speeds jump between
   saved snapshots. The views, the pseudocode, the live formula, the filmstrip and the charts follow one playhead.
   Charts can average many runs, computed in the background; a sandbox lets you draw the world yourself. */
(function (RL) {
  "use strict";
  const { esc, lab } = RL;

  const ICON = {
    play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/></svg>',
    pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor"/></svg>',
    step: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5.5 5.5v13l9-6.5zM15.5 5H18v14h-2.5z" fill="currentColor"/></svg>',
    restart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h2.5v14H6zM18.5 5.5v13l-9-6.5z" fill="currentColor"/></svg>',
    dice: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="4" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="9" cy="9" r="1.6" fill="currentColor"/><circle cx="15" cy="15" r="1.6" fill="currentColor"/><circle cx="15" cy="9" r="1.6" fill="currentColor"/><circle cx="9" cy="15" r="1.6" fill="currentColor"/></svg>',
  };

  // What one unit is called, in this world and for this algorithm.
  const NOUNS = { episode: ["episode", "episodes"], sweep: ["sweep", "sweeps"], step: ["step", "steps"], pull: ["pull", "pulls"], hand: ["hand", "hands"] };
  // Walking speeds replay a unit event by event ("step" stops at each update); rates jump between snapshots.
  const walking = (step, every) => [{ id: "line", label: "Line by line", every: every[0] }, { id: "step", label: step, every: every[1] }];
  const rates = (...list) => list.map((rate) => ({ id: `r${rate}`, rate }));
  const SPEEDS = {
    episode: [...walking("Step by step", [560, 180]), ...rates(1, 10, 50)],
    sweep: [...walking("State by state", [380, 130]), ...rates(1, 4, 20)],
    step: [...walking("Pull by pull", [480, 300]), ...rates(10, 50, 250)],
  };
  // The knobs a preset can show as sliders. alpha = 0 means sample averages (1/n), where an algorithm allows it.
  const KNOBS = {
    alpha: { sym: "α", name: "step size", min: 0.01, max: 1, step: 0.01 },
    epsilon: { sym: "ε", name: "exploration", min: 0, max: 0.5, step: 0.01 },
    gamma: { sym: "γ", name: "discount", min: 0.5, max: 1, step: 0.01 },
    c: { sym: "c", name: "confidence", min: 0, max: 5, step: 0.1 },
    q0: { sym: "Q₁", name: "first estimate", min: -2, max: 10, step: 0.5 },
    theta: { sym: "θ", name: "tolerance", choices: [0.1, 0.01, 0.001, 0.0001] },
    n: { sym: "n", name: "steps ahead", choices: [1, 2, 3, 4, 8, 16, 32, 64] },
    lambda: { sym: "λ", name: "trace decay", min: 0, max: 1, step: 0.01 },
    planning: { sym: "n", name: "planning steps", choices: [0, 1, 5, 10, 20, 50, 100] },
    kappa: { sym: "κ", name: "exploration bonus", choices: [0, 0.0001, 0.001, 0.01] },
  };
  const AVERAGING = new Set(["epsilon-greedy", "optimistic-init", "ucb", "mc-prediction", "exploring-starts", "mc-control"]);
  // What a chart can plot.
  const METRICS = {
    return: { title: (n) => `Reward per ${n}`, smooth: 10 },
    steps: { title: (n) => `Steps per ${n}`, smooth: 10, log: true }, // a first episode of 1000 steps would flatten the rest
    optimal: { title: () => "How often the best arm is pulled", percent: true },
    left: { title: () => "How often the agent goes left from A", percent: true },
    delta: { title: () => "Largest change in a sweep", log: true },
    error: { title: () => "Error: distance from the true values (RMS)", zero: true },
    "optimal-error": { title: () => "Distance from the optimal values (RMS)", log: true },
    match: { title: () => "States where the greedy action is optimal", percent: true },
    greedy: { title: (n, env) => (env.ice ? "Chance the greedy policy reaches the gem" : "Return of the greedy policy"), percent: true },
  };
  const LADDER = [10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 200000, 500000];
  const signed = (v, d = 2) => (v < 0 ? "−" : v > 0 ? "+" : "") + Math.abs(v).toFixed(d);
  // Events after which the views redraw what the algorithm knows.
  const SHOWN = new Set(["update", "improve", "trace", "plan", "world"]);
  const plural = (n, [one, many]) => `${n.toLocaleString("en")} ${n === 1 ? one : many}`;

  RL.views.lab = function (host, presetId) {
    const preset = RL.content.presets[presetId];
    if (!preset) {
      host.innerHTML = '<section class="lab-missing card"><h1>There is no such lab yet</h1><p><a href="#/">Back to the map</a></p></section>';
      return {};
    }
    const sandbox = preset.env === "sandbox" ? RL.sandbox(sandboxHost()) : null;
    const make = () => (sandbox ? lab.grid(sandbox.spec()) : lab.make(preset.env));
    let env = make();
    let racers = preset.racers.map((r) => ({ ...r, algorithm: lab.algorithms[r.algorithm] }));
    const unitOf = () => racers[0].algorithm.unit;
    const noun = () => NOUNS[env.unitName || (unitOf() === "step" ? "pull" : unitOf())];
    // Shared knobs: the preset's, minus those every racer sets for itself.
    const knobs = { units: preset.units, seed: preset.seed, runs: preset.runs, ...preset.params };
    const shown = () => Object.keys(KNOBS).filter((k) => k in preset.params && racers.some((r) => !(k in r.params)));
    const paramsOf = (r) => ({ ...preset.params, ...pick(knobs, Object.keys(KNOBS)), ...r.params });
    const view = { seeds: false, show: {} };
    const from = RL.app?.from?.name === "entry" ? racers.findIndex((r) => r.algorithm.id === RL.app.from.id) : -1;
    const P = { e: 0, playing: false, speed: "step", acc: 0, wait: 0, walkers: null, focus: Math.max(0, from >= 0 ? from : racers.length - 1) };
    let runs = [], stages = [], views = [], job = null, film = null, charts = [];

    host.innerHTML = `
      <section class="lab" data-kind="${env.kind}">
        <header class="lab-head">
          <div class="lab-title">
            <div class="eyebrow">Lab · <span class="world-name">${esc(env.title)}</span></div>
            <h1>${esc(preset.title)}</h1>
            <p class="intro">${esc(preset.intro || "")}</p>
          </div>
          <div class="knobs card"></div>
        </header>
        ${sandbox ? '<section class="sandbox card"></section>' : ""}
        <div class="lab-main">
          <div class="lab-flow">
            <div class="stages"></div>
            <div class="transport card">
              <button class="icon-btn restart" type="button" aria-label="Back to the start">${ICON.restart}</button>
              <button class="icon-btn big play" type="button" aria-label="Play">${ICON.play}</button>
              <button class="icon-btn step" type="button" aria-label="One step forward">${ICON.step}</button>
              <select class="speed" aria-label="Speed"></select>
              <input class="scrub" type="range" min="0" value="0" aria-label="Position in the run">
              <span class="pos"></span>
            </div>
            <section class="film card" hidden><div class="film-head"><h3>Filmstrip · <span class="film-name as-is"></span></h3><span class="faint">click a frame to jump there</span></div><div class="film-host"></div></section>
            <section class="chart-card card">
              <div class="chart-head">
                <div class="legend"></div>
                <div class="chart-mode"></div>
              </div>
              <div class="chart-grid"></div>
              <p class="faint chart-note"></p>
              <p class="summary"></p>
            </section>
          </div>
          <aside class="lab-side">
            <section class="panel card"><h3>Pseudocode · <span class="algo-name as-is"></span></h3><div class="pseudo-host"></div></section>
            <section class="panel card live">
              <h3>This step</h3>
              <div class="live-sym"></div>
              <div class="live-num"></div>
              <p class="live-note faint">Play <b>line by line</b> or <b class="walk-name">step by step</b> to see every update with its numbers.</p>
            </section>
            <section class="panel card show-panel"><h3>Show</h3><div class="show-host"></div></section>
          </aside>
        </div>
      </section>`;

    const q = (sel) => host.querySelector(sel);
    const playBtn = q(".play"), scrub = q(".scrub"), pos = q(".pos"), pseudo = q(".pseudo-host");
    const liveSym = q(".live-sym"), liveNum = q(".live-num"), liveNote = q(".live-note");

    // ---- building the page around the current world and racers ----
    function knobPanel() {
      const box = q(".knobs");
      const slider = (k) => {
        const d = KNOBS[k], v = knobs[k];
        if (d.choices) return `<label class="knob"><span class="sym">${d.sym}</span><span class="name">${d.name}</span>
          <select data-knob="${k}">${d.choices.map((c) => `<option value="${c}"${c === v ? " selected" : ""}>${c}</option>`).join("")}</select></label>`;
        const min = k === "alpha" && racers.every((r) => AVERAGING.has(r.algorithm.id)) ? 0 : d.min;
        return `<label class="knob"><span class="sym">${d.sym}</span><span class="name">${d.name}</span>
          <input type="range" data-knob="${k}" min="${min}" max="${d.max}" step="${d.step}" value="${v}"><output>${fmtKnob(k, v)}</output></label>`;
      };
      const options = LADDER.filter((n) => n >= preset.units / 10 && n <= preset.units * 5);
      if (!options.includes(knobs.units)) options.push(knobs.units);
      options.sort((a, b) => a - b);
      box.innerHTML = shown().map(slider).join("") +
        `<label class="knob"><span class="name">${noun()[1]}</span><select data-knob="units">${options.map((n) => `<option value="${n}"${n === knobs.units ? " selected" : ""}>${n.toLocaleString("en")}</option>`).join("")}</select></label>` +
        (preset.runs > 1 ? `<label class="knob"><span class="name">averaged over</span><select data-knob="runs">${[...new Set([10, 100, preset.runs, 500, 2000])].sort((a, b) => a - b).map((n) => `<option value="${n}"${n === knobs.runs ? " selected" : ""}>${n} runs</option>`).join("")}</select></label>` : "") +
        `<button class="pill seed" type="button" title="Run again with new randomness">${ICON.dice}<span>Seed <b>${knobs.seed}</b></span></button>`;
    }
    const fmtKnob = (k, v) => (k === "alpha" && v === 0 ? "1/n" : String(v));

    function build() {
      env = make();
      runs = [];
      q(".lab").dataset.kind = env.kind;
      q(".world-name").textContent = env.title;
      views.forEach((v) => v.destroy());
      const View = RL.labViews[env.kind];
      const box = q(".stages");
      box.className = `stages n${racers.length}`;
      box.dataset.shape = env.kind === "grid" && env.cols / env.rows < 2 ? "boxy" : ""; // boxy grids sit side by side
      box.innerHTML = racers.map((r, i) => `
        <figure class="stage card" data-i="${i}">
          <figcaption><i class="key" style="--k: var(--s${i + 1})"></i><b>${esc(r.name)}</b><span class="stat"></span></figcaption>
          <div class="view-host"></div>
        </figure>`).join("");
      stages = Array.from(box.querySelectorAll(".stage"));
      view.show.agent = unitOf() !== "sweep"; // dynamic programming has no agent walking about
      views = stages.map((st) => new View(st.querySelector(".view-host"), env, view.show));
      stages.forEach((st, i) => st.addEventListener("click", () => focus(i)));
      const displays = racers.map((r) => {
        const p = paramsOf(r), { m } = lab.memory(r.algorithm.memory(env, p));
        return { ...r.algorithm.show(m, env, p, 0), sweeps: r.algorithm.unit === "sweep" };
      });
      const opts = View.options ? View.options(env, displays) : [];
      for (const o of opts) if (!(o.key in view.show)) view.show[o.key] = o.value;
      q(".show-panel").hidden = !opts.length;
      q(".show-host").innerHTML = opts.map((o) => o.type === "seg"
        ? `<div class="seg" role="group" aria-label="${esc(o.label)}">${o.choices.map(([v, text]) => `<button type="button" data-opt="${o.key}" data-v="${v}" class="${view.show[o.key] === v ? "on" : ""}">${esc(text)}</button>`).join("")}</div>`
        : `<label class="check"><input type="checkbox" data-opt="${o.key}"${view.show[o.key] ? " checked" : ""}> ${esc(o.label)}${o.swatch ? ` <i class="swatch ${o.swatch}"></i>` : ""}</label>`).join("") +
        (View.legend ? View.legend(env) : "");
      views.forEach((v) => v.setOptions(view.show));
      const speeds = SPEEDS[unitOf()];
      if (!speeds.some((s) => s.id === P.speed)) P.speed = "step";
      q(".speed").innerHTML = speeds.map((s) => `<option value="${s.id}"${s.id === P.speed ? " selected" : ""}>${s.rate ? `${s.rate} ${noun()[s.rate === 1 ? 0 : 1]} / s` : s.label}</option>`).join("");
      q(".walk-name").textContent = speeds[1].label.toLowerCase();
      q(".legend").innerHTML = racers.map((r, i) => `<span><i class="key" style="--k: var(--s${i + 1})"></i>${esc(r.name)}</span>`).join("");
      knobPanel();
      chartsFor();
      focus(Math.min(P.focus, racers.length - 1));
      simulate();
    }

    // ---- the runs ----
    function simulate() {
      job?.cancel();
      P.walkers = null;
      runs = racers.map((r) => lab.simulate({ world: make, algorithm: r.algorithm, params: paramsOf(r), units: knobs.units, seed: knobs.seed, measures: preset.measures }));
      views.forEach((v, i) => { v.env = runs[i].env; }); // each view looks at its own run's world (its bandit's machines, its cards)
      scrub.max = knobs.units;
      filmstrip();
      drawCharts();
      summary();
      seek(Math.min(P.e, knobs.units));
      if (knobs.runs > 1 || view.seeds) average();
    }

    // Many runs in the background, a few at a time, so the page stays responsive; the charts fill in as they come.
    function average() {
      const n = knobs.runs > 1 ? knobs.runs : 10, seeds = Array.from({ length: n }, (_, k) => knobs.seed + k);
      const sums = racers.map(() => ({})), lines = racers.map(() => []);
      let done = 0, cancelled = false;
      const note = q(".chart-note");
      const chunk = () => {
        if (cancelled) return;
        const t0 = performance.now();
        while (done < n && performance.now() - t0 < 14) {
          racers.forEach((r, i) => {
            const { metrics } = lab.simulate({ world: make, algorithm: r.algorithm, params: paramsOf(r), units: knobs.units, seed: seeds[done], snapshots: false, measures: preset.measures });
            for (const k of preset.charts) {
              const s = (sums[i][k] ||= new Float64Array(knobs.units));
              for (let t = 0; t < knobs.units; t++) s[t] += metrics[k][t];
            }
            if (knobs.runs <= 1) lines[i].push(metrics);
          });
          done++;
        }
        drawCharts({ sums, lines, done });
        note.textContent = done < n ? `Running ${done} of ${n} runs…` : chartNote(n);
        if (done < n) timer = setTimeout(chunk, 0);
      };
      let timer = setTimeout(chunk, 30);
      job = { cancel() { cancelled = true; clearTimeout(timer); } };
    }

    // ---- charts ----
    function chartsFor() {
      charts.forEach((c) => c.destroy());
      const grid = q(".chart-grid");
      grid.innerHTML = preset.charts.map((k) => `<div class="chart-box"><h3>${esc(METRICS[k].title(noun()[0], env))}</h3><div class="chart-host"></div></div>`).join("");
      grid.className = `chart-grid n${preset.charts.length}`;
      charts = preset.charts.map((k, j) => new RL.LineChart(grid.querySelectorAll(".chart-host")[j], {
        label: METRICS[k].title(noun()[0], env), percent: METRICS[k].percent, log: METRICS[k].log, zero: METRICS[k].zero,
        domain: k === "return" ? preset.params.domain : null, noun: noun(), logX: knobs.units > 20000,
        onSeek: (t) => { pause(); seek(t); },
      }));
      const mode = q(".chart-mode");
      mode.innerHTML = preset.runs > 1 ? "" : `<div class="seg" role="group" aria-label="How many runs the charts show">
        <button type="button" data-seeds="0" class="${view.seeds ? "" : "on"}">This run</button><button type="button" data-seeds="1" class="${view.seeds ? "on" : ""}">10 seeds</button></div>`;
    }

    function chartNote(n) {
      if (knobs.runs > 1) return `Each line is the average of ${n} runs, each with its own seed; the views above play the run with seed ${knobs.seed}. Click a chart to jump there.`;
      if (view.seeds) return `Thin lines: 10 runs with seeds ${knobs.seed} to ${knobs.seed + 9}, the same settings each time. Thick lines: their average. Click a chart to jump there.`;
      const smooth = preset.charts.some((k) => METRICS[k].smooth);
      return `${smooth ? "Smoothed over 10 " + noun()[1] + ". " : ""}Click a chart to jump there.`;
    }

    function drawCharts(avg) {
      const many = knobs.runs > 1, n = avg ? avg.done : 0;
      preset.charts.forEach((k, j) => {
        const m = METRICS[k], series = [];
        racers.forEach((r, i) => {
          const color = `--s${i + 1}`;
          if (many) {
            if (n) series.push({ name: r.name, color, values: avg.sums[i][k].map((v) => v / n), smooth: k === "return" && unitOf() === "episode" ? 5 : 1 });
          } else if (view.seeds && avg) {
            for (const metrics of avg.lines[i]) series.push({ name: r.name, color, values: metrics[k], smooth: m.smooth, faint: true });
            if (n) series.push({ name: `${r.name}, average`, color, values: avg.sums[i][k].map((v) => v / n), smooth: m.smooth });
          } else series.push({ name: r.name, color, values: runs[i].metrics[k], smooth: m.smooth });
        });
        charts[j].set(series, references(k));
      });
      if (!avg) q(".chart-note").textContent = many || view.seeds ? "Starting the runs…" : chartNote(1);
    }

    // Reference lines: the best a chart can reach, where it is known exactly.
    function references(k) {
      const p = paramsOf(racers[0]);
      if (k === "left") return [{ value: p.epsilon / 2, label: `best possible with ε = ${p.epsilon}: ${(50 * p.epsilon).toFixed(0)}%` }];
      if (k === "greedy" && env.model) {
        const best = lab.evaluate(env, lab.greedyPolicy(env, lab.optimalValues(env, p), p.gamma, 1e-6), p.judge ?? p.gamma, { theta: 1e-8 })[env.start];
        return [{ value: best, label: `the optimal policy: ${(100 * best).toFixed(0)}%` }];
      }
      if (k === "match" || k === "optimal") return [{ value: 1, label: "" }];
      return [];
    }

    // One sentence on where each run ended up.
    function summary() {
      const out = [], last = Math.min(100, Math.max(1, Math.floor(knobs.units / 2))); // the second half at most: early episodes are a search
      const each = (f) => runs.map((r, i) => `<b>${esc(racers[i].name)}</b> ${f(r, i)}`).join(" · ");
      const pct = (v) => `${Math.round(100 * v)}%`;
      if (env.kind === "bandit") {
        out.push(`This run's last ${last} pulls: ${each((r) => `${lab.mean(r.metrics.return, knobs.units - last).toFixed(2)} per pull, a best arm ${pct(lab.mean(r.metrics.optimal, knobs.units - last))} of the time`)}.`);
      }
      if (env.kind === "graph") out.push(`Went left from A in this run: ${each((r) => `${pct(lab.mean(r.metrics.left))} of the episodes`)}.`);
      if (env.kind === "blackjack" && unitOf() === "episode") out.push(`Hands won in this run: ${each((r) => pct(r.metrics.return.reduce((n, g) => n + (g > 0), 0) / knobs.units))}.`);
      if (env.kind === "grid" && unitOf() === "episode") {
        if (env.ice) out.push(`Reached the gem in the last ${last} episodes: ${each((r) => pct(lab.mean(r.metrics.return, knobs.units - last)))}.`);
        else if (preset.charts.includes("steps")) out.push(`${knobs.runs > 1 ? `In the run with seed ${knobs.seed}, average` : "Average"} steps per episode over the last ${last}: ${each((r) => lab.mean(r.metrics.steps, knobs.units - last).toFixed(1))}.`);
        else out.push(`Average reward per episode over the last ${last}: ${each((r) => signed(lab.mean(r.metrics.return, knobs.units - last), 0))}.`);
        const finals = runs.map((r) => r.algorithm.show(r.at(knobs.units), r.env, r.params));
        if (finals[0].t !== undefined) env.setTime?.(finals[0].t); // a maze whose walls moved: follow the final layout
        if (!env.slip && finals.every((d) => d.Q)) {
          out.push(`Greedy path at the end: ${each((r, i) => { const g = lab.greedyPath(env, finals[i].Q); if (g.reached) return plural(g.path.length - 1, ["step", "steps"]);
            // Still reaching the goal while learning, yet no greedy path: the greedy moves go round in circles
            // (Dyna-Q+'s bonuses can do this: its values include the pull of moves not tried in a while).
            return lab.mean(r.metrics.steps, Math.max(0, knobs.units - 10)) < 200 ? "none, its greedy moves go round in circles" : "none yet"; })}.`);
        }
      }
      if (unitOf() === "sweep") {
        out.push(`Settled (largest change below θ = ${knobs.theta}) after: ${each((r) => { const t = r.metrics.converged.findIndex((c) => c); return t < 0 ? "not yet" : plural(t + 1, noun()); })}.`);
      }
      for (const k of preset.measures) {
        const f = METRICS[k].percent ? (v) => `${(100 * v).toFixed(0)}%` : (v) => v.toFixed(3);
        out.push(`${METRICS[k].title(noun()[0], env)}, at the end: ${each((r) => f(r.metrics[k][knobs.units - 1]))}.`);
      }
      q(".summary").innerHTML = out.join("<br>");
    }

    // ---- the filmstrip: what the focused racer knew at a few moments ----
    function filmstrip() {
      const View = RL.labViews[env.kind], frames = [...new Set((preset.film || []).map((t) => Math.min(t, knobs.units)))];
      q(".film").hidden = !frames.length || !View.thumb;
      if (q(".film").hidden) return;
      const r = runs[P.focus], a = racers[P.focus].algorithm, p = paramsOf(racers[P.focus]);
      q(".film-name").textContent = racers[P.focus].name;
      film ||= new RL.Film(q(".film-host"), { onSeek: (t) => { pause(); seek(t); } });
      film.set(frames.map((t) => ({ t, label: t === 0 ? "at the start" : `after ${plural(t, noun())}`, html: View.thumb(env, a.show(r.at(t), env, p, t), p) })));
    }

    // ---- moving the playhead: t is the start of unit t; t = units is the end of the run ----
    function seek(t, draw = false) {
      P.e = Math.max(0, Math.min(knobs.units, t));
      P.walkers = null;
      P.acc = 0;
      P.wait = 0;
      runs.forEach((r, i) => {
        const p = paramsOf(racers[i]);
        views[i].show(racers[i].algorithm.show(r.at(P.e), r.env, p, P.e), p);
        views[i].rest(P.e > 0 ? [...r.replay(P.e - 1).events] : [], draw);
        caption(i);
      });
      mark(null);
      position();
      if (P.e >= knobs.units) pause();
    }

    function position() {
      charts.forEach((c) => c.playhead(P.e));
      film?.at(P.e);
      scrub.value = P.e;
      const [one] = noun();
      pos.textContent = P.walkers ? `${cap(one)} ${(P.e + 1).toLocaleString("en")} of ${knobs.units.toLocaleString("en")}` : `${P.e.toLocaleString("en")} of ${plural(knobs.units, noun())} played`;
    }
    const cap = (s) => s[0].toUpperCase() + s.slice(1);

    // The line under a racer's name: how its last unit went, or how the current one is going.
    function caption(i, w) {
      const stat = stages[i].querySelector(".stat"), r = runs[i], [one] = noun();
      if (w) { stat.textContent = `${cap(one)} ${P.e + 1}: ${live(w)}`; return; }
      if (!P.e) { stat.textContent = "Ready: nothing learned yet"; return; }
      const t = P.e - 1, M = r.metrics;
      let text;
      if (unitOf() === "sweep") text = M.improved?.[t] ? (M.changed[t] ? `improvement: ${plural(M.changed[t], ["state", "states"])} changed` : "improvement: nothing changed, done") : M.converged[t] && racers[i].algorithm.id === "policy-iteration" ? "done" : `largest change ${M.delta[t].toFixed(3)}`;
      else if (env.kind === "bandit") text = `reward ${signed(M.return[t])}${M.optimal[t] ? " · a best arm" : ""}`;
      else if (env.kind === "blackjack") text = M.return[t] > 0 ? "won (+1)" : M.return[t] < 0 ? "lost (−1)" : "a draw (0)";
      else if (env.kind === "graph") text = `${M.left?.[t] ? "went left" : "went right"} · return ${signed(M.return[t])}`;
      else text = `${signed(M.return[t], 0)} · ${plural(M.steps[t], ["step", "steps"])}${M.falls[t] ? ` · ${plural(M.falls[t], ["fall", "falls"])}` : ""}`;
      stat.textContent = `${cap(one)} ${P.e}: ${text}`;
    }
    const live = (w) => (unitOf() === "sweep" ? `state ${w.n}` : env.kind === "bandit" ? "pulling" : `${signed(w.G, env.kind === "grid" ? 0 : 2)} so far · ${plural(w.n, ["step", "steps"])}`);

    // ---- walking through a unit, event by event ----
    function walk(toUpdate) {
      if (!P.walkers) {
        if (P.e >= knobs.units) { pause(); return; }
        P.walkers = runs.map((r, i) => ({ ...r.replay(P.e), done: false, G: 0, n: 0, p: paramsOf(racers[i]) }));
      }
      let wait = 0;
      P.walkers.forEach((w, i) => {
        while (!w.done) {
          const { value: ev, done } = w.events.next();
          if (done) { w.done = true; break; }
          wait = Math.max(wait, on(i, w, ev));
          if (!toUpdate || ev.type === "update" || ev.type === "improve" || ev.type === "plan") break;
        }
      });
      P.wait = wait;
      if (P.walkers.every((w) => w.done)) {
        P.e += 1;
        P.walkers = null;
        P.wait = Math.max(P.wait, unitOf() === "episode" ? 700 : 250);
        runs.forEach((_, i) => caption(i));
        if (P.e >= knobs.units) pause();
      }
      position();
    }

    // Show one event of racer i. Returns how long to pause after it (a fall, a hand being dealt…).
    function on(i, w, ev) {
      const a = racers[i].algorithm;
      if (ev.type === "move") { w.G += ev.r; w.n += 1; }
      if (ev.type === "sweep") w.n += 1;
      if (SHOWN.has(ev.type)) views[i].show(a.show(w.m, runs[i].env, w.p, P.e), w.p);
      const pauseFor = views[i].event(ev, { line: P.speed === "line", p: w.p }) || 0;
      if (i === P.focus) {
        mark(ev.line);
        if (ev.type === "update") explain(ev, w.p);
        else if (ev.type === "plan") planned(ev);
      }
      caption(i, w);
      return pauseFor;
    }

    // Planning updates have no single formula: say how many there were and how much they changed.
    function planned(ev) {
      const n = ev.list.length;
      let big = 0;
      for (const u of ev.list) if (Math.abs(u.delta) > Math.abs(big)) big = u.delta;
      liveNote.innerHTML = !n ? "Nothing in the queue is worth an update: no planning this step"
        : `<b>${plural(n, ["planning update", "planning updates"])}</b> on remembered moves${ev.ordered ? `, the most urgent first; ${plural(ev.left, ["pair waits", "pairs wait"])} in the queue` : ", picked at random"} · the largest surprise was <b class="q-err">${signed(big, 3)}</b>${ev.list.some((u) => u.bonus > 1e-9) ? " · bonuses for moves not tried in a while included" : ""}`;
      liveNote.classList.remove("faint");
    }

    // ---- the live formula ----
    function explain(ev, p) {
      const a = racers[P.focus].algorithm;
      if (!a.numbers) return;
      liveNum.innerHTML = RL.math.tex(a.numbers(ev, p), true);
      const where = env.describe ? env.describe(ev.s, ev.a ?? -1) : "";
      let note;
      if (a.note) note = a.note(ev, where, signed, p);
      else if (ev.baseline !== undefined) note = `Reward <b>${signed(ev.r)}</b>, baseline <b>${signed(ev.baseline)}</b>: the difference <b class="q-err">${signed(ev.delta)}</b> pushes ${where}'s preference ${ev.delta >= 0 ? "up" : "down"}, and every other arm's the other way`;
      else if (unitOf() === "sweep") note = `${cap(where)}: from <b>${signed(ev.old)}</b> to <b>${signed(ev.value)}</b>`;
      else if (ev.target !== undefined && ev.n !== undefined) note = `${a.unit === "step" ? "Reward" : "Return"} <b>${signed(ev.target)}</b> · surprise <b class="q-err">${signed(ev.delta)}</b> · visit ${ev.n} · ${where}`;
      else if (ev.W !== undefined) note = `Return <b>${signed(ev.target)}</b> · weight W = <b>${+ev.W.toFixed(3)}</b> · ${where}`;
      else note = `Target <b>${signed(ev.target)}</b> · surprise <b class="q-err">δ = ${signed(ev.delta)}</b> · ${where}`;
      liveNote.innerHTML = note;
      liveNote.classList.remove("faint");
    }

    function mark(line) {
      for (const li of pseudo.querySelectorAll("li.on")) li.classList.remove("on");
      if (line) pseudo.querySelector(`li[data-line="${line}"]`)?.classList.add("on");
    }

    function focus(i) {
      P.focus = i;
      stages.forEach((st, k) => st.classList.toggle("focus", k === i && racers.length > 1));
      const a = racers[i].algorithm, p = paramsOf(racers[i]);
      q(".algo-name").textContent = racers[i].name;
      pseudo.innerHTML = RL.entry(a.id)?.pseudocode || '<p class="faint">The pseudocode of this algorithm is not written yet.</p>';
      liveSym.innerHTML = RL.math.tex(typeof a.rule === "function" ? a.rule(p) : a.rule, true);
      liveNum.innerHTML = "";
      RL.math.render(pseudo);
      if (runs.length) filmstrip();
    }

    // ---- playing ----
    function play() {
      if (P.e >= knobs.units) seek(0);
      P.playing = true;
      playBtn.innerHTML = ICON.pause;
      playBtn.setAttribute("aria-label", "Pause");
    }
    function pause() {
      P.playing = false;
      playBtn.innerHTML = ICON.play;
      playBtn.setAttribute("aria-label", "Play");
    }
    const speed = () => SPEEDS[unitOf()].find((s) => s.id === P.speed);
    const rate = () => speed().rate || 0;
    function stepOnce() {
      pause();
      if (rate()) seek(P.e + 1, true);
      else walk(P.speed === "step");
    }

    let raf = 0, last = 0;
    function frame(t) {
      const dt = last ? Math.min(100, t - last) : 0;
      last = t;
      if (P.playing) tick(dt);
      raf = requestAnimationFrame(frame);
    }
    function tick(dt) {
      if (rate()) {
        P.acc += (dt / 1000) * rate();
        if (P.acc >= 1) {
          const k = Math.floor(P.acc);
          seek(P.e + k, rate() <= 1);
          P.acc -= k;
        }
        return;
      }
      if (P.wait > 0) { P.wait -= dt; return; }
      P.acc += dt;
      if (P.acc >= speed().every) { P.acc = 0; walk(P.speed === "step"); }
    }

    // ---- controls ----
    let pending = 0;
    q(".knobs").addEventListener("input", (e) => {
      const input = e.target.closest("[data-knob]");
      if (!input) return;
      const k = input.dataset.knob;
      knobs[k] = parseFloat(input.value);
      const out = input.parentElement.querySelector("output");
      if (out) out.textContent = fmtKnob(k, knobs[k]);
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(simulate);
    });
    q(".knobs").addEventListener("click", (e) => {
      if (!e.target.closest(".seed")) return;
      knobs.seed = 1 + Math.floor(Math.random() * 99999);
      q(".seed b").textContent = knobs.seed;
      simulate();
    });
    q(".show-host").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-opt]");
      if (!b) return;
      view.show[b.dataset.opt] = b.dataset.v;
      b.parentElement.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      views.forEach((v) => v.setOptions(view.show));
    });
    q(".show-host").addEventListener("change", (e) => {
      const box = e.target.closest("input[data-opt]");
      if (!box) return;
      view.show[box.dataset.opt] = box.checked;
      views.forEach((v) => v.setOptions(view.show));
      if (!P.walkers) seek(P.e);
    });
    q(".chart-mode").addEventListener("click", (e) => {
      const b = e.target.closest("[data-seeds]");
      if (!b || +b.dataset.seeds === +view.seeds) return;
      view.seeds = b.dataset.seeds === "1";
      b.parentElement.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      job?.cancel();
      drawCharts();
      if (view.seeds) average();
    });
    playBtn.addEventListener("click", () => (P.playing ? pause() : play()));
    q(".step").addEventListener("click", stepOnce);
    q(".restart").addEventListener("click", () => { pause(); seek(0); });
    q(".speed").addEventListener("change", (e) => {
      const wasWalking = !rate();
      P.speed = e.target.value;
      if (wasWalking && rate() && P.walkers) seek(P.e);
    });
    scrub.addEventListener("input", () => { pause(); seek(parseInt(scrub.value, 10)); });
    function onKey(e) {
      if (e.target.closest("input, select, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === " ") { e.preventDefault(); P.playing ? pause() : play(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); stepOnce(); }
    }
    addEventListener("keydown", onKey);

    // The sandbox redraws the world and may change the racers; everything is built again.
    function sandboxHost() {
      return {
        host: () => q(".sandbox"),
        views: () => views,
        algorithms: () => racers.map((r) => r.algorithm.id),
        setRacers(ids) { racers = ids.map((id) => ({ algorithm: lab.algorithms[id], name: lab.algorithms[id].title, params: {} })); },
        changed(spec, ids) {
          pause();
          if (ids) this.setRacers(ids);
          build();
        },
      };
    }

    sandbox?.mount();
    build();
    raf = requestAnimationFrame(frame);

    return {
      destroy() {
        cancelAnimationFrame(raf);
        cancelAnimationFrame(pending);
        job?.cancel();
        removeEventListener("keydown", onKey);
        charts.forEach((c) => c.destroy());
        views.forEach((v) => v.destroy());
        film?.destroy();
        sandbox?.destroy();
      },
    };
  };

  function pick(o, keys) {
    const out = {};
    for (const k of keys) if (k in o) out[k] = o[k];
    return out;
  }
  RL.labViews = RL.labViews || {};
})(globalThis.RL = globalThis.RL || {});
