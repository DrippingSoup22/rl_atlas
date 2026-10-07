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
  const NOUNS = { episode: ["episode", "episodes"], sweep: ["sweep", "sweeps"], step: ["step", "steps"], pull: ["pull", "pulls"], hand: ["hand", "hands"], round: ["round", "rounds"], throw: ["throw", "throws"], block: ["block", "blocks"], pass: ["pass", "passes"] };
  // A ladder of speeds, each showing what can be seen at that pace. Walking speeds replay a unit event by event:
  // "line" stops at every line of the pseudocode, the others at each update ("step" slowly enough to read its numbers,
  // "fast" and "faster" so the agent runs while the pseudocode still lights up). Rates jump from unit to unit.
  const walking = (step, every) => [{ id: "line", label: "Line by line", every: every[0] }, { id: "step", label: step, every: every[1] }];
  const quick = (fast, faster) => [{ id: "fast", label: "Fast", every: fast, quick: true }, ...(faster ? [{ id: "faster", label: "Faster", every: faster, quick: true }] : [])];
  const rates = (...list) => list.map((rate) => ({ id: `r${rate}`, rate }));
  const SPEEDS = {
    episode: [...walking("Step by step", [560, 320]), ...quick(100, 25), ...rates(1, 10, 50)],
    sweep: [...walking("State by state", [380, 130]), ...quick(40), ...rates(1, 4, 20)],
    step: [...walking("Pull by pull", [480, 300]), ...rates(10, 50, 250)],
    round: [...walking("Step by step", [520, 320]), ...quick(110, 35), ...rates(1, 4, 20)], // every worker steps at once
    pass: [...walking("Step by step", [520, 300]), ...quick(90, 30), ...rates(1, 4, 10)], // the log being made, then passes over it
    block: [{ id: "line", label: "Step by step", every: 80 }, { id: "step", label: "Quickly", every: 12 }, ...rates(1, 4, 10)], // a recorded test episode
  };
  // The knobs a preset can show as sliders. alpha = 0 means sample averages (1/n), where an algorithm allows it.
  // sweep: the values a sweep of the odds tries for a slider (a knob with choices tries its choices).
  const KNOBS = {
    alpha: { sym: "α", name: "step size", min: 0.01, max: 1, step: 0.01, sweep: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] },
    epsilon: { sym: "ε", name: "exploration", min: 0, max: 0.5, step: 0.01, sweep: [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5] },
    gamma: { sym: "γ", name: "discount", min: 0.5, max: 1, step: 0.01, sweep: [0.5, 0.7, 0.9, 0.95, 0.99, 1] },
    c: { sym: "c", name: "confidence", min: 0, max: 5, step: 0.1, sweep: [0, 0.5, 1, 2, 3, 5] },
    q0: { sym: "Q₁", name: "first estimate", min: -2, max: 10, step: 0.5, sweep: [-2, 0, 1, 2, 5, 10] },
    theta: { sym: "θ", name: "tolerance", choices: [0.1, 0.01, 0.001, 0.0001] },
    n: { sym: "n", name: "steps ahead", choices: [1, 2, 3, 4, 8, 16, 32, 64] },
    lambda: { sym: "λ", name: "trace decay", min: 0, max: 1, step: 0.01, sweep: [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1] },
    planning: { sym: "n", name: "planning steps", choices: [0, 1, 5, 10, 20, 50, 100] },
    kappa: { sym: "κ", name: "exploration bonus", choices: [0, 0.0001, 0.001, 0.01] },
    alphaW: { sym: "αw", name: "critic step size", min: 0.01, max: 1, step: 0.01, sweep: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] },
    beta: { sym: "β", name: "entropy bonus", choices: [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] },
    workers: { sym: "N", name: "workers", choices: [1, 2, 4, 8, 16] },
    epochs: { sym: "K", name: "passes over each batch", choices: [1, 2, 4, 10, 20] },
    clip: { sym: "ε", name: "clip range (0: no clip)", choices: [0, 0.1, 0.2, 0.3, 0.5] },
    delta: { sym: "δ", name: "trust region (KL)", choices: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2] },
    noise: { sym: "σ", name: "exploration noise (degrees)", min: 0, max: 30, step: 1, sweep: [0, 1, 2, 5, 10, 20, 30] },
    logEpisodes: { sym: "E", name: "episodes in the log", choices: [1, 2, 5, 10, 20, 50, 100, 200] },
    rewardScale: { sym: "c", name: "rewards × c (other units)", choices: [0.01, 0.1, 1, 10, 100] },
    normalize: { sym: "Â", name: "normalize advantages (1: yes)", choices: [0, 1] },
    doubt: { sym: "d", name: "doubt: extra cost of an unseen tile", min: 0, max: 3, step: 0.1, sweep: [0, 0.2, 0.4, 0.6, 0.8, 1, 2] },
  };
  const ALPHA_LADDER = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05];
  const AVERAGING = new Set(["epsilon-greedy", "optimistic-init", "ucb", "mc-prediction", "exploring-starts", "mc-control"]);
  // What a chart can plot.
  const METRICS = {
    return: { title: (n, env, a) => (a?.recorded ? "Training episodes: their average return in each block" : `Reward per ${n}`), smooth: 10 },
    steps: { title: (n) => `Steps per ${n}`, smooth: 10, log: true }, // a first episode of 1000 steps would flatten the rest
    optimal: { title: () => "How often the best arm is pulled", percent: true },
    left: { title: () => "How often the agent goes left from A", percent: true },
    delta: { title: () => "Largest change in a sweep", log: true },
    error: { title: () => "Error: distance from the true values (RMS)", zero: true },
    "optimal-error": { title: () => "Distance from the optimal values (RMS)", log: true },
    match: { title: () => "States where the greedy action is optimal", percent: true },
    greedy: { title: (n, env) => (env.ice ? "Chance the greedy policy reaches the gem" : "Return of the greedy policy"), percent: true },
    ve: { title: () => "Value error √VE, weighted by time spent in each state", zero: true },
    weights: { title: () => "Size of the weights ‖w‖", log: true },
    "policy-value": { title: (n, env) => (env.kind === "corridor" ? "Value of the start under the current policy, J(θ)" : env.ice ? "Chance the current policy reaches the gem" : "Value of the current policy from the start, J(θ)") },
    right: { title: () => "Chance of stepping right, π(right)", percent: true },
    deployed: { title: () => "Return of the learned policy, played once from the start" },
    labels: { title: () => "States the expert has labeled" },
    aim: { title: () => "Where the policy aims: its angle μ, in degrees" },
    kl: { title: () => "How far each update moved the policy: KL divergence (log scale)", log: true },
    clipped: { title: () => "Samples the clip left alone in the last pass", percent: true },
    // what recorded runs chart (one seed: the one played back)
    test: { title: (n, env) => (env.kind === "cartpole" ? "Test episode after each block: steps balanced" : "Test episode after each block: its return") },
    loss: { title: () => "Loss: the squared TD error, averaged over the block", log: true },
    td: { title: () => "Size of the TD error |δ|, averaged over the block", log: true },
    // DQN logs the largest Q-value of each state in a batch; the actor–critics, the target of each sample
    q: { title: (n, env, alg) => (alg?.kind === "ac" ? "The critics' targets in the batches, on average" : "The largest Q-value of each state in the batches, on average") },
    eps: { title: () => "Exploration ε at the end of the block", percent: true },
  };
  const LADDER = [10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 200000, 500000];
  const signed = (v, d = 2) => (v < 0 ? "−" : v > 0 ? "+" : "") + Math.abs(v).toFixed(d);
  // Events after which the views redraw what the algorithm knows.
  const SHOWN = new Set(["update", "improve", "trace", "plan", "world", "critic"]);
  const fmtBig = (v) => (!Number.isFinite(v) ? "beyond any number" : v >= 1e5 ? v.toExponential(1).replace("e+", " × 10^") : v.toFixed(v >= 100 ? 0 : 1));
  const plural = (n, [one, many]) => `${n.toLocaleString("en")} ${n === 1 ? one : many}`;

  RL.views.lab = function (host, presetId) {
    const base = RL.content.presets[presetId];
    // the working copy: switching to another world (the world panel) changes its world, units, charts and odds rule
    const preset = base && { ...base, params: { ...base.params } };
    if (!preset) {
      host.innerHTML = '<section class="lab-missing card"><h1>There is no such lab yet</h1><p><a href="#/">Back to the map</a></p></section>';
      return {};
    }
    const sandbox = preset.env === "sandbox" ? RL.sandbox(sandboxHost()) : null;
    const make = () => (sandbox ? lab.grid(sandbox.spec()) : lab.make(preset.env));
    let env = make();
    // Racers that name a recording play back runs trained offline (recorder/record.py) instead of computing them.
    const recRuns = preset.racers.map((r) => (r.recording ? lab.recordedRun(RL.recordings[r.recording]) : null));
    const recorded = recRuns.some(Boolean), rec0 = recorded ? RL.recordings[preset.racers[0].recording] : null, sweepData = rec0 ? RL.sweeps?.[rec0.name] : null;
    let racers = preset.racers.map((r, i) => ({ ...r, algorithm: recRuns[i] ? recRuns[i].algorithm : lab.algorithms[r.algorithm] }));
    const unitOf = () => racers[0].algorithm.unit;
    const noun = () => NOUNS[env.unitName || (unitOf() === "step" ? "pull" : unitOf())];
    // Shared knobs: the preset's, minus those every racer sets for itself.
    // seed: "typical" (the bench's median run for these settings) or a number the reader chose
    const knobs = { units: recorded ? recRuns[0].units : preset.units, seed: recorded ? recRuns[0].seed : "typical", runs: preset.runs, ...preset.params };
    // The knobs on the bar: the lab's settings that not every racer sets for itself. In another world, step sizes
    // the racers all set are shown too: the knob is the reference their ratios apply to (lab.carryRacer).
    const shown = () => Object.keys(KNOBS).filter((k) => k in preset.params && (racers.some((r) => !(k in r.params)) || (preset.env !== base.env && (k === "alpha" || k === "alphaW"))));
    // A racer's settings: the lab's, the knobs', then its own (adapted when it runs in another world than the lab's).
    // "α = 2⁻¹³" or "αw = 0.1" in a racer's name, written again for the step sizes p it runs with
    const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    const stepText = (v) => {
      const e = Math.log2(v);
      return Number.isInteger(e) && e < -3 ? `2${[...String(e)].map((c) => SUP[c]).join("")}` : String(+v.toPrecision(2));
    };
    const stepNamed = (name, p) => name
      .replace(/αw = [^,)·]+/, (m) => (p.alphaW > 0 ? `αw = ${stepText(p.alphaW)}` : m))
      .replace(/α = (?!1\/n)[^,)·]+/, (m) => (p.alpha > 0 ? `α = ${stepText(p.alpha)}` : m));
    const paramsOf = (r) => {
      const own = preset.env === base.env ? r.params : lab.carryRacer(r.params, { env, base: base.params, racers: racers.map((x) => x.params), knobs });
      return { ...preset.params, ...pick(knobs, Object.keys(KNOBS)), ...own };
    };
    const view = { seeds: !!preset.seeds, mode: "bench", show: {} }; // seeds: a recording's thin lines; mode: "bench" or "run"
    const from = RL.app?.from?.name === "entry" ? racers.findLastIndex((r) => r.algorithm.id === RL.app.from.id) : -1;
    const P = { e: 0, playing: false, speed: "step", acc: 0, wait: 0, walkers: null, focus: Math.max(0, from >= 0 ? from : racers.length - 1) };
    let runs = [], stages = [], views = [], job = null, film = null, charts = [], avg = null, sweepJob = null, sweepCharts = [], perRun = 0, runsChosen = false;
    // the bench: the same settings on seeds 1 to 20 (or as many as the charts average), run in the background
    let bench = null, benchTimer = 0, benchDrawn = 0, strip = null, played = 1, pendingTypical = false;

    host.innerHTML = `
      <section class="lab" data-kind="${env.kind}" data-drawer="closed">
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
              <button class="settings-btn" type="button" aria-controls="lab-drawer" aria-expanded="false" title="World and display">
                <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 5h14M3 10h9M3 15h5"/><circle cx="15" cy="10" r="2"/><circle cx="11" cy="15" r="2"/></svg>
                <span class="sb-label">World &amp; display</span>
              </button>
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
            <section class="odds-card card" hidden>
              <h3 class="odds-title">The odds</h3>
              <p class="odds-tally"></p>
              <div class="strip-host"></div>
              <p class="faint strip-note"></p>
              <div class="sweep-bar">
                <label>Sweep <select class="sweep-knob" aria-label="Knob to sweep"></select></label>
                <label>over <select class="sweep-runs" aria-label="Runs per value">${[10, 20, 50, 100].map((n) => `<option value="${n}"${n === 20 ? " selected" : ""}>${n}</option>`).join("")}</select> runs per value</label>
                <button class="pill sweep-go" type="button">Run</button>
              </div>
              <div class="chart-grid sweep-grid"></div>
              <p class="faint sweep-note"></p>
            </section>
          </div>
          <aside class="lab-side">
            <section class="panel card pseudo-panel"><h3>Pseudocode · <span class="algo-name as-is"></span></h3><div class="pseudo-tabs" hidden></div><div class="pseudo-host"></div><p class="pseudo-diff faint" hidden></p></section>
            <section class="panel card live">
              <h3>This step</h3>
              <div class="live-sym"></div>
              <div class="live-num"></div>
              <p class="live-note faint"></p>
            </section>
          </aside>
        </div>
        <aside class="lab-drawer" id="lab-drawer" aria-label="World and display">
          <div class="ld-head">
            <span class="eyebrow">Settings</span>
            <button class="icon-btn ld-close" type="button" aria-label="Close the settings" title="Close (Esc)"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15"/></svg></button>
          </div>
          <section class="ld-sec world-panel" hidden><h3>World</h3><div class="world-list"></div></section>
          <section class="ld-sec show-panel"><h3>Display</h3><div class="show-host"></div></section>
        </aside>
      </section>`;

    const q = (sel) => host.querySelector(sel);
    const playBtn = q(".play"), scrub = q(".scrub"), pos = q(".pos"), pseudo = q(".pseudo-host");
    const liveSym = q(".live-sym"), liveNum = q(".live-num"), liveNote = q(".live-note");

    // ---- building the page around the current world and racers ----
    function knobPanel() {
      const box = q(".knobs");
      if (recorded) { // nothing to turn: the runs were trained offline
        box.innerHTML = `<p class="rec-note">Trained offline on Gymnasium's ${esc(env.title)}: ${rec0.steps.toLocaleString("en")} steps in ${plural(knobs.units, noun())} of ${rec0.block.toLocaleString("en")}, ${plural(rec0.seeds.length, ["seed", "seeds"])}. After each block the network played one test episode, from the same start every time: that is what plays here.</p>`;
        return;
      }
      const slider = (k) => {
        // an algorithm can name a knob its own way (A2C's n is the steps between updates, PPO's λ belongs to GAE)
        const own = racers.find((r) => r.algorithm.knobs?.[k])?.algorithm.knobs[k], d = { ...KNOBS[k], ...own }, v = knobs[k];
        // Linear methods with many features step in tiny amounts: their α is picked from a ladder instead of a slider.
        if (k === "alpha" && preset.params.alpha < 0.01) return `<label class="knob"><span class="sym">${d.sym}</span><span class="name">${d.name}</span>
          <select data-knob="${k}">${[...new Set([...ALPHA_LADDER, v])].sort((a, b) => a - b).map((c) => `<option value="${c}"${c === v ? " selected" : ""}>${c}</option>`).join("")}</select></label>`;
        if (d.choices) return `<label class="knob"><span class="sym">${d.sym}</span><span class="name">${d.name}</span>
          <select data-knob="${k}">${[...new Set([...d.choices, v])].sort((a, b) => a - b).map((c) => `<option value="${c}"${c === v ? " selected" : ""}>${c}</option>`).join("")}</select></label>`;
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
        `<div class="seedbox" role="group" aria-label="Seed">
          <span class="name">seed</span>
          <button type="button" class="seed-typ" title="Play the typical run of these settings: the median of the 20 bench seeds">Typical</button>
          <input class="seed-in" type="number" min="1" max="999999" step="1" inputmode="numeric" aria-label="Seed number" title="Type a seed and press Enter">
          <button type="button" class="seed-dice" title="Roll a seed" aria-label="Roll a seed">${ICON.dice}</button>
        </div>`;
      seedBox();
    }
    // The seed control shows the seed playing, and whether it is the typical one.
    function seedBox() {
      const typ = q(".seed-typ"), input = q(".seed-in");
      if (!typ) return;
      const typical = knobs.seed === "typical";
      typ.classList.toggle("on", typical);
      typ.classList.toggle("finding", typical && !bench?.complete);
      typ.classList.toggle("ready", pendingTypical);
      typ.textContent = pendingTypical ? `Typical: seed ${bench.typical()}` : "Typical";
      typ.title = pendingTypical ? "The typical run is ready: click to play it" : typical && !bench?.complete ? "Finding the typical run: the bench is still running" : "Play the typical run of these settings: the median of the bench seeds";
      if (document.activeElement !== input) input.value = played;
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
      box.innerHTML = racers.map((r, i) => `
        <figure class="stage card" data-i="${i}" style="--k: var(--s${i + 1})">
          <figcaption><span class="who"><i class="key"></i><b>${esc(r.name)}</b></span><span class="odds-badge" hidden></span><span class="stat"></span></figcaption>
          <div class="view-host"></div>
        </figure>`).join("");
      stages = Array.from(box.querySelectorAll(".stage"));
      view.show.agent = unitOf() !== "sweep"; // dynamic programming has no agent walking about
      views = stages.map((st) => new View(st.querySelector(".view-host"), env, view.show));
      views.forEach((v) => v.setPace?.(speed()?.quick ? P.speed : ""));
      stages.forEach((st, i) => st.addEventListener("click", () => focus(i)));
      const displays = racers.map((r, i) => {
        if (recRuns[i]) return { ...recRuns[i].at(0) };
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
      const speeds = speedList();
      if (!speeds.some((s) => s.id === P.speed)) P.speed = "step";
      q(".speed").innerHTML = speeds.map((s) => `<option value="${s.id}"${s.id === P.speed ? " selected" : ""}>${s.rate ? `${s.rate} ${noun()[s.rate === 1 ? 0 : 1]} / s` : s.label}</option>`).join("");
      idleNote();
      q(".legend").innerHTML = racers.map((r, i) => `<span><i class="key" style="--k: var(--s${i + 1})"></i>${esc(r.name)}</span>`).join("");
      knobPanel();
      worldPanel();
      q(".settings-btn").hidden = q(".world-panel").hidden && q(".show-panel").hidden;
      if (q(".settings-btn").hidden) setDrawer(false);
      oddsPanel();
      chartsFor();
      pseudoTabs();
      focus(Math.min(P.focus, racers.length - 1));
      simulate();
      fitStages();
    }

    // ---- the world panel: every shared world these racers may run, and this lab's own ----
    function worldPanel() {
      const box = q(".world-panel");
      const groups = recorded || sandbox || base.home ? [] : lab.worldsFor(racers.map((r) => r.algorithm));
      const listed = groups.some((g) => g.worlds.some((w) => w.id === base.env));
      const own = listed ? [] : [{ group: "own", title: "This lab", worlds: [{ id: base.env, title: lab.make(base.env).title, blurb: "the world this lab was made for" }] }];
      const all = [...own, ...groups], count = all.reduce((n, g) => n + g.worlds.length, 0);
      box.hidden = count < 2 && !base.home; // a lab tied to its world says why
      if (box.hidden) return;
      // a world where the study saw these racers' algorithms rarely end well is marked "hard": a run there may fail
      const tip = (w) => w.blurb + (w.hard?.length ? `. Hard for ${w.hard.join(" and ")}: in our tests, with the settings it starts from, it rarely ended well here` : "");
      const anyHard = all.some((g) => g.worlds.some((w) => w.hard?.length));
      q(".world-list").innerHTML = all.map((g) => `<div class="wgroup"><h4>${esc(g.title)}</h4><div class="wchips">${g.worlds.map((w) =>
        `<button type="button" class="wchip${w.id === preset.env ? " on" : ""}${w.hard?.length ? " hard" : ""}" data-world="${w.id}" title="${esc(tip(w))}">${esc(w.title)}${w.id === base.env && listed ? '<i class="home" aria-label="this lab\'s world"></i>' : ""}${w.hard?.length ? '<i class="hardmark">hard</i>' : ""}</button>`).join("")}</div></div>`).join("") +
        (anyHard ? '<p class="faint world-note"><i class="hardmark">hard</i> our tests rarely saw it end well there: a run may struggle or fail, which is worth seeing too</p>' : "") +
        (base.home ? `<p class="faint world-note">${esc(base.home)}</p>` : "");
    }
    function switchWorld(id) {
      if (id === preset.env) return;
      pause();
      const before = Object.keys(preset.params || {});
      if (id === base.env) {
        for (const k of Object.keys(preset)) delete preset[k];
        Object.assign(preset, base, { params: { ...base.params } });
        knobs.units = base.units;
      } else {
        const prof = lab.worldProfile(id, racers.map((r) => r.algorithm));
        Object.assign(preset, { env: id, units: prof.units, charts: prof.charts, measures: prof.measures, success: prof.success, film: prof.film, runs: prof.runs, sweep: null });
        preset.params = { ...base.params, ...prof.params };
        // racers that all bring their own step sizes get a reference one here, for their ratios to apply to
        for (const k of ["alpha", "alphaW"]) if (!(k in preset.params) && racers.some((r) => k in r.params)) preset.params[k] = 0.1;
        if (prof.gamma !== undefined && "gamma" in base.params) preset.params.gamma = prof.gamma;
        if (prof.domain) preset.params.domain = prof.domain; else delete preset.params.domain;
        if (prof.maxSteps) preset.params.maxSteps = prof.maxSteps; else delete preset.params.maxSteps; // the world's own limit
        knobs.units = prof.units;
      }
      for (const k of before) if (!(k in preset.params)) delete knobs[k]; // nothing stays from the world before
      Object.assign(knobs, preset.params); // the knobs start from the world's settings (a step size made for its features)
      knobs.runs = preset.runs;
      knobs.seed = "typical";
      P.e = 0;
      // a racer named by its step size is renamed by the one it has here (its home world's numbers would be wrong)
      racers.forEach((r) => { r.baseName ??= r.name; r.name = id === base.env ? r.baseName : stepNamed(r.baseName, paramsOf(r)); });
      q(".lab-title h1").textContent = id === base.env ? base.title : `${racers.map((r) => r.name).join(" vs ")} · ${lab.make(id).title}`;
      q(".intro").textContent = id === base.env ? base.intro || "" : `${lab.worldBlurb(id)}. The settings start from this world's defaults; the knobs are yours.`;
      build();
    }

    // ---- the stages: every racer on screen at once ----
    // The panes keep their world's shape. The arrangement (how many across) is the one that gives each pane the most
    // room while all of them, and the play bar under them, fit in the window below the header; if that would make them
    // too small, they may take the whole window instead (a scroll down to them).
    function fitStages() {
      const box = q(".stages");
      if (!box || !stages.length) return;
      box.style.gridTemplateColumns = "minmax(0, 1fr)";
      if (innerWidth < 900) return; // narrow screens: one under the other, full width
      const st = stages[0], host = st.querySelector(".view-host"), content = host.firstElementChild;
      const w0 = Math.min(host.clientWidth, content ? content.getBoundingClientRect().width || host.clientWidth : host.clientWidth);
      const padX = st.offsetWidth - host.clientWidth;
      const W = box.clientWidth, n = stages.length, gap = 16, bar = q(".transport").offsetHeight;
      // Some views reflow when narrowed (a recorded run's two panels stack), so nothing is predicted: each arrangement
      // is laid out and measured, its panes narrowed until everything fits, and the one with the widest panes wins.
      const apply = (cols, w) => { box.style.gridTemplateColumns = `repeat(${cols}, ${Math.floor(w + padX)}px)`; return box.offsetHeight; };
      const MIN = 260;
      const widest = (cols, H) => {
        let hi = Math.min(w0, (W - gap * (cols - 1)) / cols - padX), lo = MIN;
        if (hi < lo) return 0;
        if (apply(cols, hi) <= H) return hi;
        if (apply(cols, lo) > H) {
          // too tall even at the narrowest, yet a view that reflows may fit wider (panels side by side): look down
          // from the widest in small steps for a width that fits
          let w = hi * 0.93;
          while (w > lo && apply(cols, w) > H) { hi = w; w *= 0.93; }
          if (w <= lo) return 0;
          lo = w;
        }
        for (let k = 0; k < 7; k++) { const mid = (lo + hi) / 2; if (apply(cols, mid) > H) hi = mid; else lo = mid; }
        return lo;
      };
      const pick = (H) => {
        let best = { cols: 0, w: 0 };
        for (let cols = 1; cols <= n; cols++) { const w = widest(cols, H); if (w > best.w + 1) best = { cols, w }; }
        return best.cols ? best : null;
      };
      const top = box.getBoundingClientRect().top + scrollY;
      // below the header if they fit there; else the whole window (a scroll down to them); else as many across as fit
      const best = pick(innerHeight - top - bar - 28) || pick(innerHeight - 64 - bar - 28) ||
        { cols: Math.max(1, Math.min(n, Math.floor((W + gap) / (MIN + padX + gap)))), w: MIN };
      const w = best.w;
      apply(best.cols, w);
      box.classList.toggle("narrow", w < 430); // narrow panes keep their headers short
    }
    let lastWidth = 0;
    const resized = new ResizeObserver(() => {
      const w = q(".lab-main").clientWidth;
      if (Math.abs(w - lastWidth) > 4) { lastWidth = w; fitStages(); }
    });
    resized.observe(q(".lab-main"));

    // ---- the pseudocode: a tab per algorithm when the racers run different ones, the lines where they differ marked ----
    const lineText = (html) => {
      const box = document.createElement("div");
      box.innerHTML = html;
      return [...box.querySelectorAll("li")].map((li) => li.textContent.replace(/\s+/g, " ").trim());
    };
    function pseudoTabs() {
      const ids = [...new Set(racers.map((r) => r.algorithm.station || r.algorithm.id))];
      const tabs = q(".pseudo-tabs");
      tabs.hidden = ids.length < 2;
      tabs.innerHTML = ids.length < 2 ? "" : racers.map((r, i) => `<button type="button" data-i="${i}" style="--k: var(--s${i + 1})"><i class="key"></i>${esc(r.name)}</button>`).join("");
    }

    // The speeds of this world: some worlds walk faster (a Mountain Car episode has hundreds of steps) or play more
    // units a second (Baird's counterexample, whose units are single steps).
    function speedList() {
      const list = SPEEDS[unitOf()], first = list.findIndex((s) => s.rate);
      return list.map((s, i) => (s.rate ? (env.rates ? { ...s, id: `r${env.rates[i - first]}`, rate: env.rates[i - first] } : s) : env.pace?.[s.id] ? { ...s, every: env.pace[s.id] } : s));
    }

    // ---- the runs ----
    // New settings: a new bench, and the run to play: the typical one once the bench knows it (seed 1 until then),
    // or the seed the reader chose.
    function simulate() {
      job?.cancel();
      startBench();
      played = recorded ? knobs.seed : knobs.seed === "typical" ? (bench?.complete ? bench.typical() : bench?.seeds[0] ?? 1) : knobs.seed;
      playRuns();
      clearSweep(sweepCharts.length ? "The settings changed: run the sweep again to see the odds with them." : sweepIdle());
      autoRuns();
      if (recorded && (view.seeds || preset.success)) averageRecorded();
    }

    // Compute the played runs (seed `played`) with their snapshots, and show them. A run that takes long (a policy
    // gradient on CartPole, planning on the large lake) is computed in short slices, the page staying responsive,
    // with its progress under each racer's name; until then nothing plays.
    function playRuns() {
      P.walkers = null;
      pendingTypical = false;
      job?.cancel();
      runs = [];
      const jobs = racers.map((r, i) => (recRuns[i] ? null : lab.simulateJob({ world: make, algorithm: r.algorithm, params: paramsOf(r), units: knobs.units, seed: played, measures: preset.measures })));
      let spent = 0, timer = 0;
      const slice = (ms) => {
        const t0 = performance.now(), next = jobs.find((j) => j && !j.result);
        next?.step(ms);
        spent += performance.now() - t0;
        return jobs.every((j) => !j || j.result);
      };
      const ready = () => {
        job = null;
        stages.forEach((st) => st.classList.remove("training"));
        runs = racers.map((_, i) => recRuns[i] || jobs[i].result);
        perRun = spent / racers.length; // how long one run takes here, for the sweep's default
        showRuns();
      };
      if (slice(120)) return ready(); // most runs take a few milliseconds: no waiting at all
      pause();
      const progress = () => stages.forEach((st, i) => {
        st.classList.add("training");
        st.querySelector(".stat").textContent = jobs[i] ? `Training this run: ${Math.round(100 * jobs[i].done)}%` : "";
      });
      progress();
      const tick = () => { if (slice(30)) ready(); else { progress(); timer = setTimeout(tick, 0); } };
      job = { cancel() { clearTimeout(timer); job = null; } };
      timer = setTimeout(tick, 0);
    }
    // The played runs are ready: show them.
    function showRuns() {
      views.forEach((v, i) => { v.env = runs[i].env; }); // each view looks at its own run's world (its bandit's machines, its cards)
      scrub.max = knobs.units;
      filmstrip();
      drawCharts();
      summary();
      seek(Math.min(P.e, knobs.units));
      seedBox();
      odds();
    }

    // The rule a run is judged by: the preset's success rule, or its main measure over the last tenth.
    const rule = () => preset.success || { metric: preset.charts[0] };

    function startBench() {
      clearTimeout(benchTimer);
      bench = null;
      if (recorded) return;
      const many = knobs.runs > 1;
      bench = new lab.Bench({
        world: make, racers: racers.map((r) => ({ algorithm: r.algorithm, params: paramsOf(r) })), units: knobs.units, measures: preset.measures, rule: rule(),
        keys: preset.charts, smooth: Object.fromEntries(preset.charts.map((k) => [k, many ? (k === "return" && unitOf() === "episode" ? 5 : 1) : METRICS[k].smooth || 1])),
        seeds: Array.from({ length: many ? knobs.runs : 20 }, (_, i) => i + 1), giveUp: GIVE_UP, key: sandbox ? null : `${presetId}@${preset.env}`,
      });
      benchDrawn = 0;
      const chunk = () => {
        const done = bench.step(14);
        benched(done);
        if (!done) benchTimer = setTimeout(chunk, 0);
      };
      if (bench.complete) return; // kept from before: simulate() plays its typical seed at once
      benchTimer = setTimeout(chunk, 30);
    }

    // The bench made progress: redraw what depends on it (at most every 200 ms until it is done), and once it is done,
    // play the typical run if that is what is asked and the reader is not in the middle of watching another.
    function benched(done) {
      const now = performance.now();
      if (!done && now - benchDrawn < 200) return;
      benchDrawn = now;
      if (done && knobs.seed === "typical" && bench.typical() !== played) {
        if (!P.playing && !P.walkers && P.e === 0) { played = bench.typical(); playRuns(); return; }
        pendingTypical = true;
      }
      drawCharts();
      odds();
      seedBox();
    }

    // A recording keeps the training curve of every seed: the average, the thin lines and the odds come at once.
    function averageRecorded() {
      // the training returns of every seed, and any other curve every seed kept (DQN's Q-values)
      const keyOf = (k) => (k === "return" ? "train" : k), kept = [...new Set([...preset.charts, preset.success?.metric].filter(Boolean))].filter((k) => recRuns.every((r) => r.rec.curves.every((c) => c[keyOf(k)])));
      const per = recRuns.map((r) => Object.fromEntries(kept.map((k) => [k, lab.recordedCurves(r.rec, keyOf(k))]))), n = Math.min(...recRuns.map((r) => r.rec.curves.length));
      const sums = per.map((c) => Object.fromEntries(kept.map((k) => [k, c[k].mean.map((v) => v * n)])));
      const lines = per.map((c) => Array.from({ length: n }, (_, j) => Object.fromEntries(kept.map((k) => [k, c[k].seeds[j]]))));
      const wins = per.map((c) => (preset.success ? c[preset.success.metric].seeds.filter((v) => lab.success(preset.success, { [preset.success.metric]: v }).ok).length : 0));
      avg = { sums, lines, wins, done: n, n };
      if (knobs.runs > 1 || view.seeds) { drawCharts(avg); q(".chart-note").textContent = chartNote(n); }
      tally();
    }

    // ---- charts ----
    function chartsFor() {
      charts.forEach((c) => c.destroy());
      const grid = q(".chart-grid");
      grid.innerHTML = preset.charts.map((k) => `<div class="chart-box"><h3>${RL.asIs(METRICS[k].title(noun()[0], env, racers[0].algorithm))}</h3><div class="chart-host"></div></div>`).join("");
      grid.className = `chart-grid n${preset.charts.length}`;
      charts = preset.charts.map((k, j) => new RL.LineChart(grid.querySelectorAll(".chart-host")[j], {
        label: METRICS[k].title(noun()[0], env, racers[0].algorithm), percent: METRICS[k].percent, log: METRICS[k].log, zero: METRICS[k].zero,
        domain: k === "return" || k === "policy-value" ? preset.params.domain : null, noun: noun(), logX: knobs.units > 20000,
        onSeek: (t) => { pause(); seek(t); },
      }));
      const mode = q(".chart-mode");
      mode.innerHTML = preset.runs > 1 ? "" : recorded ? `<div class="seg" role="group" aria-label="How many runs the charts show">
        <button type="button" data-seeds="0" class="${view.seeds ? "" : "on"}">This run</button><button type="button" data-seeds="1" class="${view.seeds ? "on" : ""}">${rec0.seeds.length} seeds</button></div>`
        : `<div class="seg" role="group" aria-label="What the charts show">
        <button type="button" data-mode="bench" class="${view.mode === "bench" ? "on" : ""}">With the spread</button><button type="button" data-mode="run" class="${view.mode === "run" ? "on" : ""}">This run</button></div>`;
    }

    function chartNote(n) {
      const smooth = !recorded && knobs.runs <= 1 && preset.charts.some((k) => METRICS[k].smooth) ? `Smoothed over 10 ${noun()[1]}. ` : "";
      if (knobs.runs > 1) return `Each line is the average of ${n < knobs.runs ? `the ${n} runs done so far (of ${knobs.runs})` : `${knobs.runs} runs`}, with seeds 1 to ${knobs.runs}; the views above play seed ${played}. Click a chart to jump there.`;
      if (view.seeds && recorded) return `Thin lines: the ${n} seeds the recording trained, the same settings each time. Thick line: their average. The other charts follow the seed played above, ${knobs.seed}. Click a chart to jump there.`;
      if (!recorded && view.mode === "bench") return `${smooth}The line${racers.length > 1 ? "s are the runs" : " is the run"} playing above, seed ${played}. The soft band${racers.length > 2 ? ` (${racers[P.focus].name}'s; click another racer above to see its own)` : ""} holds the middle half of ${n < 20 ? `the ${n} bench seeds done so far` : "the 20 bench seeds"}, the dashed line their median. Click a chart to jump there.`;
      return `${smooth}Click a chart to jump there.`;
    }

    function drawCharts(avg) {
      if (!runs.length) return; // the played runs are still training
      const st = bench && bench.seedsDone ? bench.stats() : null, many = knobs.runs > 1;
      preset.charts.forEach((k, j) => {
        const m = { ...METRICS[k], smooth: recorded ? 1 : METRICS[k].smooth }, series = [];
        racers.forEach((r, i) => {
          const color = `--s${i + 1}`, band = st?.[i].band[k];
          if (recorded) {
            if (view.seeds && avg && avg.sums[i][k]) {
              for (const metrics of avg.lines[i]) series.push({ name: r.name, color, values: metrics[k], smooth: m.smooth, faint: true });
              series.push({ name: `${r.name}, average`, color, values: avg.sums[i][k].map((v) => v / avg.done), smooth: m.smooth });
            } else series.push({ name: r.name, color, values: runs[i].metrics[k], smooth: m.smooth });
          } else if (many) {
            if (band) series.push({ name: r.name, color, values: band.mean, units: knobs.units });
          } else {
            // the spread of every racer when there are one or two; with more, only the focused racer's, or they would blur
            if (view.mode === "bench" && band && st[i].n >= 5 && (racers.length <= 2 || i === P.focus)) series.push({ name: `${r.name}, middle half of the seeds`, color, band, units: knobs.units });
            series.push({ name: r.name, color, values: runs[i].metrics[k], smooth: m.smooth });
          }
        });
        charts[j].set(series, references(k));
      });
      const n = recorded ? avg?.done ?? 0 : st?.[0].n ?? 0;
      q(".chart-note").textContent = many && !n ? "Starting the runs…" : chartNote(n);
    }

    // Reference lines: the best a chart can reach, where it is known exactly.
    function references(k) {
      const p = paramsOf(racers[0]);
      if (k === "left") return [{ value: p.epsilon / 2, label: `best possible with ε = ${p.epsilon}: ${(50 * p.epsilon).toFixed(0)}%` }];
      if (env.kind === "corridor" && (k === "return" || k === "policy-value")) return [{ value: env.bestValue, label: `best possible on average: ${signed(env.bestValue, 1)}` }];
      if (k === "right") return [{ value: env.best, label: `the best policy: ${(100 * env.best).toFixed(0)}% right` }];
      if (k === "aim") return [{ value: env.continuousActions.best, label: `the best angle: ${env.continuousActions.best}°` }];
      if (k === "policy-value" && env.model) {
        const best = lab.optimalValues(env, p)[env.start];
        return [{ value: best, label: `an optimal policy: ${signed(best, Math.abs(best) < 10 ? 2 : 0)}` }];
      }
      if (k === "kl") { // each TRPO racer's own trust region
        const deltas = [...new Set(racers.filter((r) => r.algorithm.id === "trpo").map((r) => paramsOf(r).delta ?? 0.01))];
        return deltas.map((d) => ({ value: d, label: `δ = ${d}` }));
      }
      if (k === "greedy" && env.model) {
        const best = lab.evaluate(env, lab.greedyPolicy(env, lab.optimalValues(env, p), p.gamma, 1e-6), p.judge ?? p.gamma, { theta: 1e-8 })[env.start];
        return [{ value: best, label: `the optimal policy: ${(100 * best).toFixed(0)}%` }];
      }
      if (k === "match" || k === "optimal") return [{ value: 1, label: "" }];
      if (k === "q" && env.kind === "cartpole") return [{ value: 100, label: "the most a state can be worth: 1/(1 − γ) = 100" }];
      if (k === "test" && env.kind === "cartpole") return [{ value: 500, label: "the longest an episode lasts: 500" }];
      return [];
    }

    // ---- the odds: how often runs end well, and how that moves with a knob ----
    // The knobs of recorded runs (recorder/record.py), as their sweeps name them.
    const DEEP_KNOBS = { lr: { sym: "α", name: "learning rate" }, target_every: { sym: "C", name: "steps between target updates (0: none)" },
      buffer: { sym: "N", name: "replay memory (128: none)" }, tau: { sym: "τ", name: "how fast the target copies follow" },
      noise: { sym: "σ", name: "exploration noise" }, delay: { sym: "d", name: "critic updates per actor update" },
      policy_noise: { sym: "σ′", name: "noise on the target action" }, alpha: { sym: "α", name: "entropy weight (auto: tuned)" }, clip: { sym: "ε", name: "clip range (0: no clip)" }, epochs: { sym: "K", name: "passes over each batch" },
      steps: { sym: "n", name: "steps per worker between updates" }, delta: { sym: "δ", name: "trust region (KL)" }, gamma: { sym: "γ", name: "discount" },
      huber: { sym: "L", name: "loss (0: squared error, 1: Huber)" }, reward_scale: { sym: "c", name: "reward scale (rewards × c for learning)" } };
    const knobOf = (k) => (recorded ? DEEP_KNOBS[k] || { sym: k, name: k } : { ...KNOBS[k], ...racers.find((r) => r.algorithm.knobs?.[k])?.algorithm.knobs[k] });
    // The values a sweep tries: a knob's choices or its sweep list. Tiny step sizes (linear methods, policy gradients)
    // try the ladder around the ones in use; averaging methods also try 1/n.
    function sweepValues(k) {
      if (recorded) return sweepData?.knobs[k]?.values || [];
      if (preset.sweep?.[k]) return preset.sweep[k]; // a preset can pick the values that matter in its world
      const now = [...(k in preset.params ? [knobs[k]] : []), ...racers.filter((r) => k in r.params).map((r) => r.params[k])];
      if ((k === "alpha" || k === "alphaW") && Math.max(...now) < 0.01) return ALPHA_LADDER.filter((a) => a >= Math.min(...now) / 30 && a <= Math.max(...now) * 30);
      const list = knobOf(k).choices || knobOf(k).sweep || [];
      return k === "alpha" && racers.every((r) => AVERAGING.has(r.algorithm.id)) ? [0, ...list] : list;
    }
    const sweepable = () => (recorded ? Object.keys(sweepData?.knobs || {}) : null) || Object.keys(KNOBS).filter((k) => (k in preset.params || racers.some((r) => k in r.params)) && sweepValues(k).length > 1);
    const fmtSweep = (k, v) => (k === "alpha" && v === 0 ? "1/n" : v > 0 && v < 0.001 ? v.toExponential(0) : String(v));
    // The odds give up on a run whose episodes keep running to the step limit: it is stuck, and judged as it stands.
    const GIVE_UP = 20;
    const stuckNote = (k) => `${plural(k, ["run", "runs"])} got stuck: ${GIVE_UP} episodes in a row ran to the limit of ${(preset.params.maxSteps ?? 5000).toLocaleString("en")} steps, so ${k === 1 ? "it was" : "they were"} stopped there and judged as if ${k === 1 ? "it" : "they"} stayed stuck.`;
    const sweepIdle = () => `Run the sweep to see how ${preset.success ? "the odds move" : "the final score moves"} with a knob; every value gets the same seeds.`;

    function oddsPanel() {
      const ks = sweepable();
      q(".odds-card").hidden = recorded && !ks.length && !preset.success;
      q(".odds-tally").hidden = recorded && !preset.success;
      q(".odds-title").textContent = preset.success ? "The odds" : "How the seeds end";
      strip?.destroy();
      strip = recorded ? null : new RL.SeedStrip(q(".strip-host"), { label: "How each seed of the bench ends", onPick: (seed) => chooseSeed(seed) });
      q(".sweep-bar").hidden = !ks.length;
      q(".sweep-runs").parentElement.hidden = recorded; // a recording's seeds are fixed
      q(".sweep-knob").innerHTML = ks.map((k) => `<option value="${k}">${esc(knobOf(k).sym)} · ${esc(knobOf(k).name)}</option>`).join("");
      clearSweep(ks.length ? sweepIdle() : "");
    }

    // The bench, in words and as a strip of dots: how many seeds end well, and where the run playing above falls.
    function odds() {
      if (!runs.length) return; // the played runs are still training
      if (recorded || !bench || !strip) return;
      const st = bench.stats(), n = st[0].n, R = rule(), m = METRICS[R.metric] || {}, total = bench.seeds.length;
      const still = n < total ? ` (${total - n} still to come)` : "";
      let text;
      if (!n) text = `Running the ${total} bench seeds…`;
      else if (preset.success) {
        text = `Out of ${n} seeds with these settings${still}, how many ${esc(preset.success.text)}: ` + racers.map((r, i) => `<b>${esc(r.name)}</b> ${st[i].wins}`).join(" · ") + ".";
        if (racers.length === 2 && n === total) { // the same seed is the same luck for both: compare them seed by seed
          const w = [bench.pairedWins(0, 1).wins, bench.pairedWins(1, 0).wins], k = w[1] > w[0] ? 1 : 0, ties = n - w[0] - w[1];
          text += ` Seed by seed, with the same luck: <b>${esc(racers[k].name)}</b> ends better on ${w[k]} of ${n}${ties ? `, ${ties} ${ties === 1 ? "is a tie" : "are ties"}` : ""}.`;
        }
      } else text = `How the ${n} bench seeds${still} end: ${esc((m.title ? m.title(noun()[0], env, racers[0].algorithm) : R.metric).toLowerCase())}, averaged over the last tenth of the run.`;
      const stuck = st.reduce((k, x) => k + x.stuck, 0);
      q(".odds-tally").innerHTML = text + (stuck ? ` ${stuckNote(stuck)}` : "");
      const own = racers.map((_, i) => lab.success(R, runs[i].metrics).score);
      stages.forEach((stg, i) => {
        const b = stg.querySelector(".odds-badge");
        b.hidden = !preset.success || !n;
        if (!b.hidden) { b.innerHTML = `<b>${st[i].wins}</b>/${n}<span class="long"> seeds end well</span>`; b.title = `${st[i].wins} of ${n} seeds end well`; b.classList.toggle("good", st[i].wins >= n / 2); }
      });
      strip.set({
        rows: racers.map((r, i) => ({ name: r.name, color: `--s${i + 1}`, seeds: bench.seeds.slice(0, n), scores: st[i].scores, ok: bench.results[i].slice(0, n).map((d) => d.ok), played: { seed: played, score: own[i] } })),
        threshold: R.min ?? R.max ?? null, lower: bench.lower, percent: !!m.percent,
      });
      const inBench = bench.seeds.includes(played), typical = knobs.seed === "typical" && bench.complete && played === bench.typical(), bars = n > 60;
      const where = n && !inBench ? ` It is not one of the bench seeds; ${racers.length === 1 ? "it" : "each racer's run"} ends better than ${racers.map((r, i) => { const pl = bench.place(i, own[i]); return `${pl.beats} of ${pl.of}${racers.length > 1 ? ` (${esc(r.name)})` : ""}`; }).join(", ")}.` : "";
      q(".strip-note").innerHTML = !n ? "" : bars
        ? `Each bar counts the seeds that end there${preset.success ? ", filled where they end well" : ""}. The black mark: the run playing above, seed ${played}${typical ? ", the typical one" : ""}.${where} Click anywhere on a row to play the seed that ends closest.`
        : `One dot per seed${preset.success ? ": filled if it ends well, hollow if not" : ""}. Ringed: the run playing above, seed ${played}${typical ? ", the typical one" : ""}.${where} Click a dot to play its seed.`;
    }

    // How many of the runs so far ended as the success rule asks, for each racer (recorded runs).
    function tally() {
      if (!preset.success || !avg) return;
      const { wins, done, n, stuck } = avg, pct = (w) => Math.round((100 * w) / Math.max(1, done));
      q(".odds-tally").innerHTML = `Out of ${done} runs with these settings${done < n ? ` (${n - done} still to come)` : ""}, how many ${esc(preset.success.text)}: ` +
        racers.map((r, i) => `<b>${esc(r.name)}</b> ${wins[i]} (${pct(wins[i])}%)`).join(" · ") + "." + (stuck ? ` ${stuckNote(stuck)}` : "");
    }

    // Unless the reader picked a number, as many runs per value as fit in about 20 seconds of computing (at least 10).
    function autoRuns() {
      if (runsChosen || recorded) return;
      const values = sweepValues(q(".sweep-knob").value).length || 1;
      q(".sweep-runs").value = String([100, 50, 20, 10].find((n) => values * racers.length * n * perRun <= 20000) || 10);
    }

    function clearSweep(note) {
      sweepJob?.cancel();
      sweepCharts.forEach((c) => c.destroy());
      sweepCharts = [];
      q(".sweep-grid").innerHTML = "";
      q(".sweep-note").textContent = note;
    }

    // Every value of one knob, the same seeds for each, the other knobs as set: the share of runs that end well and
    // their average score. Racers that differ only in this knob become one line.
    function sweep() {
      clearSweep("");
      const k = q(".sweep-knob").value, values = sweepValues(k), n = +q(".sweep-runs").value;
      const rule = preset.success || { metric: preset.charts[0] }, m = METRICS[rule.metric] || {};
      const groups = [];
      racers.forEach((r, i) => {
        const base = { ...paramsOf(r) };
        delete base[k];
        const key = r.algorithm.id + JSON.stringify(base), same = groups.find((g) => g.key === key);
        if (same) same.name = r.algorithm.title;
        else groups.push({ key, r, i, base, name: k in r.params ? r.algorithm.title : r.name }); // its own value is overridden
      });
      const kinds = preset.success ? ["ok", "score"] : ["score"], current = recorded ? sweepData.knobs[k].current ?? rec0.config[k] ?? null : k in preset.params ? knobs[k] : null;
      const titles = { ok: `Runs that ${rule.text}`, score: `${m.title ? m.title(noun()[0], env, racers[0].algorithm) : rule.metric}, averaged over ${rule.window ? `${noun()[1]} ${rule.window[0]} to ${rule.window[1]}` : "the last tenth"}` };
      const grid = q(".sweep-grid");
      grid.className = `chart-grid sweep-grid n${kinds.length}`;
      grid.innerHTML = kinds.map((c) => `<div class="chart-box"><h3>${esc(titles[c])}</h3><div class="chart-host"></div></div>`).join("");
      sweepCharts = kinds.map((c, j) => new RL.SweepChart(grid.querySelectorAll(".chart-host")[j], {
        label: titles[c], percent: c === "ok" || m.percent, log: c === "score" && m.log, values, fmtX: (v) => fmtSweep(k, v), current,
      }));
      const res = groups.map(() => values.map(() => ({ runs: 0, wins: 0, sum: 0, scored: 0, stuck: 0 })));
      const total = values.length * groups.length * n;
      let done = 0, stuck = 0, cancelled = false, timer;
      const draw = () => kinds.forEach((c, j) => sweepCharts[j].set(groups.map((g, gi) => ({
        name: g.name, color: `--s${g.i + 1}`,
        points: res[gi].map((x) => (c === "ok" ? (x.runs ? x.wins / x.runs : NaN) : x.scored ? x.sum / x.scored : NaN)),
        notes: res[gi].map((x) => (c === "ok" ? `${x.wins} of ${x.runs} runs${x.stuck ? ` (${x.stuck} stuck)` : ""}` : x.scored ? `average ${sweepCharts[j].fmt(x.sum / x.scored)} over ${x.scored} runs` : "")),
      }))));
      if (recorded) { // trained offline: every value's seeds are in the sweep file
        const d = sweepData.knobs[k], judged = rule.metric === "test" ? d.test : d.train; // the curves the rule judges
        judged.forEach((curves, vi) => curves.forEach((c) => {
          const { ok, score } = lab.success(rule, { [rule.metric]: Float64Array.from(c, (x) => x ?? NaN) }), x = res[0][vi];
          x.runs++;
          if (ok) x.wins++;
          if (!Number.isNaN(score)) { x.sum += score; x.scored++; }
        }));
        draw();
        q(".sweep-note").textContent = `${plural(d.train[0].length, ["seed", "seeds"])} per value, ${sweepData.steps.toLocaleString("en")} steps each, trained offline with the recording's other settings. The shaded value is the recording's. Hover a dot for its numbers.`;
        return;
      }
      let one = null; // the run under way, spread over chunks when it is long
      const chunk = () => {
        if (cancelled) return;
        const t0 = performance.now();
        while (done < total && performance.now() - t0 < 14) {
          const vi = Math.floor(done / (groups.length * n)), si = Math.floor(done / groups.length) % n, gi = done % groups.length, g = groups[gi];
          one ||= lab.simulateJob({ world: make, algorithm: g.r.algorithm, params: { ...g.base, [k]: values[vi] }, units: knobs.units, seed: 1 + si, snapshots: false, measures: preset.measures, giveUp: GIVE_UP });
          if (!one.step(Math.max(1, 14 - (performance.now() - t0)))) break;
          const { metrics, stopped } = one.result;
          one = null;
          const { ok, score } = lab.success(rule, metrics), x = res[gi][vi];
          if (stopped) { x.stuck++; stuck++; }
          x.runs++;
          if (ok) x.wins++;
          if (!Number.isNaN(score)) { x.sum += score; x.scored++; }
          done++;
        }
        draw();
        q(".sweep-note").textContent = done < total ? `Running ${done} of ${total} runs…`
          : `${n} runs per value, with seeds 1 to ${n} and the other knobs as set above.${current !== null ? " The shaded value is the one the Lab is set to." : ""}${stuck ? ` ${stuckNote(stuck)}` : ""} Hover a dot for its numbers.`;
        if (done < total) timer = setTimeout(chunk, 0);
      };
      timer = setTimeout(chunk, 30);
      sweepJob = { cancel() { cancelled = true; clearTimeout(timer); } };
    }

    // One sentence on where each run ended up.
    function summary() {
      if (!runs.length) return; // the played runs are still training
      const out = [], last = Math.min(100, Math.max(1, Math.floor(knobs.units / 2))); // the second half at most: early episodes are a search
      const each = (f) => runs.map((r, i) => `<b>${esc(racers[i].name)}</b> ${f(r, i)}`).join(" · ");
      const pct = (v) => `${Math.round(100 * v)}%`;
      if (recorded) {
        const end = knobs.units - 1;
        out.push(`After the last block, the test episode lasts ${each((r) => plural(r.metrics.steps[end], ["step", "steps"]))}${env.kind === "cartpole" ? " (500 at most)" : `, return ${each((r) => signed(r.metrics.test[end], 0))} (0 is the best there is)`}.`);
        out.push(`Training episodes in the last block, on average: ${each((r) => signed(r.metrics.return[end], 0))}.`);
      }
      if (env.kind === "bandit") {
        out.push(`This run's last ${last} pulls: ${each((r) => `${lab.mean(r.metrics.return, knobs.units - last).toFixed(2)} per pull, a best arm ${pct(lab.mean(r.metrics.optimal, knobs.units - last))} of the time`)}.`);
      }
      if (env.kind === "graph") out.push(`Went left from A in this run: ${each((r) => `${pct(lab.mean(r.metrics.left))} of the episodes`)}.`);
      if (env.kind === "blackjack" && unitOf() === "round") out.push(`Average reward per hand over the last ${last} rounds: ${each((r) => signed(lab.mean(r.metrics.return, knobs.units - last), 2))} (+1 a win, −1 a loss; the best play averages about −0.05).`);
      if (env.kind === "blackjack" && unitOf() === "episode") out.push(`Hands won in this run: ${each((r) => pct(r.metrics.return.reduce((n, g) => n + (g > 0), 0) / knobs.units))}.`);
      if (env.kind === "grid" && unitOf() === "episode") {
        if (env.ice) out.push(`Reached the gem in the last ${last} episodes: ${each((r) => pct(lab.mean(r.metrics.return, knobs.units - last)))}.`);
        else if (preset.charts.includes("steps")) out.push(`${knobs.runs > 1 ? `In the run with seed ${played}, average` : "Average"} steps per episode over the last ${last}: ${each((r) => lab.mean(r.metrics.steps, knobs.units - last).toFixed(1))}.`);
        else out.push(`Average reward per episode over the last ${last}: ${each((r) => signed(lab.mean(r.metrics.return, knobs.units - last), 0))}.`);
        const finals = runs.map((r) => r.algorithm.show(r.at(knobs.units), r.env, r.params));
        if (finals[0].t !== undefined) env.setTime?.(finals[0].t); // a maze whose walls moved: follow the final layout
        if (!env.slip && env.start !== undefined && finals.every((d) => d.Q)) { // Taxi starts anywhere: no one path to follow
          out.push(`Greedy path at the end: ${each((r, i) => { const g = lab.greedyPath(env, finals[i].Q); if (g.reached) return plural(g.path.length - 1, ["step", "steps"]);
            // Still reaching the goal while learning, yet no greedy path: the greedy moves go round in circles
            // (Dyna-Q+'s bonuses can do this: its values include the pull of moves not tried in a while).
            return lab.mean(r.metrics.steps, Math.max(0, knobs.units - 10)) < 200 ? "none, its greedy moves go round in circles" : "none yet"; })}.`);
        }
      }
      if (env.kind === "corridor") {
        out.push(`${knobs.runs > 1 ? `In the run with seed ${played}, average` : "Average"} reward per episode over the last ${last}: ${each((r) => signed(lab.mean(r.metrics.return, knobs.units - last), 1))} (the best possible is ${signed(env.bestValue, 1)}).`);
      }
      if (env.kind === "throw") {
        out.push(`Where the policy aims at the end: ${each((r) => { const d = r.algorithm.show(r.at(knobs.units), r.env, r.params); return d.deterministic ? `${d.mu.toFixed(1)}°, thrown with ±${d.sd.toFixed(0)}° of noise` : `${d.mu.toFixed(1)}° ± ${d.sd.toFixed(1)}°`; })}; average distance over the last ${last} throws: ${each((r) => `${lab.mean(r.metrics.return, knobs.units - last).toFixed(1)} m`)} (40 m at best).`);
      }
      if (env.kind === "grid" && unitOf() === "round") {
        out.push(`${knobs.runs > 1 ? `In the run with seed ${played}, average` : "Average"} ${preset.charts.includes("steps") ? "steps" : "reward"} per episode over the last ${last} rounds: ${each((r) => (preset.charts.includes("steps") ? lab.mean(r.metrics.steps, knobs.units - last).toFixed(1) : lab.mean(r.metrics.return, knobs.units - last).toFixed(2)))}.`);
      }
      if (env.kind === "grid" && !env.slip) {
        const finals = runs.map((r) => r.algorithm.show(r.at(knobs.units), r.env, r.params));
        if (finals.every((d) => d.P && !d.Q && (d.theta || !d.V))) out.push(`Most likely path at the end: ${each((r, i) => { const g = likelyPath(env, finals[i].P); return g.reached ? plural(g.steps, ["step", "steps"]) : "none, its most likely moves go round in circles"; })}.`);
      }
      // the shared worlds of the World panel (over episodes, or rounds of several workers)
      const lastN = `the last ${plural(last, noun())}`, avgOf = (k, d) => each((r) => lab.mean(r.metrics[k], knobs.units - last).toFixed(d));
      if (env.kind === "taxi") out.push(`Average reward per trip over ${lastN}: ${each((r) => signed(lab.mean(r.metrics.return, knobs.units - last), 1))} (about +8 is the best possible).`);
      if (env.kind === "catch") out.push(`Caught over ${lastN}: ${each((r) => pct((lab.mean(r.metrics.return, knobs.units - last) + 1) / 2))} of the balls.`);
      if (env.kind === "cartpole" && !recorded) out.push(`Average steps the pole stays up, over ${lastN}: ${avgOf("steps", 0)} (500 at most).`);
      if (env.kind === "acrobot") out.push(`Average steps to swing the tip over the line, over ${lastN}: ${avgOf("steps", 0)} (500: cut short; a good swing takes under 100).`);
      if (env.kind === "line") out.push(`Value error √VE at the end: ${each((r) => r.metrics.ve ? r.metrics.ve[knobs.units - 1].toFixed(3) : "—")} (0 would be a perfect fit; how close the features allow is in the textbook).`);
      if (env.kind === "car") out.push(`${knobs.runs > 1 ? `In the run with seed ${played}, average` : "Average"} steps per episode over the last ${last}: ${each((r) => lab.mean(r.metrics.steps, knobs.units - last).toFixed(0))} (the best possible from a typical start is a little over 100).`);
      if (env.kind === "star") out.push(`Size of the weights at the end: ${each((r) => fmtBig(r.metrics.weights[knobs.units - 1]))}, from ${fmtBig(runs[0].metrics.weights[0])} after the first step.`);
      if (unitOf() === "sweep") {
        out.push(`Settled (largest change below θ = ${knobs.theta}) after: ${each((r) => { const t = r.metrics.converged.findIndex((c) => c); return t < 0 ? "not yet" : plural(t + 1, noun()); })}.`);
      }
      for (const k of preset.measures || []) {
        if (k === "aim" && env.kind === "throw") continue; // said above, with the spread
        if (k === "ve" && env.kind === "line") continue; // said above
        const f = METRICS[k].percent ? (v) => `${(100 * v).toFixed(0)}%` : (v) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(Math.abs(v) >= 10 ? 1 : 3)}${k === "aim" ? "°" : ""}`;
        out.push(`${METRICS[k].title(noun()[0], env, racers[0].algorithm)}, at the end: ${each((r) => f(r.metrics[k][knobs.units - 1]))}.`);
      }
      q(".summary").innerHTML = out.join("<br>");
    }

    // The greedy path from the start, while it reaches a goal; null if it goes round in circles.
    function greedyStates(Q) {
      const g = lab.greedyPath(env, Q);
      return g.reached ? g.path : null;
    }

    // Follow the most likely action of a learned policy from the start, as far as it reaches.
    function likelyPath(world, P) {
      let s = world.start, steps = 0;
      const seen = new Set([s]);
      while (!world.terminal(s) && steps < 200) {
        let a = 0;
        for (let b = 1; b < world.nA; b++) if (P[s * world.nA + b] > P[s * world.nA + a]) a = b;
        const { s2 } = world.step(s, a, lab.rng(1));
        steps++;
        if (seen.has(s2) && !world.terminal(s2)) return { reached: false };
        seen.add(s2);
        s = s2;
      }
      return { reached: world.terminal(s), steps };
    }

    // ---- the filmstrip: what the focused racer knew at a few moments ----
    function filmstrip() {
      if (!runs.length) return; // the played runs are still training
      const View = RL.labViews[env.kind], frames = [...new Set((preset.film || []).map((t) => Math.min(t, knobs.units)))];
      q(".film").hidden = !frames.length || !View.thumb;
      if (q(".film").hidden) return;
      const r = runs[P.focus], a = racers[P.focus].algorithm, p = paramsOf(racers[P.focus]);
      q(".film-name").textContent = racers[P.focus].name;
      film ||= new RL.Film(q(".film-host"), { onSeek: (t) => { pause(); seek(t); } });
      film.set(frames.map((t) => ({ t, label: t === 0 ? "at the start" : `after ${plural(t, noun())}`, html: View.thumb(env, a.show(r.at(t), env, p, t), p) })));
    }

    // ---- moving the playhead: t is the start of unit t; t = units is the end of the run ----
    // heat: "keep" when the units since the last position were already added to the views' maps of where the agent
    // goes; otherwise the maps are rebuilt from the 20 units before the new position.
    function seek(t, draw = false, heat = "rebuild") {
      if (!runs.length) return; // the played runs are still training
      P.e = Math.max(0, Math.min(knobs.units, t));
      P.walkers = null;
      P.acc = 0;
      P.wait = 0;
      let shown = [];
      const fast = rate() >= 10;
      runs.forEach((r, i) => {
        const p = paramsOf(racers[i]), events = P.e > 0 ? [...r.replay(P.e - 1).events] : [];
        const d = racers[i].algorithm.show(r.at(P.e), r.env, p, P.e);
        views[i].show(d, p);
        if (views[i].heatEpisode && heat === "rebuild") {
          views[i].heat.clear();
          for (let u = Math.max(0, P.e - 20); u < P.e; u++) views[i].heatEpisode(u === P.e - 1 ? events : [...r.replay(u).events], { draw: false });
          views[i].drawHeat();
        }
        views[i].rest(events, draw, { trail: !fast });
        // at quick rates single episodes cannot be followed: the greedy path shows where learning is heading
        if (views[i].path && env.kind === "grid") views[i].path(fast && d.Q && !env.slip ? greedyStates(d.Q) : null);
        caption(i);
        if (i === P.focus) shown = events;
      });
      // The panel follows the jump: a recorded run's last step, whose numbers the view shows; otherwise its prompt.
      const a = racers[P.focus].algorithm, last = a.recorded && shown.findLast((ev) => ev.type === "choose");
      if (last) { texInto(liveNum, a.numbers(last)); liveNote.innerHTML = a.note(last); liveNote.classList.remove("faint"); }
      else idleNote();
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
      if (!runs[i]) return;
      const stat = stages[i].querySelector(".stat"), r = runs[i], [one] = noun();
      if (w) { stat.textContent = `${cap(one)} ${P.e + 1}: ${live(w)}`; return; }
      if (!P.e) { stat.textContent = "Ready: nothing learned yet"; return; }
      const t = P.e - 1, M = r.metrics;
      let text;
      if (unitOf() === "sweep") text = M.improved?.[t] ? (M.changed[t] ? `improvement: ${plural(M.changed[t], ["state", "states"])} changed` : "improvement: nothing changed, done") : M.converged[t] && racers[i].algorithm.id === "policy-iteration" ? "done" : `largest change ${M.delta[t].toFixed(3)}`;
      else if (env.kind === "bandit") text = `reward ${signed(M.return[t])}${M.optimal[t] ? " · a best arm" : ""}`;
      else if (env.kind === "blackjack") text = M.return[t] > 0 ? "won (+1)" : M.return[t] < 0 ? "lost (−1)" : "a draw (0)";
      else if (env.kind === "star") text = `weights ‖w‖ = ${fmtBig(M.weights[t])}`;
      else if (env.kind === "car") text = `${plural(M.steps[t], ["step", "steps"])} to the flag${M.steps[t] >= runs[i].params.maxSteps ? " (cut short)" : ""}`;
      else if (env.kind === "graph") text = `${M.left?.[t] ? "went left" : "went right"} · return ${signed(M.return[t])}`;
      else if (env.kind === "throw") text = `thrown at ${M.angle[t].toFixed(1)}°: ${M.return[t].toFixed(1)} m`;
      else if (recorded) text = `test episode: ${plural(M.steps[t], ["step", "steps"])}${env.kind === "cartpole" ? "" : `, return ${signed(M.test[t], 0)}`}`;
      else if (unitOf() === "round") text = `${plural(paramsOf(racers[i]).workers ?? 4, ["episode", "episodes"])}, ${M.steps[t].toFixed(M.steps[t] < 100 ? 1 : 0)} steps on average · reward ${signed(M.return[t], env.rewards?.small !== undefined ? 2 : 0)}`;
      else text = `${signed(M.return[t], 0)} · ${plural(M.steps[t], ["step", "steps"])}${M.falls[t] ? ` · ${plural(M.falls[t], ["fall", "falls"])}` : ""}`;
      stat.textContent = `${cap(one)} ${P.e}: ${text}`;
    }
    const live = (w) => (recorded ? `step ${w.n} of the test episode` : unitOf() === "sweep" ? `state ${w.n}` : env.kind === "bandit" ? "pulling" : env.kind === "star" ? "one step" : env.kind === "throw" ? "throwing"
      : env.kind === "car" ? `${plural(w.n, ["step", "steps"])} so far` : unitOf() === "round" ? `${plural(w.n, ["move", "moves"])} so far, all workers together`
        : `${signed(w.G, env.kind === "grid" ? 0 : 2)} so far · ${plural(w.n, ["step", "steps"])}`);

    // ---- walking through a unit, event by event ----
    function walk(toUpdate) {
      if (!runs.length) return; // the played runs are still training
      if (!P.walkers) {
        if (P.e >= knobs.units) { pause(); return; }
        P.walkers = runs.map((r, i) => ({ ...r.replay(P.e), done: false, G: 0, n: 0, p: paramsOf(racers[i]), log: [] }));
        if (recorded) P.walkers.forEach((w, i) => views[i].show(w.m, w.p)); // the network that plays this test episode
      }
      let wait = 0;
      // A tick shows one event ("line by line"), or runs to the next update, but never past a second move: an
      // algorithm that updates only after the episode (Monte Carlo, REINFORCE), or n steps later, still walks it step
      // by step, its pseudocode in step with the agent. (A round's workers all move in the same tick.)
      P.walkers.forEach((w, i) => {
        let moved = false;
        while (!w.done) {
          let ev = w.held;
          w.held = null;
          if (!ev) {
            const next = w.events.next();
            if (next.done) { w.done = true; break; }
            ev = next.value;
          }
          if (toUpdate && moved && ev.type === "move" && ev.w === undefined) { w.held = ev; break; }
          wait = Math.max(wait, on(i, w, ev));
          if (ev.type === "move") moved = true;
          if (!toUpdate || ev.type === "update" || ev.type === "improve" || ev.type === "plan" || ev.type === "tick" || ev.type === "advantage") break;
        }
      });
      P.wait = wait;
      if (P.walkers.every((w) => w.done)) {
        P.walkers.forEach((w, i) => views[i].heatEpisode?.(w.log));
        P.e += 1;
        P.walkers = null;
        P.wait = Math.max(P.wait, unitOf() === "episode" ? 700 : 250);
        runs.forEach((_, i) => caption(i));
        if (P.e >= knobs.units) pause();
      }
      if (speed().quick) P.wait *= speed().every / 400; // the quick paces keep the pauses short too
      position();
    }

    // Show one event of racer i. Returns how long to pause after it (a fall, a hand being dealt…).
    function on(i, w, ev) {
      const a = racers[i].algorithm;
      if (ev.type === "move") { w.G += ev.r; w.n += 1; w.log.push(ev); }
      if (ev.type === "sweep") w.n += 1;
      if (SHOWN.has(ev.type)) views[i].show(a.show(w.m, runs[i].env, w.p, P.e), w.p);
      const pauseFor = views[i].event(ev, { line: P.speed === "line", p: w.p }) || 0;
      if (i === P.focus) {
        mark(ev.line);
        if (ev.type === "update") explain(ev, w.p);
        else if (ev.type === "plan") planned(ev);
        else if (ev.type === "advantage") batchAdvantages(ev);
        else if (ev.type === "choose" && a.recorded) { texInto(liveNum, a.numbers(ev)); liveNote.innerHTML = a.note(ev); liveNote.classList.remove("faint"); }
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
        : `<b>${plural(n, ["planning update", "planning updates"])}</b> on remembered moves${ev.ordered ? `, the most urgent first${ev.left !== undefined ? `; ${plural(ev.left, ["pair waits", "pairs wait"])} in the queue` : ""}` : ", picked at random"} · the largest surprise was <b class="q-err">${signed(big, 3)}</b>${ev.list.some((u) => u.bonus > 1e-9) ? " · bonuses for moves not tried in a while included" : ""}`;
      liveNote.classList.remove("faint");
    }

    // A batch of experience, before the update: how many steps, and how its advantages came out.
    function batchAdvantages(ev) {
      let up = 0;
      for (const u of ev.list) if (u.adv > 0) up++;
      liveNum.innerHTML = "";
      liveNote.innerHTML = `The round is over: <b>${ev.samples.toLocaleString("en")}</b> steps, each with its advantage, the GAE estimate of how much better its action did than expected · <b>${Math.round((100 * up) / Math.max(1, ev.samples))}%</b> better (blue sparks), the rest worse (orange)`;
      liveNote.classList.remove("faint");
    }

    // ---- the live formula ----
    // At rest, the panel says how to fill it. The note is rewritten whole: stepping replaces its contents.
    function idleNote() {
      liveNum.innerHTML = "";
      liveNote.innerHTML = recorded ? "Play <b>step by step</b> to see, at every step of a test episode, what the network makes of each move."
        : `Play <b>line by line</b> or <b>${speedList()[1].label.toLowerCase()}</b> to see every update with its numbers.`;
      liveNote.classList.add("faint");
    }

    let explained = 0;
    function explain(ev, p) {
      const a = racers[P.focus].algorithm;
      if (!a.numbers) return;
      // at the quick paces the numbers change too fast to read: refresh them a few times a second
      if (speed().quick) { const now = performance.now(); if (now - explained < 250) return; explained = now; }
      texInto(liveNum, a.numbers(ev, p));
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

    // A display formula in the side panel, fitted to its width: it shrinks a little, then stacks the parts written
    // side by side (\\qquad), as Textbook formulas do.
    function texInto(box, src) {
      box.innerHTML = '<div class="tex tex-display"></div>';
      const div = box.firstChild;
      div.dataset.src = src;
      div.dataset.done = "1";
      div.innerHTML = RL.math.tex(src, true);
      RL.math.fit(box);
    }

    function mark(line) {
      const li = line ? pseudo.querySelector(`li[data-line="${line}"]`) : null;
      for (const x of pseudo.querySelectorAll("li.on")) if (x !== li) x.classList.remove("on", "flash");
      if (!li) return;
      li.classList.add("on");
      li.classList.remove("flash");
      void li.offsetWidth; // restart the flash: the simulation came to this line again
      li.classList.add("flash");
    }

    function focus(i) {
      P.focus = i;
      stages.forEach((st, k) => st.classList.toggle("focus", k === i && racers.length > 1));
      const a = racers[i].algorithm, p = paramsOf(racers[i]);
      q(".algo-name").textContent = racers[i].name;
      const code = (r) => RL.entry(r.algorithm.station || r.algorithm.id)?.pseudocode || "";
      pseudo.innerHTML = code(racers[i]) || '<p class="faint">The pseudocode of this algorithm is not written yet.</p>';
      // the lines of this algorithm that none of the other racers' algorithms has
      const others = racers.filter((r) => (r.algorithm.station || r.algorithm.id) !== (a.station || a.id));
      const known = new Set(others.flatMap((r) => lineText(code(r))));
      const lines = lineText(pseudo.innerHTML);
      pseudo.querySelectorAll("li").forEach((li, k) => li.classList.toggle("differs", others.length > 0 && !known.has(lines[k])));
      q(".pseudo-diff").hidden = !others.length || !pseudo.querySelector("li.differs");
      q(".pseudo-diff").innerHTML = `Marked: where ${esc(racers[i].name)} differs from ${others.map((r) => esc(r.name)).filter((v, k, l) => l.indexOf(v) === k).join(" and ")}.`;
      q(".pseudo-tabs").querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.i === i));
      pseudo.closest(".panel").style.setProperty("--k-focus", `var(--s${i + 1})`);
      texInto(liveSym, typeof a.rule === "function" ? a.rule(p) : a.rule);
      liveNum.innerHTML = "";
      RL.math.render(pseudo);
      if (runs.length) { filmstrip(); if (racers.length > 2 && view.mode === "bench" && bench?.seedsDone) drawCharts(); }
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
    const speed = () => speedList().find((s) => s.id === P.speed);
    const rate = () => speed().rate || 0;
    function stepOnce() {
      pause();
      if (rate()) seek(P.e + 1, true);
      else walk(P.speed !== "line");
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
          const k = Math.min(Math.floor(P.acc), knobs.units - P.e);
          // every unit passed adds to the views' maps of where the agent goes, the last one through seek
          runs.forEach((r, i) => { if (views[i].heatEpisode) for (let u = P.e; u < P.e + k - 1; u++) views[i].heatEpisode([...r.replay(u).events], { draw: false }); });
          runs.forEach((r, i) => views[i].heatEpisode?.(P.e + k - 1 >= 0 && k > 0 ? [...r.replay(P.e + k - 1).events] : []));
          seek(P.e + k, rate() <= 1, "keep");
          P.acc -= Math.floor(P.acc);
        }
        return;
      }
      if (P.wait > 0) { P.wait -= dt; return; }
      P.acc += dt;
      if (P.acc >= speed().every) { P.acc = 0; walk(P.speed !== "line"); }
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
    // Play another seed with the same settings: the bench stays, only the played run changes.
    function chooseSeed(seed) {
      knobs.seed = seed;
      played = seed === "typical" ? (bench?.complete ? bench.typical() : played) : seed;
      pause();
      playRuns();
    }
    q(".knobs").addEventListener("click", (e) => {
      if (e.target.closest(".seed-dice")) chooseSeed(1 + Math.floor(Math.random() * 99999));
      else if (e.target.closest(".seed-typ")) chooseSeed("typical");
    });
    q(".knobs").addEventListener("change", (e) => {
      if (!e.target.closest(".seed-in")) return;
      const v = Math.round(+e.target.value);
      if (Number.isFinite(v) && v >= 1) chooseSeed(Math.min(999999, v));
      else e.target.value = played;
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
      const mb = e.target.closest("[data-mode]");
      if (mb) {
        view.mode = mb.dataset.mode;
        mb.parentElement.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === mb));
        drawCharts();
        return;
      }
      const b = e.target.closest("[data-seeds]");
      if (!b || +b.dataset.seeds === +view.seeds) return;
      view.seeds = b.dataset.seeds === "1";
      b.parentElement.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      if (avg && avg.done === avg.n) { // the runs are all done (counted for the odds): only the charts change
        drawCharts(view.seeds ? avg : undefined);
        if (view.seeds) q(".chart-note").textContent = chartNote(avg.n);
        return;
      }
      job?.cancel();
      drawCharts();
      if (view.seeds || preset.success) averageRecorded();
    });
    q(".world-list").addEventListener("click", (e) => { const b = e.target.closest("[data-world]"); if (b) switchWorld(b.dataset.world); });
    q(".pseudo-tabs").addEventListener("click", (e) => { const b = e.target.closest("[data-i]"); if (b) focus(+b.dataset.i); });
    q(".sweep-go").addEventListener("click", sweep);
    q(".sweep-runs").addEventListener("change", () => { runsChosen = true; });
    q(".sweep-knob").addEventListener("change", autoRuns);
    playBtn.addEventListener("click", () => (P.playing ? pause() : play()));
    q(".step").addEventListener("click", stepOnce);
    q(".restart").addEventListener("click", () => { pause(); seek(0); });
    q(".speed").addEventListener("change", (e) => {
      const wasWalking = !rate();
      P.speed = e.target.value;
      views.forEach((v) => v.setPace?.(speed().quick ? P.speed : ""));
      if (!wasWalking || rate()) seek(P.e, false, "keep"); // the trail and the greedy path follow the new pace
      if (wasWalking && rate() && P.walkers) seek(P.e);
    });
    scrub.addEventListener("input", () => { pause(); seek(parseInt(scrub.value, 10)); });
    // ---- the settings drawer: the world and how the views draw, out of the way until asked for ----
    const drawerBtn = q(".settings-btn"), drawer = q(".lab-drawer");
    function setDrawer(open, focusIt = false) {
      if ((host.querySelector(".lab").dataset.drawer === "open") === open) return;
      host.querySelector(".lab").dataset.drawer = open ? "open" : "closed";
      drawerBtn.setAttribute("aria-expanded", String(open));
      if (focusIt) (open ? drawer.querySelector(".wchip.on, button, input") : drawerBtn)?.focus({ preventScroll: true });
    }
    // focus follows the drawer only for the keyboard (a click has a detail count; Enter and Space do not)
    drawerBtn.addEventListener("click", (e) => setDrawer(host.querySelector(".lab").dataset.drawer !== "open", e.detail === 0));
    q(".ld-close").addEventListener("click", (e) => setDrawer(false, e.detail === 0));
    // a click anywhere else closes it
    function onOutside(e) {
      if (host.querySelector(".lab")?.dataset.drawer === "open" && !drawer.contains(e.target) && !drawerBtn.contains(e.target)) setDrawer(false);
    }
    document.addEventListener("pointerdown", onOutside);
    function onKey(e) {
      if (e.key === "Escape" && host.querySelector(".lab").dataset.drawer === "open") { setDrawer(false, true); return; }
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
        clearTimeout(benchTimer);
        strip?.destroy();
        clearSweep("");
        removeEventListener("keydown", onKey);
        document.removeEventListener("pointerdown", onOutside);
        resized.disconnect();
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
