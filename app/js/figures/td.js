/* Figures of Part 5, temporal-difference learning: the cliff, the random walk, Expected SARSA and maximization bias. */
(function (RL) {
  "use strict";
  const { lab } = RL;

  // ---- the cliff world (Sutton & Barto, Example 6.6) ----
  const CLIFF = { alpha: 0.5, epsilon: 0.1, gamma: 1 };
  const cliffRun = (id, seed) => lab.simulate({ world: "cliff", algorithm: lab.algorithms[id], params: CLIFF, units: 500, seed });
  const ROUTES = [
    { id: "sarsa", label: "SARSA: the safe path", dashed: true, dx: -3 },
    { id: "q-learning", label: "Q-learning: the optimal path", dashed: false, dx: 3 },
  ];

  // The grid, the cliff, and the path each algorithm's greedy policy takes after 500 episodes.
  RL.demos["cliff-paths"] = function (host) {
    const env = lab.make("cliff"), T = 34, X = 12, Y = 10, W = env.cols * T + 2 * X, H = env.rows * T + Y + 50;
    const id = RL.fig.uid();
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="The cliff world and the paths SARSA and Q-learning learn"></svg>`;
    const svg = host.firstElementChild;
    RL.fig.whenVisible(host, () => {
      const cx = (s) => X + (s % env.cols) * T + T / 2, cy = (s) => Y + Math.floor(s / env.cols) * T + T / 2;
      const bottom = Y + env.rows * T, S = env.start, G = env.nS - 1;
      let g = `<defs>
          <pattern id="hatch-${id}" class="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7"/></pattern>
          <marker id="tip-${id}" class="tip-mark" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker>
        </defs>
        <rect class="cliff" x="${X + T}" y="${Y + 3 * T}" width="${10 * T}" height="${T}" fill="url(#hatch-${id})"/>`;
      for (let r = 0; r <= env.rows; r++) g += `<line class="cell" x1="${X}" x2="${X + env.cols * T}" y1="${Y + r * T}" y2="${Y + r * T}"/>`;
      for (let c = 0; c <= env.cols; c++) g += `<line class="cell" x1="${X + c * T}" x2="${X + c * T}" y1="${Y}" y2="${bottom}"/>`;
      g += `<rect class="frame" x="${X}" y="${Y}" width="${env.cols * T}" height="${env.rows * T}"/>
        <text class="halo cliff-name" x="${X + 6 * T}" y="${cy(S) + 5}" text-anchor="middle">The Cliff</text>
        <text class="big" x="${cx(S)}" y="${cy(S) + 6}" text-anchor="middle">S</text>
        <text class="big" x="${cx(G)}" y="${cy(G) + 6}" text-anchor="middle">G</text>
        <path class="back" marker-end="url(#tip-${id})" d="M${X + 6 * T} ${bottom + 3} C${X + 6 * T} ${bottom + 34} ${cx(S)} ${bottom + 34} ${cx(S)} ${bottom + 5}"/>
        <text class="note" x="${X + 3.5 * T}" y="${bottom + 44}" text-anchor="middle">R = −100, and back to the start</text>
        <clipPath id="wipe-${id}"><rect class="wipe" x="0" y="0" width="${W}" height="${H}"/></clipPath>`;

      let routes = "";
      for (const route of ROUTES) {
        const { path } = lab.greedyPath(env, cliffRun(route.id, 7).at(500).Q);
        const pts = path.map((s) => [cx(s) + route.dx, cy(s)]);
        pts[0][1] = Y + 3 * T; // leave S and G readable: start on the edge of S, stop just above G
        pts[pts.length - 1][1] = Y + 3 * T - 4;
        routes += `<path class="route${route.dashed ? " dashed" : ""}" marker-end="url(#tip-${id})" d="M${pts.map((p) => p.join(" ")).join(" L")}"/>`;
        // The label goes under the longest straight stretch of the path.
        let best = [0, 0];
        for (let i = 0, j = 0; i < pts.length; i = j) {
          for (j = i + 1; j < pts.length && pts[j][1] === pts[i][1]; j++);
          if (j - i > best[1] - best[0]) best = [i, j];
        }
        const [a, b] = best, lx = (pts[a][0] + pts[b - 1][0]) / 2;
        routes += `<text class="halo route-name" x="${lx}" y="${pts[a][1] + 15}" text-anchor="middle">${route.label} (${path.length - 1} steps)</text>`;
      }
      svg.innerHTML = `${g}<g clip-path="url(#wipe-${id})">${routes}</g>`;
      requestAnimationFrame(() => svg.classList.add("on"));
    });
  };

  // Reward per episode while learning, averaged over many runs: the curves smooth out as runs come in.
  RL.demos["cliff-curves"] = function (host) {
    RL.fig.average(host, {
      label: "Reward per episode while learning, SARSA and Q-learning",
      x: { max: 500, ticks: [0, 100, 200, 300, 400, 500], label: "Episodes" },
      y: { min: -100, max: 0, ticks: [-100, -75, -50, -25, 0], label: "Sum of rewards during episode", digits: 1, tickDigits: 0 },
      curves: ROUTES.map((r) => ({ id: r.id, name: r.id === "sarsa" ? "SARSA" : "Q-learning", dash: r.dashed })),
      at: (e) => `Episode ${e}`,
    }, {
      runs: 100, smooth: 10,
      sample: (seed) => Object.fromEntries(ROUTES.map((r) => [r.id, cliffRun(r.id, seed).metrics.return])),
    });
  };

  // ---- the random walk (Sutton & Barto, Example 6.2) ----
  const WALK = (n) => Array.from({ length: n }, (_, i) => i + 1);
  const LETTER = (u) => "ABCDE"[u - 1] || "";

  // "values": what TD(0) has learned after 0, 1, 10 and 100 episodes, against the true values.
  // "error": the RMS error, averaged over 100 runs, for TD(0) and constant-α Monte Carlo at several step sizes.
  // "batch": the same, when all episodes so far are replayed until the estimates converge (batch training).
  RL.demos["random-walk"] = function (host, arg = "values") {
    const env = lab.make("random-walk"), truth = env.truth(), xs = WALK(env.n);
    if (arg === "values") {
      const run = lab.simulate({ world: "random-walk", algorithm: lab.algorithms.td0, params: { alpha: 0.1, gamma: 1, v0: 0.5 }, units: 100, seed: 1 });
      const p = RL.fig.plot(host, {
        label: "Values learned by TD(0) on the random walk after 0, 1, 10 and 100 episodes, and the true values",
        right: 110,
        x: { min: 0.6, max: 5.4, ticks: xs, format: LETTER, label: "State" },
        y: { min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], label: "Estimated value", digits: 2, tickDigits: 1 },
        curves: [
          { id: "truth", name: "true values", light: true, marks: false },
          { id: "e0", name: "0", dash: "dotted" }, { id: "e1", name: "1", dash: true },
          { id: "e10", name: "10", dash: true, light: true }, { id: "e100", name: "100 episodes" },
        ],
        at: (u) => `State ${LETTER(u)}`,
      });
      const pick = (V) => ({ xs, ys: xs.map((s) => V[s]) });
      p.draw({ truth: pick(truth), e0: pick(run.at(0).V), e1: pick(run.at(1).V), e10: pick(run.at(10).V), e100: pick(run.at(100).V) });
      p.status("TD(0) with α = 0.1, starting from 0.5 everywhere; one run.");
      return;
    }
    const RATES = { td0: [0.05, 0.1, 0.15], mc: [0.01, 0.02, 0.03, 0.04] };
    if (arg === "error") {
      const curves = [
        ...RATES.td0.map((a, i) => ({ id: `td${a}`, name: `TD, α = ${a}`, light: i === 2 })),
        ...RATES.mc.map((a, i) => ({ id: `mc${a}`, name: `MC, α = ${a}`, dash: true, light: i % 2 === 1 })),
      ];
      RL.fig.average(host, {
        label: "RMS error on the random walk, averaged over states and runs, for TD(0) and constant-α Monte Carlo",
        right: 120,
        x: { max: 100, ticks: [0, 25, 50, 75, 100], label: "Episodes" },
        y: { min: 0, max: 0.25, ticks: [0, 0.05, 0.1, 0.15, 0.2, 0.25], label: "RMS error", digits: 3, tickDigits: 2 },
        curves, at: (e) => `After episode ${e}`,
      }, {
        runs: 100,
        sample(seed) {
          const out = {};
          for (const a of RATES.td0) out[`td${a}`] = lab.simulate({ world: "random-walk", algorithm: lab.algorithms.td0, params: { alpha: a, gamma: 1, v0: 0.5 }, units: 100, seed, snapshots: false, measures: ["error"] }).metrics.error;
          for (const a of RATES.mc) out[`mc${a}`] = lab.simulate({ world: "random-walk", algorithm: lab.algorithms["mc-prediction"], params: { alpha: a, gamma: 1, v0: 0.5 }, units: 100, seed, snapshots: false, measures: ["error"] }).metrics.error;
          return out;
        },
      });
      return;
    }
    // Batch training: after each episode, both methods are run on all the episodes so far until they converge.
    // Batch MC converges to the average return after each state; batch TD(0) to the values of the
    // maximum-likelihood model of the chain (certainty equivalence), computed here directly.
    const n = env.n, episodes = 100;
    const walk = (rng) => { const path = [env.start]; let s = env.start; while (!env.terminal(s)) { s += rng.next() < 0.5 ? -1 : 1; path.push(s); } return path; };
    RL.fig.average(host, {
      label: "RMS error under batch training on the random walk, for TD(0) and Monte Carlo",
      right: 100,
      x: { max: episodes, ticks: [0, 25, 50, 75, 100], label: "Episodes" },
      y: { min: 0, max: 0.25, ticks: [0, 0.05, 0.1, 0.15, 0.2, 0.25], label: "RMS error", digits: 3, tickDigits: 2 },
      curves: [{ id: "td", name: "batch TD(0)" }, { id: "mc", name: "batch MC", dash: true }],
      at: (e) => `After episode ${e}`,
    }, {
      runs: 100,
      sample(seed) {
        const rng = lab.rng(seed), sums = new Float64Array(n + 2), counts = new Float64Array(n + 2);
        const moves = Array.from({ length: n + 2 }, () => new Float64Array(n + 2)), out = { td: new Float64Array(episodes), mc: new Float64Array(episodes) };
        for (let e = 0; e < episodes; e++) {
          const path = walk(rng), G = path[path.length - 1] === n + 1 ? 1 : 0;
          for (let i = 0; i < path.length - 1; i++) { sums[path[i]] += G; counts[path[i]] += 1; moves[path[i]][path[i + 1]] += 1; }
          const mc = new Float64Array(n + 2), td = new Float64Array(n + 2);
          for (let s = 1; s <= n; s++) { mc[s] = counts[s] ? sums[s] / counts[s] : 0.5; td[s] = 0.5; }
          // the maximum-likelihood chain: from each visited state, the observed frequencies of moving left and right
          for (let k = 0; k < 2000; k++) {
            let change = 0;
            for (let s = 1; s <= n; s++) {
              if (!counts[s]) continue;
              const v = (moves[s][s - 1] * (s - 1 === 0 ? 0 : td[s - 1]) + moves[s][s + 1] * (s + 1 === n + 1 ? 1 : td[s + 1])) / counts[s];
              change = Math.max(change, Math.abs(v - td[s]));
              td[s] = v;
            }
            if (change < 1e-9) break;
          }
          out.mc[e] = lab.rms(mc, truth, env);
          out.td[e] = lab.rms(td, truth, env);
        }
        return out;
      },
    });
  };

  // ---- Sutton & Barto, Figure 6.3: SARSA, Expected SARSA and Q-learning on the cliff, by step size ----
  RL.demos["cliff-alpha"] = function (host) {
    const alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
    const ALGOS = [["sarsa", "SARSA"], ["expected-sarsa", "Expected SARSA"], ["q-learning", "Q-learning"]];
    const STYLE = { sarsa: { dash: true }, "expected-sarsa": {}, "q-learning": { dash: "dotted" } };
    const p = RL.fig.plot(host, {
      label: "Average reward per episode on the cliff, over the first 100 episodes and over episodes 401 to 500, by step size",
      right: 200,
      x: { min: 0.1, max: 1, ticks: alphas, label: "Step size α" },
      y: { min: -160, max: 0, ticks: [-160, -120, -80, -40, 0], label: "Reward per episode", digits: 1, tickDigits: 0 },
      curves: ALGOS.flatMap(([id, name]) => [
        { id: `${id}:early`, name: `${name}, 1–100`, ...STYLE[id], light: true },
        { id: `${id}:late`, name: `${name}, 401–500`, ...STYLE[id] },
      ]),
      at: (a) => `α = ${a}`,
    });
    const early = 20, late = 10, sums = {}, counts = {};
    for (const [id] of ALGOS) for (const k of ["early", "late"]) { sums[`${id}:${k}`] = new Float64Array(alphas.length); counts[`${id}:${k}`] = new Float64Array(alphas.length); }
    const jobs = [];
    for (let r = 0; r < Math.max(early, late); r++) for (const [id] of ALGOS) alphas.forEach((a, i) => jobs.push([id, a, i, r]));
    RL.fig.whenVisible(host, () => {
      let j = 0;
      const more = () => {
        if (!host.isConnected) return;
        if (RL.scrolling()) return void setTimeout(more, 120); // the reader is scrolling: later
        const t0 = performance.now();
        while (j < jobs.length && performance.now() - t0 < 30) {
          const [id, alpha, i, r] = jobs[j++], units = r < late ? 500 : 100;
          const ret = lab.simulate({ world: "cliff", algorithm: lab.algorithms[id], params: { alpha, epsilon: 0.1, gamma: 1 }, units, seed: 2000 + r, snapshots: false }).metrics.return;
          if (r < early) { sums[`${id}:early`][i] += lab.mean(ret, 0, 100); counts[`${id}:early`][i] += 1; }
          if (r < late) { sums[`${id}:late`][i] += lab.mean(ret, 400, 500); counts[`${id}:late`][i] += 1; }
        }
        const lines = {};
        for (const k in sums) {
          const xs = [], ys = [];
          alphas.forEach((a, i) => { if (counts[k][i]) { xs.push(a); ys.push(sums[k][i] / counts[k][i]); } });
          if (xs.length) lines[k] = { xs, ys };
        }
        p.draw(lines);
        p.status(j < jobs.length ? `Running ${Math.round((100 * j) / jobs.length)}%…` : `ε = 0.1. Episodes 1–100 averaged over ${early} runs, episodes 401–500 over ${late} runs.`);
        if (j < jobs.length) setTimeout(more, 16);
      };
      more();
    });
  };

  // ---- Sutton & Barto, Figure 6.5: maximization bias, Q-learning against Double Q-learning ----
  RL.demos["max-bias"] = function (host) {
    const params = { alpha: 0.1, epsilon: 0.1, gamma: 1 }, units = 300;
    RL.fig.average(host, {
      label: "How often each method goes left from A, the worse choice, episode by episode",
      right: 130,
      x: { max: units, ticks: [0, 100, 200, 300], label: "Episodes" },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], label: "% left actions from A", percent: true },
      curves: [{ id: "q", name: "Q-learning" }, { id: "dq", name: "Double Q-learning", dash: true }],
      refs: [{ value: params.epsilon / 2, label: "best possible: 5%", at: 0 }],
      at: (e) => `Episode ${e}`,
    }, {
      runs: 1000,
      sample: (seed) => ({
        q: lab.simulate({ world: "max-bias", algorithm: lab.algorithms["q-learning"], params, units, seed, snapshots: false }).metrics.left,
        dq: lab.simulate({ world: "max-bias", algorithm: lab.algorithms["double-q"], params, units, seed, snapshots: false }).metrics.left,
      }),
    });
  };
})(globalThis.RL = globalThis.RL || {});
