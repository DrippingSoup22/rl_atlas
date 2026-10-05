/* Figures of Part 6, n-step methods and eligibility traces: the parameter studies on the 19-state random walk,
   the weights of the λ-return, and what one episode teaches one-step and n-step SARSA in the maze. */
(function (RL) {
  "use strict";
  const { lab } = RL;

  // ---- a maze drawn as a dry line figure, shared with Part 7 ----
  // Returns { svg, cx, cy, W, H }: the grid, its walls (hatched), S and the gem, at tile size T, its corner at (X, Y).
  RL.fig.maze = function (env, { T = 22, X = 6, Y = 6, id = RL.fig.uid() } = {}) {
    const W = env.cols * T + 12, H = env.rows * T + 12; // the size of the maze itself, wherever (X, Y) puts it
    const cx = (s) => X + (s % env.cols) * T + T / 2, cy = (s) => Y + Math.floor(s / env.cols) * T + T / 2;
    let svg = `<defs><pattern id="hatch-${id}" class="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5"/></pattern>
      <marker id="tip-${id}" class="tip-mark" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>`;
    for (let s = 0; s < env.nS; s++) if (env.tile(s) === "#") svg += `<rect class="wall" x="${cx(s) - T / 2}" y="${cy(s) - T / 2}" width="${T}" height="${T}" fill="url(#hatch-${id})"/>`;
    for (let r = 0; r <= env.rows; r++) svg += `<line class="cell" x1="${X}" x2="${X + env.cols * T}" y1="${Y + r * T}" y2="${Y + r * T}"/>`;
    for (let c = 0; c <= env.cols; c++) svg += `<line class="cell" x1="${X + c * T}" x2="${X + c * T}" y1="${Y}" y2="${Y + env.rows * T}"/>`;
    const goal = Array.from({ length: env.nS }, (_, s) => s).find((s) => env.tile(s) === "G");
    svg += `<rect class="frame" x="${X}" y="${Y}" width="${env.cols * T}" height="${env.rows * T}"/>
      <text class="maze-mark" x="${cx(env.start)}" y="${cy(env.start) + 5}" text-anchor="middle">S</text>
      <text class="maze-mark" x="${cx(goal)}" y="${cy(goal) + 5}" text-anchor="middle">G</text>`;
    return { svg, cx, cy, W, H, id, T };
  };
  // An arrow inside tile s pointing in direction a (up, right, down, left).
  RL.fig.mazeArrow = (m, s, a, cls = "pol-arrow") => {
    const [dx, dy] = [[0, -1], [1, 0], [0, 1], [-1, 0]][a], r = m.T * 0.34;
    return `<line class="${cls}" x1="${m.cx(s) - dx * r}" y1="${m.cy(s) - dy * r}" x2="${m.cx(s) + dx * r}" y2="${m.cy(s) + dy * r}" marker-end="url(#tip-${m.id})"/>`;
  };

  // ---- parameter studies on the 19-state random walk (Sutton & Barto, Figures 7.2, 12.3 and 12.6) ----
  // Each point: the RMS error over the 19 states, averaged over the first 10 episodes and over many runs.
  const ALPHAS = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
  function study(host, { label, curves, run, ylabel, top = 0.55 }) {
    const p = RL.fig.plot(host, {
      label, right: 92,
      x: { min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], format: (v) => String(v), label: "α, the step size" },
      y: { min: 0.25, max: top, ticks: [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55].filter((v) => v <= top), label: ylabel || "Average RMS error, first 10 episodes", digits: 3, tickDigits: 2 },
      curves: curves.map((c, i) => ({ id: c.id, name: c.name, light: i % 2 === 1 })),
      at: (a) => `α = ${+a.toFixed(2)}`,
    });
    const runs = 100, xs = ALPHAS.slice(1), sums = curves.map(() => new Float64Array(xs.length));
    RL.fig.whenVisible(host, () => {
      let done = 0;
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (done < runs && performance.now() - t0 < 30) {
          curves.forEach((c, j) => xs.forEach((alpha, i) => {
            const { metrics } = lab.simulate({ world: "random-walk-19", algorithm: lab.algorithms[c.algorithm], params: { gamma: 1, alpha, ...c.params }, units: 10, seed: 1000 + done, snapshots: false, measures: ["error"] });
            const e = lab.mean(metrics.error);
            sums[j][i] += Number.isFinite(e) ? Math.min(e, 1) : 1; // a run that blew up counts as an error of 1, off the chart
          }));
          done++;
        }
        // Points off the top of the chart are left out, as in the book: the curve leaves the chart and does not come back.
        p.draw(Object.fromEntries(curves.map((c, j) => {
          const keep = xs.map((a, i) => i).filter((i) => sums[j][i] / done <= top + 0.02);
          return [c.id, { xs: keep.map((i) => xs[i]), ys: keep.map((i) => sums[j][i] / done) }];
        })));
        p.status(done < runs ? `Averaging run ${done} of ${runs}…` : `Each point: the average of ${runs} runs of 10 episodes. ${run}`);
        if (done < runs) setTimeout(more, 16);
      };
      more();
    });
  }

  RL.demos["n-step-study"] = function (host) {
    study(host, {
      label: "Average RMS error of n-step TD on the 19-state random walk, for several n, against the step size",
      curves: [1, 2, 4, 8, 16, 32, 64, 256].map((n) => ({ id: `n${n}`, name: `n = ${n}`, algorithm: "n-step-td", params: { n } })),
      run: "Points above the chart are left out.",
    });
  };

  RL.demos["lambda-study"] = function (host, arg = "offline") {
    const online = arg === "online";
    study(host, {
      label: `Average RMS error of ${online ? "TD(λ)" : "the offline λ-return algorithm"} on the 19-state random walk, for several λ, against the step size`,
      curves: [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1].map((l) => ({ id: `l${l}`, name: `λ = ${l}`, algorithm: online ? "td-lambda" : "offline-lambda", params: { lambda: l } })),
      run: online ? "Points above the chart are left out; for large λ and α the estimates grow without bound." : "Points above the chart are left out.",
    });
  };

  // ---- the weights of the λ-return: (1 − λ)λ^(n−1) on the n-step return, and what is left on the full return ----
  RL.demos["lambda-weights"] = function (host) {
    const steps = 12, W = 640, H = 270, M = { l: 62, r: 20, t: 16, b: 50 }, pw = W - M.l - M.r, ph = H - M.t - M.b;
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="The weights the λ-return gives to each n-step return"></svg>
      <label class="fig-knob"><span>λ</span><input type="range" min="0" max="1" step="0.01" value="0.8"><output>0.8</output></label>`;
    const svg = host.querySelector("svg"), input = host.querySelector("input"), out = host.querySelector("output");
    const bw = pw / steps, y = (v) => M.t + (1 - v) * ph;
    function draw() {
      const lam = +input.value;
      out.textContent = lam.toFixed(2);
      let g = "";
      for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="gridline" x1="${M.l}" x2="${M.l + pw}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
      for (let n = 1; n <= steps; n++) {
        const w = n < steps ? (1 - lam) * lam ** (n - 1) : lam ** (steps - 1), x = M.l + (n - 1) * bw + 4;
        g += `<rect class="bar${n === steps ? " last" : ""}" x="${x}" y="${y(w)}" width="${bw - 8}" height="${Math.max(0, y(0) - y(w))}"/>
          <text class="tick" x="${x + (bw - 8) / 2}" y="${M.t + ph + 18}" text-anchor="middle">${n < steps ? n : "end"}</text>`;
        if (w > 0.005) g += `<text class="num small" x="${x + (bw - 8) / 2}" y="${y(w) - 5}" text-anchor="middle">${w.toFixed(2)}</text>`;
      }
      g += `<line class="axis" x1="${M.l}" x2="${M.l + pw}" y1="${y(0)}" y2="${y(0)}"/><line class="axis" x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${y(0)}"/>
        <text class="axis-name" x="${M.l + pw / 2}" y="${H - 8}" text-anchor="middle">n, the number of real rewards in the return (the episode ends after ${steps})</text>
        <text class="axis-name" transform="translate(16 ${M.t + ph / 2}) rotate(-90)" text-anchor="middle">weight</text>`;
      svg.innerHTML = g;
    }
    input.addEventListener("input", draw);
    draw();
  };

  // ---- the trace of one state, visited at a few moments: accumulating and replacing (after Figure 12.4 and §12.5) ----
  RL.demos["trace-shapes"] = function (host) {
    const visits = new Set([2, 4, 5, 6, 14]), fade = 0.8, T = 24, acc = new Float64Array(T + 1), rep = new Float64Array(T + 1);
    for (let t = 1; t <= T; t++) {
      acc[t] = fade * acc[t - 1] + (visits.has(t) ? 1 : 0);
      rep[t] = visits.has(t) ? 1 : fade * rep[t - 1];
    }
    const p = RL.fig.plot(host, {
      label: "The eligibility trace of one state visited at steps 2, 4, 5, 6 and 14, accumulating and replacing, with γλ = 0.8",
      h: 240, right: 120,
      x: { min: 0, max: T, from: 0, ticks: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], label: "Time step (the state is visited at steps 2, 4, 5, 6 and 14)" },
      y: { min: 0, max: 2.5, ticks: [0, 0.5, 1, 1.5, 2, 2.5], label: "Trace z", digits: 2, tickDigits: 1 },
      curves: [{ id: "acc", name: "accumulating" }, { id: "rep", name: "replacing", dash: true }],
      at: (t) => `Step ${t}`,
    });
    p.draw({ acc, rep });
    p.status("Every step the trace fades by γλ = 0.8; a visit adds 1 (accumulating) or resets it to 1 (replacing).");
  };

  // ---- one episode in the maze: the moves that one-step and n-step SARSA strengthen (after Figure 7.4) ----
  // arg "lambda": the third panel shows SARSA(λ) with λ = 0.9 instead, each arrow as dark as the value it gained.
  RL.demos["n-step-paths"] = function (host, arg) {
    const env = lab.make("dyna-maze"), params = { alpha: 0.1, epsilon: 0.1, gamma: 0.95 };
    // A seed whose first episode is short enough to read: the path is the same for every n (all values are 0 until the gem).
    let seed = 1, path = [];
    for (; seed < 500; seed++) {
      path = lab.simulate({ world: "dyna-maze", algorithm: lab.algorithms["n-step-sarsa"], params: { ...params, n: 1 }, units: 1, seed }).trail(0);
      if (path.length > 22 && path.length < 34) break;
    }
    const panels = [{ title: "Path taken" }, { title: "One-step SARSA", n: 1 },
      arg === "lambda" ? { title: "SARSA(λ), λ = 0.9", lambda: 0.9 } : { title: "10-step SARSA", n: 10 }];
    const id = RL.fig.uid(), parts = [];
    let x0 = 0, W = 0, H = 0;
    for (const panel of panels) {
      const m = RL.fig.maze(env, { T: 20, X: x0 + 6, Y: 26, id: `${id}-${parts.length}` });
      let g = m.svg + `<text class="note" x="${x0 + m.W / 2}" y="16" text-anchor="middle">${panel.title}</text>`;
      if (!panel.n) g += `<path class="route" marker-end="url(#tip-${m.id})" d="M${path.map((s) => `${m.cx(s)} ${m.cy(s)}`).join(" L")}"/>`;
      else {
        const Q = panel.lambda
          ? lab.simulate({ world: "dyna-maze", algorithm: lab.algorithms["sarsa-lambda"], params: { ...params, lambda: panel.lambda }, units: 1, seed }).at(1).Q
          : lab.simulate({ world: "dyna-maze", algorithm: lab.algorithms["n-step-sarsa"], params: { ...params, n: panel.n }, units: 1, seed }).at(1).Q;
        const top = Math.max(...Q);
        for (let s = 0; s < env.nS; s++) for (let a = 0; a < 4; a++) {
          if (Q[s * 4 + a] <= 1e-4) continue;
          const arrow = RL.fig.mazeArrow(m, s, a);
          g += panel.lambda ? arrow.replace("<line ", `<line style="opacity:${(0.25 + 0.75 * Math.sqrt(Q[s * 4 + a] / top)).toFixed(2)}" `) : arrow;
        }
      }
      parts.push(g);
      x0 += m.W + 22;
      W = x0 - 22;
      H = Math.max(H, m.H + 26);
    }
    host.innerHTML = `<svg class="fig maze" viewBox="0 0 ${W} ${H}" role="img" aria-label="One episode in the maze, and the moves one-step and 10-step SARSA strengthen after it">${parts.join("")}</svg>`;
  };
})(globalThis.RL = globalThis.RL || {});
