/* Figures of Part 3, dynamic programming: sweeps of policy evaluation on the 4 × 4 gridworld, policy iteration and
   value iteration on Frozen Lake, and the picture of generalized policy iteration. All values come from the Lab. */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const ARROW = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  const fmt = (v, d) => (Math.abs(v) < 0.5 * 10 ** -d ? "0" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(d));

  // A small grid world drawn as a figure: values as numbers, a policy as arrows (tied actions share the cell).
  // V, P: arrays or null. Returns SVG markup placed at (x, y), cells T pixels wide.
  function miniGrid(env, { V = null, P = null, x = 0, y = 0, T = 30, digits = 1, tip }) {
    let g = "";
    for (let s = 0; s < env.nS; s++) {
      const [r, c] = env.rc(s), cx = x + c * T + T / 2, cy = y + r * T + T / 2, k = env.tile(s);
      const end = k === "T" || k === "H" || k === "G";
      g += `<rect class="cell${k === "T" ? " shade" : k === "H" ? " hole" : ""}" x="${x + c * T}" y="${y + r * T}" width="${T}" height="${T}"/>`;
      if (k === "G") g += `<text class="big" x="${cx}" y="${cy + 6}" text-anchor="middle">G</text>`;
      if (end) continue;
      if (V) g += `<text class="num small" x="${cx}" y="${cy + (P ? 11 : 4.5)}" text-anchor="middle">${fmt(V[s], digits)}</text>`;
      if (P) {
        const len = V ? 7 : 10, oy = V ? -5 : 0;
        for (let a = 0; a < 4; a++) {
          if (!(P[s * 4 + a] > 0)) continue;
          const [dx, dy] = ARROW[a];
          g += `<path class="pol-arrow" marker-end="url(#${tip})" d="M${cx + dx * 2} ${cy + oy + dy * 2} L${cx + dx * len} ${cy + oy + dy * len}"/>`;
        }
      }
    }
    return g + `<rect class="frame" x="${x}" y="${y}" width="${env.cols * T}" height="${env.rows * T}"/>`;
  }
  const marker = (id) => `<defs><marker id="${id}" class="tip-mark" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>`;

  // Panels in rows of perRow: each { V, P, label }. Returns the SVG.
  function panels(env, list, { T = 30, gap = 22, digits = 1, label = "", perRow = list.length } = {}) {
    const tip = `dp-tip-${RL.fig.uid()}`, w = env.cols * T, h = env.rows * T, rowH = h + 40;
    const cols = Math.min(perRow, list.length), rows = Math.ceil(list.length / cols);
    const W = cols * w + (cols - 1) * gap + 8, H = rows * rowH - 6;
    let g = marker(tip);
    list.forEach((p, i) => {
      const x = 4 + (i % cols) * (w + gap), y = 4 + Math.floor(i / cols) * rowH;
      g += miniGrid(env, { ...p, x, y, T, digits: p.digits ?? digits, tip });
      g += `<text class="note" x="${x + w / 2}" y="${y + h + 22}" text-anchor="middle">${p.label}</text>`;
    });
    return `<svg class="fig dp" viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}">${g}</svg>`;
  }

  // ---- Sutton & Barto, Figure 4.1: sweeps of policy evaluation (random policy, two arrays) and the greedy policies ----
  RL.demos["dp-sweeps"] = function (host) {
    const env = lab.make("small-gridworld");
    const run = lab.simulate({ world: "small-gridworld", algorithm: lab.algorithms["policy-evaluation"], params: { gamma: 1, theta: 1e-6, sync: true, policy: "random" }, units: 400, seed: 1 });
    const moments = [[0, "k = 0"], [1, "k = 1"], [2, "k = 2"], [3, "k = 3"], [10, "k = 10"], [400, "k = ∞"]];
    const list = moments.map(([k, label]) => {
      const V = run.at(k).V;
      return { V, P: lab.greedyPolicy(env, V, 1, 1e-9), label: `after ${label}` };
    });
    host.innerHTML = panels(env, list, { T: 42, gap: 30, perRow: 3, label: "Values of the random policy after k sweeps, and the greedy policy with respect to them" });
  };

  // ---- Frozen Lake: policy iteration's policies and values, or the optimal policy alone ----
  RL.demos.frozen = function (host, arg = "pi") {
    const env = lab.make("frozen-lake"), gamma = 0.99;
    if (arg === "optimal") {
      const V = lab.valueIteration(env, gamma, { theta: 1e-12 }), P = lab.greedyPolicy(env, V, gamma, 1e-6);
      const win = lab.evaluate(env, P, 1, { theta: 1e-12 });
      host.innerHTML = panels(env, [
        { V, digits: 2, label: "v*, with γ = 0.99" }, { P, label: "π*" },
        { V: win, P, digits: 2, label: "chance of reaching G under π*" },
      ], { T: 44, gap: 30, label: "The optimal values and policy of Frozen Lake" });
      return;
    }
    const run = lab.simulate({ world: "frozen-lake", algorithm: lab.algorithms["policy-iteration"], params: { gamma, theta: 1e-3 }, units: 200, seed: 1 });
    const list = [];
    let k = 0;
    for (let t = 0; t < 200; t++) {
      if (!run.metrics.improved[t]) continue;
      const d = run.algorithm.show(run.at(t), env, run.params);
      list.push({ V: d.V, P: d.P, digits: 2, label: `π${sub(k)} and its values` });
      k += 1;
      if (!run.metrics.changed[t]) break;
    }
    host.innerHTML = panels(env, list, { T: 40, gap: 26, label: "The policies of policy iteration on Frozen Lake, each with its values" });
  };
  const sub = (n) => String(n).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]);

  // ---- Policy iteration against value iteration on Frozen Lake, sweep by sweep ----
  RL.demos["dp-race"] = function (host) {
    const units = 120, params = { gamma: 0.99, theta: 1e-3 };
    const run = (id) => lab.simulate({ world: "frozen-lake", algorithm: lab.algorithms[id], params, units, seed: 1, snapshots: false, measures: ["optimal-error"] }).metrics;
    const pi = run("policy-iteration"), vi = run("value-iteration");
    const p = RL.fig.plot(host, {
      label: "Distance from the optimal values, sweep by sweep, for policy iteration and value iteration",
      right: 136,
      x: { max: units, from: 1, ticks: [0, 20, 40, 60, 80, 100, 120], label: "Sweeps through the states" },
      y: { min: 1e-3, max: 1, log: true, ticks: [0.001, 0.01, 0.1, 1], label: "RMS distance from v*", digits: 3 },
      curves: [{ id: "vi", name: "value iteration" }, { id: "pi", name: "policy iteration", dash: true }],
      at: (t) => `After sweep ${t}`,
    });
    p.draw({ vi: vi["optimal-error"], pi: pi["optimal-error"] });
    // Mark policy iteration's improvement steps.
    const marks = [];
    for (let t = 0; t < units; t++) if (pi.improved[t]) marks.push(t);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.innerHTML = marks.map((t) => `<line class="ref-line" x1="${p.x(t + 1)}" x2="${p.x(t + 1)}" y1="${p.y(1)}" y2="${p.y(1e-3)}"/>`).join("") +
      `<text class="note" x="${p.x(marks[0] + 1) + 5}" y="${p.y(0.7)}">improvement steps</text>`;
    p.svg.insertBefore(g, p.svg.firstChild);
    p.status(`γ = 0.99, θ = 0.001. Thin vertical lines: policy iteration's improvement steps (sweeps ${marks.map((t) => t + 1).join(", ")}).`);
  };

  // ---- Generalized policy iteration: evaluation and improvement pull toward the same point ----
  RL.demos.gpi = function (host) {
    const id = RL.fig.uid(), W = 560, H = 300;
    const top = [280, 34], L0 = [60, 270], R0 = [500, 270];
    const onLine = (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    // Beside a line, outside the triangle: a point at fraction f along it, pushed out by d along its normal.
    const beside = (a, f, side, d = 15) => {
      const [x, y] = onLine(a, top, f), len = Math.hypot(top[0] - a[0], top[1] - a[1]);
      const ux = (top[0] - a[0]) / len, uy = (top[1] - a[1]) / len;
      return side < 0 ? [x + uy * d, y - ux * d] : [x - uy * d, y + ux * d];
    };
    const angle = (a) => (Math.atan2(top[1] - a[1], top[0] - a[0]) * 180) / Math.PI;
    // a zigzag from the bottom, alternating between the two lines, closing in on the top
    const pts = [[236, 262]];
    let f = 0.12;
    for (let i = 0; i < 6; i++, f += (1 - f) * 0.36) pts.push(i % 2 === 0 ? onLine(L0, top, f) : onLine(R0, top, f));
    const [lx, ly] = beside(L0, 0.34, -1), [rx, ry] = beside(R0, 0.24, 1);
    let g = `<defs><marker id="gpi-tip-${id}" class="tip-mark" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>
      <line class="gpi-line" x1="${L0[0]}" y1="${L0[1]}" x2="${top[0]}" y2="${top[1]}"/>
      <line class="gpi-line" x1="${R0[0]}" y1="${R0[1]}" x2="${top[0]}" y2="${top[1]}"/>
      <text class="note" x="${lx}" y="${ly}" text-anchor="middle" transform="rotate(${angle(L0)} ${lx} ${ly})">v = v<tspan baseline-shift="sub" font-size="10">π</tspan></text>
      <text class="note" x="${rx}" y="${ry}" text-anchor="middle" transform="rotate(${angle(R0) + 180} ${rx} ${ry})">π = greedy(v)</text>
      <circle class="gpi-goal" cx="${top[0]}" cy="${top[1]}" r="6"/>
      <text class="big" x="${top[0]}" y="${top[1] - 14}" text-anchor="middle">v*, π*</text>
      <text class="note" x="${pts[0][0] + 10}" y="${pts[0][1] + 18}" text-anchor="middle">start: v, π</text>`;
    for (let i = 1; i < pts.length; i++) {
      const [a, b] = [pts[i - 1], pts[i]];
      g += `<path class="gpi-step" marker-end="url(#gpi-tip-${id})" d="M${a[0]} ${a[1]} L${b[0]} ${b[1]}"/>`;
    }
    const mid = (i) => [(pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2];
    const [e1, e2] = mid(0), [i1, i2] = mid(1);
    g += `<text class="note" x="${e1}" y="${e2 + 17}" text-anchor="middle">evaluation</text>
      <text class="note" x="${i1 + 6}" y="${i2 - 10}" text-anchor="middle">improvement</text>`;
    host.innerHTML = `<svg class="fig gpi" viewBox="0 0 ${W} ${H}" role="img" aria-label="Generalized policy iteration: evaluation and improvement alternate and meet at the optimal values and policy">${g}</svg>`;
  };
})(globalThis.RL = globalThis.RL || {});
