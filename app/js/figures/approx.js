/* Figures of Part 8, function approximation: what features look like, "touch a tile" (an update moves every state
   that shares a tile), the width of coarse features, the 1000-state walk learned with groups, tiles, polynomials and
   cosines, Mountain Car's cost-to-go and learning curves, and the weights of Baird's counterexample running away.
   As everywhere, every number is computed by the Lab when the figure scrolls into view. */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const WALK = "walk-1000";
  const CAR = { gamma: 1, epsilon: 0, features: "tiles", tilings: 8, cells: 8 };
  const later = (fn) => setTimeout(fn, 16);

  // ---- touch a tile: a two-number state space with tile coding; a click teaches one point ----
  // The demo uses the Lab's own tile coder on the unit square. Each click moves the weights of the tiles under the
  // pointer, one per tiling, by a step that takes the estimate there halfway to the target; everything that shares
  // those tiles moves with it.
  RL.demos["touch-tiles"] = function (host) {
    const S = 340, P = 30, G = 34; // drawing size, margin, heat-map cells per side
    const st = { tilings: 4, cells: 4, sign: 1, hover: null, last: null };
    let F = null, w = null;
    const env = () => ({ key: `touch-${st.tilings}-${st.cells}`, dims: 2, continuous: true, coords: (s) => s });
    host.innerHTML = `<div class="touch">
      <div class="touch-controls">
        <div class="seg" role="group" aria-label="How many tilings">${[1, 2, 4, 8].map((t) => `<button type="button" data-k="tilings" data-v="${t}">${t} tiling${t > 1 ? "s" : ""}</button>`).join("")}</div>
        <div class="seg" role="group" aria-label="Tile size">${[[4, "big tiles"], [8, "small tiles"]].map(([c, n]) => `<button type="button" data-k="cells" data-v="${c}">${n}</button>`).join("")}</div>
        <div class="seg" role="group" aria-label="What a click teaches">${[[1, "click raises"], [-1, "click lowers"]].map(([v, n]) => `<button type="button" data-k="sign" data-v="${v}">${n}</button>`).join("")}</div>
        <button class="pill touch-clear" type="button">Forget everything</button>
      </div>
      <svg class="fig touch-map" viewBox="0 0 ${S + P + 10} ${S + P + 6}" role="img" aria-label="A two-dimensional state space covered by tilings; click to teach a point">
        <g class="cells"></g><g class="tiles"></g><rect class="frame" x="${P}" y="4" width="${S}" height="${S}"/>
        <text class="tick" x="${P}" y="${S + 20}">0</text><text class="tick" x="${P + S}" y="${S + 20}" text-anchor="end">1</text>
        <text class="axis-name" x="${P + S / 2}" y="${S + 22}" text-anchor="middle">first number (say, position)</text>
        <text class="axis-name" transform="translate(14 ${4 + S / 2}) rotate(-90)" text-anchor="middle">second number (say, speed)</text>
        <circle class="touch-dot" r="4" opacity="0"/>
      </svg>
      <p class="fig-status touch-note"></p>
    </div>`;
    const svg = host.querySelector("svg"), cellsG = svg.querySelector(".cells"), tilesG = svg.querySelector(".tiles"), note = host.querySelector(".touch-note"), dot = svg.querySelector(".touch-dot");
    const c = S / G, rects = [];
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", P + i * c); r.setAttribute("y", 4 + (G - 1 - j) * c); r.setAttribute("width", c + 0.3); r.setAttribute("height", c + 0.3);
      cellsG.appendChild(r);
      rects.push(r);
    }
    const toUnit = (e) => {
      const box = svg.getBoundingClientRect(), k = (S + P + 10) / box.width;
      const u = ((e.clientX - box.left) * k - P) / S, v = 1 - ((e.clientY - box.top) * k - 4) / S;
      return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? [u, v] : null;
    };
    const value = (s) => lab.dot(w, F.of(s));
    // The tiles that hold a point: one square per tiling, where the coder puts it.
    function tilesAt(s) {
      const out = [];
      for (let t = 0; t < st.tilings; t++) {
        const box = [0, 1].map((j) => {
          const shift = ((t * (2 * j + 1)) % st.tilings) / st.tilings, i = Math.min(st.cells, Math.floor(s[j] * st.cells + shift));
          return [Math.max(0, (i - shift) / st.cells), Math.min(1, (i + 1 - shift) / st.cells)];
        });
        out.push(box);
      }
      return out;
    }
    function draw() {
      const cs = getComputedStyle(host), pos = cs.getPropertyValue("--v-pos"), neg = cs.getPropertyValue("--v-neg"), mid = cs.getPropertyValue("--v-mid");
      for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
        const v = value([(i + 0.5) / G, (j + 0.5) / G]), a = Math.round(Math.min(1, Math.abs(v)) * 100);
        rects[j * G + i].setAttribute("fill", `color-mix(in oklab, ${v >= 0 ? pos : neg} ${a}%, ${mid})`);
      }
      const at = st.hover || st.last;
      tilesG.innerHTML = at ? tilesAt(at).map(([[x0, x1], [y0, y1]], t) =>
        `<rect class="tile-box" style="--t:${t}" x="${P + x0 * S}" y="${4 + (1 - y1) * S}" width="${(x1 - x0) * S}" height="${(y1 - y0) * S}"/>`).join("") : "";
      if (at) { dot.setAttribute("cx", P + at[0] * S); dot.setAttribute("cy", 4 + (1 - at[1]) * S); dot.setAttribute("opacity", 1); }
      else dot.setAttribute("opacity", 0);
    }
    function reset() {
      F = lab.features(env(), { features: "tiles", tilings: st.tilings, cells: st.cells });
      w = new Float64Array(F.n);
      st.last = null;
      host.querySelectorAll("[data-k]").forEach((b) => b.classList.toggle("on", +b.dataset.v === st[b.dataset.k]));
      note.innerHTML = `Each tiling cuts the square into ${st.cells} × ${st.cells} tiles, shifted a little from the others. A point lies in ${st.tilings === 1 ? "one tile" : `${st.tilings} tiles, one from each tiling`}. <b>Click anywhere</b> to teach that point.`;
      draw();
    }
    svg.addEventListener("pointermove", (e) => { st.hover = toUnit(e); draw(); });
    svg.addEventListener("pointerleave", () => { st.hover = null; draw(); });
    svg.addEventListener("click", (e) => {
      const s = toUnit(e);
      if (!s) return;
      const x = F.of(s), before = lab.dot(w, x), target = st.sign;
      lab.addTo(w, x, (0.5 / st.tilings) * (target - before)); // α = 0.5 / tilings: halfway to the target
      st.last = s;
      note.innerHTML = `The estimate at (${s[0].toFixed(2)}, ${s[1].toFixed(2)}) went from <b>${before.toFixed(2)}</b> to <b>${lab.dot(w, x).toFixed(2)}</b>, halfway to ${target > 0 ? "+1" : "−1"}. ${st.tilings === 1 ? "Its whole tile moved with it, and nothing else." : `Its ${st.tilings} tiles moved with it: points that share all of them moved as much, points that share some moved less, by the share of tiles in common.`}`;
      draw();
    });
    host.querySelector(".touch-controls").addEventListener("click", (e) => {
      const b = e.target.closest("[data-k]");
      if (b) { st[b.dataset.k] = +b.dataset.v; if (b.dataset.k !== "sign") reset(); else host.querySelectorAll('[data-k="sign"]').forEach((x) => x.classList.toggle("on", x === b)); }
      if (e.target.closest(".touch-clear")) reset();
    });
    reset();
  };

  // ---- what the features of the 1000-state walk look like ----
  RL.demos["feature-shapes"] = function (host) {
    const W = 640, PW = 290, PH = 92, GAP = 40, x = (u, x0) => x0 + u * PW;
    const panel = (x0, y0, title, body) => `<g><text class="note" x="${x0}" y="${y0 - 8}">${title}</text>
      <line class="axis" x1="${x0}" x2="${x0 + PW}" y1="${y0 + PH}" y2="${y0 + PH}"/>${body}
      <text class="tick" x="${x0}" y="${y0 + PH + 15}">1</text><text class="tick" x="${x0 + PW}" y="${y0 + PH + 15}" text-anchor="end">1000</text></g>`;
    const curve = (f, x0, y0, lo, hi, cls) => {
      let d = "";
      for (let k = 0; k <= 120; k++) { const u = k / 120, v = f(u); d += `${k ? "L" : "M"}${x(u, x0).toFixed(1)} ${(y0 + PH - ((v - lo) / (hi - lo)) * PH).toFixed(1)}`; }
      return `<path class="curve${cls ? ` ${cls}` : ""}" d="${d}"/>`;
    };
    // groups: five boxes
    let groups = "";
    for (let g = 0; g < 5; g++) groups += `<rect class="bar${g % 2 ? " last" : ""}" x="${x(g / 5, 0) + 1}" y="${26 + 18}" width="${PW / 5 - 2}" height="${PH - 18}"/>`;
    // tiles: three tilings as rows of bricks, each shifted by a third of a tile; the tiles holding state 430 are dark
    let tiles = "";
    const s0 = 0.43;
    for (let t = 0; t < 3; t++) for (let i = 0; i <= 5; i++) {
      const a = Math.max(0, (i - t / 3) / 5), b = Math.min(1, (i + 1 - t / 3) / 5);
      if (b <= a) continue;
      const on = s0 >= a && s0 < b, y0 = 26 + 18 + t * 25;
      tiles += `<rect class="bar${on ? "" : " last"}" x="${x(a, PW + GAP) + 1}" y="${y0}" width="${(b - a) * PW - 2}" height="20"/>`;
    }
    tiles += `<line class="ref-line" x1="${x(s0, PW + GAP)}" x2="${x(s0, PW + GAP)}" y1="34" y2="${26 + PH}"/><text class="note" x="${x(s0, PW + GAP)}" y="31" text-anchor="middle">state 430</text>`;
    const y2 = 26 + PH + 56;
    let poly = "", four = "";
    for (let k = 0; k <= 4; k++) {
      poly += curve((u) => u ** k, 0, y2, -0.05, 1.05, k % 2 ? "light" : "");
      four += curve((u) => Math.cos(Math.PI * k * u), PW + GAP, y2, -1.1, 1.1, k % 2 ? "light" : "");
    }
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${W} ${y2 + PH + 22}" role="img" aria-label="The features of four constructions over the 1000 states">
      ${panel(0, 26, "State aggregation: 5 groups, one feature each", groups)}
      ${panel(PW + GAP, 26, "Tile coding: 3 tilings, each shifted a third of a tile", tiles)}
      ${panel(0, y2, "Polynomials: 1, s, s², s³, s⁴ (s scaled to [0, 1])", poly)}
      ${panel(PW + GAP, y2, "Fourier: cos(πcs) for c = 0, 1, 2, 3, 4", four)}
    </svg>`;
  };

  // ---- the width of coarse features: learning a square pulse with narrow, medium and broad intervals (after Figure 9.8) ----
  RL.demos["coarse-widths"] = function (host) {
    const widths = [["narrow", 0.06], ["medium", 0.2], ["broad", 0.5]], counts = [10, 40, 160, 640, 2560, 10240], N = 50;
    const f = (u) => (u >= 0.3 && u < 0.7 ? 1 : 0);
    const CW = 176, CH = 44, L = 74, T = 40, GX = 22, GY = 16;
    const learn = (wd) => {
      const centers = Array.from({ length: N }, (_, k) => -wd / 2 + ((1 + wd) * (k + 0.5)) / N), w = new Float64Array(N), rng = lab.rng(7), shots = [];
      const active = (u) => centers.map((c, k) => (Math.abs(u - c) < wd / 2 ? k : -1)).filter((k) => k >= 0);
      const v = (u) => active(u).reduce((sum, k) => sum + w[k], 0);
      for (let t = 1; t <= counts[counts.length - 1]; t++) {
        const u = rng.next(), on = active(u), err = f(u) - v(u), step = 0.2 / Math.max(1, on.length);
        for (const k of on) w[k] += step * err;
        if (counts.includes(t)) shots.push(Array.from({ length: 101 }, (_, i) => v(i / 100)));
      }
      return shots;
    };
    const rows = widths.map(([, wd]) => learn(wd));
    let g = "";
    widths.forEach(([name, wd], col) => {
      const x0 = L + col * (CW + GX);
      g += `<text class="note" x="${x0 + CW / 2}" y="14" text-anchor="middle">${name} features</text>
        <line class="axis" x1="${x0 + CW / 2 - (wd * CW) / 2}" x2="${x0 + CW / 2 + (wd * CW) / 2}" y1="26" y2="26" stroke-width="4"/>`;
      counts.forEach((n, r) => {
        const y0 = T + r * (CH + GY), Y = (v) => y0 + CH - Math.max(-0.2, Math.min(1.3, v)) * (CH / 1.2);
        let target = "", fit = "";
        for (let i = 0; i <= 100; i++) {
          const X = x0 + (i / 100) * CW;
          target += `${i ? "L" : "M"}${X.toFixed(1)} ${Y(f(i / 100 - 1e-9 * (i === 100))).toFixed(1)}`;
          fit += `${i ? "L" : "M"}${X.toFixed(1)} ${Y(rows[col][r][i]).toFixed(1)}`;
        }
        g += `<line class="gridline" x1="${x0}" x2="${x0 + CW}" y1="${Y(0)}" y2="${Y(0)}"/><path class="curve light dashed" d="${target}"/><path class="curve" d="${fit}"/>`;
        if (col === 0) g += `<text class="tick" x="${L - 10}" y="${y0 + CH / 2 + 4}" text-anchor="end">${n.toLocaleString("en")}</text>`;
      });
    });
    const H = T + counts.length * (CH + GY);
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${L + 3 * CW + 2 * GX + 6} ${H + 4}" role="img" aria-label="A square pulse learned with narrow, medium and broad features, after more and more examples">${g}
      <text class="axis-name" transform="translate(14 ${T + (H - T) / 2}) rotate(-90)" text-anchor="middle">examples seen</text></svg>`;
  };

  // ---- the walk learned with 10 groups: Monte Carlo and TD against the truth, over the time spent in each state ----
  // (after Figures 9.1 and 9.2). arg "mc": Monte Carlo only.
  RL.demos["walk-fit"] = function (host, arg) {
    const env = lab.make(WALK), truth = env.truth(), mu = env.mu();
    const curves = [{ id: "truth", name: "true value", dash: true }, { id: "mc", name: "gradient MC" }];
    if (arg !== "mc") curves.push({ id: "td", name: "semi-gradient TD", dash: "dotted" });
    const p = RL.fig.plot(host, {
      label: "Values of the 1000-state walk: the truth, and what gradient Monte Carlo and semi-gradient TD settle on with 10 groups",
      right: 128, h: 320,
      x: { min: 1, max: 1000, ticks: [1, 200, 400, 600, 800, 1000], label: "State" },
      y: { min: -1, max: 1, ticks: [-1, -0.5, 0, 0.5, 1], label: "Value", digits: 3, tickDigits: 1 },
      curves, at: (s) => `State ${s}`,
    });
    // μ as bars along the bottom, scaled to the lower third of the plot
    const top = Math.max(...Array.from(mu).filter((_, s) => Math.abs(s - env.start) > 2));
    let bars = "";
    for (let s = 1; s <= 1000; s += 10) bars += `<rect class="mu-bar" x="${p.x(s)}" y="${p.y(-1) - Math.min(1.3, mu[s] / top) * 50}" width="${p.x(s + 10) - p.x(s) - 0.6}" height="${Math.min(1.3, mu[s] / top) * 50}"/>`;
    p.svg.insertAdjacentHTML("afterbegin", `<g class="mu">${bars}<text class="note" x="${p.x(1000) + 8}" y="${p.y(-1) - 30}">time spent</text><text class="note" x="${p.x(1000) + 8}" y="${p.y(-1) - 14}">in each state (μ)</text></g>`);
    const lines = { truth: truth.subarray(1, 1001) };
    p.draw(lines);
    const settle = (id, alpha) => {
      const r = lab.simulate({ world: WALK, algorithm: lab.algorithms[id], params: { gamma: 1, features: "groups", cells: 10, alpha }, units: 10000, seed: 5 });
      const w = new Float64Array(10), marks = [];
      for (let t = 5000; t <= 10000; t += 100) marks.push(t);
      for (const t of marks) r.at(t).w.forEach((v, i) => { w[i] += v / marks.length; });
      return Float64Array.from({ length: 1000 }, (_, s) => w[Math.floor(s / 100)]);
    };
    p.status("Running 10,000 walks…");
    RL.fig.whenVisible(host, () => later(() => {
      lines.mc = settle("gradient-mc", 5e-4);
      p.draw(lines);
      const done = () => p.status("One run of 10,000 walks with α = 0.0005 (Monte Carlo)" + (arg === "mc" ? "" : " and α = 0.002 (TD)") + "; the weights are averaged over the second half, where they only jitter about where they settled.");
      if (arg === "mc") return done();
      later(() => { lines.td = settle("semi-gradient-td", 2e-3); p.draw(lines); done(); });
    }));
  };

  // ---- one-step and n-step semi-gradient TD with 20 groups: error in the first 10 walks against α (after Figure 9.2) ----
  RL.demos["walk-n-study"] = function (host) {
    const NS = [1, 2, 4, 8, 16, 32, 64, 128], xs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], top = 0.55, runs = 40;
    const p = RL.fig.plot(host, {
      label: "Average RMS error of n-step semi-gradient TD on the 1000-state walk with 20 groups, for several n, against the step size",
      right: 92,
      x: { min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], format: (v) => String(v), label: "α, the step size" },
      y: { min: 0.25, max: top, ticks: [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], label: "Average RMS error, first 10 walks", digits: 3, tickDigits: 2 },
      curves: NS.map((n, i) => ({ id: `n${n}`, name: `n = ${n}`, light: i % 2 === 1 })), at: (a) => `α = ${a}`,
    });
    const sums = NS.map(() => new Float64Array(xs.length));
    RL.fig.whenVisible(host, () => {
      let done = 0;
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (done < runs && performance.now() - t0 < 30) {
          NS.forEach((n, j) => xs.forEach((alpha, i) => {
            const { metrics } = lab.simulate({ world: WALK, algorithm: lab.algorithms["semi-gradient-td"], params: { gamma: 1, alpha, n, features: "groups", cells: 20 }, units: 10, seed: 1000 + done, snapshots: false, measures: ["error"] });
            const e = lab.mean(metrics.error);
            sums[j][i] += Number.isFinite(e) ? Math.min(e, 1) : 1;
          }));
          done++;
        }
        p.draw(Object.fromEntries(NS.map((n, j) => {
          const keep = xs.map((a, i) => i).filter((i) => sums[j][i] / done <= top + 0.02);
          return [`n${n}`, { xs: keep.map((i) => xs[i]), ys: keep.map((i) => sums[j][i] / done) }];
        })));
        p.status(done < runs ? `Averaging run ${done} of ${runs}…` : `Each point: the average of ${runs} runs of 10 walks, the error measured over all 1000 states. Points above the chart are left out.`);
        if (done < runs) later(more);
      };
      more();
    });
  };

  // Learning curves of gradient Monte Carlo on the walk, one run at a time so the page never waits long:
  // √VE after each walk, averaged over runs.
  function walkCurves(host, { label, curves, episodes, runs, note }) {
    const step = Math.max(1, Math.round(episodes / 300)), n = Math.floor(episodes / step);
    const p = RL.fig.plot(host, {
      label, right: 150,
      x: { min: 0, max: episodes, from: 0, ticks: [0, episodes / 5, (2 * episodes) / 5, (3 * episodes) / 5, (4 * episodes) / 5, episodes], label: "Walks (episodes)" },
      y: { min: 0, max: 0.4, ticks: [0, 0.1, 0.2, 0.3, 0.4], label: "√VE, averaged over runs", digits: 3, tickDigits: 1 },
      curves: curves.map(({ id, name, dash, light }) => ({ id, name, dash, light, marks: false })), at: (e) => `After ${e} walks`,
    });
    const sums = curves.map(() => new Float64Array(n)), jobs = [];
    for (let r = 0; r < runs; r++) curves.forEach((c, j) => jobs.push([j, 1000 + r]));
    RL.fig.whenVisible(host, () => {
      let k = 0;
      const more = () => {
        if (!host.isConnected) return;
        const [j, seed] = jobs[k++];
        const { metrics } = lab.simulate({ world: WALK, algorithm: lab.algorithms["gradient-mc"], params: { gamma: 1, ...curves[j].params }, units: episodes, seed, snapshots: false, measures: ["ve"] });
        for (let i = 0; i < n; i++) sums[j][i] += metrics.ve[i * step + step - 1];
        const done = Math.floor(k / curves.length), lines = {};
        curves.forEach((c, i) => {
          const runsDone = done + (k % curves.length > i ? 1 : 0);
          if (runsDone) lines[c.id] = { xs: Array.from({ length: n }, (_, t) => (t + 1) * step), ys: Array.from(sums[i], (v) => v / runsDone) };
        });
        p.draw(lines);
        p.status(k < jobs.length ? `Averaging run ${done + 1} of ${runs}…` : note);
        if (k < jobs.length) later(more);
      };
      more();
    });
    return p;
  }

  // ---- polynomials against Fourier cosines (after Figure 9.5) ----
  RL.demos["basis-study"] = function (host) {
    walkCurves(host, {
      label: "Value error of gradient Monte Carlo on the 1000-state walk with polynomial and Fourier features of orders 5, 10 and 20",
      episodes: 5000, runs: 10,
      curves: [5, 10, 20].flatMap((o, i) => [
        { id: `p${o}`, name: `polynomial, ${o}`, dash: true, light: i === 1, params: { features: "poly", order: o, alpha: 1e-4 } },
        { id: `f${o}`, name: `Fourier, ${o}`, light: i === 1, params: { features: "fourier", order: o, alpha: 5e-5 } },
      ]),
      note: "Average of 10 runs of 5,000 walks; α = 0.0001 for polynomials and 0.00005 for cosines. Dashed: polynomials.",
    });
  };

  // ---- many shifted tilings against one (after Figure 9.10) ----
  RL.demos["tiling-study"] = function (host) {
    walkCurves(host, {
      label: "Value error of gradient Monte Carlo on the 1000-state walk with one tiling of 200-state tiles and with 50 such tilings",
      episodes: 5000, runs: 10,
      curves: [
        { id: "one", name: "one tiling (5 groups)", dash: true, params: { features: "groups", cells: 5, alpha: 1e-4 } },
        { id: "fifty", name: "50 tilings", params: { features: "tiles", tilings: 50, cells: 5, alpha: 1e-4 / 50 } },
      ],
      note: "Average of 10 runs of 5,000 walks. Every tile spans 200 states; the 50 tilings are shifted 4 states from one another. α = 0.0001 for one tiling, 0.0001/50 for fifty, so that each update moves the estimate of the visited state equally far.",
    });
  };

  // ---- Mountain Car: the cost-to-go after 1, 12, 104 and 1000 episodes of one run (after Figure 10.1) ----
  RL.demos["car-surfaces"] = function (host) {
    const marks = [1, 12, 104, 1000];
    host.innerHTML = `<div class="surface-row">${marks.map((t) => `<figure class="surface-cell"><div class="surface-host"></div><figcaption>after ${t.toLocaleString("en")} episode${t > 1 ? "s" : ""}</figcaption></figure>`).join("")}</div><p class="fig-status">Running 1,000 episodes…</p>`;
    RL.fig.whenVisible(host, () => later(() => {
      const r = lab.simulate({ world: "mountain-car", algorithm: lab.algorithms["semi-gradient-sarsa"], params: { ...CAR, alpha: 0.5 / 8 }, units: 1000, seed: 1 });
      const env = r.env, N = 31, F = lab.features(env, r.params);
      host.querySelectorAll(".surface-host").forEach((box, k) => {
        const w = r.at(marks[k]).w, Z = new Float64Array(N * N);
        let top = 0;
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
          const x = F.of([env.xMin + ((env.xMax - env.xMin) * i) / (N - 1), -env.vMax + (2 * env.vMax * j) / (N - 1)]);
          let best = -Infinity;
          for (let a = 0; a < 3; a++) best = Math.max(best, lab.dot(w, x, a * F.n));
          Z[j * N + i] = Math.max(0, -best);
          top = Math.max(top, Z[j * N + i]);
        }
        const zMax = Math.max(10, Math.ceil(top / 10) * 10);
        box.innerHTML = RL.Surface3D.svg(Z, N, N, zMax, { w: 220, h: 150, cls: "fig-surface" }) + `<span class="surface-top">highest: ${Math.round(top)} steps</span>`;
      });
      host.querySelector(".fig-status").textContent = `Seen from the same angle as the Lab's landscape: position runs to the right, speed toward the back. The first episode took ${r.metrics.steps[0]} steps, the last ${r.metrics.steps[999]}. Each panel has its own height scale.`;
    }));
  };

  // ---- Mountain Car: steps per episode for three step sizes, or for one-step and eight-step SARSA (Figures 10.2, 10.3) ----
  RL.demos["car-curves"] = function (host, arg) {
    const set = arg === "n"
      ? [{ id: "n1", name: "n = 1, α = 0.5/8", n: 1, alpha: 0.5 / 8, dash: true }, { id: "n8", name: "n = 8, α = 0.3/8", n: 8, alpha: 0.3 / 8 }]
      : [0.1, 0.2, 0.5].map((a, i) => ({ id: `a${a}`, name: `α = ${a}/8`, alpha: a / 8, dash: i === 0, light: i === 1 }));
    RL.fig.average(host, {
      label: arg === "n" ? "Steps per episode on Mountain Car, one-step and eight-step semi-gradient SARSA" : "Steps per episode on Mountain Car, semi-gradient SARSA with three step sizes",
      right: 150,
      x: { min: 1, max: 500, ticks: [1, 100, 200, 300, 400, 500], label: "Episodes" },
      y: { min: 100, max: 1000, log: true, ticks: [100, 200, 400, 1000], label: "Steps per episode (log scale)", digits: 0 },
      curves: set.map(({ id, name, dash, light }) => ({ id, name, dash, light })), at: (e) => `Episode ${e}`,
    }, {
      runs: 30, smooth: 5,
      sample: (seed) => Object.fromEntries(set.map((c) => [c.id, lab.simulate({ world: "mountain-car", algorithm: lab.algorithms["semi-gradient-sarsa"], params: { ...CAR, alpha: c.alpha, n: c.n || 1 }, units: 500, seed, snapshots: false }).metrics.steps])),
      note: "Average of 30 runs, smoothed over 5 episodes. Tile coding with 8 tilings of 8 × 8 tiles; ε = 0, all estimates start at 0.",
    });
  };

  // ---- Baird's counterexample: the weights of semi-gradient off-policy TD and of semi-gradient DP (after Figure 11.2) ----
  RL.demos["baird-weights"] = function (host) {
    const T = 1000;
    host.innerHTML = '<div class="fig-panel"></div><div class="fig-panel"></div>';
    const [a, b] = host.querySelectorAll(".fig-panel");
    // sampled: the Lab's off-policy semi-gradient TD
    const r = lab.simulate({ world: "baird", algorithm: lab.algorithms["semi-gradient-td"], params: { gamma: 0.99, alpha: 0.01, features: "own", policy: "target", behavior: "behavior" }, units: T, seed: 1 });
    const td = Array.from({ length: 8 }, () => new Float64Array(T + 1));
    for (let t = 0; t <= T; t++) r.at(t).w.forEach((v, i) => { td[i][t] = v; });
    // expected: every state updated at once with the target policy's expected target, in proportion 1/7 each
    const env = lab.make("baird"), F = lab.features(env, { features: "own" }), w = new Float64Array(8);
    F.init(w);
    const dp = Array.from({ length: 8 }, () => new Float64Array(T + 1));
    for (let t = 0; t <= T; t++) {
      w.forEach((v, i) => { dp[i][t] = v; });
      const step = new Float64Array(8), v7 = lab.dot(w, F.of(env.LOW));
      for (let s = 0; s < 7; s++) { const x = F.of(s); lab.addTo(step, x, (0.01 / 7) * (0.99 * v7 - lab.dot(w, x))); }
      w.forEach((v, i) => { w[i] = v + step[i]; });
    }
    const draw = (box, W, title, unit) => {
      const top = Math.ceil(Math.max(...W.flatMap((L) => Array.from(L))) / 50) * 50, low = Math.min(0, Math.floor(Math.min(...W.flatMap((L) => Array.from(L))) / 50) * 50);
      const ticks = [];
      for (let v = low; v <= top; v += top - low > 400 ? 100 : 50) ticks.push(v);
      const p = RL.fig.plot(box, {
        label: title, h: 250, right: 92,
        x: { min: 0, max: T, from: 0, ticks: [0, 200, 400, 600, 800, 1000], label: unit },
        y: { min: low, max: top, ticks, label: `Weights: ${title}`, digits: 1, tickDigits: 0 },
        curves: [...[1, 2, 3, 4, 5, 6].map((i) => ({ id: `w${i}`, name: i === 1 ? "w₁ … w₆" : "", light: true })), { id: "w7", name: "w₇", dash: true }, { id: "w8", name: "w₈" }],
        at: (t) => `${unit.replace(/s$/, "")} ${t}`,
      });
      p.draw(Object.fromEntries(W.map((L, i) => [`w${i + 1}`, L])));
      return p;
    };
    draw(a, td, "off-policy TD", "Steps");
    draw(b, dp, "expected updates (DP)", "Sweeps").status("α = 0.01, γ = 0.99; the weights start at (1, 1, 1, 1, 1, 1, 10, 1). Top: one run of sampled transitions under the behavior policy, each weighted by its importance ratio. Bottom: every state updated at once, with the target policy's expected target, no sampling at all.");
  };
})(globalThis.RL = globalThis.RL || {});
