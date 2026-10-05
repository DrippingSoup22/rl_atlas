/* Figures of Part 7, planning: Dyna-Q in the maze, the blocking and shortcut mazes, and prioritized sweeping. */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const MAZE = { alpha: 0.1, epsilon: 0.1, gamma: 0.95 };
  const steps = (world, id, params, units, seed) => lab.simulate({ world, algorithm: lab.algorithms[id], params, units, seed, snapshots: false }).metrics.steps;

  // ---- the Dyna architecture: real experience feeds the values directly and through the model ----
  RL.demos["dyna-architecture"] = function (host) {
    const W = 560, H = 210, id = RL.fig.uid();
    const box = (x, y, w, h, text) => `<rect class="arch-box" x="${x}" y="${y}" width="${w}" height="${h}" rx="10"/><text x="${x + w / 2}" y="${y + h / 2 + 5}" text-anchor="middle">${text}</text>`;
    const arrow = (d, label, lx, ly, anchor = "middle") => `<path class="arch-arrow" d="${d}" marker-end="url(#tip-${id})"/>` +
      label.split("\n").map((t, i) => `<text class="note" x="${lx}" y="${ly + i * 16}" text-anchor="${anchor}">${t}</text>`).join("");
    host.innerHTML = `<svg class="fig arch" viewBox="0 0 ${W} ${H}" role="img" aria-label="The Dyna architecture">
      <defs><marker id="tip-${id}" class="tip-mark" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>
      ${box(200, 14, 160, 50, "values / policy")}
      ${box(20, 146, 150, 50, "experience")}
      ${box(390, 146, 150, 50, "model")}
      ${arrow("M200 39 C110 39 60 70 62 142", "acting", 98, 52, "end")}
      ${arrow("M128 146 C138 108 180 80 222 68", "direct RL", 182, 122, "start")}
      ${arrow("M174 171 H386", "model learning", 280, 163)}
      ${arrow("M465 146 C465 90 420 42 364 40", "planning\n(simulated experience)", 446, 100, "end")}
    </svg>`;
  };

  // ---- steps per episode with 0, 5 and 50 planning steps (Sutton & Barto, Figure 8.2) ----
  RL.demos["dyna-curves"] = function (host) {
    const N = [0, 5, 50];
    RL.fig.average(host, {
      label: "Steps per episode in the Dyna maze, for Dyna-Q with 0, 5 and 50 planning steps",
      right: 150, x: { min: 2, max: 50, from: 2, ticks: [2, 10, 20, 30, 40, 50], label: "Episodes" }, // the first episode is a random search for all of them
      y: { min: 0, max: 800, ticks: [0, 200, 400, 600, 800], label: "Steps per episode", digits: 0 },
      curves: N.map((n, i) => ({ id: `n${n}`, name: n ? `${n} planning steps` : "0 (Q-learning)", dash: i === 0, light: i === 1 })),
      at: (e) => `Episode ${e}`,
    }, {
      runs: 30,
      sample: (seed) => Object.fromEntries(N.map((n) => [`n${n}`, steps("dyna-maze", "dyna-q", { ...MAZE, planning: n }, 50, seed).subarray(1)])),
    });
  };

  // ---- the greedy policies halfway through the second episode, without and with planning (after Figure 8.3) ----
  RL.demos["dyna-midway"] = function (host) {
    const env = lab.make("dyna-maze"), id = RL.fig.uid(), parts = [];
    let x0 = 0, W = 0, H = 0;
    for (const n of [0, 50]) {
      const r = lab.simulate({ world: "dyna-maze", algorithm: lab.algorithms["dyna-q"], params: { ...MAZE, planning: n }, units: 2, seed: 3 });
      // Walk the second episode up to its middle; the copy of the memory follows every update.
      const { m, events } = r.replay(1), half = Math.floor(r.metrics.steps[1] / 2);
      let moves = 0, at = env.start;
      for (const ev of events) { if (ev.type === "move") { moves++; at = ev.s2; } if (moves >= half && ev.type === "next") break; }
      const mz = RL.fig.maze(env, { T: 26, X: x0 + 6, Y: 26, id: `${id}-${n}` });
      let g = mz.svg + `<text class="note" x="${x0 + mz.W / 2}" y="16" text-anchor="middle">${n ? `With planning (n = ${n})` : "Without planning (n = 0)"}</text>`;
      for (let s = 0; s < env.nS; s++) {
        if (env.terminal(s) || env.blocked(s) || lab.maxQ(m.Q, s, env) <= 0) continue;
        g += RL.fig.mazeArrow(mz, s, lab.greedy(m.Q, s, env));
      }
      g += `<circle class="agent-dot" cx="${mz.cx(at)}" cy="${mz.cy(at)}" r="${mz.T * 0.22}"/>`;
      parts.push(g);
      x0 += mz.W + 30;
      W = x0 - 30;
      H = Math.max(H, mz.H + 26);
    }
    host.innerHTML = `<svg class="fig maze" viewBox="0 0 ${W} ${H}" role="img" aria-label="Greedy policies halfway through the second episode, without and with planning">${parts.join("")}</svg>`;
  };

  // ---- the blocking and shortcut mazes: cumulative reward against time steps (Figures 8.4 and 8.5) ----
  // From the steps each episode took: the cumulative reward at step k is the number of episodes finished by then.
  const CHANGING = {
    blocking: { world: "blocking-maze", budget: 3000, units: 400, planning: 10, top: 150, ticks: [0, 50, 100, 150] },
    shortcut: { world: "shortcut-maze", budget: 6000, units: 800, planning: 50, top: 400, ticks: [0, 100, 200, 300, 400] },
  };
  RL.demos["changing-maze"] = function (host, arg = "blocking") {
    const c = CHANGING[arg], env = lab.make(c.world), id = RL.fig.uid();
    const params = { alpha: 1, epsilon: 0.1, gamma: 0.95, planning: c.planning, kappa: 0.001 };
    // The maze before and after its change, side by side, above the curves.
    let mazes = "", x0 = 0, H = 0;
    for (const [t, title] of [[0, `Steps 1–${env.changes.toLocaleString("en")}`], [env.changes, `After step ${env.changes.toLocaleString("en")}`]]) {
      env.setTime(t);
      const mz = RL.fig.maze(env, { T: 16, X: x0 + 4, Y: 22, id: `${id}-${t}` });
      mazes += mz.svg + `<text class="note" x="${x0 + mz.W / 2}" y="14" text-anchor="middle">${title}</text>`;
      x0 += mz.W + 40;
      H = Math.max(H, mz.H + 22);
    }
    host.innerHTML = `<svg class="fig maze pair" viewBox="0 0 ${x0 - 40} ${H}" role="img" aria-label="The maze before and after it changes">${mazes}</svg><div class="fig-curves"></div>`;
    RL.fig.average(host.querySelector(".fig-curves"), {
      label: `Cumulative reward in the ${arg} maze for Dyna-Q and Dyna-Q+`,
      right: 100, x: { min: 0, max: c.budget, from: 1, ticks: Array.from({ length: c.budget / 1000 + 1 }, (_, i) => i * 1000), label: "Time steps" },
      y: { min: 0, max: c.top, ticks: c.ticks, label: "Cumulative reward", digits: 1, tickDigits: 0 },
      curves: [{ id: "plus", name: "Dyna-Q+" }, { id: "q", name: "Dyna-Q", dash: true }],
      at: (k) => `Step ${Math.round(k).toLocaleString("en")}`,
    }, {
      runs: 20,
      sample(seed) {
        const out = {};
        for (const [key, algo] of [["q", "dyna-q"], ["plus", "dyna-q-plus"]]) {
          const ep = steps(c.world, algo, params, c.units, seed), cum = new Float64Array(c.budget);
          let k = 0, done = 0;
          for (const n of ep) { for (let j = 0; j < n && k < c.budget; j++) cum[k++] = done; done++; if (k >= c.budget) break; }
          while (k < c.budget) cum[k++] = done;
          out[key] = cum;
        }
        return out;
      },
    });
  };

  // ---- expected against sample updates (after Figure 8.7), worked out exactly ----
  // A state–action pair with b equally likely successors whose values are known exactly, and an estimate that starts
  // off by 1. An expected update fixes it with b computations; t sample updates (sample averages) leave an error of
  // √((b − 1)/(b t)). The x axis counts computations in units of b.
  RL.demos["expected-vs-sample"] = function (host) {
    const B = [2, 10, 100, 1000];
    const p = RL.fig.plot(host, {
      label: "RMS error after expected and sample updates, against the computation spent, for several branching factors",
      right: 130, x: { min: 0, max: 2, ticks: [0, 0.5, 1, 1.5, 2], format: (v) => (v === 1 ? "1b" : v === 2 ? "2b" : v === 0 ? "0" : `${v}b`), label: "Number of max Q(s′, a′) computations" },
      y: { min: -0.05, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], label: "RMS error in the estimate", digits: 3, tickDigits: 2 }, // a little room under 0 keeps the names off the axis
      curves: [{ id: "expected", name: "expected update", dash: true, marks: false }, ...B.map((b, i) => ({ id: `b${b}`, name: `sample, b = ${b}`, marks: false, light: i % 2 === 1 }))],
      at: (x) => `${(+x).toFixed(2)} b computations`,
    });
    const lines = { expected: { xs: [0, 1, 1, 2], ys: [1, 1, 0, 0] } };
    for (const b of B) {
      const xs = [0], ys = [1];
      for (let t = 1; t <= 2 * b; t++) { xs.push(t / b); ys.push(Math.sqrt((b - 1) / (b * t))); }
      lines[`b${b}`] = { xs, ys };
    }
    p.draw(lines);
    p.status("Worked out exactly, not simulated.");
  };

  // ---- prioritized sweeping against Dyna-Q, same number of planning updates per step ----
  RL.demos["sweeping-curves"] = function (host) {
    RL.fig.average(host, {
      label: "Steps per episode in the Dyna maze for Dyna-Q and prioritized sweeping, 5 planning updates per step",
      right: 160, x: { min: 2, max: 30, from: 2, ticks: [2, 10, 20, 30], label: "Episodes" },
      y: { min: 0, max: 200, ticks: [0, 50, 100, 150, 200], label: "Steps per episode", digits: 0 },
      curves: [{ id: "ps", name: "prioritized sweeping" }, { id: "dq", name: "Dyna-Q", dash: true }],
      at: (e) => `Episode ${e}`,
    }, {
      runs: 30,
      sample: (seed) => ({
        ps: steps("dyna-maze", "prioritized-sweeping", { ...MAZE, alpha: 0.5, planning: 5 }, 30, seed).subarray(1),
        dq: steps("dyna-maze", "dyna-q", { ...MAZE, alpha: 0.5, planning: 5 }, 30, seed).subarray(1),
      }),
    });
  };
})(globalThis.RL = globalThis.RL || {});
