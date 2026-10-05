/* Figures of Part 2, bandits: the 10-armed testbed, sample averages and step sizes, and the learning curves of
   Sutton & Barto's chapter 2, every one computed by the Lab when it scrolls into view. */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const STORY_SEED = 58; // the testbed problem the stories of Part 2 play on
  const fraction = (u) => (u >= 1 ? String(u) : `1/${Math.round(1 / u)}`);

  // ---- the 10-armed testbed (Sutton & Barto, Figure 2.1): each arm's reward distribution, a normal around q*(a) ----
  RL.demos.testbed = function (host) {
    const env = lab.make("testbed");
    env.init(lab.rng(STORY_SEED));
    const W = 640, H = 300, M = { l: 62, r: 20, t: 14, b: 46 }, pw = W - M.l - M.r, ph = H - M.t - M.b;
    const lo = -4, hi = 4, y = (v) => M.t + ((hi - v) / (hi - lo)) * ph, cw = pw / env.k, cx = (a) => M.l + (a + 0.5) * cw;
    const clip = `tb-clip-${RL.fig.uid()}`;
    let g = `<clipPath id="${clip}"><rect x="${M.l}" y="${M.t}" width="${pw}" height="${ph}"/></clipPath>`;
    for (let v = lo; v <= hi; v += 2) g += `<line class="gridline" x1="${M.l}" x2="${M.l + pw}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${RL.fig.num(v, 0)}</text>`;
    for (let a = 0; a < env.k; a++) {
      const q = env.q[a];
      let right = "", left = "";
      for (let i = 0; i <= 40; i++) {
        const v = q - 3 + (6 * i) / 40, w = 0.42 * cw * Math.exp(-0.5 * (v - q) ** 2);
        right += `${i ? "L" : "M"}${(cx(a) + w).toFixed(1)} ${y(v).toFixed(1)}`;
        left = `L${(cx(a) - w).toFixed(1)} ${y(v).toFixed(1)}${left}`;
      }
      g += `<path class="violin" clip-path="url(#${clip})" d="${right}${left}Z"/><line class="mean-line" x1="${cx(a) - 0.3 * cw}" x2="${cx(a) + 0.3 * cw}" y1="${y(q)}" y2="${y(q)}"/>`;
      g += `<text class="tick" x="${cx(a)}" y="${M.t + ph + 20}" text-anchor="middle">${a + 1}</text>`;
    }
    g += `<line class="axis" x1="${M.l}" x2="${M.l + pw}" y1="${M.t + ph}" y2="${M.t + ph}"/><line class="axis" x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${M.t + ph}"/>
      <text class="axis-name" x="${M.l + pw / 2}" y="${H - 6}" text-anchor="middle">Arm</text>
      <text class="axis-name" transform="translate(16 ${M.t + ph / 2}) rotate(-90)" text-anchor="middle">Reward distribution</text>`;
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="The reward distributions of the ten arms of one testbed problem">${g}</svg>`;
  };

  // ---- sample averages: three runs on the same arm close in on its mean, within about one over the square root of n ----
  RL.demos["sample-average"] = function (host) {
    const n = 1000, q = 1, runs = [11, 12, 13];
    const p = RL.fig.plot(host, {
      label: "The sample average of an arm's rewards, in three runs",
      x: { max: n, log: true, ticks: [1, 10, 100, 1000], label: "Number of rewards averaged, n" },
      y: { min: -1, max: 3, ticks: [-1, 0, 1, 2, 3], label: "Estimate Q", digits: 1 },
      curves: [
        { id: "a", name: "run 1" }, { id: "b", name: "run 2", dash: true }, { id: "c", name: "run 3", dash: "dotted" },
        { id: "up", name: "q* + 1/√n", light: true }, { id: "down", name: "q* − 1/√n", light: true },
      ],
      at: (k) => `After ${k} ${k === 1 ? "reward" : "rewards"}`,
    });
    const lines = { up: new Float64Array(n), down: new Float64Array(n) };
    runs.forEach((seed, j) => {
      const rng = lab.rng(seed), L = (lines["abc"[j]] = new Float64Array(n));
      let Q = 0;
      for (let k = 1; k <= n; k++) {
        Q += (q + rng.normal() - Q) / k;
        L[k - 1] = Q;
      }
    });
    for (let k = 1; k <= n; k++) { lines.up[k - 1] = q + 1 / Math.sqrt(k); lines.down[k - 1] = q - 1 / Math.sqrt(k); }
    p.draw(lines);
    p.status("Rewards drawn from a normal distribution with mean q* = 1 and standard deviation 1.");
  };

  // ---- step sizes: how much each past reward counts in the estimate after 20 rewards ----
  RL.demos["step-weights"] = function (host) {
    const n = 20, xs = Array.from({ length: n }, (_, i) => i + 1);
    const weight = (alpha) => xs.map((i) => (alpha ? alpha * (1 - alpha) ** (n - i) : 1 / n));
    const p = RL.fig.plot(host, {
      label: "The weight of each of 20 rewards in the estimate, for three step sizes",
      x: { min: 1, max: n, ticks: [1, 5, 10, 15, 20], label: "Reward number i (20 is the latest)" },
      y: { min: 0, max: 0.5, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], label: "Weight of reward i", digits: 2 },
      curves: [{ id: "avg", name: "α = 1/n" }, { id: "a1", name: "α = 0.1", dash: true }, { id: "a5", name: "α = 0.5", dash: "dotted" }],
      at: (i) => `Reward ${i} of 20`,
    });
    p.draw({ avg: { xs, ys: weight(0) }, a1: { xs, ys: weight(0.1) }, a5: { xs, ys: weight(0.5) } });
    p.status("The weight left on the first estimate Q₁ is 0 for α = 1/n, 0.9²⁰ ≈ 0.12 for α = 0.1 and 0.5²⁰ ≈ 0.000001 for α = 0.5.");
  };

  // ---- step sizes: averages and a constant step size, on an arm that stays put and on one whose mean wanders ----
  RL.demos["step-sizes"] = function (host) {
    const n = 1000, rules = [["avg", 0], ["a1", 0.1]];
    const panel = (title, drift, seed) => {
      const rng = lab.rng(seed), lines = { truth: new Float64Array(n) }, Q = rules.map(() => 0);
      for (const [id] of rules) lines[id] = new Float64Array(n);
      let q = 1;
      for (let k = 1; k <= n; k++) {
        const r = q + rng.normal();
        rules.forEach(([id, alpha], j) => { Q[j] += (alpha || 1 / k) * (r - Q[j]); lines[id][k - 1] = Q[j]; });
        lines.truth[k - 1] = q;
        q += drift * rng.normal();
      }
      return { title, lines };
    };
    const panels = [panel("An arm that stays put", 0, 21), panel("An arm whose mean wanders", 0.04, 4)];
    host.replaceChildren();
    for (const { title, lines } of panels) {
      const box = document.createElement("div");
      box.className = "fig-panel";
      host.appendChild(box);
      const p = RL.fig.plot(box, {
        label: title, h: 240,
        x: { min: 1, max: n, ticks: [1, 250, 500, 750, 1000], label: `Rewards · ${title.toLowerCase()}` },
        y: { min: -1, max: 3, ticks: [-1, 0, 1, 2, 3], label: "Estimate Q", digits: 1 },
        curves: [{ id: "truth", name: "true mean q*", light: true }, { id: "avg", name: "α = 1/n" }, { id: "a1", name: "α = 0.1", dash: true }],
        at: (k) => `After ${k} ${k === 1 ? "reward" : "rewards"}`,
      });
      p.draw(lines);
    }
  };

  // ---- learning curves on the testbed, averaged over many problems (Sutton & Barto, Figures 2.2 to 2.5, Exercise 2.5) ----
  const simulate = (world, id, params, units, seed) => lab.simulate({ world, algorithm: lab.algorithms[id], params, units, seed, snapshots: false }).metrics;
  const steps = (units) => ({ max: units, ticks: units > 1000 ? [0, 2500, 5000, 7500, 10000] : [0, 250, 500, 750, 1000], label: "Steps" });
  const REWARD = { ticks: [0, 0.5, 1, 1.5], label: "Average reward", digits: 2 };
  const OPTIMAL = { min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], label: "% optimal action", percent: true };
  const at = (t) => `Step ${t.toLocaleString("en")}`;
  // Each figure: its racers (a curve each: world, algorithm, knobs, style), what it plots, how many runs it averages.
  const CURVES = {
    epsilon: {
      racers: [
        { id: "e1", name: "ε = 0.1", algorithm: "epsilon-greedy", params: { epsilon: 0.1, alpha: 0 } },
        { id: "e2", name: "ε = 0.01", algorithm: "epsilon-greedy", params: { epsilon: 0.01, alpha: 0 }, dash: true },
        { id: "e0", name: "ε = 0 (greedy)", algorithm: "epsilon-greedy", params: { epsilon: 0, alpha: 0 }, dash: "dotted" },
      ],
      plots: ["return", "optimal"], runs: 1000,
    },
    optimistic: {
      racers: [
        { id: "opt", name: "optimistic, greedy", algorithm: "optimistic-init", params: { q0: 5, epsilon: 0, alpha: 0.1 } },
        { id: "real", name: "realistic, ε-greedy", algorithm: "epsilon-greedy", params: { q0: 0, epsilon: 0.1, alpha: 0.1 }, dash: true },
      ],
      plots: ["optimal"], runs: 1000,
    },
    ucb: {
      racers: [
        { id: "ucb", name: "UCB, c = 2", algorithm: "ucb", params: { c: 2, alpha: 0 } },
        { id: "eps", name: "ε-greedy, ε = 0.1", algorithm: "epsilon-greedy", params: { epsilon: 0.1, alpha: 0 }, dash: true },
      ],
      plots: ["return"], runs: 1000, reward: { min: -0.2, max: 1.6 },
    },
    gradient: {
      world: "testbed-4",
      racers: [
        { id: "b1", name: "α = 0.1, baseline", algorithm: "gradient-bandit", params: { alpha: 0.1, baseline: true } },
        { id: "b4", name: "α = 0.4, baseline", algorithm: "gradient-bandit", params: { alpha: 0.4, baseline: true }, light: true },
        { id: "n1", name: "α = 0.1, no baseline", algorithm: "gradient-bandit", params: { alpha: 0.1, baseline: false }, dash: true },
        { id: "n4", name: "α = 0.4, no baseline", algorithm: "gradient-bandit", params: { alpha: 0.4, baseline: false }, dash: true, light: true },
      ],
      plots: ["optimal"], runs: 500, right: 168,
    },
    drift: {
      world: "drifting", units: 10000,
      racers: [
        { id: "const", name: "constant α = 0.1", algorithm: "epsilon-greedy", params: { epsilon: 0.1, alpha: 0.1 } },
        { id: "avg", name: "sample averages", algorithm: "epsilon-greedy", params: { epsilon: 0.1, alpha: 0 }, dash: true },
      ],
      plots: ["return", "optimal"], runs: 100, smooth: 100,
    },
  };

  RL.demos["bandit-curves"] = function (host, name = "epsilon") {
    const f = CURVES[name];
    if (!f) { RL.warn(`no bandit curves '${name}'`); return; }
    const world = f.world || "testbed", units = f.units || 1000;
    const specs = f.plots.map((metric) => ({
      label: `${metric === "return" ? "Average reward" : "% optimal action"} over ${units.toLocaleString("en")} steps`,
      h: f.plots.length > 1 ? 250 : 300, right: f.right, x: steps(units),
      y: metric === "return" ? { min: 0, max: 1.6, ...REWARD, ...f.reward } : OPTIMAL,
      curves: f.racers.map((r) => ({ id: `${metric}:${r.id}`, name: r.name, dash: r.dash, light: r.light })),
      at,
    }));
    RL.fig.average(host, specs.length > 1 ? specs : specs[0], {
      runs: f.runs, smooth: f.smooth || 1,
      note: `Average of ${f.runs.toLocaleString("en")} runs, each on a new ${world === "drifting" ? "drifting " : ""}problem${f.smooth ? `; smoothed over ${f.smooth} steps` : ""}.`,
      sample(seed) {
        const out = {};
        for (const r of f.racers) {
          const m = simulate(world, r.algorithm, r.params, units, seed);
          for (const metric of f.plots) out[`${metric}:${r.id}`] = m[metric];
        }
        return out;
      },
    });
  };

  // ---- the parameter study (Sutton & Barto, Figure 2.6): average reward over the first 1000 steps, by each method's knob ----
  const powers = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => 2 ** (a + i));
  const STUDY = [
    { id: "eps", name: "ε-greedy (ε)", algorithm: "epsilon-greedy", xs: powers(-7, -2), params: (v) => ({ epsilon: v, alpha: 0 }) },
    { id: "grad", name: "gradient bandit (α)", algorithm: "gradient-bandit", xs: powers(-5, 2), params: (v) => ({ alpha: v, baseline: true }), dash: true },
    { id: "ucb", name: "UCB (c)", algorithm: "ucb", xs: powers(-4, 2), params: (v) => ({ c: v, alpha: 0 }), dash: "dotted" },
    { id: "opt", name: "optimistic greedy (Q₁)", algorithm: "optimistic-init", xs: powers(-2, 2), params: (v) => ({ q0: v, epsilon: 0, alpha: 0.1 }), light: true },
  ];
  RL.demos["bandit-study"] = function (host) {
    const p = RL.fig.plot(host, {
      label: "Average reward over the first 1000 steps for each bandit method and setting of its knob",
      right: 168,
      x: { min: 1 / 128, max: 4, log2: true, ticks: powers(-7, 2), format: fraction, label: "ε, α, c or Q₁ (each method's own knob)" },
      y: { min: 0.7, max: 1.55, ticks: [0.8, 1, 1.2, 1.4], label: "Average reward, steps 1–1000", digits: 2 },
      curves: STUDY.map((s) => ({ id: s.id, name: s.name, dash: s.dash, light: s.light })),
      at: (v) => `Knob = ${fraction(v)}`,
    });
    const runs = 200, sums = STUDY.map((s) => new Float64Array(s.xs.length));
    RL.fig.whenVisible(host, () => {
      let done = 0;
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (done < runs && performance.now() - t0 < 30) {
          STUDY.forEach((s, j) => s.xs.forEach((v, i) => { sums[j][i] += lab.mean(simulate("testbed", s.algorithm, s.params(v), 1000, 1000 + done).return); }));
          done++;
        }
        p.draw(Object.fromEntries(STUDY.map((s, j) => [s.id, { xs: s.xs, ys: Array.from(sums[j], (v) => v / done) }])));
        p.status(done < runs ? `Averaging run ${done} of ${runs}…` : `Each point: the average of ${runs} runs of 1000 steps, each on a new problem.`);
        if (done < runs) setTimeout(more, 16);
      };
      more();
    });
  };
})(globalThis.RL = globalThis.RL || {});
