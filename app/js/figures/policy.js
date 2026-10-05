/* Figures of Part 10, policy gradients: the short corridor's landscape, softmax and Gaussian policies to play with,
   single-episode gradient estimates, REINFORCE and its baseline (after Sutton & Barto, Figures 13.1 and 13.2), the
   critic's step size, the entropy bonus, GAE's weights and its λ, TRPO's surrogate and trust region, and PPO's clip.
   As everywhere, the numbers are worked out exactly or computed by the Lab when the figure comes into view. */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const later = (fn) => setTimeout(fn, 16);
  const minus = (v, d = 1) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;
  const NS = "http://www.w3.org/2000/svg";

  // ---- the short corridor, worked out exactly: values, visits, advantages ----
  const corridor = lab.corridor();
  const next = (s, a) => corridor.step(s, a).s2;
  // Expected visits to each cell in one episode, stepping right with chance p (the start counts as a visit).
  function visits(p) {
    const k = (p * p) / (1 - (1 - p) ** 2), d0 = 1 / (p - k), d1 = (p * d0) / (1 - (1 - p) ** 2);
    return [d0, d1, (1 - p) * d1];
  }
  // Advantages under p: A(s, a) = −1 + v(s′) − v(s).
  function advantage(p) {
    const v = corridor.values(p);
    return [0, 1, 2].map((s) => [0, 1].map((a) => -1 + v[next(s, a)] - v[s]));
  }
  const klOf = (p0, p) => p0 * Math.log(p0 / p) + (1 - p0) * Math.log((1 - p0) / (1 - p));
  const SUB = "₀₁₂₃₄₅₆₇₈₉", sub = (n) => String(n).replace(/\d/g, (c) => SUB[c]);

  // ---- the corridor's landscape: the value of the start for every chance of stepping right (Example 13.1) ----
  RL.demos["corridor-values"] = function (host) {
    const xs = [], ys = [];
    for (let k = 1; k < 1000; k++) { const u = k / 1000; if (corridor.J(u) >= -100) { xs.push(u); ys.push(corridor.J(u)); } }
    const p = RL.fig.plot(host, {
      label: "The value of the start of the short corridor for every chance p of stepping right", right: 40, h: 300,
      x: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], format: (u) => `${Math.round(u * 100)}%`, label: "p, the chance of stepping right (the same in every cell)" },
      y: { min: -100, max: 0, ticks: [-100, -80, -60, -40, -20, 0], label: "Value of the start, J(p)", digits: 1, tickDigits: 0 },
      curves: [{ id: "j", name: "", marks: false }],
      at: (u) => `Step right ${Math.round(u * 100)}% of the time`,
    });
    p.draw({ j: { xs, ys } });
    p.svg.insertAdjacentHTML("beforeend", `<g class="pts">${[0.05, corridor.best, 0.95].map((u) => `<circle cx="${p.x(u)}" cy="${p.y(corridor.J(u))}" r="4"/>`).join("")}</g>`);
    const put = (u, text, dx, dy, anchor) => p.svg.insertAdjacentHTML("beforeend", `<text class="note halo" x="${p.x(u) + dx}" y="${p.y(corridor.J(u)) + dy}" text-anchor="${anchor}">${text}</text>`);
    put(0.05, `ε-greedy, preferring left: ${minus(corridor.J(0.05))}`, 10, 4, "start");
    put(0.95, `ε-greedy, preferring right: ${minus(corridor.J(0.95))}`, -10, 4, "end");
    put(corridor.best, `best: ${minus(corridor.bestValue, 2)} at p = 2 − √2 ≈ ${corridor.best.toFixed(3)}`, 0, 22, "middle");
    p.status("Worked out exactly: J(p) = −2(2 − p) / (p(1 − p)). Both ε-greedy policies use ε = 0.1; a deterministic policy (p = 0 or 1) never reaches the goal.");
  };

  // ---- softmax: three preferences, their probabilities ----
  RL.demos["softmax-play"] = function (host) {
    const names = ["left", "stay", "right"], h = [0.5, 0, -1];
    host.innerHTML = `<p class="fig-formula"><span class="tex">\\pol{\\pi(a)} = e^{\\pol{h(a)}} \\big/ \\textstyle\\sum_b e^{\\pol{h(b)}}</span>: the chances always add up to 1, and none is ever exactly 0.</p>
      <svg class="fig" viewBox="0 0 640 220" role="img" aria-label="Three action preferences and the probabilities their softmax gives"></svg>
      <div class="fig-knobs">${names.map((n, i) => `<label class="fig-knob"><span>h(${n})</span><input type="range" min="-4" max="4" step="0.1" value="${h[i]}" data-i="${i}"><output></output></label>`).join("")}
      <div class="fig-buttons"><button class="pill" type="button" data-add="1">add 1 to every preference</button><button class="pill" type="button" data-scale="2">double them all</button></div></div>`;
    const svg = host.querySelector("svg"), inputs = host.querySelectorAll("input"), outs = host.querySelectorAll("output");
    function draw() {
      const pi = lab.softmax(h), W = 640, base = 186, top = 24, bw = 110;
      let g = `<line class="axis" x1="60" x2="${W - 40}" y1="${base}" y2="${base}"/>`;
      names.forEach((n, i) => {
        const x = 110 + i * 170, hgt = (base - top) * pi[i];
        g += `<rect class="bar" x="${x}" y="${base - hgt}" width="${bw}" height="${hgt}"/>
          <text class="num" x="${x + bw / 2}" y="${base - hgt - 7}" text-anchor="middle">${(100 * pi[i]).toFixed(1)}%</text>
          <text class="tick" x="${x + bw / 2}" y="${base + 20}" text-anchor="middle">${n}</text>`;
      });
      svg.innerHTML = g;
      inputs.forEach((inp, i) => { inp.value = h[i]; outs[i].textContent = `${h[i] < 0 ? "−" : ""}${Math.abs(h[i]).toFixed(1)}`; });
    }
    RL.math.render(host);
    host.querySelector(".fig-knobs").addEventListener("input", (e) => { const i = +e.target.dataset.i; if (!Number.isNaN(i)) { h[i] = +e.target.value; draw(); } });
    host.querySelector(".fig-buttons").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      for (let i = 0; i < 3; i++) h[i] = Math.max(-4, Math.min(4, b.dataset.add ? h[i] + 1 : h[i] * 2));
      draw();
    });
    draw();
  };

  // ---- a Gaussian policy: the bell, and how a single action would push its mean and its spread ----
  // Below the bell, the two parts of ∇ ln π(a) for every action a, each drawn to its own scale (their shapes and signs
  // are what matter): a sample that did better than expected moves θ along them, one that did worse against them.
  RL.demos["gaussian-play"] = function (host) {
    const st = { mu: 30, sd: 10 };
    host.innerHTML = `<svg class="fig" viewBox="0 0 640 410" role="img" aria-label="A Gaussian policy over angles, and the gradient of ln π for its mean and its spread"></svg>
      <div class="fig-knobs"><label class="fig-knob"><span>μ</span><input type="range" min="10" max="80" step="1" value="30" data-k="mu"><output></output></label>
      <label class="fig-knob"><span>σ</span><input type="range" min="3" max="20" step="0.5" value="10" data-k="sd"><output></output></label></div>`;
    const svg = host.querySelector("svg"), L = 64, R = 616, x = (a) => L + (a / 90) * (R - L);
    function panel(y0, h, f, cap, title, note) {
      // f(a) → value; drawn between −cap and +cap around the panel's middle line, clipped at the edges
      const mid = y0 + h / 2, y = (v) => mid - (Math.max(-cap, Math.min(cap, v)) / cap) * (h / 2);
      let d = "";
      for (let a = 0; a <= 90; a += 0.5) d += `${d ? "L" : "M"}${x(a).toFixed(1)} ${y(f(a)).toFixed(1)}`;
      return `<text class="note" x="${L}" y="${y0 - 8}">${title}</text>
        <line class="gridline" x1="${L}" x2="${R}" y1="${mid}" y2="${mid}"/><line class="axis" x1="${L}" x2="${L}" y1="${y0}" y2="${y0 + h}"/>
        <text class="tick" x="${L - 7}" y="${y0 + 12}" text-anchor="end">+</text><text class="tick" x="${L - 7}" y="${mid + 4}" text-anchor="end">0</text><text class="tick" x="${L - 7}" y="${y0 + h}" text-anchor="end">−</text>
        <path class="curve" d="${d}"/><text class="note" x="${R}" y="${y0 + h + 14}" text-anchor="end">${note}</text>`;
    }
    function draw() {
      const { mu, sd } = st, top = 30, base = 140, peak = (base - top) * Math.min(1, 6 / sd);
      let bell = `M${x(0)} ${base}`;
      for (let a = 0; a <= 90; a += 0.5) bell += `L${x(a).toFixed(1)} ${(base - peak * Math.exp(-0.5 * ((a - mu) / sd) ** 2)).toFixed(1)}`;
      const lo = Math.max(0, mu - sd), hi = Math.min(90, mu + sd);
      let g = `<rect class="region" x="${x(lo)}" y="${top - 6}" width="${x(hi) - x(lo)}" height="${388 - top}"/>
        <path class="bell-fill" d="${bell}L${x(90)} ${base}Z"/><line class="axis" x1="${L}" x2="${R}" y1="${base}" y2="${base}"/>
        <line class="ref-line" x1="${x(mu)}" x2="${x(mu)}" y1="${top - 6}" y2="388"/><text class="note halo" x="${x(mu)}" y="${top - 10}" text-anchor="middle">μ = ${mu}°</text>
        <text class="note halo" x="${x(hi) + 4}" y="${top + 8}">±σ = ${sd}°</text>`;
      for (let a = 0; a <= 90; a += 15) g += `<text class="tick" x="${x(a)}" y="${base + 17}" text-anchor="middle">${a}°</text>`;
      g += `<text class="axis-name" x="${R}" y="${base + 17}" text-anchor="end" dy="16">the action, an angle, drawn from π</text>`;
      g += panel(206, 64, (a) => (a - mu) / (sd * sd), 2.5 / sd, "Its push on the mean, (a − μ)/σ²: up for actions above μ, down below", "");
      g += panel(316, 64, (a) => ((a - mu) / sd) ** 2 - 1, 3, "Its push on the spread, (a − μ)²/σ² − 1: narrower within ±σ, wider beyond", "keeps growing beyond the scale");
      svg.innerHTML = g;
      host.querySelectorAll("output")[0].textContent = `${mu}°`;
      host.querySelectorAll("output")[1].textContent = `${sd}°`;
    }
    host.querySelector(".fig-knobs").addEventListener("input", (e) => { const k = e.target.dataset.k; if (k) { st[k] = +e.target.value; draw(); } });
    draw();
  };

  // ---- single-episode estimates of the gradient, with and without a baseline (the corridor at p = 0.3) ----
  RL.demos["pg-estimates"] = function (host) {
    const p = 0.3, N = 50000, exact = (2 * (p * p - 4 * p + 2)) / (p * (1 - p)), v = corridor.values(p), d = visits(p);
    const b = (d[0] * v[0] + d[1] * v[1] + d[2] * v[2]) / (d[0] + d[1] + d[2]); // what a one-weight baseline learns
    const rng = lab.rng(11), plain = [], based = [];
    for (let i = 0; i < N; i++) {
      let s = 0;
      const A = [];
      while (s !== 3 && A.length < 5000) { const a = rng.next() < p ? 1 : 0; A.push(a); s = next(s, a); }
      let g = 0, gb = 0;
      A.forEach((a, t) => { const G = -(A.length - t), score = a - p; g += G * score; gb += (G - b) * score; });
      plain.push(g); based.push(gb);
    }
    const lo = -60, hi = 110, bins = 34, W = 640, L = 60, R = 616, x = (u) => L + ((u - lo) / (hi - lo)) * (R - L);
    const stats = (xs) => { const m = lab.mean(xs); return { m, sd: Math.sqrt(lab.mean(xs.map((u) => (u - m) ** 2))) }; };
    const row = (xs, y0, title) => {
      const c = new Float64Array(bins);
      for (const u of xs) c[Math.max(0, Math.min(bins - 1, Math.floor(((u - lo) / (hi - lo)) * bins)))]++;
      const top = Math.max(...c), { m, sd } = stats(xs), bw = (R - L) / bins;
      let g = `<text class="note" x="${L}" y="${y0 - 70}">${title}: average ${m.toFixed(1)}, spread ${sd.toFixed(1)}</text>`;
      c.forEach((k, i) => { const hgt = (k / top) * 60; g += `<rect class="bar" x="${L + i * bw + 1}" y="${y0 - hgt}" width="${bw - 2}" height="${hgt}"/>`; });
      g += `<line class="axis" x1="${L}" x2="${R}" y1="${y0}" y2="${y0}"/><line class="mean-line" x1="${x(m)}" x2="${x(m)}" y1="${y0 - 66}" y2="${y0 + 4}"/>`;
      return g;
    };
    let g = row(plain, 110, "Without a baseline") + row(based, 230, `With a baseline of ${minus(b, 1)}`);
    for (let u = -60; u <= 100; u += 20) g += `<text class="tick" x="${x(u)}" y="248" text-anchor="middle">${u < 0 ? "−" : ""}${Math.abs(u)}</text>`;
    g += `<line class="ref-line" x1="${x(exact)}" x2="${x(exact)}" y1="40" y2="236"/><text class="note halo" x="${x(exact) + 8}" y="142">the true gradient: ${exact.toFixed(2)}</text>
      <text class="axis-name" x="${(L + R) / 2}" y="268" text-anchor="middle">the estimate from one episode, Σ Gₜ ∇ ln π(Aₜ | Sₜ), along the preference for stepping right</text>`;
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${W} 276" role="img" aria-label="Histograms of single-episode gradient estimates with and without a baseline">${g}</svg>
      <p class="fig-status">${N.toLocaleString("en")} episodes with p = 0.3; the spread is the standard deviation. The baseline is the average value of the cells visited, what a learned single-number baseline settles on. Each histogram is drawn to its own height, and the few estimates beyond the scale are counted at its ends. Solid lines: the averages.</p>`;
  };

  // ---- REINFORCE on the short corridor (after Figures 13.1 and 13.2) ----
  const CORRIDOR = { gamma: 1, features: "own", maxSteps: 1000 };
  const corridorReturns = (id, params, units, seed) => lab.simulate({ world: "corridor", algorithm: lab.algorithms[id], params: { ...CORRIDOR, ...params }, units, seed, snapshots: false }).metrics.return;
  function corridorCurves(host, { label, curves, note }) {
    RL.fig.average(host, {
      label, right: 132,
      x: { min: 1, max: 1000, ticks: [1, 200, 400, 600, 800, 1000], label: "Episodes" },
      y: { min: -90, max: -10, ticks: [-90, -70, -50, -30, -10], label: "Total reward per episode", digits: 1, tickDigits: 0 },
      curves: curves.map(({ id, name, dash, light }) => ({ id, name, dash, light })),
      refs: [{ value: corridor.bestValue, label: `best possible on average: ${minus(corridor.bestValue)}`, at: 1 }],
      at: (e) => `Episode ${e}`,
    }, {
      runs: 100, smooth: 10, note,
      sample: (seed) => Object.fromEntries(curves.map((c) => [c.id, corridorReturns(c.algorithm, c.params, 1000, seed)])),
    });
  }
  RL.demos["reinforce-alpha"] = function (host) {
    corridorCurves(host, {
      label: "REINFORCE on the short corridor with three step sizes",
      curves: [
        { id: "a12", name: "α = 2⁻¹²", algorithm: "reinforce", params: { alpha: 2 ** -12 } },
        { id: "a13", name: "α = 2⁻¹³", algorithm: "reinforce", params: { alpha: 2 ** -13 }, dash: true },
        { id: "a14", name: "α = 2⁻¹⁴", algorithm: "reinforce", params: { alpha: 2 ** -14 }, light: true },
      ],
      note: "Average of 100 runs, smoothed over 10 episodes; every run starts from the policy that steps right 5% of the time. With α = 2⁻¹², a few runs are pushed by one long episode to a policy that always steps the same way: it never arrives (each episode is cut at 1000 steps), its gradient is zero, and it stays there, dragging the average down.",
    });
  };
  RL.demos["baseline-curves"] = function (host) {
    corridorCurves(host, {
      label: "REINFORCE with and without a learned baseline on the short corridor",
      curves: [
        { id: "plain", name: "REINFORCE", algorithm: "reinforce", params: { alpha: 2 ** -13 }, dash: true },
        { id: "base", name: "with baseline", algorithm: "baseline", params: { alpha: 2 ** -9, alphaW: 2 ** -6 } },
      ],
      note: "Average of 100 runs, smoothed over 10 episodes. Without a baseline, α = 2⁻¹³, about the largest that stays safe; with it, αᶿ = 2⁻⁹ for the policy and αʷ = 2⁻⁶ for the baseline, a single learned number.",
    });
  };

  // ---- actor–critic on the cliff: the critic's step size ----
  RL.demos["critic-speed"] = function (host) {
    const set = [0.02, 0.1, 0.5].map((a, i) => ({ id: `w${i}`, name: `αʷ = ${a}`, alphaW: a, dash: i === 0, light: i === 2 }));
    RL.fig.average(host, {
      label: "Actor–critic on the cliff with three step sizes for the critic", right: 120,
      x: { min: 1, max: 300, ticks: [1, 50, 100, 150, 200, 250, 300], label: "Episodes" },
      y: { min: -100, max: 0, ticks: [-100, -75, -50, -25, 0], label: "Reward per episode", digits: 1, tickDigits: 0 },
      curves: set.map(({ id, name, dash, light }) => ({ id, name, dash, light })), at: (e) => `Episode ${e}`,
    }, {
      runs: 30, smooth: 10,
      sample: (seed) => Object.fromEntries(set.map((c) => [c.id, lab.simulate({ world: "cliff", algorithm: lab.algorithms["actor-critic"], params: { gamma: 1, alpha: 0.1, alphaW: c.alphaW, maxSteps: 2000 }, units: 300, seed, snapshots: false }).metrics.return])),
      note: "Average of 30 runs, smoothed over 10 episodes; the actor's step size is 0.1 throughout. Early episodes below −100 are cut off.",
    });
  };

  // ---- the entropy bonus on two gems: how often A2C ends up at the big gem, and what it earns ----
  RL.demos["entropy-study"] = function (host) {
    const betas = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], runs = 16, units = 150;
    const p = RL.fig.plot(host, {
      label: "A2C on the two gems: the share of runs that end at the big gem, and the reward per episode at the end, for each entropy bonus", right: 150,
      x: { min: 0, max: 6, ticks: [0, 1, 2, 3, 4, 5, 6], format: (i) => String(betas[i]), label: "β, the weight of the entropy bonus" },
      y: { min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], label: "Share, and reward", digits: 2, tickDigits: 1 },
      curves: [{ id: "share", name: "runs at the big gem" }, { id: "reward", name: "reward per episode", dash: true }],
      at: (i) => `β = ${betas[i]}`,
    });
    const share = new Float64Array(betas.length), reward = new Float64Array(betas.length);
    RL.fig.whenVisible(host, () => {
      let done = 0;
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (done < runs && performance.now() - t0 < 30) {
          betas.forEach((beta, i) => {
            const m = lab.simulate({ world: "two-gems", algorithm: lab.algorithms.a2c, params: { gamma: 0.95, maxSteps: 500, alpha: 2, alphaW: 0.3, workers: 4, n: 5, beta }, units, seed: 1000 + done, snapshots: false }).metrics;
            const end = lab.mean(m.return, units - 10);
            share[i] += end > 0.5 ? 1 : 0;
            reward[i] += end;
          });
          done++;
        }
        const xs = betas.map((_, i) => i);
        p.draw({ share: { xs, ys: Array.from(share, (v) => v / done) }, reward: { xs, ys: Array.from(reward, (v) => v / done) } });
        p.status(done < runs ? `Averaging run ${done} of ${runs}…` : `Each point: ${runs} runs of ${units} rounds of four episodes; the reward is that of the last 10 rounds (1 for the big gem, 0.3 for the small one, less for a policy that still wanders).`);
        if (done < runs) later(more);
      };
      more();
    });
  };

  // ---- GAE: the weights of the TD errors, and the bias–variance trade of λ ----
  RL.demos["gae-weights"] = function (host) {
    const steps = 16, gamma = 0.99, W = 640, H = 260, M = { l: 62, r: 20, t: 16, b: 50 }, pw = W - M.l - M.r, ph = H - M.t - M.b;
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="The weight GAE gives to each TD error after a step"></svg>
      <label class="fig-knob"><span>λ</span><input type="range" min="0" max="1" step="0.01" value="0.9"><output>0.9</output></label>`;
    const svg = host.querySelector("svg"), input = host.querySelector("input"), out = host.querySelector("output"), bw = pw / steps, y = (v) => M.t + (1 - v) * ph;
    function draw() {
      const lam = +input.value;
      out.textContent = lam.toFixed(2);
      let g = "";
      for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="gridline" x1="${M.l}" x2="${M.l + pw}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
      for (let k = 0; k < steps; k++) {
        const wgt = (gamma * lam) ** k, x0 = M.l + k * bw + 4;
        g += `<rect class="bar" x="${x0}" y="${y(wgt)}" width="${bw - 8}" height="${Math.max(0, y(0) - y(wgt))}"/><text class="tick" x="${x0 + (bw - 8) / 2}" y="${M.t + ph + 18}" text-anchor="middle">${k ? `δₜ₊${sub(k)}` : "δₜ"}</text>`;
        if (wgt > 0.005) g += `<text class="num small" x="${x0 + (bw - 8) / 2}" y="${y(wgt) - 5}" text-anchor="middle">${wgt.toFixed(2)}</text>`;
      }
      g += `<line class="axis" x1="${M.l}" x2="${M.l + pw}" y1="${y(0)}" y2="${y(0)}"/><line class="axis" x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${y(0)}"/>
        <text class="axis-name" x="${M.l + pw / 2}" y="${H - 8}" text-anchor="middle">the TD errors after step t, each weighted (γλ)ᵏ with γ = ${gamma}</text>
        <text class="note halo" x="${M.l + pw - 6}" y="${M.t + 14}" text-anchor="end">${lam === 0 ? "λ = 0: only the step's own TD error" : lam === 1 ? "λ = 1: every TD error, fading only by γ: the return minus the critic" : `the weights fade by γλ = ${(gamma * lam).toFixed(3)} a step; over a long episode they add up to ${(1 / (1 - gamma * lam)).toFixed(1)}`}</text>`;
      svg.innerHTML = g;
    }
    input.addEventListener("input", draw);
    draw();
  };
  RL.demos["gae-study"] = function (host) {
    const lambdas = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1], runs = 16, units = 100;
    const p = RL.fig.plot(host, {
      label: "PPO on Frozen Lake with GAE advantages: how good the policy becomes, for each λ", right: 40,
      x: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], format: (u) => String(u), label: "λ" },
      y: { min: 0, max: 0.2, ticks: [0, 0.05, 0.1, 0.15, 0.2], label: "Value of the policy, rounds 1–100", digits: 3, tickDigits: 2 },
      curves: [{ id: "v", name: "" }], at: (u) => `λ = ${u}`,
    });
    const sums = new Float64Array(lambdas.length);
    RL.fig.whenVisible(host, () => {
      let done = 0;
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (done < runs && performance.now() - t0 < 30) {
          lambdas.forEach((lambda, i) => {
            const m = lab.simulate({ world: "frozen-lake", algorithm: lab.algorithms.ppo, params: { gamma: 0.99, maxSteps: 200, alpha: 0.1, alphaW: 0.1, workers: 4, epochs: 4, lambda, clip: 0.2 }, units, seed: 1000 + done, snapshots: false, measures: ["policy-value"] }).metrics;
            sums[i] += lab.mean(m["policy-value"]);
          });
          done++;
        }
        p.draw({ v: { xs: lambdas, ys: Array.from(sums, (s) => s / done) } });
        p.status(done < runs ? `Averaging run ${done} of ${runs}…` : `Each point: the exact value of the start under the learned policy (γ = 0.99), averaged over 100 rounds of four episodes and ${runs} runs. Frozen Lake's ice makes returns noisy.`);
        if (done < runs) later(more);
      };
      more();
    });
  };

  // ---- the surrogate objective and the trust region, worked out exactly on the corridor ----
  // arg "clip": PPO's clipped objective instead of the KL trust region.
  RL.demos["trust-region"] = function (host, arg) {
    const p0 = 0.2, delta = 0.02, eps = 0.2, d = visits(p0), A = advantage(p0), clip = arg === "clip";
    const at = (p) => {
      let L = 0, Lc = 0;
      for (let s = 0; s < 3; s++) for (const a of [0, 1]) {
        const po = a ? p0 : 1 - p0, pn = a ? p : 1 - p, r = pn / po;
        L += d[s] * pn * A[s][a];
        Lc += d[s] * po * Math.min(r * A[s][a], Math.max(1 - eps, Math.min(1 + eps, r)) * A[s][a]);
      }
      return { L, Lc, gain: corridor.J(p) - corridor.J(p0) };
    };
    const ps = [], gain = [], L = [], Lc = [];
    for (let k = 8; k <= 92; k++) { const p = k / 100, r = at(p); ps.push(p); gain.push(r.gain); L.push(r.L); Lc.push(r.Lc); }
    const keep = (ys) => { const xs = [], vs = []; ps.forEach((p, i) => { if (ys[i] <= 30 && ys[i] >= -30) { xs.push(p); vs.push(ys[i]); } }); return { xs, ys: vs }; };
    // the trust region: where KL(π_old ‖ π) ≤ δ
    const edge = (dir) => { let lo = p0, hi = dir > 0 ? 0.999 : 0.001; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (klOf(p0, m) <= delta) lo = m; else hi = m; } return lo; };
    const left = edge(-1), right = edge(1);
    const curves = [{ id: "gain", name: "the true change of J", marks: false }, { id: "L", name: "the surrogate L", dash: true, marks: false }];
    if (clip) curves.push({ id: "Lc", name: "the clipped L", dash: "dotted", marks: false });
    const plot = RL.fig.plot(host, {
      label: clip ? "On the short corridor: the true change of the objective, the surrogate, and PPO's clipped surrogate, for every new policy" : "On the short corridor: the true change of the objective and the surrogate TRPO maximizes, with the trust region",
      right: 150, h: 320,
      x: { min: 0.08, max: 0.92, ticks: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], format: (u) => `${Math.round(u * 100)}%`, label: "p, the new policy's chance of stepping right (the old one: 20%)" },
      y: { min: -30, max: 30, ticks: [-30, -20, -10, 0, 10, 20, 30], label: "Change from the old policy", digits: 2, tickDigits: 0 },
      curves, at: (u) => `New policy: p = ${Math.round(u * 100)}%`,
    });
    const lines = { gain: keep(gain), L: keep(L) };
    if (clip) lines.Lc = keep(Lc);
    plot.draw(lines);
    const y0 = plot.y(30), y1 = plot.y(-30);
    const band = clip
      ? [p0 * (1 - eps), p0 * (1 + eps)]
      : [left, right];
    plot.svg.insertAdjacentHTML("afterbegin", `<rect class="region" x="${plot.x(band[0])}" y="${y0}" width="${plot.x(band[1]) - plot.x(band[0])}" height="${y1 - y0}"/>`);
    const best = ps[gain.indexOf(Math.max(...gain))];
    plot.svg.insertAdjacentHTML("beforeend", `<text class="note halo" x="${plot.x((band[0] + band[1]) / 2)}" y="${y0 + 14}" text-anchor="middle">${clip ? `ratio within 1 ± ${eps}` : `KL ≤ δ = ${delta}`}</text>
      <line class="ref-line" x1="${plot.x(best)}" x2="${plot.x(best)}" y1="${plot.y(at(best).gain) + 6}" y2="${plot.y(0)}"/>
      <text class="note halo" x="${plot.x(best)}" y="${plot.y(0) + 16}" text-anchor="middle">best: ${Math.round(best * 100)}%</text>`);
    const step = clip ? ps[Lc.indexOf(Math.max(...Lc))] : right;
    plot.status(clip
      ? `Exact, from the corridor's visits and advantages under the old policy (p = 20%). The clipped surrogate peaks at p = ${Math.round(step * 100)}%, close to the old policy: past the band its gains are capped, while the loss in the swapped cell, where stepping right is worse, is not.`
      : `Exact, from the corridor's visits and advantages under the old policy (p = 20%). The surrogate is a straight line here: maximized without a limit it would jump to the far end, where the true change is a large loss. Within the trust region (p from ${Math.round(left * 100)}% to ${Math.round(right * 100)}%) it stays close to the truth, and TRPO's step goes to the region's edge, p = ${Math.round(right * 100)}%.`);
  };

  // ---- PPO's clipped objective for one sample, as a function of the probability ratio ----
  RL.demos["ppo-clip"] = function (host) {
    const eps = 0.2;
    host.innerHTML = '<div class="fig-pair"><div></div><div></div></div>';
    const [a, b] = host.querySelectorAll(".fig-pair > div");
    for (const [box, sign] of [[a, 1], [b, -1]]) {
      const rs = Array.from({ length: 201 }, (_, i) => i / 100);
      const plot = RL.fig.plot(box, {
        label: `The clipped objective for one sample with ${sign > 0 ? "a positive" : "a negative"} advantage`, w: 320, h: 260, right: 16,
        x: { min: 0, max: 2, ticks: [0, 0.5, 1, 1.5, 2], format: (u) => String(u), label: `ratio r, advantage Â ${sign > 0 ? "> 0" : "< 0"}` },
        y: { min: -2, max: 2, ticks: [-2, -1, 0, 1, 2], label: "objective, in units of |Â|", digits: 2, tickDigits: 0 },
        curves: [{ id: "u", name: "", dash: true, light: true, marks: false }, { id: "c", name: "", marks: false }], at: (r) => `r = ${r}`,
      });
      plot.draw({ u: { xs: rs, ys: rs.map((r) => r * sign) }, c: { xs: rs, ys: rs.map((r) => Math.min(r * sign, Math.max(1 - eps, Math.min(1 + eps, r)) * sign)) } });
      const x = sign > 0 ? 1 + eps : 1 - eps;
      plot.svg.insertAdjacentHTML("beforeend", `<line class="ref-line" x1="${plot.x(x)}" x2="${plot.x(x)}" y1="${plot.y(2)}" y2="${plot.y(-2)}"/>
        <text class="note halo" x="${plot.x(x) + 6}" y="${plot.y(sign > 0 ? -1.4 : 1.6)}" text-anchor="start">${sign > 0 ? "1 + ε" : "1 − ε"}: no further push</text>`);
    }
    host.insertAdjacentHTML("beforeend", `<p class="fig-status">Solid: min(r Â, clip(r, 1 − ε, 1 + ε) Â) with ε = ${eps}; dashed: r Â, the unclipped surrogate. The objective is flat, so its gradient is zero, only where the change has already gone far enough in the direction the advantage favors.</p>`);
  };
})(globalThis.RL = globalThis.RL || {});
