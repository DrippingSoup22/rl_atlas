/* Story scene stream: one quantity estimated from a stream of noisy samples. The samples are dots over time, each rule's
   running estimate a line, and the true value a dashed line (it moves when the target drifts).
   cfg.source: { kind: "normal", mean, sd, drift (the spread of the mean's random step after each sample), n, seed }, or
   { kind: "loop", n, seed }: episodes of Sutton & Barto's one-state example of infinite variance, played by a behavior
   policy that picks each of two actions half the time and judged for a target policy that always goes left; each
   sample is then ρ·G, the episode's return times its importance ratio, and a 0 when the behavior went right.
   cfg.rules: { name: { rule: "average" | "constant" | "unbiased" | "ordinary" | "weighted", alpha, q0, label } }.
   Step keys: upto (samples shown so far; the time axis spans them, at least 10), rules (names drawn), truth (the dashed true value), domain ([lo, hi]),
   mark (a sample to ring, 1-based), note. */
(function (RL) {
  "use strict";
  const { formula, FORMULA } = RL.sceneKit;
  const W = 640, H = 300, M = { l: 44, r: 16, t: 18, b: 34 };

  function samples(src) {
    const rng = RL.lab.rng(src.seed ?? 1), n = src.n ?? 100, out = [];
    if (src.kind === "loop") {
      for (let i = 0; i < n; i++) {
        let rho = 1, G = 0, steps = 0;
        for (;;) {
          steps++;
          if (rng.next() < 0.5) { rho = 0; break; } // right: the target never goes right
          rho *= 2; // left, with probability 1 under the target and 1/2 under the behavior
          if (rng.next() < 0.1) { G = 1; break; } // the left loop ends with +1 one time in ten
        }
        out.push({ x: rho * G, rho, G, steps, truth: 1 });
      }
      return out;
    }
    let mean = src.mean ?? 0;
    for (let i = 0; i < n; i++) {
      out.push({ x: mean + (src.sd ?? 1) * rng.normal(), truth: mean });
      if (src.drift) mean += src.drift * rng.normal();
    }
    return out;
  }

  // Each rule's estimate after each sample (estimate[k] after k + 1 samples), from its first estimate q0.
  function estimate(xs, r) {
    const out = [];
    let q = r.q0 ?? 0, o = 0, sum = 0, wsum = 0, n = 0;
    for (const s of xs) {
      n++;
      if (r.rule === "average") q += (s.x - q) / n;
      else if (r.rule === "constant") q += r.alpha * (s.x - q);
      else if (r.rule === "unbiased") { o += r.alpha * (1 - o); q += (r.alpha / o) * (s.x - q); }
      else if (r.rule === "ordinary") { sum += s.x; q = sum / n; }
      else if (r.rule === "weighted") { sum += s.x; wsum += s.rho; q = wsum ? sum / wsum : r.q0 ?? 0; }
      out.push(q);
    }
    return out;
  }

  RL.scenes.stream = {
    create(card, cfg) {
      const xs = samples(cfg.source || {}), names = Object.keys(cfg.rules || {}), est = Object.fromEntries(names.map((k) => [k, estimate(xs, cfg.rules[k])]));
      card.innerHTML = `<div class="stream-legend"></div><svg class="stream" viewBox="0 0 ${W} ${H}" role="img" aria-label="Samples and running estimates"></svg>${FORMULA}
        <div class="scene-foot"><span class="scene-note"></span></div>`;
      const svg = card.querySelector("svg"), note = card.querySelector(".scene-note"), legend = card.querySelector(".stream-legend"), showFormula = formula(card, cfg);
      const n = xs.length, pw = W - M.l - M.r, ph = H - M.t - M.b;

      return {
        apply(st) {
          const upto = Math.min(n, st.upto ?? n), on = st.rules || names;
          // the time axis spans the samples shown so far (at least 10), so the first few are spread out
          const span = Math.max(10, upto), X = (k) => M.l + ((k + 0.5) / span) * pw;
          const [lo, hi] = st.domain || cfg.domain || [Math.min(...xs.map((s) => s.x)), Math.max(...xs.map((s) => s.x))];
          const Y = (v) => M.t + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * ph;
          let g = "";
          for (const v of RL.ticks ? RL.ticks(lo, hi, 4) : [lo, (lo + hi) / 2, hi]) g += `<line class="grid" x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}"/><text class="tick" x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${+v.toFixed(2)}</text>`;
          g += `<text class="tick" x="${M.l}" y="${H - 10}">1</text><text class="tick" x="${W - M.r}" y="${H - 10}" text-anchor="end">${span} samples</text>`;
          if (st.truth !== false) g += `<path class="truth" d="${xs.slice(0, span).map((s, k) => `${k ? "L" : "M"}${X(k).toFixed(1)} ${Y(s.truth).toFixed(1)}`).join("")}"/>`;
          const labels = []; // off-scale samples far beyond it get their value, the largest first, where a label has room
          xs.forEach((s, k) => {
            if (k >= upto) return;
            const out = s.x > hi || s.x < lo, far = s.x >= 4 * hi || s.x <= lo - 3 * (hi - lo);
            g += `<circle class="dot${out ? " out" : ""}${st.mark === k + 1 ? " mark" : ""}" cx="${X(k).toFixed(1)}" cy="${Y(s.x).toFixed(1)}" r="${st.mark === k + 1 ? 6 : upto <= 30 ? 4.5 : 3}"/>`;
            if (out && (far || upto <= 30)) labels.push({ x: X(k), s });
          });
          const placed = [];
          for (const { x, s } of labels.sort((a, b) => Math.abs(b.s.x) - Math.abs(a.s.x))) {
            if (placed.some((p) => Math.abs(p - x) < 26)) continue;
            placed.push(x);
            g += `<text class="out-t" x="${x.toFixed(1)}" y="${s.x > hi ? M.t - 4 : H - M.b + 12}" text-anchor="middle">${+s.x.toFixed(0)}</text>`;
          }
          const color = (name) => `var(--s${names.indexOf(name) + 1})`; // each rule keeps its color from step to step
          on.forEach((name) => {
            const e = est[name].slice(0, upto);
            if (!e.length) return;
            g += `<path class="est" style="stroke: ${color(name)}" d="${e.map((v, k) => `${k ? "L" : "M"}${X(k).toFixed(1)} ${Y(v).toFixed(1)}`).join("")}"/>`;
            g += `<circle class="end" style="fill: ${color(name)}" cx="${X(upto - 1).toFixed(1)}" cy="${Y(e[upto - 1]).toFixed(1)}" r="4.5"/>`;
          });
          svg.innerHTML = g;
          legend.innerHTML = on.map((name) => `<span><i class="key" style="--k: ${color(name)}"></i>${RL.esc(cfg.rules[name].label || name)}: <b>${est[name][upto - 1].toFixed(2)}</b></span>`).join("") +
            (st.truth !== false ? `<span><i class="key truth-key"></i>true value: <b>${xs[upto - 1].truth.toFixed(2)}</b></span>` : "");
          showFormula(st);
          note.textContent = st.note || `after ${upto} sample${upto === 1 ? "" : "s"}`;
        },
        destroy() {},
      };
    },
  };
})(globalThis.RL = globalThis.RL || {});
