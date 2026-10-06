/* Story scene surprise: TD errors in a conditioning task, trial after trial, as in the recordings of dopamine neurons.
   A cue appears at an unpredictable moment; `delay` steps later a reward of 1 arrives. TD(0) learns the value of each
   moment after the cue (step size `alpha`, discount `gamma`); before the cue nothing can be predicted, so that moment
   is worth 0. Each row shows one trial's TD errors over time, a bar per moment: up for a pleasant surprise, down for
   a disappointment. Step keys: trials (the trials shown as rows, counted from 1), omit (n: add a trial after the n-th
   whose reward never comes), values (n: the predictions after n trials, as a row of dots under the bars). */
(function (RL) {
  "use strict";
  const { formula, FORMULA, signed } = RL.sceneKit;
  const ROWS = 7, X0 = 104, DX = 66, TOP = 46, RH = 44, BW = 22;

  // Every trial's TD errors, and the predictions after each trial (V[n]: after n trials), for normal trials.
  function learn(cfg, trials) {
    const D = cfg.delay ?? 4, a = cfg.alpha ?? 0.4, g = cfg.gamma ?? 1, V = new Float64Array(D);
    const run = (v, omit) => {
      const d = [0, g * v[0]]; // before the cue (nothing to predict), and the cue appearing
      for (let k = 0; k < D; k++) {
        const r = k === D - 1 && !omit ? 1 : 0, next = k === D - 1 ? 0 : v[k + 1], delta = r + g * next - v[k];
        d.push(delta);
        v[k] += a * delta;
      }
      d.push(0); // after the trial: nothing new
      return d;
    };
    const deltas = [], values = [Float64Array.from(V)];
    for (let t = 0; t < trials; t++) { deltas.push(run(V, false)); values.push(Float64Array.from(V)); }
    return { D, deltas, values, omitted: (n) => run(Float64Array.from(values[n]), true) };
  }

  RL.scenes.surprise = {
    create(card, cfg) {
      const D = cfg.delay ?? 4, cols = D + 3, W = X0 + cols * DX, H = TOP + ROWS * RH + 26;
      const x = (k) => X0 + k * DX + DX / 2;
      card.innerHTML = `<svg class="surprise" viewBox="0 0 ${W} ${H}" role="img" aria-label="TD errors over the moments of each trial"></svg>${FORMULA}
        <div class="scene-foot"><span class="scene-note"></span></div>`;
      const svg = card.querySelector("svg"), note = card.querySelector(".scene-note"), showFormula = formula(card, cfg);
      const names = ["before", "cue", ...Array.from({ length: D - 1 }, (_, k) => `+${k + 1}`), "reward", "after"];
      let g = `<g class="head">`;
      names.forEach((n, k) => { g += `<text class="moment${k === 1 ? " cue" : k === D + 1 ? " rew" : ""}" x="${x(k)}" y="18" text-anchor="middle">${n}</text>`; });
      g += `<rect class="mark cue" x="${x(1) - DX / 2 + 4}" y="26" width="${DX - 8}" height="${ROWS * RH + 14}" rx="8"/>`;
      g += `<rect class="mark rew" x="${x(D + 1) - DX / 2 + 4}" y="26" width="${DX - 8}" height="${ROWS * RH + 14}" rx="8"/></g>`;
      for (let r = 0; r < ROWS; r++) {
        const base = TOP + r * RH + RH / 2;
        g += `<g class="row" data-r="${r}"><text class="row-name" x="12" y="${base + 4}"></text><line class="base" x1="${X0}" x2="${W - 8}" y1="${base}" y2="${base}"/>`;
        for (let k = 0; k < cols; k++) g += `<rect class="bar" data-k="${k}" x="${x(k) - BW / 2}" width="${BW}" rx="3" y="${base}" height="0"/><text class="num" data-k="${k}" x="${x(k)}" text-anchor="middle"></text>`;
        g += `<g class="dots"></g></g>`;
      }
      svg.innerHTML = g;
      const most = Math.max(60, ...(cfg.maxTrials ? [cfg.maxTrials] : []));
      const L = learn(cfg, most);

      // A row at its baseline, with bars up to BAR high: the rows shown share the height, so a few rows get tall bars.
      function row(r, label, d, base, v, BAR) {
        const el = svg.querySelector(`.row[data-r="${r}"]`);
        el.classList.toggle("off", !d);
        if (!d) return;
        el.querySelector(".row-name").textContent = label;
        el.querySelector(".row-name").setAttribute("y", base + 4);
        const line = el.querySelector(".base");
        line.setAttribute("y1", base);
        line.setAttribute("y2", base);
        el.querySelectorAll(".bar").forEach((b, k) => {
          const v0 = d[k] ?? 0, h = Math.min(1, Math.abs(v0)) * BAR;
          b.style.y = `${v0 >= 0 ? base - h : base}px`;
          b.style.height = `${Math.max(h, 0.001)}px`;
          b.classList.toggle("neg", v0 < 0);
        });
        el.querySelectorAll(".num").forEach((t, k) => {
          const v0 = d[k] ?? 0;
          t.textContent = Math.abs(v0) >= 0.05 ? signed(v0, 2) : "";
          t.setAttribute("y", v0 >= 0 ? base - Math.min(1, Math.abs(v0)) * BAR - 4 : base + Math.min(1, Math.abs(v0)) * BAR + 12);
        });
        // the predictions, as dots on the moments after the cue (0 to 1, bottom to top of the row)
        el.querySelector(".dots").innerHTML = v ? Array.from(v, (p, k) => `<circle class="pred" cx="${x(k + 1)}" cy="${base + BAR - 2 * BAR * Math.max(0, Math.min(1, p))}" r="3.5"/>`).join("") +
          `<text class="pred-name" x="12" y="${base + 20}">• predictions</text>` : "";
      }

      return {
        apply(st) {
          const list = st.trials || [1];
          const rows = list.map((t) => ({ label: `trial ${t}`, d: L.deltas[t - 1], v: st.values ? L.values[t - 1] : null }));
          if (st.omit) rows.push({ label: `trial ${st.omit + 1}`, d: L.omitted(st.omit), v: null, omit: true });
          const rh = Math.min(150, (ROWS * RH) / rows.length), bar = Math.min(40, rh * 0.3);
          for (let r = 0; r < ROWS; r++) row(r, rows[r]?.label, rows[r]?.d, TOP + 8 + r * rh + rh / 2, rows[r]?.v, bar);
          svg.querySelectorAll(".row").forEach((el, r) => el.classList.toggle("omit", !!rows[r]?.omit));
          showFormula(st);
          note.textContent = st.note || (st.omit ? `trial ${st.omit + 1}: the cue, and no reward` : "");
        },
        destroy() {},
      };
    },
  };
})(globalThis.RL = globalThis.RL || {});
