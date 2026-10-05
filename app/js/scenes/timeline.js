/* Story scene timeline: the rewards that follow time t, how much each one counts, and what they add up to. */
(function (RL) {
  "use strict";
  const { formula, FORMULA, signed } = RL.sceneKit;

  // ---- timeline: the rewards that follow time t, how much each one counts, and what they add up to ----
  RL.scenes.timeline = {
    create(card, cfg) {
      const N = 10, X0 = 74, DX = 44, BW = 26, SUM_X = 556, W_BASE = 266, W_MAX = 48;
      const x = (k) => X0 + k * DX;
      card.innerHTML = `<svg class="timeline" viewBox="0 0 600 290" role="img" aria-label="Rewards along time and the return"></svg>${FORMULA}
        <div class="scene-foot"><span class="scene-note"></span></div>`;
      const svg = card.querySelector("svg"), note = card.querySelector(".scene-note"), showFormula = formula(card, cfg);
      let g = "";
      for (let r = 0; r < 2; r++) {
        g += `<g class="row" data-r="${r}"><text class="row-name" x="14"></text><line class="base" x1="${x(0) - 24}" x2="${x(N - 1) + 24}"/>`;
        for (let k = 0; k < N; k++) {
          g += `<g class="col" data-k="${k}"><rect class="ghost" x="${x(k) - BW / 2}" width="${BW}" rx="4"/><rect class="bar" x="${x(k) - BW / 2}" width="${BW}" rx="4"/>
            <text class="val" x="${x(k)}" text-anchor="middle"></text></g>`;
        }
        g += `<text class="more" x="${x(N - 1) + 34}" text-anchor="middle">…</text><text class="sum-name" x="${SUM_X}" text-anchor="middle"></text><text class="sum" x="${SUM_X}" text-anchor="middle"></text></g>`;
      }
      for (let k = 0; k < N; k++) g += `<text class="time" data-k="${k}" x="${x(k)}" y="206" text-anchor="middle">t+${k + 1}</text>`;
      g += `<g class="weights"><text class="row-name" x="14" y="${W_BASE - 18}">γᵏ</text><line class="base" x1="${x(0) - 24}" x2="${x(N - 1) + 24}" y1="${W_BASE}" y2="${W_BASE}"/>`;
      for (let k = 0; k < N; k++) g += `<rect class="w" data-k="${k}" x="${x(k) - BW / 2 + 5}" width="${BW - 10}" rx="3"/><text class="w-val" data-k="${k}" x="${x(k)}" y="${W_BASE + 15}" text-anchor="middle"></text>`;
      g += `</g><g class="recursion"><path class="bracket"/><text class="br-name rest" text-anchor="middle"></text><text class="br-name first" text-anchor="middle">Rₜ₊₁</text></g>`;
      svg.innerHTML = g;
      const shown = new Map(); // the sum each row last showed, so the next one counts from there

      function count(el, from, to, text) {
        const t0 = performance.now(), ms = RL.reducedMotion() ? 1 : 600;
        const frame = (t) => {
          const u = Math.min(1, (t - t0) / ms);
          el.textContent = text(from + (to - from) * (1 - (1 - u) ** 3));
          if (u < 1 && el.isConnected) requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
      }

      return {
        apply(st) {
          const names = [].concat(st.seq || Object.keys(cfg.sequences)[0]);
          const n = Math.min(N, st.count ?? N), gamma = st.gamma, two = names.length > 1;
          const unit = two ? 4.4 : 8, bases = two ? [74, 172] : [124];
          svg.querySelectorAll(".row").forEach((row, r) => {
            const seq = cfg.sequences[names[r]];
            row.style.display = seq ? "" : "none";
            if (!seq) return;
            const base = bases[r];
            row.querySelector(".row-name").textContent = two ? names[r] : "R";
            row.querySelector(".row-name").setAttribute("y", base - 6);
            const line = row.querySelector(".base");
            line.setAttribute("y1", base);
            line.setAttribute("y2", base);
            let G = 0, top = base;
            row.querySelectorAll(".col").forEach((col, k) => {
              const R = seq[k] ?? 0, weight = gamma === undefined ? 1 : gamma ** k, v = st.discount ? R * weight : R;
              const on = k < n && k < seq.length;
              if (on) G += v;
              const h = Math.abs(v) * unit, gh = Math.abs(R) * unit;
              col.classList.toggle("off", !on);
              col.classList.toggle("hot", st.highlight === k);
              const bar = col.querySelector(".bar"), ghost = col.querySelector(".ghost"), val = col.querySelector(".val");
              bar.style.y = `${v >= 0 ? base - h : base}px`;
              bar.style.height = `${Math.max(h, 0.001)}px`;
              bar.classList.toggle("neg", v < 0);
              ghost.style.y = `${R >= 0 ? base - gh : base}px`;
              ghost.style.height = `${Math.max(gh, 0.001)}px`;
              ghost.style.opacity = st.discount && R ? 1 : 0;
              val.textContent = R ? signed(v, st.discount && weight < 1 && k ? 1 : 0) : "0";
              val.setAttribute("y", v >= 0 ? base - h - 6 : base + h + 14);
              if (on && v > 0) top = Math.min(top, base - h);
              if (on && st.discount && R > 0) top = Math.min(top, base - gh);
            });
            // A stream that goes on forever repeats its last reward; with γ < 1 the endless tail still adds up.
            if (st.infinite && gamma !== undefined && st.discount) G += seq[seq.length - 1] * gamma ** n / (1 - gamma);
            row.querySelector(".more").setAttribute("y", base + 4);
            row.querySelector(".more").style.opacity = st.infinite ? 1 : 0;
            const sum = row.querySelector(".sum"), sumName = row.querySelector(".sum-name");
            sum.setAttribute("y", base + 2);
            sumName.setAttribute("y", base - 24);
            sumName.textContent = st.sum ? (two ? `G (${names[r]})` : "Gₜ") : "";
            const key = names[r], prev = shown.get(key) ?? 0, infinite = st.infinite && gamma === undefined;
            if (!st.sum) sum.textContent = "";
            else if (infinite) sum.textContent = "∞";
            else count(sum, prev, G, (v) => signed(v, st.discount ? 2 : 0).replace(/^\+/, ""));
            shown.set(key, st.sum && !infinite ? G : 0);
            if (r === 0) svg.dataset.top = top;
          });
          svg.querySelectorAll(".time").forEach((t, k) => t.classList.toggle("off", k >= n));
          const weights = svg.querySelector(".weights");
          weights.style.opacity = st.weights ? 1 : 0;
          weights.querySelectorAll(".w").forEach((b, k) => {
            const wgt = gamma === undefined ? 1 : gamma ** k, h = wgt * W_MAX;
            b.style.y = `${W_BASE - h}px`;
            b.style.height = `${Math.max(h, 0.001)}px`;
            b.classList.toggle("off", k >= n);
          });
          weights.querySelectorAll(".w-val").forEach((t, k) => {
            t.textContent = gamma === undefined ? "1" : (gamma ** k).toFixed(2).replace(/^0/, "");
            t.classList.toggle("off", k >= n);
          });
          const rec = svg.querySelector(".recursion"), y = Math.max(16, +svg.dataset.top - 16);
          rec.style.opacity = st.recursion && !two ? 1 : 0;
          rec.querySelector(".bracket").setAttribute("d", `M${x(1) - 16} ${y + 8} V${y} H${x(n - 1) + 16} V${y + 8}`);
          const rest = rec.querySelector(".rest");
          rest.setAttribute("x", (x(1) + x(n - 1)) / 2);
          rest.setAttribute("y", y - 6);
          rest.textContent = gamma === undefined ? "Gₜ₊₁" : `γ · Gₜ₊₁`;
          const first = rec.querySelector(".first");
          first.setAttribute("x", x(0));
          first.setAttribute("y", y - 6);
          showFormula(st);
          note.textContent = gamma === undefined ? "" : `γ = ${gamma}: a reward k steps after the next one counts γᵏ of its size`;
        },
        destroy() {},
      };
    },
  };
})(globalThis.RL = globalThis.RL || {});
