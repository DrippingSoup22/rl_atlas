/* Story scene predictor: a handful of recorded episodes, and what two predictors make of them (Sutton & Barto's
   Example 6.4, "You are the predictor"). cfg.episodes is a list of episodes, each a list of [state, reward] pairs:
   the state, then the reward of leaving it. Monte Carlo predicts each state by the average return that followed it;
   TD, run on the batch until it settles, predicts each state by the average reward and next prediction: a guess built
   from a guess. Step keys: show (how many episodes are listed, all by default), mark (a state whose visits are
   outlined), mc and td (show each predictor's values), link (an arrow from B's TD value to A's: where it came from). */
(function (RL) {
  "use strict";
  const { formula, FORMULA, signed } = RL.sceneKit;

  // Both predictors on a batch of episodes: Monte Carlo averages returns; batch TD(0), with a small step, settles on
  // the values of the Markov model fitted to the batch (certainty equivalence).
  function predict(episodes, gamma = 1) {
    const states = [...new Set(episodes.flatMap((ep) => ep.map(([s]) => s)))];
    const G = Object.fromEntries(states.map((s) => [s, []]));
    for (const ep of episodes) {
      let g = 0;
      for (let t = ep.length - 1; t >= 0; t--) { g = ep[t][1] + gamma * g; G[ep[t][0]].push(g); }
    }
    const mc = Object.fromEntries(states.map((s) => [s, G[s].reduce((a, b) => a + b, 0) / G[s].length]));
    const td = Object.fromEntries(states.map((s) => [s, 0]));
    for (let sweep = 0; sweep < 20000; sweep++) {
      let change = 0;
      for (const ep of episodes) for (let t = 0; t < ep.length; t++) {
        const [s, r] = ep[t], next = t + 1 < ep.length ? td[ep[t + 1][0]] : 0, d = r + gamma * next - td[s];
        td[s] += 0.001 * d;
        change = Math.max(change, Math.abs(d * 0.001));
      }
      if (change < 1e-9) break;
    }
    return { states, mc, td };
  }

  RL.scenes.predictor = {
    create(card, cfg) {
      const eps = cfg.episodes || [], { states, mc, td } = predict(eps, cfg.gamma ?? 1);
      const RH = 26, top = 38, H = Math.max(260, top + eps.length * RH + 20), W = 680, PX = 340, PW = 250;
      card.innerHTML = `<svg class="predictor" viewBox="0 0 ${W} ${H}" role="img" aria-label="Recorded episodes and two predictions"></svg>${FORMULA}
        <div class="scene-foot"><span class="scene-note"></span></div>`;
      const svg = card.querySelector("svg"), note = card.querySelector(".scene-note"), showFormula = formula(card, cfg);
      let g = `<text class="head" x="16" y="18">${eps.length} episodes</text>`;
      eps.forEach((ep, i) => {
        const y = top + i * RH;
        g += `<g class="ep" data-i="${i}">`;
        ep.forEach(([s, r], t) => {
          const x = 30 + t * 104;
          g += `<g class="tok" data-s="${s}"><circle cx="${x}" cy="${y}" r="10"/><text x="${x}" y="${y + 4}" text-anchor="middle">${s}</text></g>`;
          g += `<line class="arr" x1="${x + 12}" x2="${x + 66}" y1="${y}" y2="${y}"/><text class="r${r ? " pay" : ""}" x="${x + 39}" y="${y - 5}" text-anchor="middle">${signed(r, 0)}</text>`;
          if (t === ep.length - 1) g += `<rect class="end" x="${x + 70}" y="${y - 9}" width="26" height="18" rx="4"/><text class="end-t" x="${x + 83}" y="${y + 4}" text-anchor="middle">end</text>`;
        });
        g += `</g>`;
      });
      // the two panels of predictions, one bar per state
      [["mc", "Monte Carlo: the returns seen"], ["td", "TD: reward + next guess"]].forEach(([k, title], p) => {
        const y0 = 40 + p * 110;
        g += `<g class="panel" data-k="${k}"><text class="ptitle" x="${PX}" y="${y0}">${title}</text>`;
        states.forEach((s, j) => {
          const y = y0 + 14 + j * 34;
          g += `<text class="sname" x="${PX}" y="${y + 15}">V(${s})</text><rect class="track" x="${PX + 44}" y="${y + 3}" width="${PW - 90}" height="16" rx="4"/>
            <rect class="vbar" data-s="${s}" x="${PX + 44}" y="${y + 3}" width="0" height="16" rx="4"/><text class="vnum" data-s="${s}" x="${PX + PW - 40}" y="${y + 16}"></text>`;
        });
        g += `</g>`;
      });
      g += `<defs><marker id="pred-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="head-mark"/></marker></defs>
        <path class="link" d="" marker-end="url(#pred-head)"/><text class="link-t" text-anchor="start"></text>`;
      svg.innerHTML = g;
      const barX = (v) => PX + 44 + Math.max(0, Math.min(1, v)) * (PW - 90);

      return {
        apply(st) {
          const n = st.show ?? eps.length;
          svg.querySelectorAll(".ep").forEach((el, i) => {
            el.classList.toggle("off", i >= n);
            el.classList.toggle("dim", !!st.mark && !eps[i].some(([s]) => s === st.mark));
          });
          svg.querySelectorAll(".tok").forEach((el) => el.classList.toggle("mark", el.dataset.s === st.mark));
          for (const [k, vals] of [["mc", mc], ["td", td]]) {
            const on = !!st[k], panel = svg.querySelector(`.panel[data-k="${k}"]`);
            panel.classList.toggle("off", !on);
            panel.querySelectorAll(".vbar").forEach((b) => b.setAttribute("width", on ? Math.max(0.5, barX(vals[b.dataset.s]) - PX - 44) : 0));
            panel.querySelectorAll(".vnum").forEach((t) => { t.textContent = on ? vals[t.dataset.s].toFixed(2) : ""; });
          }
          // the arrow: TD's value of A is borrowed from its value of B
          const link = svg.querySelector(".link");
          if (st.link && st.td) {
            const j = (s) => states.indexOf(s), y = (s) => 40 + 110 + 14 + j(s) * 34 + 11, [from, to] = st.link;
            const xr = PX + PW + 6; // an arc in the margin, from B's value up to A's
            link.setAttribute("d", `M${xr - 6} ${y(from)} C ${xr + 26} ${y(from)}, ${xr + 26} ${y(to)}, ${xr - 2} ${y(to)}`);
            link.style.opacity = 1;
            const label = svg.querySelector(".link-t");
            label.setAttribute("x", xr + 24);
            label.setAttribute("y", (y(from) + y(to)) / 2 + 4);
            label.textContent = `0 + V(${from})`;
            label.style.opacity = 1;
          } else { link.style.opacity = 0; svg.querySelector(".link-t").style.opacity = 0; }
          showFormula(st);
          note.textContent = st.note || "";
        },
        destroy() {},
      };
    },
  };
})(globalThis.RL = globalThis.RL || {});
