/* Textbook figures: plain line drawings, computed by the Lab when they first scroll into view, so the numbers in a
   figure are always the ones the Lab really produces. This file holds what the figures share: drawing on first
   sight, and a dry line plot that fills in as runs are averaged. The figures live in one file per part. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  let uid = 0;

  function whenVisible(host, draw) {
    const io = new IntersectionObserver(([it]) => {
      if (it.isIntersecting) { io.disconnect(); draw(); }
    }, { rootMargin: "200px" });
    io.observe(host);
  }
  const num = (v, d) => {
    const a = Math.abs(v), digits = d ?? (a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3);
    return (v < 0 ? "−" : "") + a.toFixed(digits);
  };
  const el = (tag, attrs = {}) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };

  // A dry line plot. spec: { w, h, x: { max, from, log, ticks, label }, y: { min, max, ticks, label, percent, log },
  // curves: [{ id, name, dash }] }. Values[i] sits at x = from + i (from = 1 unless said: "after i + 1 episodes").
  // Returns { draw(lines), status(text) }, where lines maps a curve's id to its values.
  function plot(host, spec) {
    const W = spec.w || 640, H = spec.h || 300, M = { l: 62, r: spec.right ?? 118, t: 14, b: 46 };
    const pw = W - M.l - M.r, ph = H - M.t - M.b, X = spec.x, Y = spec.y, from = X.from ?? 1;
    const x = X.log ? (u) => M.l + (Math.log10(Math.max(1, u)) / Math.log10(X.max)) * pw : (u) => M.l + ((u - (X.min ?? 0)) / (X.max - (X.min ?? 0))) * pw;
    const yv = Y.log ? (v) => (Math.log10(Y.max) - Math.log10(Math.max(Y.min, v))) / (Math.log10(Y.max) - Math.log10(Y.min)) : (v) => (Y.max - Math.min(Y.max, Math.max(Y.min, v))) / (Y.max - Y.min);
    const y = (v) => M.t + yv(v) * ph;
    const fy = Y.percent ? (v) => `${Math.round(v * 100)}%` : (v) => num(v, Y.digits);
    const svg = el("svg", { class: "fig plot", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.label || "" });
    let g = "";
    for (const v of Y.ticks) g += `<line class="gridline" x1="${M.l}" x2="${M.l + pw}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${fy(v)}</text>`;
    for (const u of X.ticks) g += `<text class="tick" x="${x(u)}" y="${M.t + ph + 20}" text-anchor="middle">${u.toLocaleString("en")}</text>`;
    for (const r of spec.refs || []) g += `<line class="ref-line" x1="${M.l}" x2="${M.l + pw}" y1="${y(r.value)}" y2="${y(r.value)}"/><text class="note" x="${M.l + pw - 4}" y="${y(r.value) - 6}" text-anchor="end">${r.label}</text>`;
    g += `<line class="axis" x1="${M.l}" x2="${M.l + pw}" y1="${M.t + ph}" y2="${M.t + ph}"/><line class="axis" x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${M.t + ph}"/>
      <text class="axis-name" x="${M.l + pw / 2}" y="${H - 6}" text-anchor="middle">${X.label}</text>
      <text class="axis-name" transform="translate(16 ${M.t + ph / 2}) rotate(-90)" text-anchor="middle">${Y.label}</text>`;
    g += spec.curves.map((c) => `<path class="curve${c.dash ? ` ${c.dash === true ? "dashed" : c.dash}` : ""}" data-id="${c.id}"/><text class="curve-name" data-id="${c.id}" x="${M.l + pw + 8}">${c.name}</text>`).join("");
    g += `<line class="cross" y1="${M.t}" y2="${M.t + ph}"/><rect class="hit" x="${M.l}" y="${M.t}" width="${pw}" height="${ph}"/>`;
    svg.innerHTML = g;
    host.replaceChildren(svg);
    const status = document.createElement("p");
    status.className = "fig-status";
    host.appendChild(status);
    let lines = {};

    function draw(next) {
      lines = next;
      const ends = [];
      for (const c of spec.curves) {
        const L = lines[c.id];
        if (!L) continue;
        const n = L.length, step = Math.max(1, Math.floor(n / (pw * 1.5)));
        let d = "";
        for (let i = 0; i < n; i += step) {
          let sum = 0, k = 0;
          for (let j = i; j < Math.min(n, i + step); j++) { sum += L[j]; k++; }
          d += `${d ? "L" : "M"}${x(from + i + (k - 1) / 2).toFixed(1)} ${y(sum / k).toFixed(1)}`;
        }
        svg.querySelector(`.curve[data-id="${c.id}"]`).setAttribute("d", d);
        ends.push({ id: c.id, y: y(L[n - 1]) + 4 });
      }
      // Names at the right end of their curves, nudged apart where they would overlap.
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
      for (const e of ends) svg.querySelector(`.curve-name[data-id="${e.id}"]`).setAttribute("y", e.y);
    }

    const cross = svg.querySelector(".cross"), hit = svg.querySelector(".hit");
    hit.addEventListener("pointermove", (ev) => {
      const any = Object.values(lines)[0];
      if (!any) return;
      const box = svg.getBoundingClientRect(), px = ((ev.clientX - box.left) / box.width) * W;
      const u = X.log ? 10 ** (((px - M.l) / pw) * Math.log10(X.max)) : (X.min ?? 0) + ((px - M.l) / pw) * (X.max - (X.min ?? 0));
      const i = Math.max(0, Math.min(any.length - 1, Math.round(u - from)));
      cross.setAttribute("x1", x(from + i));
      cross.setAttribute("x2", x(from + i));
      cross.style.opacity = 1;
      RL.tip.show(ev.clientX, ev.clientY, `<div class="head">${spec.at ? spec.at(from + i) : `${X.label} ${from + i}`}</div>` +
        spec.curves.filter((c) => lines[c.id]).map((c) => `<div class="row"><b>${fy(lines[c.id][i])}</b><span>${c.name}</span></div>`).join(""));
    });
    hit.addEventListener("pointerleave", () => { cross.style.opacity = 0; RL.tip.hide(); });
    return { draw, status: (text) => { status.textContent = text; }, svg, x, y };
  }

  // Average sample(seed) over many runs, a few at a time, redrawing as they come in. sample returns { id: values }.
  // smooth: a trailing moving average over that many points, applied to the averages.
  function average(host, spec, { runs, sample, smooth = 1, note }) {
    const p = plot(host, spec);
    whenVisible(host, () => {
      let sums = null, done = 0;
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (done < runs && performance.now() - t0 < 30) {
          const out = sample(1000 + done);
          sums ||= Object.fromEntries(Object.entries(out).map(([k, v]) => [k, new Float64Array(v.length)]));
          for (const k in out) for (let i = 0; i < out[k].length; i++) sums[k][i] += out[k][i];
          done++;
        }
        const lines = {};
        for (const k in sums) {
          const L = (lines[k] = new Float64Array(sums[k].length));
          let acc = 0;
          for (let i = 0; i < L.length; i++) {
            acc += sums[k][i] / done;
            if (i >= smooth) acc -= sums[k][i - smooth] / done;
            L[i] = acc / Math.min(i + 1, smooth);
          }
        }
        p.draw(lines);
        p.status(done < runs ? `Averaging run ${done} of ${runs}…` : note || `Average of ${runs} runs, each with its own seed${smooth > 1 ? `; smoothed over ${smooth} points` : ""}.`);
        if (done < runs) setTimeout(more, 16);
      };
      more();
    });
    return p;
  }

  RL.fig = { whenVisible, plot, average, num, uid: () => ++uid };
})(globalThis.RL = globalThis.RL || {});
