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

  // A dry line plot. spec: { w, h, x: { min, max, from, log, log2, ticks, label, format }, y: { min, max, ticks, label,
  // percent, log }, curves: [{ id, name, dash: true | "dotted", light }] }. Values[i] sits at x = from + i (from = 1 unless
  // said: "after i + 1 episodes"); a curve can also be points, { xs, ys }, drawn with a mark at each one.
  // x.log runs from 1 to max on a log10 scale, x.log2 from min to max on a log2 scale.
  // Returns { draw(lines), status(text) }, where lines maps a curve's id to its values.
  function plot(host, spec) {
    const W = spec.w || 640, H = spec.h || 300, M = { l: 62, r: spec.right ?? 118, t: 14, b: 46 };
    const pw = W - M.l - M.r, ph = H - M.t - M.b, X = spec.x, Y = spec.y, from = X.from ?? 1;
    const lo = X.min ?? 0, l2 = X.log2 ? [Math.log2(X.min), Math.log2(X.max)] : null;
    const x = X.log2 ? (u) => M.l + ((Math.log2(u) - l2[0]) / (l2[1] - l2[0])) * pw
      : X.log ? (u) => M.l + (Math.log10(Math.max(1, u)) / Math.log10(X.max)) * pw : (u) => M.l + ((u - lo) / (X.max - lo)) * pw;
    // and back: the x value under a point of the drawing
    const xAt = (px) => {
      const f = (px - M.l) / pw;
      return X.log2 ? 2 ** (l2[0] + f * (l2[1] - l2[0])) : X.log ? 10 ** (f * Math.log10(X.max)) : lo + f * (X.max - lo);
    };
    const fx = X.format || ((u) => u.toLocaleString("en"));
    const yv = Y.log ? (v) => (Math.log10(Y.max) - Math.log10(Math.max(Y.min, v))) / (Math.log10(Y.max) - Math.log10(Y.min)) : (v) => (Y.max - Math.min(Y.max, Math.max(Y.min, v))) / (Y.max - Y.min);
    const y = (v) => M.t + yv(v) * ph;
    const fy = Y.percent ? (v) => `${Math.round(v * 100)}%` : (v) => num(v, Y.digits);
    const svg = el("svg", { class: "fig plot", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.label || "" });
    let g = "";
    for (const v of Y.ticks) g += `<line class="gridline" x1="${M.l}" x2="${M.l + pw}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${fy(v)}</text>`;
    for (const u of X.ticks) g += `<text class="tick" x="${x(u)}" y="${M.t + ph + 20}" text-anchor="middle">${fx(u)}</text>`;
    for (const r of spec.refs || []) g += `<line class="ref-line" x1="${M.l}" x2="${M.l + pw}" y1="${y(r.value)}" y2="${y(r.value)}"/><text class="note" x="${M.l + pw - 4}" y="${y(r.value) - 6}" text-anchor="end">${r.label}</text>`;
    g += `<line class="axis" x1="${M.l}" x2="${M.l + pw}" y1="${M.t + ph}" y2="${M.t + ph}"/><line class="axis" x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${M.t + ph}"/>
      <text class="axis-name" x="${M.l + pw / 2}" y="${H - 6}" text-anchor="middle">${X.label}</text>
      <text class="axis-name" transform="translate(16 ${M.t + ph / 2}) rotate(-90)" text-anchor="middle">${Y.label}</text>`;
    g += spec.curves.map((c) => `<path class="curve${c.dash ? ` ${c.dash === true ? "dashed" : c.dash}` : ""}${c.light ? " light" : ""}" data-id="${c.id}"/><g class="pts${c.light ? " light" : ""}" data-id="${c.id}"></g><text class="curve-name" data-id="${c.id}" x="${M.l + pw + 8}">${c.name}</text>`).join("");
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
        let d = "";
        if (L.xs) {
          d = L.xs.map((u, i) => `${i ? "L" : "M"}${x(u).toFixed(1)} ${y(L.ys[i]).toFixed(1)}`).join("");
          if (c.marks !== false) svg.querySelector(`.pts[data-id="${c.id}"]`).innerHTML = L.xs.map((u, i) => `<circle cx="${x(u).toFixed(1)}" cy="${y(L.ys[i]).toFixed(1)}" r="3"/>`).join("");
          ends.push({ id: c.id, y: y(L.ys[L.ys.length - 1]) + 4, x: x(L.xs[L.xs.length - 1]) + 9 }); // points end anywhere: name by the last one
        } else {
          const n = L.length, step = Math.max(1, Math.floor(n / (pw * 1.5)));
          if (X.log && n > pw * 1.5) {
            // On a log axis, equal buckets would average the early points away: draw log-spaced points instead.
            let last = -1;
            for (let j = 0; j <= pw * 1.5; j++) {
              const i = Math.min(n - 1, Math.round(n ** (j / (pw * 1.5))) - 1);
              if (i === last) continue;
              last = i;
              d += `${d ? "L" : "M"}${x(from + i).toFixed(1)} ${y(L[i]).toFixed(1)}`;
            }
          } else {
            for (let i = 0; i < n; i += step) {
              let sum = 0, k = 0;
              for (let j = i; j < Math.min(n, i + step); j++) { sum += L[j]; k++; }
              d += `${d ? "L" : "M"}${x(from + i + (k - 1) / 2).toFixed(1)} ${y(sum / k).toFixed(1)}`;
            }
          }
          ends.push({ id: c.id, y: y(L[n - 1]) + 4 });
        }
        svg.querySelector(`.curve[data-id="${c.id}"]`).setAttribute("d", d);
      }
      // Names at the right end of their curves, nudged apart where they would overlap (only names that share an x).
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) {
        const prev = ends.slice(0, i).reverse().find((e) => Math.abs((e.x || 0) - (ends[i].x || 0)) < 60);
        if (prev) ends[i].y = Math.max(ends[i].y, prev.y + 15);
      }
      let floor = M.t + ph + 2; // and none below the axis, where the tick labels are
      for (let i = ends.length - 1; i >= 0; i--) {
        if (ends[i].x) continue;
        ends[i].y = Math.min(ends[i].y, floor);
        floor = ends[i].y - 15;
      }
      for (const e of ends) {
        const name = svg.querySelector(`.curve-name[data-id="${e.id}"]`);
        name.setAttribute("y", e.y);
        if (e.x) name.setAttribute("x", e.x);
      }
    }

    const cross = svg.querySelector(".cross"), hit = svg.querySelector(".hit");
    hit.addEventListener("pointermove", (ev) => {
      const shown = spec.curves.filter((c) => lines[c.id]);
      if (!shown.length) return;
      const box = svg.getBoundingClientRect(), px = ((ev.clientX - box.left) / box.width) * W;
      let at, rows;
      if (lines[shown[0].id].xs) { // points: the nearest x any curve has a point at
        const all = [...new Set(shown.flatMap((c) => lines[c.id].xs))];
        at = all.reduce((a, b) => (Math.abs(x(b) - px) < Math.abs(x(a) - px) ? b : a));
        rows = shown.filter((c) => lines[c.id].xs.includes(at)).map((c) => [lines[c.id].ys[lines[c.id].xs.indexOf(at)], c.name]);
      } else {
        const n = lines[shown[0].id].length;
        at = from + Math.max(0, Math.min(n - 1, Math.round(xAt(px) - from)));
        rows = shown.map((c) => [lines[c.id][at - from], c.name]);
      }
      cross.setAttribute("x1", x(at));
      cross.setAttribute("x2", x(at));
      cross.style.opacity = 1;
      RL.tip.show(ev.clientX, ev.clientY, `<div class="head">${spec.at ? spec.at(at) : `${X.label} ${fx(at)}`}</div>` +
        rows.map(([v, name]) => `<div class="row"><b>${fy(v)}</b><span>${name}</span></div>`).join(""));
    });
    hit.addEventListener("pointerleave", () => { cross.style.opacity = 0; RL.tip.hide(); });
    return { draw, status: (text) => { status.textContent = text; status.hidden = !text; }, svg, x, y };
  }

  // Average sample(seed) over many runs, a few at a time, redrawing as they come in. sample returns { id: values }.
  // smooth: a trailing moving average over that many points, applied to the averages.
  // spec can be a list: several plots, one under the other, each drawing the curves it names from the same runs.
  function average(host, spec, { runs, sample, smooth = 1, note }) {
    const specs = [].concat(spec);
    let plots = [];
    if (specs.length === 1) plots = [plot(host, specs[0])];
    else {
      host.replaceChildren();
      for (const s of specs) {
        const box = document.createElement("div");
        box.className = "fig-panel";
        host.appendChild(box);
        plots.push(plot(box, s));
      }
    }
    const p = { draw: (lines) => plots.forEach((q) => q.draw(lines)), status: (text) => plots.forEach((q, i) => q.status(i === plots.length - 1 ? text : "")) };
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
    return plots.length === 1 ? plots[0] : plots;
  }

  RL.fig = { whenVisible, plot, average, num, uid: () => ++uid };
})(globalThis.RL = globalThis.RL || {});
