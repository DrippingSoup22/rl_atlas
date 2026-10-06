/* Line chart of one number per unit (reward per episode, error after each sweep…) for one or more runs.
   The part already played is solid and the rest faint; a playhead marks "now" and a click jumps there.
   A soft band can show the spread of many runs (their middle half) behind a line; dashed lines mark the best that
   can be reached. A series may be shorter than the run (the bench keeps at most a few hundred points): it then
   spreads over the run's units.
   Long runs are drawn at screen resolution: each point of the line averages the units under one pixel. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const M = { l: 52, r: 18, t: 12, b: 28 };
  let uid = 0;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  // Trailing moving average, so each point only uses units already played.
  function smooth(values, w) {
    if (w <= 1) return values;
    const out = new Float64Array(values.length);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
      sum += values[i];
      if (i >= w) sum -= values[i - w];
      out[i] = sum / Math.min(i + 1, w);
    }
    return out;
  }
  // Round numbers for the axis: steps of 1, 2, 2.5 or 5 times a power of ten.
  function ticks(lo, hi, count = 5) {
    const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 2.2 ? 2 : f < 3 ? 2.5 : f < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  RL.ticks = ticks;
  // Ticks on a log axis: the powers of ten in range, with 2 and 5 times them when the range is short, and both ends.
  function logTicks(lo, hi) {
    const short = Math.log10(hi / lo) <= 2.5, out = new Set(short ? [lo, hi] : []), mults = short ? [1, 2, 5] : [1];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of mults) { const v = m * 10 ** e; if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.add(v); }
    return [...out].sort((a, b) => a - b);
  }
  const trim = (t) => (t.includes(".") ? t.replace(/0+$/, "").replace(/\.$/, "") : t); // 2.50 → 2.5, but 100 stays 100
  const num = (v) => {
    const a = Math.abs(v), d = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : a >= 0.01 ? 3 : 4;
    return (v < 0 ? "−" : "") + (a === 0 ? "0" : a < 1e-4 ? a.toExponential(0) : trim(a.toFixed(d)));
  };
  // Axis labels: the chart's own format, unless it rounds two ticks to the same text (−123.5 and −124.5 both "−124"
  // on a narrow range); then as many decimals as the ticks need, the same for all.
  const tickLabels = (vals, fmt) => {
    const out = vals.map(fmt);
    if (new Set(out).size === out.length) return out;
    const places = (v) => { for (let d = 0; d < 4; d++) if (Math.abs(v - +v.toFixed(d)) < 1e-9) return d; return 4; };
    const d = Math.max(0, ...vals.map(places));
    return vals.map((v) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d));
  };

  class LineChart {
    // label: what is plotted; percent: values are shares (0–1); log: a log scale for small positive numbers;
    // zero: the axis starts at 0; domain: fixed [low, high]; logX: a log scale for long runs; noun: [unit, units].
    constructor(host, { height = 190, label = "", percent = false, log = false, zero = false, domain = null, logX = false, noun = ["episode", "episodes"], onSeek } = {}) {
      Object.assign(this, { host, height, percent, log, zero, domain, logX, noun, onSeek });
      this.series = [];
      this.refs = [];
      this.e = 0;
      this.clip = `chart-clip-${++uid}`;
      this.svg = el("svg", { class: "linechart", role: "img", "aria-label": label }, host);
      this.ro = new ResizeObserver(() => this.render());
      this.ro.observe(host);
    }

    // series: [{ name, color, values, smooth, faint, units }] or bands [{ band: { lo, mid, hi }, color, units }]
    set(series, refs = []) {
      this.series = series.map((s) => (s.band ? { ...s, line: s.band.mid, values: s.band.mid } : { ...s, line: smooth(s.values, s.smooth || 1) }));
      this.refs = refs;
      this.render();
    }

    playhead(e) {
      this.e = e;
      this._place();
    }

    fmt(v) { return this.percent ? `${Math.round(v * 100)}%` : num(v); }

    _domain() {
      if (this.domain) return this.domain;
      if (this.percent) return [0, 1];
      let lo = Infinity, hi = -Infinity;
      for (const s of this.series) for (const L of s.band ? [s.band.lo, s.band.hi] : [s.line]) for (const v of L) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      for (const r of this.refs) { lo = Math.min(lo, r.value); hi = Math.max(hi, r.value); }
      if (!Number.isFinite(lo)) return this.log ? [1e-3, 1] : [0, 1];
      if (this.log) {
        lo = Math.max(lo, hi * 1e-6, 1e-9);
        // Round to 1, 2 or 5 times a power of ten, so the scale ends close to the data instead of a whole decade away.
        const round = (v, up) => {
          const mag = 10 ** Math.floor(Math.log10(v)), steps = [1, 2, 5, 10];
          return mag * (up ? steps.find((m) => m * mag >= v * (1 - 1e-9)) : [...steps].reverse().find((m) => m * mag <= v * (1 + 1e-9)) || 1);
        };
        hi = Math.max(hi, lo * 3);
        // Over many decades the axis shows powers of ten only, so it ends on one.
        if (Math.log10(hi / lo) > 2.5) return [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi))];
        return [round(lo, false), round(hi, true)];
      }
      if (this.zero) lo = Math.min(0, lo);
      if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
      const pad = (hi - lo) * 0.06, t = ticks(lo - (this.zero ? 0 : pad), hi + pad);
      return [Math.min(t[0], lo), Math.max(t[t.length - 1], hi)];
    }

    render() {
      const W = Math.max(280, this.host.clientWidth), H = this.height, svg = this.svg;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.setAttribute("width", W);
      svg.setAttribute("height", H);
      svg.replaceChildren();
      const n = Math.max(1, ...this.series.map((s) => s.units || s.values.length));
      const [lo, hi] = this._domain(), pw = W - M.l - M.r, ph = H - M.t - M.b;
      const lx = Math.log10(Math.max(2, n));
      const x = this.logX ? (u) => M.l + (Math.log10(Math.max(1, u)) / lx) * pw : (u) => M.l + (u / n) * pw;
      const unitAt = this.logX ? (px) => 10 ** (((px - M.l) / pw) * lx) : (px) => ((px - M.l) / pw) * n;
      // values beyond the axis, infinite ones included, are drawn at its ends; a missing value (NaN) at the bottom
      const yv = this.log ? (v) => (Math.log10(hi) - Math.log10(Math.min(hi, Math.max(lo, v)))) / (Math.log10(hi) - Math.log10(lo)) : (v) => (hi - Math.min(hi, Math.max(lo, v))) / (hi - lo);
      const y = (v) => M.t + (Number.isNaN(v) ? 1 : yv(v)) * ph;
      Object.assign(this, { n, x, y, unitAt, W, H, pw });

      // axes and grid
      const grid = el("g", { class: "grid" }, svg);
      const yt = this.log ? logTicks(lo, hi) : ticks(lo, hi, 4);
      const yl = tickLabels(yt, (v) => this.fmt(v));
      yt.forEach((v, i) => {
        el("line", { x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }, grid);
        el("text", { class: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end" }, grid).textContent = yl[i];
      });
      const xt = this.logX ? Array.from({ length: Math.floor(lx) + 1 }, (_, k) => 10 ** k) : ticks(0, n, W < 520 ? 2 : 5).filter((u) => u <= n); // fewer ticks on narrow charts, so labels never collide
      xt.forEach((u, k) => {
        const last = k === xt.length - 1;
        el("text", { class: "tick", x: x(u), y: H - 8, "text-anchor": k === 0 && !this.logX ? "start" : last ? "end" : "middle" }, grid)
          .textContent = last ? `${u.toLocaleString("en")} ${this.noun[1]}` : u.toLocaleString("en");
      });
      for (const r of this.refs) {
        el("line", { class: "ref", x1: M.l, x2: W - M.r, y1: y(r.value), y2: y(r.value) }, grid);
        if (r.label) el("text", { class: "ref-label", x: W - M.r - 4, y: y(r.value) - 5, "text-anchor": "end" }, grid).textContent = r.label;
      }

      // the lines, each point averaging the units under one pixel
      const defs = el("defs", {}, svg);
      this.clipRect = el("rect", { x: 0, y: 0, height: H, width: 0 }, el("clipPath", { id: this.clip }, defs));
      const buckets = Math.max(1, Math.floor(pw / 1.5));
      // a series shorter than the run: point i stands for the units it covers, drawn at their middle
      const ux = (s, i) => (s.line.length === n ? i + 1 : ((i + 0.5) * n) / s.line.length);
      for (const s of this.series.filter((b) => b.band)) { // bands first, under every line
        const { lo: L0, hi: L1, mid } = s.band, idx = [...L0.keys()].filter((i) => Number.isFinite(L0[i]) && Number.isFinite(L1[i]));
        if (idx.length < 2) continue;
        const top = idx.map((i) => `${x(ux(s, i)).toFixed(1)} ${y(L1[i]).toFixed(1)}`), bottom = idx.reverse().map((i) => `${x(ux(s, i)).toFixed(1)} ${y(L0[i]).toFixed(1)}`);
        el("path", { class: "band", d: `M${top.join("L")}L${bottom.join("L")}Z`, style: `fill: var(${s.color})` }, svg);
        el("path", { class: "band-mid", d: "M" + [...mid.keys()].filter((i) => Number.isFinite(mid[i])).map((i) => `${x(ux(s, i)).toFixed(1)} ${y(mid[i]).toFixed(1)}`).join("L"), style: `stroke: var(${s.color})` }, svg);
      }
      for (const s of this.series) {
        if (s.band) continue;
        const L = s.line, len = L.length;
        let d = "";
        if (len <= buckets * 2 || this.logX) {
          for (let i = 0; i < len; i++) d += `${d ? "L" : "M"}${x(ux(s, i)).toFixed(1)} ${y(L[i]).toFixed(1)}`;
        } else {
          for (let b = 0; b < buckets; b++) {
            const i0 = Math.floor((b * len) / buckets), i1 = Math.max(i0 + 1, Math.floor(((b + 1) * len) / buckets));
            let sum = 0;
            for (let i = i0; i < i1; i++) sum += L[i];
            d += `${d ? "L" : "M"}${x((i0 + i1) / 2).toFixed(1)} ${y(sum / (i1 - i0)).toFixed(1)}`;
          }
        }
        const style = `stroke: var(${s.color})`, cls = s.faint ? " faint" : "";
        el("path", { class: `future${cls}`, d, style }, svg);
        el("path", { class: `past${cls}`, d, style, "clip-path": `url(#${this.clip})` }, svg);
      }
      this.head = el("line", { class: "playhead", y1: M.t - 4, y2: M.t + ph }, svg);
      this.dots = this.series.map((s) => (s.faint || s.band ? null : el("circle", { class: "dot", r: 4.5, style: `fill: var(${s.color})` }, svg)));
      this.cross = el("line", { class: "cross", y1: M.t, y2: M.t + ph }, svg);
      const hit = el("rect", { class: "hit", x: M.l, y: 0, width: pw, height: H }, svg);
      hit.addEventListener("pointermove", (ev) => this._hover(ev));
      hit.addEventListener("pointerleave", () => { this.cross.style.opacity = 0; RL.tip.hide(); });
      hit.addEventListener("click", (ev) => this.onSeek?.(this._unitAt(ev) + 1));
      this._place();
    }

    // The unit (0-based) under the pointer.
    _unitAt(ev) {
      const box = this.svg.getBoundingClientRect(), px = ((ev.clientX - box.left) / box.width) * this.W;
      return Math.max(0, Math.min(this.n - 1, Math.round(this.unitAt(px)) - 1));
    }

    _place() {
      if (!this.head) return;
      const e = Math.min(this.e, this.n), px = e ? this.x(e) : M.l;
      this.clipRect.setAttribute("width", Math.max(0, px));
      this.head.setAttribute("x1", px);
      this.head.setAttribute("x2", px);
      this.series.forEach((s, k) => {
        const dot = this.dots[k];
        if (!dot) return;
        const v = s.line[Math.min(s.line.length - 1, Math.floor(((e - 1) * s.line.length) / this.n))];
        if (e < 1 || !Number.isFinite(v)) { dot.style.opacity = 0; return; }
        dot.style.opacity = 1;
        dot.setAttribute("cx", px);
        dot.setAttribute("cy", this.y(v));
      });
    }

    _hover(ev) {
      const i = this._unitAt(ev), px = this.x(i + 1);
      this.cross.setAttribute("x1", px);
      this.cross.setAttribute("x2", px);
      this.cross.style.opacity = 1;
      const at = (s) => Math.min(s.line.length - 1, Math.floor((i * s.line.length) / this.n));
      const rows = this.series.filter((s) => !s.faint).map((s) => {
        const j = at(s);
        if (s.band) return `<div class="row"><i class="key band-key" style="--k: var(${s.color})"></i><b>${this.fmt(s.band.lo[j])} to ${this.fmt(s.band.hi[j])}</b><span>${RL.esc(s.name)}</span></div>`;
        return `<div class="row"><i class="key" style="--k: var(${s.color})"></i><b>${this.fmt(s.line[j])}</b><span>${RL.esc(s.name)}</span>${(s.smooth || 1) > 1 && s.values.length === this.n ? `<span class="faint">this ${this.noun[0]} ${this.fmt(s.values[i])}</span>` : ""}</div>`;
      });
      RL.tip.show(ev.clientX, ev.clientY, `<div class="head">${this.noun[0][0].toUpperCase() + this.noun[0].slice(1)} ${(i + 1).toLocaleString("en")}</div>${rows.join("")}`);
    }

    destroy() {
      this.ro.disconnect();
      RL.tip.hide();
      this.svg.remove();
    }
  }

  RL.LineChart = LineChart;

  // The odds against one knob: a dot per value the knob was swept over (evenly spaced, the value the Lab is set to
  // shaded), joined by lines, one series per racer: the share of runs that succeeded, or their average score.
  class SweepChart {
    constructor(host, { height = 170, label = "", percent = false, log = false, values = [], fmtX = String, current = null } = {}) {
      Object.assign(this, { host, height, percent, log, values, fmtX, current });
      this.series = [];
      this.svg = el("svg", { class: "linechart sweep", role: "img", "aria-label": label }, host);
      this.ro = new ResizeObserver(() => this.render());
      this.ro.observe(host);
    }

    // series: [{ name, color, points: [a number or NaN per value], notes: [what a dot's tooltip says] }]
    set(series) {
      this.series = series;
      this.render();
    }

    fmt(v) { return this.percent ? `${Math.round(v * 100)}%` : num(v); }

    render() {
      const W = Math.max(280, this.host.clientWidth), H = this.height, svg = this.svg, k = this.values.length;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.setAttribute("width", W);
      svg.setAttribute("height", H);
      svg.replaceChildren();
      let lo = Infinity, hi = -Infinity;
      for (const s of this.series) for (const v of s.points) if (Number.isFinite(v) && (!this.log || v > 0)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      if (this.percent) [lo, hi] = [0, 1];
      else if (!Number.isFinite(lo)) [lo, hi] = this.log ? [1, 10] : [0, 1]; // no dots yet (a log axis cannot start at 0)
      else if (this.log) [lo, hi] = [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(Math.max(hi, lo * 1.01)))];
      else {
        if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
        const t = ticks(lo, hi, 4);
        [lo, hi] = [Math.min(t[0], lo), Math.max(t[t.length - 1], hi)];
      }
      const pw = W - M.l - M.r, ph = H - M.t - M.b, slot = pw / Math.max(1, k);
      const x = (i) => M.l + (i + 0.5) * slot;
      const yv = this.log ? (v) => (Math.log10(hi) - Math.log10(Math.min(hi, Math.max(lo, v)))) / (Math.log10(hi) - Math.log10(lo)) : (v) => (hi - Math.min(hi, Math.max(lo, v))) / (hi - lo);
      const y = (v) => M.t + yv(v) * ph;
      const grid = el("g", { class: "grid" }, svg);
      const at = this.values.indexOf(this.current);
      if (at >= 0) el("rect", { class: "now-band", x: M.l + at * slot, y: M.t, width: slot, height: ph }, grid);
      const yt = this.log ? logTicks(lo, hi) : ticks(lo, hi, 4), yl = tickLabels(yt, (v) => this.fmt(v));
      yt.forEach((v, i) => {
        el("line", { x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }, grid);
        el("text", { class: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end" }, grid).textContent = yl[i];
      });
      const every = slot < 34 ? 2 : 1; // crowded labels: every other one, always keeping the current value
      this.values.forEach((v, i) => {
        if (i % every && i !== at) return;
        el("text", { class: `tick${i === at ? " now" : ""}`, x: x(i), y: H - 8, "text-anchor": "middle" }, grid).textContent = this.fmtX(v);
      });
      for (const s of this.series) {
        const pts = s.points.map((v, i) => [i, v]).filter(([, v]) => Number.isFinite(v));
        if (pts.length > 1) el("polyline", { class: "past", points: pts.map(([i, v]) => `${x(i)},${y(v)}`).join(" "), style: `stroke: var(${s.color})` }, svg);
        for (const [i, v] of pts) {
          const dot = el("circle", { class: "dot", r: 4, cx: x(i), cy: y(v), style: `fill: var(${s.color})` }, svg);
          el("title", {}, dot).textContent = `${s.name}, ${this.fmtX(this.values[i])}: ${s.notes?.[i] ?? this.fmt(v)}`;
        }
      }
    }

    destroy() {
      this.ro.disconnect();
      this.svg.remove();
    }
  }
  RL.SweepChart = SweepChart;

  // How the runs of the bench end: per racer one row, a dot per seed at its final score. A dashed mark splits the
  // seeds that end well (filled) from the others (hollow); the seed playing above is ringed. A click on a dot plays
  // that seed. Dots that would overlap are stacked a little above and below the row.
  class SeedStrip {
    constructor(host, { onPick, label = "" } = {}) {
      Object.assign(this, { host, onPick });
      this.data = null;
      this.svg = el("svg", { class: "seedstrip", role: "img", "aria-label": label }, host);
      this.ro = new ResizeObserver(() => this.render());
      this.ro.observe(host);
    }

    // rows: [{ name, color, seeds, scores, ok, played: { seed, score } }]; threshold: a number or null; lower: less is
    // better; percent: scores are shares; fmt: a number's text.
    set(data) {
      this.data = data;
      this.render();
    }

    render() {
      const d = this.data, svg = this.svg;
      svg.replaceChildren();
      if (!d || !d.rows.length) return;
      const W = Math.max(280, this.host.clientWidth), ROW = d.rows.some((r) => r.scores.length > 40) ? 44 : 34, TOP = 22, H = TOP + d.rows.length * ROW + 24;
      const L = Math.min(150, Math.max(80, ...d.rows.map((r) => 8 + 7 * r.name.length))), R = 70;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.setAttribute("width", W);
      svg.setAttribute("height", H);
      const all = d.rows.flatMap((r) => [...r.scores, r.played?.score]).filter(Number.isFinite);
      if (Number.isFinite(d.threshold)) all.push(d.threshold);
      let lo = Math.min(...all), hi = Math.max(...all);
      if (d.percent) { lo = Math.min(lo, 0); hi = Math.max(hi, 1); }
      if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
      const t = ticks(lo, hi, W < 520 ? 3 : 5);
      lo = Math.min(lo, t[0]); hi = Math.max(hi, t[t.length - 1]);
      // the better end on the right, so "further right" always reads "better"
      const flip = d.lower, x = (v) => L + ((flip ? hi - v : v - lo) / (hi - lo)) * (W - L - R);
      const fmt = d.fmt || ((v) => (d.percent ? `${Math.round(v * 100)}%` : num(v)));
      const grid = el("g", { class: "grid" }, svg);
      t.filter((v) => x(v) < W - R - 22).forEach((v) => el("text", { class: "tick", x: x(v), y: H - 6, "text-anchor": "middle" }, grid).textContent = fmt(v));
      el("text", { class: "tick end", x: W - R + 6, y: H - 6 }, grid).textContent = "better →";
      if (Number.isFinite(d.threshold)) {
        const tx = x(d.threshold);
        el("line", { class: "goal", x1: tx, x2: tx, y1: TOP - 6, y2: H - 20 }, grid);
        el("text", { class: "goal-label", x: tx + 5, y: TOP - 8 }, grid).textContent = `ends well: ${d.lower ? "≤" : "≥"} ${fmt(d.threshold)}`;
      }
      d.rows.forEach((r, ri) => {
        const cy = TOP + ri * ROW + ROW / 2, g = el("g", { class: "row" }, svg), many = r.scores.length > 40, rad = many ? 2.6 : 4.2;
        const goal = Number.isFinite(d.threshold);
        el("line", { class: "axis", x1: L, x2: W - R, y1: cy, y2: cy }, g);
        el("text", { class: "name", x: L - 10, y: cy + 4, "text-anchor": "end" }, g).textContent = r.name;
        const wins = r.ok.filter(Boolean).length;
        el("text", { class: "count", x: W - R + 6, y: cy + 4 }, g).textContent = Number.isFinite(d.threshold) ? `${wins} of ${r.scores.length}` : `${r.scores.length} seeds`;
        if (r.scores.length > 60) { this._histogram(g, r, x, cy, ROW, goal, d, fmt); return; }
        // stack dots that would overlap: each pixel bin of a dot's width fills above and below the row in turn
        const bins = new Map(), pos = r.scores.map((v) => {
          if (!Number.isFinite(v)) return null;
          const px = x(v), b = Math.round(px / (rad * 2 + 1)), k = bins.get(b) || 0;
          bins.set(b, k + 1);
          const off = Math.min(ROW / 2 - rad - 1, Math.ceil(k / 2) * (rad * 1.6)) * (k % 2 ? -1 : 1);
          return [px, cy + off];
        });
        r.scores.forEach((v, i) => {
          if (!pos[i]) return;
          const ok = Number.isFinite(d.threshold) ? r.ok[i] : true;
          const dot = el("circle", { class: `seed-dot${ok ? "" : " miss"}`, r: rad, cx: pos[i][0], cy: pos[i][1], style: `--c: var(${r.color})` }, g);
          dot.addEventListener("pointerenter", (ev) => RL.tip.show(ev.clientX, ev.clientY, `<div class="head">Seed ${r.seeds[i]}</div><div class="row"><i class="key" style="--k: var(${r.color})"></i><b>${fmt(v)}</b><span>${RL.esc(r.name)}${Number.isFinite(d.threshold) ? (ok ? " · ends well" : " · does not") : ""}</span></div><div class="faint">Click to play this seed</div>`));
          dot.addEventListener("pointerleave", () => RL.tip.hide());
          dot.addEventListener("click", () => { RL.tip.hide(); this.onPick?.(r.seeds[i]); });
        });
        const p = r.played;
        if (p && Number.isFinite(p.score)) {
          const i = r.seeds.indexOf(p.seed), at = i >= 0 && pos[i] ? pos[i] : [x(p.score), cy];
          if (i < 0) el("circle", { class: "seed-dot own", r: rad, cx: at[0], cy: at[1], style: `--c: var(${r.color})` }, g);
          el("circle", { class: "seed-ring", r: rad + 4, cx: at[0], cy: at[1] }, g);
        }
      });
    }

    // Many seeds: a small histogram above the row instead of dots, filled where runs end well; a click plays the seed
    // whose ending is closest to where it lands.
    _histogram(g, r, x, cy, ROW, goal, d, fmt) {
      const x0 = x(d.lower ? Math.max(...r.scores.filter(Number.isFinite)) : Math.min(...r.scores.filter(Number.isFinite)));
      const W = this.svg.viewBox.baseVal.width, bw = 6, counts = new Map();
      r.scores.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        const b = Math.floor(x(v) / bw), c = counts.get(b) || { n: 0, ok: 0 };
        c.n++;
        if (r.ok[i]) c.ok++;
        counts.set(b, c);
      });
      const top = Math.max(...[...counts.values()].map((c) => c.n)), hgt = ROW - 10;
      for (const [b, c] of counts) {
        const h = Math.max(1.5, (c.n / top) * hgt), ok = goal ? c.ok >= c.n / 2 : true;
        el("rect", { class: `seed-bar${ok ? "" : " miss"}`, x: b * bw + 0.5, y: cy + ROW / 2 - 4 - h, width: bw - 1, height: h, rx: 1, style: `--c: var(${r.color})` }, g);
      }
      const hit = el("rect", { class: "hit", x: Math.min(x0, W) - 6, y: cy - ROW / 2, width: W, height: ROW }, g);
      const nearest = (ev) => {
        const box = this.svg.getBoundingClientRect(), px = ((ev.clientX - box.left) / box.width) * W;
        let best = -1, bd = Infinity;
        r.scores.forEach((v, i) => { const dd = Math.abs(x(v) - px); if (dd < bd) { bd = dd; best = i; } });
        return best;
      };
      hit.addEventListener("pointermove", (ev) => { const i = nearest(ev); if (i >= 0) RL.tip.show(ev.clientX, ev.clientY, `<div class="head">Seed ${r.seeds[i]}</div><div class="row"><i class="key" style="--k: var(${r.color})"></i><b>${fmt(r.scores[i])}</b><span>${RL.esc(r.name)}</span></div><div class="faint">Click to play this seed</div>`); });
      hit.addEventListener("pointerleave", () => RL.tip.hide());
      hit.addEventListener("click", (ev) => { const i = nearest(ev); RL.tip.hide(); if (i >= 0) this.onPick?.(r.seeds[i]); });
      const p = r.played;
      if (p && Number.isFinite(p.score)) {
        const px = x(p.score);
        el("line", { class: "seed-mark", x1: px, x2: px, y1: cy - ROW / 2 + 2, y2: cy + ROW / 2 - 2 }, g);
      }
    }

    destroy() {
      this.ro.disconnect();
      RL.tip.hide();
      this.svg.remove();
    }
  }
  RL.SeedStrip = SeedStrip;
})(globalThis.RL = globalThis.RL || {});
