/* Line chart of one number per unit (reward per episode, error after each sweep…) for one or more runs.
   The part already played is solid and the rest faint; a playhead marks "now" and a click jumps there.
   Faint thin lines show single runs behind their average; dashed lines mark the best that can be reached.
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
  const num = (v) => {
    const a = Math.abs(v), d = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : a >= 0.01 ? 3 : 4;
    return (v < 0 ? "−" : "") + (a === 0 ? "0" : a < 1e-4 ? a.toExponential(0) : a.toFixed(d).replace(/\.?0+$/, ""));
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

    set(series, refs = []) {
      this.series = series.map((s) => ({ ...s, line: smooth(s.values, s.smooth || 1) }));
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
      for (const s of this.series) for (const v of s.line) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      for (const r of this.refs) { lo = Math.min(lo, r.value); hi = Math.max(hi, r.value); }
      if (!Number.isFinite(lo)) return this.log ? [1e-3, 1] : [0, 1];
      if (this.log) {
        lo = Math.max(lo, hi * 1e-6, 1e-9);
        return [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(Math.max(hi, lo * 10)))];
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
      const n = Math.max(1, ...this.series.map((s) => s.values.length));
      const [lo, hi] = this._domain(), pw = W - M.l - M.r, ph = H - M.t - M.b;
      const lx = Math.log10(Math.max(2, n));
      const x = this.logX ? (u) => M.l + (Math.log10(Math.max(1, u)) / lx) * pw : (u) => M.l + (u / n) * pw;
      const unitAt = this.logX ? (px) => 10 ** (((px - M.l) / pw) * lx) : (px) => ((px - M.l) / pw) * n;
      const yv = this.log ? (v) => (Math.log10(hi) - Math.log10(Math.max(lo, v))) / (Math.log10(hi) - Math.log10(lo)) : (v) => (hi - Math.min(hi, Math.max(lo, v))) / (hi - lo);
      const y = (v) => M.t + yv(v) * ph;
      Object.assign(this, { n, x, y, unitAt, W, H, pw });

      // axes and grid
      const grid = el("g", { class: "grid" }, svg);
      const yt = this.log ? Array.from({ length: Math.round(Math.log10(hi / lo)) + 1 }, (_, k) => lo * 10 ** k) : ticks(lo, hi, 4);
      for (const v of yt) {
        el("line", { x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }, grid);
        el("text", { class: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end" }, grid).textContent = this.fmt(v);
      }
      const xt = this.logX ? Array.from({ length: Math.floor(lx) + 1 }, (_, k) => 10 ** k) : ticks(0, n, 5).filter((u) => u <= n);
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
      for (const s of this.series) {
        const L = s.line, len = L.length;
        let d = "";
        if (len <= buckets * 2 || this.logX) {
          for (let i = 0; i < len; i++) d += `${d ? "L" : "M"}${x(i + 1).toFixed(1)} ${y(L[i]).toFixed(1)}`;
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
      this.dots = this.series.map((s) => (s.faint ? null : el("circle", { class: "dot", r: 4.5, style: `fill: var(${s.color})` }, svg)));
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
        if (e < 1 || !Number.isFinite(s.line[e - 1])) { dot.style.opacity = 0; return; }
        dot.style.opacity = 1;
        dot.setAttribute("cx", px);
        dot.setAttribute("cy", this.y(s.line[e - 1]));
      });
    }

    _hover(ev) {
      const i = this._unitAt(ev), px = this.x(i + 1);
      this.cross.setAttribute("x1", px);
      this.cross.setAttribute("x2", px);
      this.cross.style.opacity = 1;
      const rows = this.series.filter((s) => !s.faint).map((s) => `<div class="row"><i class="key" style="--k: var(${s.color})"></i><b>${this.fmt(s.line[i])}</b><span>${RL.esc(s.name)}</span>${(s.smooth || 1) > 1 ? `<span class="faint">this ${this.noun[0]} ${this.fmt(s.values[i])}</span>` : ""}</div>`);
      RL.tip.show(ev.clientX, ev.clientY, `<div class="head">${this.noun[0][0].toUpperCase() + this.noun[0].slice(1)} ${(i + 1).toLocaleString("en")}</div>${rows.join("")}`);
    }

    destroy() {
      this.ro.disconnect();
      RL.tip.hide();
      this.svg.remove();
    }
  }

  RL.LineChart = LineChart;
})(globalThis.RL = globalThis.RL || {});
