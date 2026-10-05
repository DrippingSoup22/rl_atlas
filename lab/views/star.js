/* Star view: Baird's counterexample (Sutton & Barto, Example 11.1), shared by the Lab and the stories. Six upper
   states over one lower state; the solid action leads down to state 7, the dashed action up to one of the six at
   random. Each state shows its estimate, written as its features make it (2w₁ + w₈ …), and on the right the weights
   themselves, on a scale that grows by factors of ten so that a run that blows up stays readable. All true values
   are 0: any estimate that is not 0 is wrong. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 640, H = 260, UY = 66, LY = 204, X0 = 46, DX = 64, LX = 46 + 2.5 * 64, BX = 470, BW = 20, BY = 190, BH = 148;
  const SUB = "₀₁₂₃₄₅₆₇₈₉", sub = (n) => String(n).replace(/\d/g, (d) => SUB[d]);
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const fmt = (v) => {
    if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
    const a = Math.abs(v), s = v < 0 ? "−" : "";
    return a >= 1e5 ? `${s}${a.toExponential(1).replace("e+", "·10^")}` : `${s}${a >= 100 ? a.toFixed(0) : a.toFixed(a >= 10 ? 1 : 2)}`;
  };
  // A height that grows like the logarithm of a number's size, keeping its sign: 1 → 1 step, 10 → 2, 100 → 3, …
  const slog = (v) => Math.sign(v) * Math.log10(1 + Math.abs(Number.isFinite(v) ? v : 1e9));

  class StarView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { ...options };
      this.svg = el("svg", { class: "starview", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": env.title });
      host.appendChild(this.svg);
      const defs = el("defs", {}, this.svg), id = `sv-${Math.random().toString(36).slice(2, 7)}`;
      el("path", { d: "M0 0 L10 5 L0 10 z" }, el("marker", { id, class: "tip", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs));
      const tip = `url(#${id})`;
      this.pos = (s) => (s === env.LOW ? [LX, LY] : [X0 + s * DX, UY]);
      const edges = el("g", { class: "edges" }, this.svg);
      this.solid = [];
      for (let s = 0; s < 6; s++) {
        // from under the state's label, so the arrows never cross it
        const [x, y0] = this.pos(s), y = y0 + 42, dx = LX - x, dy = LY - y, len = Math.hypot(dx, dy);
        this.solid[s] = el("path", { class: "edge solid", d: `M${x + (dx / len) * 4} ${y + (dy / len) * 4}L${LX - (dx / len) * 25} ${LY - (dy / len) * 25}`, "marker-end": tip }, edges);
      }
      this.solid[6] = el("path", { class: "edge solid", d: `M${LX + 18} ${LY + 12}c34 22 34 -46 2 -32`, "marker-end": tip }, edges);
      // dashed: from anywhere up to the row of six (drawn once, as a bus along the top)
      this.dashed = el("path", { class: "edge dashed", d: `M${LX - 20} ${LY - 8}C${LX - 120} ${LY - 40} ${X0 - 30} ${UY + 70} ${X0 - 30} ${UY}V${UY - 34}H${X0 + 5 * DX}` }, edges);
      for (let s = 0; s < 6; s++) el("path", { class: "edge dashed", d: `M${X0 + s * DX} ${UY - 34}V${UY - 24}`, "marker-end": tip }, edges);
      el("text", { class: "edge-name", x: X0 - 34, y: UY - 40 }, this.svg).textContent = "dashed: to one of the six, at random";
      el("text", { class: "edge-name", x: LX + 50, y: LY + 30 }, this.svg).textContent = "solid: to 7";
      this.nodes = [];
      this.vals = [];
      this.forms = [];
      for (let s = 0; s < 7; s++) {
        const [x, y] = this.pos(s), g = el("g", { class: "node", transform: `translate(${x} ${y})` }, this.svg);
        el("circle", { r: 21 }, g);
        this.vals[s] = el("text", { class: "v", y: 4 }, g);
        this.forms[s] = el("text", { class: "form", y: 36 }, g);
        el("text", { class: "name", x: 15, y: -16 }, g).textContent = s + 1;
        this.nodes[s] = g;
      }
      // the weights, on a log scale either side of zero
      this.gW = el("g", { class: "weights" }, this.svg);
      el("text", { class: "w-title", x: BX - 8, y: 24 }, this.gW).textContent = "the weights";
      el("line", { class: "zero", x1: BX - 8, x2: W - 8, y1: BY, y2: BY }, this.gW);
      for (const v of [1, 10, 100, 1000]) {
        const y = BY - (slog(v) / 4) * BH;
        el("line", { class: "w-grid", x1: BX - 8, x2: W - 8, y1: y, y2: y }, this.gW);
        el("text", { class: "w-tick", x: BX - 12, y: y + 3, "text-anchor": "end" }, this.gW).textContent = v;
      }
      this.bars = [];
      this.wNums = [];
      this.bot = el("circle", { class: "bot", r: 8 }, this.svg);
      this.gFx = el("g", { class: "fx" }, this.svg);
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
      this.place(0, true);
    }

    setOptions(o) { Object.assign(this.o, o); this.bot.classList.toggle("gone", this.o.agent === false); }

    // The bars for n weights, made again when the features change (8 for Baird's, 7 for a table).
    _bars(n, own) {
      if (this.bars.length === n) return;
      this.bars.forEach((b) => b.remove());
      this.wNums.forEach((b) => b.remove());
      const step = (W - 16 - BX) / n;
      this.bars = Array.from({ length: n }, (_, i) => el("rect", { class: "w-bar", x: BX + i * step + 2, width: Math.min(BW, step - 4), rx: 2 }, this.gW));
      this.wNums = Array.from({ length: n }, (_, i) => el("text", { class: "w-name", x: BX + i * step + 2 + Math.min(BW, step - 4) / 2, y: H - 6 }, this.gW));
      this.wNums.forEach((t, i) => { t.textContent = `w${sub(i + 1)}`; });
      for (let s = 0; s < 7; s++) this.forms[s].textContent = own ? (s === this.env.LOW ? "w₇ + 2w₈" : `2w${sub(s + 1)} + w₈`) : `w${sub(s + 1)}`;
    }

    show(d) {
      if (!d.w) return;
      this.w = Float64Array.from(d.w);
      this.V = Float64Array.from(d.V);
      this._bars(d.w.length, d.F?.kind === "own");
      d.w.forEach((v, i) => {
        const h = Math.min(v >= 0 ? 1.1 : 0.3, Math.abs(slog(v)) / 4) * BH; // little room below zero: weights rarely go far negative
        this.bars[i].setAttribute("y", v >= 0 ? BY - h : BY);
        this.bars[i].setAttribute("height", Math.max(1, h));
        this.bars[i].classList.toggle("neg", v < 0);
      });
      for (let s = 0; s < 7; s++) {
        this.vals[s].textContent = fmt(d.V[s]);
        this.nodes[s].style.setProperty("--glow", Math.min(1, Math.abs(slog(d.V[s])) / 3).toFixed(3));
      }
    }

    place(s, instant = false) {
      const [x, y] = this.pos(s);
      if (instant) this.bot.classList.add("instant");
      this.bot.style.transform = `translate(${x + 20}px, ${y + 16}px)`;
      if (instant) { this.bot.getBoundingClientRect(); this.bot.classList.remove("instant"); }
    }

    flash(edge) {
      if (!edge) return;
      edge.classList.remove("chosen");
      edge.getBoundingClientRect();
      edge.classList.add("chosen");
    }

    spark(s, idx) {
      const [x, y] = this.pos(s), c = el("circle", { class: "spark", cx: x, cy: y, r: 22 }, this.gFx);
      c.animate([{ opacity: 1, transform: "scale(0.7)" }, { opacity: 0, transform: "scale(1.4)" }], { duration: 600, easing: "ease-out" }).onfinish = () => c.remove();
      for (const i of idx || []) {
        const b = this.bars[i];
        if (!b) continue;
        b.classList.remove("sparked");
        b.getBoundingClientRect();
        b.classList.add("sparked");
      }
    }

    event(ev) {
      switch (ev.type) {
        case "start": this.place(ev.s, true); return 0;
        case "choose": this.flash(ev.a === this.env.SOLID ? this.solid[ev.s] : this.dashed); return 0;
        case "move": this.place(ev.s2); return 0;
        case "update": this.spark(ev.s, ev.rho ? Array.from(ev.x.idx.subarray(0, ev.x.k)) : []); return 0;
        default: return 0;
      }
    }

    rest(events) {
      const moves = events.filter((e) => e.type === "move"), start = events.find((e) => e.type === "start");
      this.place(moves.length ? moves[moves.length - 1].s2 : start ? start.s : 0, true);
    }

    _hover(e) {
      if (!this.V) return;
      const box = this.svg.getBoundingClientRect(), px = ((e.clientX - box.left) / box.width) * W, py = ((e.clientY - box.top) / box.height) * H;
      const s = [0, 1, 2, 3, 4, 5, 6].find((k) => { const [x, y] = this.pos(k); return Math.hypot(px - x, py - y) < 26; });
      if (s === undefined) { RL.tip.hide(); return; }
      RL.tip.show(e.clientX, e.clientY, `<div class="head">State ${s + 1}</div>
        <div class="row"><b>${fmt(this.V[s])}</b><span>estimate v̂ = ${this.forms[s].textContent}</span></div>
        <div class="row"><b>0</b><span>true value: nothing ever pays</span></div>`);
    }

    destroy() { RL.tip.hide(); this.svg.remove(); }

    static thumb(env, d) {
      const n = d.w.length, w = 200, h = 64, step = w / n, mid = h / 2;
      return `<svg class="thumb-star" viewBox="0 0 ${w} ${h}"><line class="t-zero" x1="0" x2="${w}" y1="${mid}" y2="${mid}"/>${Array.from(d.w, (v, i) => {
        const bh = Math.min(1, Math.abs(slog(v)) / 4) * (mid - 3);
        return `<rect class="t-bar${v < 0 ? " neg" : ""}" x="${i * step + 3}" width="${step - 6}" y="${v >= 0 ? mid - bh : mid}" height="${Math.max(1, bh)}"/>`;
      }).join("")}</svg>`;
    }
  }

  RL.StarView = StarView;
  (RL.labViews = RL.labViews || {}).star = StarView;
})(globalThis.RL = globalThis.RL || {});
