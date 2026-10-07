/* Catch view: the board, the falling ball and the paddle. A table for Catch has a row for every ball position and
   paddle column; the board shows the 50 that go with where the paddle is now: for each place the ball could be, its
   value as a color and the move the learner thinks best as an arrow. As the paddle moves, the board switches to the
   values that go with its new column. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const T = 40, PAD = 8, CAP = 28;
  const MARK = ["←", "·", "→"];
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function valueColor(v, range) {
    const t = Math.max(-1, Math.min(1, v / range)), pct = Math.round(Math.abs(t) * 100);
    return `color-mix(in oklab, var(${t < 0 ? "--v-neg" : "--v-pos"}) ${pct}%, var(--v-mid))`;
  }
  const signed = (v, d = 2) => `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(v).toFixed(d)}`;
  const cx = (c) => c * T + T / 2, cy = (r) => r * T + T / 2;

  class CatchView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { arrows: true, numbers: false, ...options };
      const W = env.cols * T, H = env.rows * T;
      this.W = W; this.H = H;
      this.Q = new Float64Array(env.nS * env.nA);
      this.px = env.cols >> 1;
      this.svg = el("svg", { class: "catchview", viewBox: `${-PAD} ${-PAD} ${W + 2 * PAD} ${H + CAP + 2 * PAD}`, role: "img", "aria-label": env.title });
      this.svg.style.maxWidth = "260px";
      host.appendChild(this.svg);
      const g = (cls) => el("g", { class: cls }, this.svg);
      this.gTiles = g("tiles"); this.gArrows = g("arrows"); this.gNums = g("nums"); this.gPieces = g("pieces"); this.gFx = g("fx");
      this.tiles = []; this.marks = []; this.nums = [];
      // the bottom row is where the ball lands: its tiles stay plain, the paddle moves along it
      for (let r = 0; r < env.rows - 1; r++) for (let c = 0; c < env.cols; c++) {
        this.tiles.push(el("rect", { class: "tile", x: c * T + 1, y: r * T + 1, width: T - 2, height: T - 2, rx: 5 }, this.gTiles));
        this.marks.push(el("text", { class: "best", x: cx(c), y: cy(r) + 6 }, this.gArrows));
        this.nums.push(el("text", { class: "num", x: cx(c), y: r * T + T - 5 }, this.gNums));
      }
      el("rect", { class: "floor", x: 0, y: (env.rows - 1) * T, width: W, height: T, rx: 5 }, this.gTiles);
      el("rect", { class: "edge", x: 0, y: 0, width: W, height: H, rx: 4 }, this.gTiles);
      this.trail = el("line", { class: "fall" }, this.gPieces);
      this.ball = el("circle", { class: "ball", r: 11 }, this.gPieces);
      this.paddle = el("rect", { class: "paddle", x: -T / 2 + 3, y: -7, width: T - 6, height: 12, rx: 5 }, this.gPieces);
      this.caption = el("text", { class: "trip", x: W / 2, y: H + 20 }, this.svg);
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
      this.setOptions({});
      this.place(env.encode(0, 2, 2), true);
    }

    setOptions(o) {
      Object.assign(this.o, o);
      this.gArrows.style.display = this.o.arrows ? "" : "none";
      this.gNums.style.display = this.o.numbers && !this.noValue ? "" : "none";
      this.gPieces.style.display = this.o.agent === false ? "none" : "";
      this._draw();
    }

    // What the learner knows: its action values, or its state values and policy.
    show(d) {
      if (d.Q) this.Q = d.Q;
      this.V = d.V || null;
      this.P = d.P || null;
      this.noValue = !d.Q && !d.V; // a policy with no critic (REINFORCE): no values to write
      this.gNums.style.display = this.o.numbers && !this.noValue ? "" : "none";
      this._draw();
    }
    value(s) {
      if (this.V) return this.V[s];
      const nA = this.env.nA;
      let best = -Infinity;
      for (let a = 0; a < nA; a++) best = Math.max(best, this.Q[s * nA + a]);
      return best;
    }
    best(s) {
      const nA = this.env.nA, src = this.P || (this.V ? null : this.Q);
      if (!src) return -1;
      let b = 0;
      for (let a = 1; a < nA; a++) if (src[s * nA + a] > src[s * nA + b] + 1e-12) b = a;
      const flat = !this.P && this.Q.subarray(s * nA, s * nA + nA).every((q) => q === this.Q[s * nA]);
      return flat ? -1 : b;
    }

    // The 45 places the ball can fall through, with the paddle where it is.
    _draw() {
      const env = this.env;
      for (let r = 0; r < env.rows - 1; r++) for (let c = 0; c < env.cols; c++) {
        const i = r * env.cols + c, s = env.encode(r, c, this.px), v = this.value(s), b = this.best(s);
        this.tiles[i].style.fill = valueColor(v, env.valueRange);
        this.marks[i].textContent = b < 0 ? "" : MARK[b];
        this.nums[i].textContent = signed(v);
      }
      // the tiles are the values of each place of the ball with the paddle where it is (said in the "Show" panel)
      this.caption.textContent = this.end ? (this.end > 0 ? "Caught: +1" : "Missed: −1") : `Paddle in column ${this.px + 1}`;
    }

    place(s, instant = false) {
      const d = this.env.decode(s), bottom = (this.env.rows - 1) * T;
      this.svg.classList.toggle("instant", instant);
      this.ball.style.transform = `translate(${cx(d.bx)}px, ${cy(d.by)}px)`;
      this.paddle.style.transform = `translate(${cx(d.px)}px, ${bottom + T / 2 + 6}px)`;
      this.trail.setAttribute("x1", cx(d.bx)); this.trail.setAttribute("x2", cx(d.bx));
      this.trail.setAttribute("y1", 4); this.trail.setAttribute("y2", Math.max(4, cy(d.by) - 12));
      if (instant) this.svg.getBoundingClientRect();
      this.svg.classList.remove("instant");
      const end = this.env.terminal(s) ? (this.env.caught(s) ? 1 : -1) : 0;
      if (d.px !== this.px || end !== this.end) { this.px = d.px; this.end = end; this._draw(); }
      this.at = s;
    }

    pop(s, text, kind) {
      const d = this.env.decode(s), t = el("text", { class: `pop k-${kind}`, x: cx(d.bx), y: cy(d.by) - 18 }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-16px)" }], { duration: 950, easing: "ease-out" }).onfinish = () => t.remove();
    }
    spark(s) {
      const d = this.env.decode(s), c = el("circle", { class: "spark", cx: cx(d.bx), cy: cy(d.by), r: 13 }, this.gFx);
      c.animate([{ opacity: 1, transform: "scale(0.3)" }, { opacity: 0, transform: "scale(1.6)" }], { duration: 600, easing: "ease-out" }).onfinish = () => c.remove();
    }

    event(ev, { line = false } = {}) {
      if (ev.w) return 0; // several workers: the view follows the first
      switch (ev.type) {
        case "start": this.place(ev.s, true); return 0;
        case "move":
          this.place(ev.s2);
          if (this.env.terminal(ev.s2)) { this.pop(ev.s2, ev.r > 0 ? "caught! +1" : "missed −1", ev.r > 0 ? "rew" : "err"); return 450; }
          return 0;
        case "update": if (ev.s !== undefined && ev.s >= 0) this.spark(ev.s); return line ? 120 : 0;
        default: return 0;
      }
    }

    rest(events) {
      const moves = events.filter((e) => e.type === "move" && !e.w), first = events.find((e) => e.type === "start" && !e.w);
      if (moves.length) this.place(moves[moves.length - 1].s2, true);
      else if (first) this.place(first.s, true);
    }

    _hover(e) {
      const box = this.svg.getBoundingClientRect(), scale = (this.W + 2 * PAD) / box.width;
      const x = (e.clientX - box.left) * scale - PAD, y = (e.clientY - box.top) * scale - PAD;
      const r = Math.floor(y / T), c = Math.floor(x / T);
      if (r < 0 || c < 0 || r >= this.env.rows - 1 || c >= this.env.cols) { RL.tip.hide(); return; }
      const s = this.env.encode(r, c, this.px), nA = this.env.nA;
      const rows = this.noValue && this.P ? this.env.actionNames.map((name, a) => `<div class="row"><b>${Math.round(100 * this.P[s * nA + a])}%</b><span>${name}</span></div>`).join("") : this.V ? `<div class="row"><b>${signed(this.V[s])}</b><span>value</span></div>` : this.env.actionNames.map((name, a) => `<div class="row"><b>${signed(this.Q[s * nA + a])}</b><span>${name}</span></div>`).join("");
      RL.tip.show(e.clientX, e.clientY, `<div class="head">Ball at row ${r + 1}, column ${c + 1}; paddle in column ${this.px + 1}</div>${rows}`);
    }

    destroy() { RL.tip.hide(); this.svg.remove(); }

    static options() {
      return [
        { key: "arrows", type: "check", label: "The best move for each place of the ball (with the paddle where it is)", value: true },
        { key: "numbers", type: "check", label: "Values as numbers", value: false },
      ];
    }

    static thumb(env, d) {
      const c = 10, cells = [], nA = env.nA;
      const val = (s) => (d.V ? d.V[s] : d.Q ? Math.max(...Array.from({ length: nA }, (_, a) => d.Q[s * nA + a])) : 0);
      for (let r = 0; r < env.rows - 1; r++) for (let k = 0; k < env.cols; k++) cells.push(`<rect x="${k * c}" y="${r * c}" width="${c - 1}" height="${c - 1}" rx="2" style="fill:${valueColor(val(env.encode(r, k, env.cols >> 1)), env.valueRange)}"/>`);
      return `<svg class="thumb-taxi" viewBox="0 0 ${env.cols * c} ${(env.rows - 1) * c}" style="max-height:110px">${cells.join("")}</svg>`;
    }
  }

  RL.CatchView = CatchView;
  (RL.labViews = RL.labViews || {}).catch = CatchView;
})(globalThis.RL = globalThis.RL || {});
