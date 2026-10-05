/* Graph view: the maximization-bias example (Sutton & Barto, Example 6.7) drawn as its states. From A, right ends
   the episode; left leads to B, whose many actions all end it with a small random loss. The two estimates out of A
   sit on their edges, and B's estimates are a row of bars: the tallest one is what Q-learning's target trusts. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 620, H = 260, Y = 168, POS = { endL: 46, B: 220, A: 420, endR: 574 }, BAR = 34, BW = 13;
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const fmt = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);

  class GraphView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { ...options };
      this.nB = env.acts(1).length;
      this.Q = new Float64Array(env.nS * env.nA);
      this.svg = el("svg", { class: "graphview", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": env.title });
      host.appendChild(this.svg);
      const defs = el("defs", {}, this.svg), id = `gv-tip-${Math.random().toString(36).slice(2, 7)}`;
      el("path", { d: "M0 0 L10 5 L0 10 z" }, el("marker", { id, class: "tip", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs));
      const tip = `url(#${id})`;
      // B's actions fan out to the left exit
      this.fan = el("g", { class: "fan" }, this.svg);
      for (let i = 0; i < this.nB; i++) {
        const yy = Y - 46 + (92 * i) / (this.nB - 1);
        el("path", { class: "edge b-edge", d: `M${POS.B - 22} ${Y + (yy - Y) * 0.25} Q${(POS.B + POS.endL) / 2} ${yy} ${POS.endL + 24} ${Y + (yy - Y) * 0.2}`, "marker-end": tip }, this.fan);
      }
      el("text", { class: "edge-name", x: (POS.B + POS.endL) / 2, y: Y + 68, "text-anchor": "middle" }, this.svg).textContent = `${this.nB} actions, each N(−0.1, 1)`;
      this.left = el("path", { class: "edge a-edge", d: `M${POS.A - 24} ${Y} H${POS.B + 26}`, "marker-end": tip }, this.svg);
      this.right = el("path", { class: "edge a-edge", d: `M${POS.A + 24} ${Y} H${POS.endR - 26}`, "marker-end": tip }, this.svg);
      el("text", { class: "edge-name", x: (POS.A + POS.B) / 2, y: Y - 10, "text-anchor": "middle" }, this.svg).textContent = "left · 0";
      el("text", { class: "edge-name", x: (POS.A + POS.endR) / 2, y: Y - 10, "text-anchor": "middle" }, this.svg).textContent = "right · 0";
      this.qLeft = el("text", { class: "q-label", x: (POS.A + POS.B) / 2, y: Y + 46, "text-anchor": "middle" }, this.svg);
      this.qRight = el("text", { class: "q-label", x: (POS.A + POS.endR) / 2, y: Y + 46, "text-anchor": "middle" }, this.svg);
      const node = (x, name, cls) => {
        const g = el("g", { class: `node ${cls}`, transform: `translate(${x} ${Y})` }, this.svg);
        if (cls === "exit") el("rect", { x: -22, y: -18, width: 44, height: 36, rx: 8 }, g);
        else el("circle", { r: 22 }, g);
        el("text", { y: 5 }, g).textContent = name;
        return g;
      };
      node(POS.endL, "end", "exit");
      node(POS.endR, "end", "exit");
      node(POS.B, "B", "state");
      node(POS.A, "A", "state start");
      // B's estimates as bars above it
      this.bars = el("g", { class: "bbars", transform: `translate(${POS.B - (this.nB * BW) / 2} 62)` }, this.svg);
      el("line", { class: "zero", x1: -4, x2: this.nB * BW, y1: 0, y2: 0 }, this.bars);
      el("text", { class: "bars-name", x: -10, y: 4, "text-anchor": "end" }, this.bars).textContent = "Q(B, ·)";
      this.barEls = Array.from({ length: this.nB }, (_, i) => el("rect", { class: "bbar", x: i * BW + 1.5, width: BW - 3, rx: 2 }, this.bars));
      this.maxLabel = el("text", { class: "max-label", "text-anchor": "start" }, this.bars);
      this.bot = el("g", { class: "bot" }, this.svg);
      el("circle", { class: "bot-head", r: 11 }, this.bot);
      el("circle", { class: "bot-eye", cx: -4, cy: -2, r: 2.2 }, this.bot);
      el("circle", { class: "bot-eye", cx: 4, cy: -2, r: 2.2 }, this.bot);
      this.gFx = el("g", { class: "fx" }, this.svg);
      this.place(POS.A, true);
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
    }

    setOptions(o) { Object.assign(this.o, o); }

    show(d) {
      this.Q.set(d.Q);
      const nA = this.env.nA, Q = this.Q, qL = Q[0], qR = Q[1];
      this.qLeft.textContent = `Q(A, left) = ${fmt(qL)}`;
      this.qRight.textContent = `Q(A, right) = ${fmt(qR)}`;
      this.left.classList.toggle("greedy", qL > qR);
      this.right.classList.toggle("greedy", qR > qL);
      const bs = this.env.acts(1).map((a) => Q[nA + a]), top = Math.max(...bs);
      bs.forEach((v, i) => {
        const h = Math.min(BAR, Math.abs(v) * BAR * 2);
        const r = this.barEls[i];
        r.setAttribute("y", v >= 0 ? -h : 0);
        r.setAttribute("height", Math.max(0.5, h));
        r.classList.toggle("neg", v < 0);
        r.classList.toggle("top", v === top && top !== 0);
      });
      const at = bs.indexOf(top);
      this.maxLabel.setAttribute("x", this.nB * BW + 8);
      this.maxLabel.setAttribute("y", 4);
      this.maxLabel.textContent = `max ${fmt(top)}`;
      this.maxLabel.dataset.at = at;
    }

    place(x, instant = false) {
      if (instant) this.bot.classList.add("instant");
      this.bot.style.transform = `translate(${x}px, ${Y - 36}px)`;
      if (instant) { this.bot.getBoundingClientRect(); this.bot.classList.remove("instant"); }
    }

    pop(x, text, kind = "rew") {
      const t = el("text", { class: `pop k-${kind}`, x, y: Y - 52 }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-18px)" }], { duration: 950, easing: "ease-out" }).onfinish = () => t.remove();
    }

    event(ev) {
      const { env } = this;
      if (ev.type === "start") { this.place(POS.A, true); return 0; }
      if (ev.type === "choose" && ev.s === 0) {
        const e = ev.a === 0 ? this.left : this.right;
        e.classList.remove("chosen");
        e.getBoundingClientRect();
        e.classList.add("chosen");
        return 0;
      }
      if (ev.type === "move") {
        const x = env.terminal(ev.s2) ? (ev.s === 0 ? POS.endR : POS.endL) : POS.B;
        this.place(x);
        if (env.terminal(ev.s2)) { this.pop(x, (ev.r >= 0 ? "+" : "−") + Math.abs(ev.r).toFixed(2)); return 350; }
        return 150;
      }
      if (ev.type === "update") {
        const label = ev.s === 0 ? (ev.a === 0 ? this.qLeft : this.qRight) : this.barEls[ev.a - 2];
        label?.classList.remove("sparked");
        label?.getBoundingClientRect();
        label?.classList.add("sparked");
      }
      return 0;
    }

    rest(events) {
      const moves = events.filter((e) => e.type === "move"), last = moves[moves.length - 1];
      if (!last) { this.place(POS.A, true); return; }
      this.place(this.env.terminal(last.s2) ? (last.s === 0 ? POS.endR : POS.endL) : POS.B, true);
    }

    _hover(e) {
      const r = e.target.closest?.(".bbar");
      if (!r) { RL.tip.hide(); return; }
      const i = this.barEls.indexOf(r), v = this.Q[this.env.nA + 2 + i];
      RL.tip.show(e.clientX, e.clientY, `<div class="head">From B, action ${i + 1}</div><div class="row"><b>${fmt(v, 3)}</b><span>estimate Q(B, b${i + 1})</span></div><div class="row"><b>−0.100</b><span>true value</span></div>`);
    }

    destroy() { RL.tip.hide(); this.svg.remove(); }
  }

  RL.GraphView = GraphView;
  (RL.labViews = RL.labViews || {}).graph = GraphView;
})(globalThis.RL = globalThis.RL || {});
