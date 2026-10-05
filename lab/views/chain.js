/* Chain view: the random walk, shared by the Lab and the stories. Below, the row of states between two exits and the
   agent walking it; above, each state's estimated value (a dot) against its true value (a dashed step), as in
   Sutton & Barto's Figure 6.2. Ghost lines can keep earlier estimates in view. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W0 = 620, PT = 26, PH = 132, CY = 214, H = 250;
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const fmt = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  const HOP = [{ transform: "translateY(0)" }, { transform: "translateY(-9px)" }, { transform: "translateY(0)" }];

  class ChainView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { truth: true, numbers: false, ...options };
      const n = env.n;
      this.lo = Math.min(0, env.exits.left);
      this.hi = 1;
      // Long chains get a wider drawing, and smaller states, so they never touch.
      this.W = Math.max(W0, 170 + 34 * (n - 1));
      this.dx = Math.min(84, (this.W - 160) / Math.max(1, n - 1));
      const R = Math.min(17, this.dx * 0.42), k = +(R / 17).toFixed(3);
      this.V = new Float64Array(env.nS);
      this.svg = el("svg", { class: "chainview", viewBox: `0 0 ${this.W} ${H}`, role: "img", "aria-label": env.title });
      if (this.W > W0) this.svg.style.maxWidth = `${Math.round(this.W * 1.25)}px`;
      host.appendChild(this.svg);
      const g = (cls) => el("g", { class: cls }, this.svg);
      const axis = g("axis");
      for (const v of this.lo < 0 ? [-1, 0, 1] : [0, 0.5, 1]) {
        el("line", { class: "grid", x1: this.x(1) - 34, x2: this.x(n) + 34, y1: this.y(v), y2: this.y(v) }, axis);
        el("text", { class: "tick", x: this.x(1) - 42, y: this.y(v) + 4, "text-anchor": "end" }, axis).textContent = v;
      }
      el("text", { class: "axis-name", x: this.x(1) - 42, y: PT - 10, "text-anchor": "start" }, axis).textContent = "value";
      this.gGhosts = g("ghosts");
      const truth = env.truth();
      this.truthEl = el("path", { class: "truth", d: this._steps(truth) }, g("truth-layer"));
      this.line = el("path", { class: "est-line" }, this.svg);
      this.dots = [];
      this.nums = [];
      for (let s = 1; s <= n; s++) {
        this.dots[s] = el("circle", { class: "est", cx: this.x(s), r: 6 }, this.svg);
        this.nums[s] = el("text", { class: "est-num", x: this.x(s) }, this.svg);
      }
      // the chain itself
      this.gTrace = g("trace-layer"); // eligibility traces: a glow around each state, brighter the more credit it is due
      this.traces = [];
      for (let s = 1; s <= n; s++) this.traces[s] = el("circle", { class: "trace", cx: this.x(s), cy: CY, r: R + 6 }, this.gTrace);
      const chain = g("chain");
      el("line", { class: "track", x1: this.x(0), x2: this.x(n + 1), y1: CY, y2: CY }, chain);
      this.nodes = [];
      for (let s = 0; s <= n + 1; s++) {
        const exit = s === 0 || s === n + 1, x = this.x(s);
        const node = el("g", { class: `node${exit ? " exit" : ""}`, transform: `translate(${x} ${CY})${k < 1 ? ` scale(${k})` : ""}` }, chain);
        if (exit) {
          el("rect", { x: -17, y: -17, width: 34, height: 34, rx: 8 }, node);
          el("text", { class: "exit-r", y: 5 }, node).textContent = s === 0 ? (env.exits.left ? "−1" : "0") : "+1";
        } else {
          el("circle", { r: 17 }, node);
          el("text", { y: 5 }, node).textContent = env.names[s];
        }
        this.nodes[s] = node;
      }
      this.gWindow = g("window-layer"); // n-step: the stretch of the walk whose rewards make the target
      this.gAgent = g("agent-layer");
      this.bot = el("g", { class: "bot" }, this.gAgent);
      this.body = el("g", { class: "bot-body" }, this.bot);
      el("circle", { class: "bot-head", r: 11 }, this.body);
      this.eyes = el("g", { class: "bot-eyes" }, this.body);
      el("circle", { cx: -4, cy: -1.5, r: 2.2 }, this.eyes);
      el("circle", { cx: 4, cy: -1.5, r: 2.2 }, this.eyes);
      this.gFx = g("fx");
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
      this.setOptions({});
      this.place(env.start, true);
    }

    x(s) { const n = this.env.n; return this.W / 2 + (s - (n + 1) / 2) * this.dx; }
    y(v) { return PT + ((this.hi - v) / (this.hi - this.lo)) * PH; }
    _steps(V) {
      let d = "";
      for (let s = 1; s <= this.env.n; s++) d += `${s === 1 ? "M" : "L"}${this.x(s) - this.dx / 2} ${this.y(V[s])}H${this.x(s) + this.dx / 2}`;
      return d;
    }

    setOptions(o) {
      Object.assign(this.o, o);
      this.svg.toggleAttribute("data-truth", !!this.o.truth);
      this.svg.toggleAttribute("data-numbers", !!this.o.numbers);
      this.bot.classList.toggle("gone", this.o.agent === false);
    }

    show(d) {
      if (!d.V) return;
      for (let s = 1; s <= this.env.n; s++) {
        const z = d.Z && this.o.traces !== false ? d.Z[s] : 0;
        this.traces[s].style.opacity = z > 0.005 ? (0.2 + 0.8 * Math.min(1, z) ** 0.7).toFixed(3) : 0;
        this.traces[s].style.strokeWidth = `${(3 + 7 * Math.min(1.5, z)).toFixed(1)}px`;
      }
      this.Z = d.Z ? Float64Array.from(d.Z) : null;
      this.V.set(d.V);
      let line = "";
      for (let s = 1; s <= this.env.n; s++) {
        this.dots[s].style.transform = `translateY(${this.y(d.V[s])}px)`;
        this.nums[s].setAttribute("y", this.y(d.V[s]) - 11);
        this.nums[s].textContent = fmt(d.V[s]);
        line += `${s === 1 ? "M" : "L"}${this.x(s)} ${this.y(d.V[s])}`;
      }
      this.line.setAttribute("d", line);
    }

    // Faint copies of earlier estimates: [{ V, label }].
    ghosts(list) {
      this.gGhosts.replaceChildren();
      for (const { V, label } of list || []) {
        let d = "";
        for (let s = 1; s <= this.env.n; s++) d += `${s === 1 ? "M" : "L"}${this.x(s)} ${this.y(V[s])}`;
        el("path", { class: "ghost", d }, this.gGhosts);
        if (label) el("text", { class: "ghost-name", x: this.x(this.env.n) + 12, y: this.y(V[this.env.n]) + 4 }, this.gGhosts).textContent = label;
      }
    }

    place(s, instant = false) {
      if (instant) this.bot.classList.add("instant");
      this.bot.style.transform = `translate(${this.x(s)}px, ${CY - 30}px)`;
      if (instant) { this.bot.getBoundingClientRect(); this.bot.classList.remove("instant"); }
      this.at = s;
    }

    look(dir) { this.eyes.style.transform = `translate(${dir * 3}px, 0)`; }

    mark(s) {
      this.nodes.forEach((n, i) => n.classList.toggle("focus", i === s));
    }

    pop(s, text, kind = "rew") {
      const t = el("text", { class: `pop k-${kind}`, x: this.x(s), y: CY - 50 }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-18px)" }], { duration: 950, easing: "ease-out" }).onfinish = () => t.remove();
    }

    // A ring around the state whose trace was just bumped up.
    traceSpark(s) {
      const c = el("circle", { class: "spark k-trc", cx: this.x(s), cy: CY, r: 20 }, this.gFx);
      c.animate([{ opacity: 1, transform: "scale(0.6)" }, { opacity: 0, transform: "scale(1.5)" }], { duration: 600, easing: "ease-out" }).onfinish = () => c.remove();
    }

    spark(s) {
      const c = el("circle", { class: "spark", cx: this.x(s), cy: this.y(this.V[s]), r: 13 }, this.gFx);
      c.animate([{ opacity: 1, transform: "scale(0.3)" }, { opacity: 0, transform: "scale(1.6)" }], { duration: 600, easing: "ease-out" }).onfinish = () => c.remove();
    }

    event(ev, { line = false } = {}) {
      const exitNames = (s) => s === 0 || s === this.env.n + 1;
      switch (ev.type) {
        case "start": this.mark(-1); this.place(ev.s, true); return 0;
        case "choose": this.look(ev.a === 1 ? 1 : -1); return 0;
        case "move":
          this.place(ev.s2);
          if (!RL.reducedMotion()) this.body.animate(HOP, { duration: 240, easing: "ease-out" });
          if (exitNames(ev.s2)) { this.pop(ev.s2, ev.r > 0 ? "+1" : ev.r < 0 ? "−1" : "0"); return 400; }
          return 0;
        case "update": this.spark(ev.s); this.mark(ev.s); if (ev.states) this.window(null); return 0;
        case "window": this.window(ev); this.mark(ev.s); return line ? 250 : 0;
        case "trace": this.traceSpark(ev.s); return 0;
        case "return": this.mark(ev.s); this.pop(ev.s, `G ${fmt(ev.G, 0)}`, "ret"); return line ? 150 : 0;
        case "skip": this.pop(ev.s, "seen", "skip"); return 0;
        default: return 0;
      }
    }

    // A bracket under the states S_τ … S_τ+k, with what the target is made of.
    window(ev) {
      this.gWindow.replaceChildren();
      if (!ev) return;
      const a = this.x(ev.states[0]), b = this.x(ev.states[ev.states.length - 1]), y = CY + 26;
      el("path", { class: "window", d: `M${a} ${y - 6}V${y}H${b}V${y - 6}` }, this.gWindow);
      el("text", { class: "window-name", x: (a + b) / 2, y: y + 14 }, this.gWindow).textContent =
        `${ev.k === 1 ? "1 reward" : `${ev.k} rewards`}${ev.boot ? " + the estimate here" : ", to the end"}`;
    }

    rest(events) {
      this.mark(-1);
      this.window(null);
      const moves = events.filter((e) => e.type === "move");
      this.place(moves.length ? moves[moves.length - 1].s2 : this.env.start, true);
    }

    _hover(e) {
      const box = this.svg.getBoundingClientRect(), px = ((e.clientX - box.left) / box.width) * this.W;
      const s = Math.round((px - this.W / 2) / this.dx + (this.env.n + 1) / 2);
      if (s < 1 || s > this.env.n) { RL.tip.hide(); return; }
      const truth = this.env.truth()[s];
      RL.tip.show(e.clientX, e.clientY, `<div class="head">State ${this.env.names[s]}</div>
        <div class="row"><b>${fmt(this.V[s], 3)}</b><span>estimate V</span></div>
        <div class="row"><b>${fmt(truth, 3)}</b><span>true value</span></div>
        <div class="row"><b>${fmt(this.V[s] - truth, 3)}</b><span>error</span></div>` +
        (this.Z ? `<div class="row"><b>${this.Z[s].toFixed(3)}</b><span>z, its eligibility trace</span></div>` : ""));
    }

    destroy() { RL.tip.hide(); this.svg.remove(); }

    static options(env, displays = []) {
      return [
        { key: "truth", type: "check", label: "True values (dashed)", value: true },
        { key: "numbers", type: "check", label: "Estimates as numbers", value: false },
        ...(displays.some((d) => d.Z) ? [{ key: "traces", type: "check", label: "Eligibility traces", swatch: "trc", value: true }] : []),
      ];
    }

    static thumb(env, d) {
      const n = env.n, S = 20, Hh = 52, lo = Math.min(0, env.exits.left), y = (v) => 4 + ((1 - v) / (1 - lo)) * Hh, truth = env.truth();
      let g = "", line = "";
      for (let s = 1; s <= n; s++) {
        const x = (s - 0.5) * S;
        g += `<line class="t-truth" x1="${x - 7}" x2="${x + 7}" y1="${y(truth[s])}" y2="${y(truth[s])}"/>`;
        line += `${s === 1 ? "M" : "L"}${x} ${y(d.V[s])}`;
      }
      return `<svg class="thumb-chain" viewBox="0 0 ${n * S} ${Hh + 8}">${g}<path class="t-line" d="${line}"/>${Array.from({ length: n }, (_, i) => `<circle class="t-dot" cx="${(i + 0.5) * S}" cy="${y(d.V[i + 1])}" r="2.6"/>`).join("")}</svg>`;
    }
  }

  RL.ChainView = ChainView;
  (RL.labViews = RL.labViews || {}).chain = ChainView;
})(globalThis.RL = globalThis.RL || {});
