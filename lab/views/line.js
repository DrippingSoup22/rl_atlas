/* Line view: a walk with too many states to draw one by one (the 1000-state random walk), shared by the Lab and the
   stories. Above, the estimate v̂ = w · x(s) of every state as one line, against the true values (dashed) and over a
   shading of how much time the walk spends in each state (μ), as in Sutton & Barto's Figure 9.1. Below it, a strip
   shows the footprint of the last update: how much each state's estimate moved with it, as a share of the updated
   state's own move. That strip is generalization made visible: a box for state aggregation, a tent for tile coding,
   waves that cover everything for polynomials and cosines. At the bottom, the track the agent jumps along. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 640, L = 48, R = 24, PT = 22, PH = 158, FT = 222, FH = 36, CY = 284, H = 300;
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const fmt = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  const clamp = (v, lo, hi) => (Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : hi);

  class LineView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { truth: true, visits: true, footprint: true, ...options };
      this.V = new Float64Array(env.nS);
      this.svg = el("svg", { class: "lineview", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": env.title });
      host.appendChild(this.svg);
      const g = (cls) => el("g", { class: cls }, this.svg);
      // μ, the share of time spent in each state, as a shading under the plot
      // (the start state gets a spike of its own, a visit every episode: the scale is set by the others, and the spike is cut)
      const mu = env.mu(), top = Math.max(...Array.from(mu).filter((_, s) => Math.abs(s - env.start) > 2));
      let area = `M${this.x(1)} ${PT + PH}`;
      for (let s = 1; s <= env.n; s++) if (s % 4 === 1 || Math.abs(s - env.start) <= 1) area += `L${this.x(s).toFixed(1)} ${(PT + PH - Math.min(1.45, mu[s] / top) * PH * 0.55).toFixed(1)}`;
      this.visits = el("path", { class: "visits", d: `${area}L${this.x(env.n)} ${PT + PH}Z` }, g("visit-layer"));
      el("text", { class: "visits-name", x: this.x(env.n * 0.75), y: PT + PH - 8, "text-anchor": "middle" }, this.svg).textContent = "time spent here (μ)";
      const axis = g("axis");
      for (const v of [-1, -0.5, 0, 0.5, 1]) {
        el("line", { class: `grid${v === 0 ? " zero" : ""}`, x1: L, x2: W - R, y1: this.y(v), y2: this.y(v) }, axis);
        el("text", { class: "tick", x: L - 8, y: this.y(v) + 4, "text-anchor": "end" }, axis).textContent = fmt(v, v % 1 ? 1 : 0);
      }
      for (const s of [1, 200, 400, 600, 800, 1000]) el("text", { class: "tick", x: this.x(s), y: PT + PH + 15, "text-anchor": "middle" }, axis).textContent = s;
      el("text", { class: "axis-name", x: L - 8, y: PT - 9, "text-anchor": "start" }, axis).textContent = "value";
      this.gBounds = g("bounds"); // the edges of the groups, for state aggregation
      this.gGhosts = g("ghosts");
      const truth = env.truth();
      let d = "";
      for (let s = 1; s <= env.n; s += 3) d += `${d ? "L" : "M"}${this.x(s).toFixed(1)} ${this.y(truth[s]).toFixed(1)}`;
      this.truthEl = el("path", { class: "truth", d: `${d}L${this.x(env.n)} ${this.y(truth[env.n])}` }, this.svg);
      this.line = el("path", { class: "est-line" }, this.svg);
      this.mark = el("circle", { class: "est-mark", r: 5 }, this.svg);
      // the footprint strip
      const strip = g("strip");
      el("line", { class: "strip-zero", x1: L, x2: W - R, y1: FT + FH, y2: FT + FH }, strip);
      el("text", { class: "strip-name", x: L, y: FT - 6 }, strip).textContent = "how far each state moved with the last update";
      this.foot = el("path", { class: "footprint" }, strip);
      this.footNeg = el("path", { class: "footprint neg" }, strip);
      // the track
      const track = g("track");
      el("line", { class: "track-line", x1: L, x2: W - R, y1: CY, y2: CY }, track);
      el("rect", { class: "exit", x: L - 24, y: CY - 10, width: 20, height: 20, rx: 5 }, track);
      el("rect", { class: "exit", x: W - R + 4, y: CY - 10, width: 20, height: 20, rx: 5 }, track);
      el("text", { class: "exit-r", x: L - 14, y: CY + 4 }, track).textContent = "−1";
      el("text", { class: "exit-r", x: W - R + 14, y: CY + 4 }, track).textContent = "+1";
      this.gJump = g("jumps");
      this.bot = el("circle", { class: "bot", r: 7 }, this.svg);
      this.gFx = g("fx");
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => { RL.tip.hide(); });
      this.setOptions({});
      this.place(env.start, true);
    }

    x(s) { return L + ((s - 0.5) / this.env.n) * (W - L - R); }
    y(v) { return PT + ((1.15 - clamp(v, -1.15, 1.15)) / 2.3) * PH; }

    setOptions(o) {
      Object.assign(this.o, o);
      this.svg.toggleAttribute("data-truth", !!this.o.truth);
      this.svg.toggleAttribute("data-visits", !!this.o.visits);
      this.svg.toggleAttribute("data-footprint", !!this.o.footprint);
      this.bot.classList.toggle("gone", this.o.agent === false);
    }

    show(d) {
      if (!d.V) return;
      this.V.set(d.V);
      this.F = d.F;
      let line = "";
      for (let s = 1; s <= this.env.n; s++) line += `${s === 1 ? "M" : "L"}${this.x(s).toFixed(1)} ${this.y(d.V[s]).toFixed(1)}`;
      this.line.setAttribute("d", line);
      if (d.F && d.F !== this.boundsFor) {
        this.boundsFor = d.F;
        this.gBounds.replaceChildren();
        if (d.F.kind === "groups") for (let i = 1; i < d.F.cells; i++) {
          const xx = L + (i / d.F.cells) * (W - L - R);
          el("line", { class: "bound", x1: xx, x2: xx, y1: PT, y2: PT + PH }, this.gBounds);
        }
      }
    }

    // Faint copies of earlier estimates: [{ V, label }].
    ghosts(list) {
      this.gGhosts.replaceChildren();
      for (const { V, label } of list || []) {
        let d = "";
        for (let s = 1; s <= this.env.n; s += 2) d += `${d ? "L" : "M"}${this.x(s).toFixed(1)} ${this.y(V[s]).toFixed(1)}`;
        el("path", { class: "ghost", d }, this.gGhosts);
        if (label) el("text", { class: "ghost-name", x: this.x(this.env.n) - 4, y: this.y(V[this.env.n]) - 6, "text-anchor": "end" }, this.gGhosts).textContent = label;
      }
    }

    // The footprint of an update of state s: x(s') · x(s) / x(s) · x(s) for every s', the share of s's move that s' got.
    footprint(s) {
      if (s == null || !this.F) { this.foot.setAttribute("d", ""); this.footNeg.setAttribute("d", ""); return; }
      const F = this.F, xs = F.of(s), dense = new Float64Array(F.n);
      let self = 0;
      for (let i = 0; i < xs.k; i++) { dense[xs.idx[i]] += xs.val[i]; self += xs.val[i] * xs.val[i]; }
      const share = (t) => { const x = F.of(t); let v = 0; for (let i = 0; i < x.k; i++) v += dense[x.idx[i]] * x.val[i]; return v / self; };
      let pos = `M${L} ${FT + FH}`, neg = pos;
      for (let t = 1; t <= this.env.n; t += 2) {
        const v = share(t), xx = this.x(t).toFixed(1);
        pos += `L${xx} ${(FT + FH - clamp(v, 0, 1.2) * FH).toFixed(1)}`;
        neg += `L${xx} ${(FT + FH - clamp(v, -1, 0) * FH * 0.5).toFixed(1)}`;
      }
      this.foot.setAttribute("d", `${pos}L${W - R} ${FT + FH}Z`);
      this.footNeg.setAttribute("d", `${neg}L${W - R} ${FT + FH}Z`);
    }

    place(s, instant = false) {
      if (instant) this.bot.classList.add("instant");
      this.bot.style.transform = `translate(${s === 0 ? L - 14 : s === this.env.n + 1 ? W - R + 14 : this.x(s)}px, ${CY}px)`;
      if (instant) { this.bot.getBoundingClientRect(); this.bot.classList.remove("instant"); }
      this.at = s;
    }

    // An arc from where the agent was to where it landed.
    jump(s, s2) {
      if (RL.reducedMotion()) return;
      const a = this.x(s), b = s2 === 0 ? L - 14 : s2 === this.env.n + 1 ? W - R + 14 : this.x(s2), h = 10 + Math.abs(b - a) * 0.35;
      const p = el("path", { class: "jump", d: `M${a} ${CY - 8}Q${(a + b) / 2} ${CY - 8 - h} ${b} ${CY - 8}` }, this.gJump);
      p.animate([{ opacity: 0.9 }, { opacity: 0 }], { duration: 700, easing: "ease-in" }).onfinish = () => p.remove();
    }

    pop(xx, y, text, kind = "rew") {
      const t = el("text", { class: `pop k-${kind}`, x: xx, y }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-16px)" }], { duration: 950, easing: "ease-out" }).onfinish = () => t.remove();
    }

    event(ev) {
      switch (ev.type) {
        case "start": this.place(ev.s, true); this.mark.style.opacity = 0; return 0;
        case "move":
          this.jump(ev.s, ev.s2);
          this.place(ev.s2);
          if (this.env.terminal(ev.s2)) { this.pop(ev.s2 === 0 ? L - 14 : W - R + 14, CY - 18, ev.r > 0 ? "+1" : "−1"); return 350; }
          return 0;
        case "update":
          this.footprint(this.o.footprint ? ev.s : null);
          this.mark.setAttribute("cx", this.x(ev.s));
          this.mark.setAttribute("cy", this.y(ev.value));
          this.mark.style.opacity = 1;
          this.place(ev.s, true);
          return 0;
        default: return 0;
      }
    }

    rest(events) {
      const moves = events.filter((e) => e.type === "move"), ups = events.filter((e) => e.type === "update");
      this.place(moves.length ? moves[moves.length - 1].s2 : this.env.start, true);
      this.mark.style.opacity = 0;
      this.footprint(ups.length && this.o.footprint ? ups[ups.length - 1].s : null);
    }

    _hover(e) {
      const box = this.svg.getBoundingClientRect(), px = ((e.clientX - box.left) / box.width) * W, n = this.env.n;
      const s = Math.round(((px - L) / (W - L - R)) * n + 0.5);
      if (s < 1 || s > n) { RL.tip.hide(); return; }
      const truth = this.env.truth()[s], mu = this.env.mu()[s];
      RL.tip.show(e.clientX, e.clientY, `<div class="head">State ${s}</div>
        <div class="row"><b>${fmt(this.V[s], 3)}</b><span>estimate v̂</span></div>
        <div class="row"><b>${fmt(truth, 3)}</b><span>true value</span></div>
        <div class="row"><b>${(100 * mu).toFixed(2)}%</b><span>of the time is spent here</span></div>`);
    }

    destroy() { RL.tip.hide(); this.svg.remove(); }

    static options() {
      return [
        { key: "truth", type: "check", label: "True values (dashed)", value: true },
        { key: "visits", type: "check", label: "Time spent in each state (μ)", value: true },
        { key: "footprint", type: "check", label: "Footprint of each update", swatch: "err", value: true },
      ];
    }

    static thumb(env, d) {
      const n = env.n, w = 200, h = 60, x = (s) => ((s - 0.5) / n) * w, y = (v) => 4 + ((1.1 - clamp(v, -1.1, 1.1)) / 2.2) * (h - 8);
      const truth = env.truth();
      let t = "", e = "";
      for (let s = 1; s <= n; s += 10) t += `${t ? "L" : "M"}${x(s).toFixed(1)} ${y(truth[s]).toFixed(1)}`;
      for (let s = 1; s <= n; s += 4) e += `${e ? "L" : "M"}${x(s).toFixed(1)} ${y(d.V[s]).toFixed(1)}`;
      return `<svg class="thumb-line" viewBox="0 0 ${w} ${h}"><line class="t-zero" x1="0" x2="${w}" y1="${y(0)}" y2="${y(0)}"/><path class="t-truth" d="${t}"/><path class="t-line" d="${e}"/></svg>`;
    }
  }

  RL.LineView = LineView;
  (RL.labViews = RL.labViews || {}).line = LineView;
})(globalThis.RL = globalThis.RL || {});
