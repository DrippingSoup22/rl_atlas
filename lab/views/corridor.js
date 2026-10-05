/* Corridor view: the short corridor of Sutton & Barto (Example 13.1), shared by the Lab and the stories.
   On top, the three cells and the goal, the agent walking them, and in every cell the policy's two arrows: they are
   the same in all three cells, because the agent cannot tell the cells apart. Below, the landscape the policy climbs:
   the value of the start, J, for every chance p of stepping right, worked out exactly. The policy is a dot on it; while
   an episode's updates play, a fading trail shows each nudge. A learned baseline is a dashed line at its height. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 640, CX = 112, CW = 96, GAP = 18, CY = 46, CH = 64, PL = 74, PR = 592, PT = 168, PB = 318, H = 348, LO = -100;
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const signed = (v, d = 1) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;
  const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const huge = (v) => { const e = Math.floor(Math.log10(Math.abs(v))); return `−${(Math.abs(v) / 10 ** e).toFixed(1)}·10${String(e).replace(/\d/g, (c) => SUP[c])}`; };
  const HOP = [{ transform: "translateY(0)" }, { transform: "translateY(-10px)" }, { transform: "translateY(0)" }];
  const BUMP = [{ transform: "translateX(0)" }, { transform: "translateX(-9px)" }, { transform: "translateX(0)" }];

  class CorridorView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { numbers: false, landscape: true, ...options };
      this.p = 0.5;
      this.history = [];
      this.svg = el("svg", { class: "corridorview", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": env.title });
      host.appendChild(this.svg);
      const g = (cls, parent = this.svg) => el("g", { class: cls }, parent);
      // ---- the corridor ----
      const hall = g("hall");
      el("rect", { class: "wall", x: CX - 12, y: CY - 6, width: 8, height: CH + 12, rx: 3 }, hall);
      this.cells = [];
      this.arrows = [];
      for (let s = 0; s < 4; s++) {
        const x = this.cx(s), cell = g(`cell${s === 3 ? " goal" : ""}${s === 1 ? " swapped" : ""}`, hall);
        el("rect", { class: "tile", x: x - CW / 2, y: CY, width: CW, height: CH, rx: 10 }, cell);
        el("text", { class: "cell-name", x, y: CY + CH + 18 }, cell).textContent = s === 3 ? "goal" : `cell ${s + 1}`;
        if (s === 3) {
          const gem = el("g", { class: "gem", transform: `translate(${x} ${CY + CH / 2})` }, cell);
          el("polygon", { points: "0,-15 14,-3 0,16 -14,-3" }, gem);
          el("polygon", { class: "shine", points: "0,-15 6,-3 0,2 -6,-3" }, gem);
        } else {
          if (s === 1) el("text", { class: "swap", x, y: CY + 15 }, cell).textContent = "⇄ swapped";
          // the policy's two arrows, the same in every cell
          const set = { left: el("g", { class: "parrow left" }, cell), right: el("g", { class: "parrow right" }, cell) };
          for (const k of ["left", "right"]) {
            set[k].halo = el("path", { class: "halo" }, set[k]);
            set[k].shaft = el("path", { class: "shaft" }, set[k]);
            set[k].num = el("text", { class: "pnum", y: CY + CH - 7 }, set[k]);
          }
          this.arrows[s] = set;
        }
        this.cells[s] = cell;
      }
      el("text", { class: "hall-note", x: this.cx(1), y: CY - 14 }, hall).textContent = "";
      this.bot = el("g", { class: "bot" }, this.svg);
      this.body = el("g", { class: "bot-body" }, this.bot);
      el("circle", { class: "bot-head", r: 12 }, this.body);
      this.eyes = el("g", { class: "bot-eyes" }, this.body);
      el("circle", { cx: -4.5, cy: -1.5, r: 2.4 }, this.eyes);
      el("circle", { cx: 4.5, cy: -1.5, r: 2.4 }, this.eyes);
      // ---- the landscape ----
      this.plot = g("land");
      const axis = g("axis", this.plot);
      for (const v of [0, -20, -40, -60, -80, -100]) {
        el("line", { class: `grid${v === 0 ? " zero" : ""}`, x1: PL, x2: PR, y1: this.y(v), y2: this.y(v) }, axis);
        el("text", { class: "tick", x: PL - 8, y: this.y(v) + 4, "text-anchor": "end" }, axis).textContent = v ? signed(v, 0) : "0";
      }
      for (const p of [0, 0.25, 0.5, 0.75, 1]) el("text", { class: "tick", x: this.x(p), y: PB + 16, "text-anchor": "middle" }, axis).textContent = `${Math.round(p * 100)}%`;
      el("text", { class: "axis-name", x: PL - 8, y: PT - 12, "text-anchor": "start" }, axis).textContent = "value of the start, J";
      el("text", { class: "axis-name", x: PR, y: PB + 30, "text-anchor": "end" }, axis).textContent = "chance of stepping right, p";
      let d = "";
      for (let k = 2; k <= 398; k++) { const p = k / 400; d += `${d ? "L" : "M"}${this.x(p).toFixed(1)} ${this.y(env.J(p)).toFixed(1)}`; }
      el("path", { class: "landscape", d }, this.plot);
      const mark = (p, label, anchor, dy) => {
        const m = el("g", { class: "mark-pt" }, this.plot);
        el("circle", { cx: this.x(p), cy: this.y(env.J(p)), r: 4 }, m);
        el("text", { x: this.x(p) + (anchor === "start" ? 8 : anchor === "end" ? -8 : 0), y: this.y(env.J(p)) + dy, "text-anchor": anchor }, m).textContent = label;
      };
      // the landmarks' names sit below their points, so that the policy's own label, above its dot, never covers them
      mark(0.05, `ε-greedy, left: ${signed(env.J(0.05), 0)}`, "start", 17);
      mark(0.95, `ε-greedy, right: ${signed(env.J(0.95), 0)}`, "end", 4);
      mark(env.best, `best: ${signed(env.bestValue, 1)} at p = ${env.best.toFixed(2)}`, "middle", 20);
      this.baseEl = el("g", { class: "baseline" }, this.plot);
      el("line", { x1: PL, x2: PR }, this.baseEl);
      this.baseText = el("text", { x: PR - 4, "text-anchor": "end" }, this.baseEl);
      this.trailEl = el("g", { class: "ptrail" }, this.plot);
      this.dotG = el("g", { class: "pdot-g" }, this.plot);
      this.dot = el("circle", { class: "pdot", r: 7 }, this.dotG);
      this.dotText = el("text", { class: "pdot-num", y: -13, "text-anchor": "middle" }, this.dotG);
      this.gFx = g("fx");
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
      this.setOptions({});
      this.place(0, true);
    }

    cx(s) { return CX + CW / 2 + s * (CW + GAP); }
    x(p) { return PL + p * (PR - PL); }
    y(v) { return PT + (Math.min(0, Math.max(LO, Number.isFinite(v) ? v : LO)) / LO) * (PB - PT); }

    setOptions(o) {
      Object.assign(this.o, o);
      this.svg.toggleAttribute("data-numbers", !!this.o.numbers);
      this.bot.classList.toggle("gone", this.o.agent === false);
    }

    // What the learner knows: its policy (one chance p of stepping right, the same in every cell) and maybe a baseline.
    show(d) {
      if (!d.P) return;
      const p = d.P[1];
      this.p = p;
      for (let s = 0; s < 3; s++) {
        const set = this.arrows[s], y = CY + CH / 2 + 3, x = this.cx(s);
        for (const [k, q, dir] of [["left", 1 - p, -1], ["right", p, 1]]) {
          const len = 6 + 34 * q, hs = 4 + 3 * q, w = 2 + 3.5 * q, tip = x + dir * len;
          const shaft = `M${x + dir * 3} ${y}H${tip}M${tip - dir * hs} ${y - hs}L${tip} ${y}L${tip - dir * hs} ${y + hs}`;
          set[k].halo.setAttribute("d", shaft);
          set[k].shaft.setAttribute("d", shaft);
          set[k].shaft.style.strokeWidth = `${w.toFixed(2)}px`;
          set[k].halo.style.strokeWidth = `${(w + 3.5).toFixed(2)}px`;
          set[k].style.opacity = q < 0.005 ? 0.25 : 1;
          set[k].num.setAttribute("x", x + dir * 24);
          set[k].num.textContent = `${Math.round(q * 100)}%`;
        }
      }
      const J = this.env.J(p);
      this.dotG.style.transform = `translate(${this.x(p).toFixed(1)}px, ${this.y(J).toFixed(1)}px)`;
      // far out on the slopes J runs to millions: the policy almost never arrives
      const pct = `${Math.round(100 * p)}% right`;
      this.dotText.textContent = !Number.isFinite(J) ? `${pct}: never arrives` : J < -1e4 ? `${pct}: J ≈ ${huge(J)}, almost never arrives` : `${pct}: J = ${signed(J)}`;
      this.dotText.setAttribute("text-anchor", p > 0.8 ? "end" : p < 0.2 ? "start" : "middle");
      this.dotText.setAttribute("x", p > 0.8 ? 6 : p < 0.2 ? -6 : 0);
      // the trail: the nudges of the updates being played, fading
      const last = this.history[this.history.length - 1];
      if (last === undefined || Math.abs(last - p) > 1e-4) this.history.push(p);
      if (this.history.length > 60) this.history.shift();
      this._trail();
      const hasBase = d.V !== undefined && d.V !== null;
      this.baseEl.style.display = hasBase ? "" : "none";
      if (hasBase) {
        const b = d.V[0];
        this.baseEl.style.transform = `translateY(${this.y(b) - PT}px)`;
        this.baseEl.querySelector("line").setAttribute("y1", PT);
        this.baseEl.querySelector("line").setAttribute("y2", PT);
        this.baseText.setAttribute("y", PT - 5);
        this.baseText.textContent = `baseline v̂ = ${signed(b)}`;
      }
    }

    _trail() {
      this.trailEl.replaceChildren();
      const n = this.history.length;
      this.history.forEach((p, i) => {
        if (i === n - 1) return;
        el("circle", { cx: this.x(p), cy: this.y(this.env.J(p)), r: 3, style: `opacity:${(0.12 + 0.55 * (i / n)).toFixed(2)}` }, this.trailEl);
      });
    }

    place(s, instant = false) {
      if (instant) this.bot.classList.add("instant");
      this.bot.style.transform = `translate(${this.cx(s)}px, ${CY - 18}px)`;
      if (instant) { this.bot.getBoundingClientRect(); this.bot.classList.remove("instant"); }
      this.at = s;
    }
    look(dir) { this.eyes.style.transform = `translate(${dir * 3}px, 0)`; }

    pop(x, y, text, kind = "rew") {
      const t = el("text", { class: `pop k-${kind}`, x, y }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-16px)" }], { duration: 950, easing: "ease-out" }).onfinish = () => t.remove();
    }
    spark(x, y, kind = "") {
      const c = el("circle", { class: `spark${kind ? ` k-${kind}` : ""}`, cx: x, cy: y, r: 12 }, this.gFx);
      c.animate([{ opacity: 1, transform: "scale(0.3)" }, { opacity: 0, transform: "scale(1.7)" }], { duration: 600, easing: "ease-out" }).onfinish = () => c.remove();
    }
    mark(s) { this.cells.forEach((c, i) => c.classList.toggle("focus", i === s)); }

    event(ev, { line = false } = {}) {
      switch (ev.type) {
        case "start": this.mark(-1); this.place(ev.s, true); return 0;
        case "choose":
          this.look(ev.a === 1 ? 1 : -1);
          this.arrows[ev.s]?.[ev.a === 1 ? "right" : "left"].classList.add("chosen");
          setTimeout(() => this.arrows[ev.s]?.[ev.a === 1 ? "right" : "left"].classList.remove("chosen"), 450);
          return 0;
        case "move":
          if (ev.s2 === ev.s) { if (!RL.reducedMotion()) this.body.animate(BUMP, { duration: 260, easing: "ease-out" }); return 0; }
          this.place(ev.s2);
          if (!RL.reducedMotion()) this.body.animate(HOP, { duration: 240, easing: "ease-out" });
          if (this.env.terminal(ev.s2)) { this.pop(this.cx(3), CY - 26, "goal!"); return 350; }
          return 0;
        case "return": this.mark(ev.s); this.pop(this.cx(ev.s), CY - 26, `G ${signed(ev.G, 0)}`, "ret"); return line ? 150 : 0;
        case "update": this.spark(this.x(this.p), this.y(this.env.J(this.p)), "pol"); return 0;
        default: return 0;
      }
    }

    // After a jump to another moment: the agent where the last episode ended, and a fresh trail.
    rest(events) {
      this.mark(-1);
      this.history = [this.p];
      this._trail();
      const moves = events.filter((e) => e.type === "move");
      this.place(moves.length ? moves[moves.length - 1].s2 : 0, true);
    }

    _hover(e) {
      const box = this.svg.getBoundingClientRect(), px = ((e.clientX - box.left) / box.width) * W, py = ((e.clientY - box.top) / box.height) * H;
      if (py < PT - 20 || px < PL || px > PR) { RL.tip.hide(); return; }
      const p = Math.max(0.01, Math.min(0.99, (px - PL) / (PR - PL))), v = this.env.values(p);
      RL.tip.show(e.clientX, e.clientY, `<div class="head">Step right ${(100 * p).toFixed(0)}% of the time</div>
        <div class="row"><b>${signed(v[0])}</b><span>value of the start, J (cell 1)</span></div>
        <div class="row"><b>${signed(v[1])}</b><span>cell 2</span></div><div class="row"><b>${signed(v[2])}</b><span>cell 3</span></div>
        <div class="row"><b>${(100 * this.p).toFixed(1)}%</b><span>the learner's p now</span></div>`);
    }

    destroy() { RL.tip.hide(); this.svg.remove(); }

    static options() {
      return [{ key: "numbers", type: "check", label: "Chances as numbers", value: true }];
    }

    static thumb(env, d) {
      const w = 200, h = 64, x = (p) => 4 + p * (w - 8), y = (v) => 4 + (Math.min(0, Math.max(LO, v)) / LO) * (h - 8);
      let path = "";
      for (let k = 4; k <= 196; k += 4) { const p = k / 200; path += `${path ? "L" : "M"}${x(p).toFixed(1)} ${y(env.J(p)).toFixed(1)}`; }
      const p = d.P ? d.P[1] : 0.5;
      return `<svg class="thumb-corridor" viewBox="0 0 ${w} ${h}"><path class="t-line" d="${path}"/><circle class="t-dot" cx="${x(p)}" cy="${y(env.J(p))}" r="5"/></svg>`;
    }
  }

  RL.CorridorView = CorridorView;
  (RL.labViews = RL.labViews || {}).corridor = CorridorView;
})(globalThis.RL = globalThis.RL || {});
