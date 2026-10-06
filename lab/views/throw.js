/* Throw view: one throw per episode, a continuous action, shared by the Lab and the stories. On top, the field: the
   thrower, the ball's arc for the angle the policy drew, where it lands (the reward), and where the last throws
   landed. Below, on one axis of angles: how far a throw at each angle flies (the world, which the learner never sees),
   and the Gaussian policy, a bell around its aim μ with spread σ, with the angles of the last throws under it. A
   learned baseline is a dashed line at the distance it expects. A deterministic policy (DPG) is only its aim: the bell is
   then the noise its throws are made with, and a critic's curve q̂(a) is drawn over the last throws' landings, with its
   slope at the aim. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 640, FX = 78, FW = 528, GY = 124, AX = 78, AW = 528, PT = 190, PB = 304, H = 336, MAXD = 45;
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const RAD = Math.PI / 180;

  class ThrowView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { truth: true, ...options };
      this.d = { mu: 20, sd: 10 };
      this.throws = []; // the last throws: { angle, land }
      this.svg = el("svg", { class: "throwview", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": env.title });
      host.appendChild(this.svg);
      const g = (cls, parent = this.svg) => el("g", { class: cls }, parent);
      // ---- the field ----
      const field = g("field");
      el("line", { class: "ground", x1: FX - 40, x2: FX + FW + 10, y1: GY, y2: GY }, field);
      for (let m = 0; m <= 40; m += 10) {
        el("line", { class: "mtick", x1: this.fx(m), x2: this.fx(m), y1: GY, y2: GY + 5 }, field);
        el("text", { class: "tick", x: this.fx(m), y: GY + 18, "text-anchor": "middle" }, field).textContent = `${m} m`;
      }
      el("line", { class: "pole", x1: this.fx(40), x2: this.fx(40), y1: GY, y2: GY - 30 }, field);
      el("path", { class: "flag", d: `M${this.fx(40)} ${GY - 30}l15 5l-15 5z` }, field);
      el("text", { class: "field-note", x: this.fx(40) + 4, y: GY - 36, "text-anchor": "middle" }, field).textContent = "best: 40 m at 45°";
      this.gLand = g("landings", field);
      // the thrower
      const man = g("thrower", field);
      el("circle", { class: "head", cx: FX - 22, cy: GY - 44, r: 7 }, man);
      el("path", { class: "body", d: `M${FX - 22} ${GY - 37}V${GY - 16}M${FX - 22} ${GY - 16}L${FX - 30} ${GY}M${FX - 22} ${GY - 16}L${FX - 14} ${GY}` }, man);
      this.arm = el("line", { class: "arm", x1: FX - 22, y1: GY - 32, x2: FX - 4, y2: GY - 32 }, man);
      this.angleText = el("text", { class: "angle-text", x: FX - 2, y: GY - 56 }, field);
      this.arc = el("path", { class: "arc", pathLength: 1 }, field);
      this.ball = el("circle", { class: "ball", r: 5, cx: 0, cy: 0 }, field);
      this.ball.style.transform = `translate(${FX - 4}px, ${GY - 32}px)`;
      this.windText = el("text", { class: "wind", x: W - 12, y: 22, "text-anchor": "end" }, field);
      // ---- the policy over angles ----
      const pane = g("pane");
      for (const m of [0, 20, 40]) {
        el("line", { class: `grid${m ? "" : " zero"}`, x1: AX, x2: AX + AW, y1: this.dy(m), y2: this.dy(m) }, pane);
        el("text", { class: "tick rew", x: AX - 26, y: this.dy(m) + 4, "text-anchor": "end" }, pane).textContent = `${m} m`;
      }
      for (let a = 0; a <= 90; a += 15) el("text", { class: "tick", x: this.ax(a), y: PB + 16, "text-anchor": "middle" }, pane).textContent = `${a}°`;
      el("text", { class: "axis-name", x: AX + AW, y: PB + 31, "text-anchor": "end" }, pane).textContent = "angle of the throw";
      let d = "";
      for (let a = 0; a <= 90; a += 1) d += `${a ? "L" : "M"}${this.ax(a).toFixed(1)} ${this.dy(env.distance(a)).toFixed(1)}`;
      this.truth = g("truth", pane);
      el("path", { class: "reach", d }, this.truth);
      el("text", { class: "reach-name", x: this.ax(76) + 6, y: this.dy(env.distance(76)) - 6, "text-anchor": "start" }, this.truth).textContent = "how far a throw flies";
      this.gPoints = g("points", pane); // a critic's data: where the last throws landed, by angle
      this.bell = el("path", { class: "bell" }, pane);
      this.criticEl = el("path", { class: "critic" }, pane);
      this.criticText = el("text", { class: "critic-name", "text-anchor": "start" }, pane);
      this.slopeEl = el("path", { class: "slope" }, pane);
      this.slopeDot = el("circle", { class: "slope-dot", r: 4 }, pane);
      this.muLine = el("line", { class: "mu", y1: PT - 8, y2: PB }, pane);
      this.muText = el("text", { class: "mu-text", y: PT - 12, "text-anchor": "middle" }, pane);
      this.sdBar = el("path", { class: "sd" }, pane);
      this.gTicks = g("ticks", pane);
      this.baseEl = g("baseline", pane);
      this.baseLine = el("line", { x1: AX, x2: AX + AW }, this.baseEl);
      this.baseText = el("text", { x: AX + 4, "text-anchor": "start" }, this.baseEl);
      this.gFx = g("fx");
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
      this.setOptions({});
      this._arm(20);
    }

    fx(m) { return FX + (Math.max(-3, Math.min(MAXD, m)) / MAXD) * FW; }
    ax(a) { return AX + (Math.max(-10, Math.min(100, a)) / 90) * AW; }
    dy(m) { return PB - (m / 42) * (PB - PT); }

    setOptions(o) {
      Object.assign(this.o, o);
      this.svg.toggleAttribute("data-truth", this.o.truth !== false);
    }

    // The policy: a bell around μ with spread σ (in degrees); its height grows as it narrows, up to the top of the pane.
    show(d) {
      if (d.mu === undefined) return;
      const redraw = !!d.critic !== !!this.d.critic;
      this.d = d;
      const { mu } = d, det = !!d.deterministic, sd = Math.max(0, d.sd);
      this.svg.toggleAttribute("data-deterministic", det);
      const top = Math.min(PB - PT + 6, 640 / Math.max(0.5, sd)), y = (a) => PB - top * Math.exp(-0.5 * ((a - mu) / Math.max(1e-6, sd)) ** 2);
      // drawn from −4° to 94°: a little of the tails the world clips, as far as the labels allow
      let path = `M${this.ax(-4)} ${PB}`;
      for (let a = -4; a <= 94; a += 0.5) path += `L${this.ax(a).toFixed(1)} ${y(a).toFixed(1)}`;
      this.bell.setAttribute("d", sd > 0 ? `${path}L${this.ax(94)} ${PB}Z` : "");
      const x = this.ax(mu);
      this.muLine.setAttribute("x1", x);
      this.muLine.setAttribute("x2", x);
      this.muText.setAttribute("x", Math.max(AX + 40, Math.min(AX + AW - 40, x)));
      this.muText.textContent = det
        ? `aim μ = ${mu.toFixed(1)}°, ${sd > 0 ? `thrown with noise σ = ${sd.toFixed(0)}°` : "no noise"}`
        : `aim μ = ${mu.toFixed(1)}°, spread σ = ${sd.toFixed(1)}°`;
      const half = PB - top * Math.exp(-0.5);
      this.sdBar.setAttribute("d", sd > 0 ? `M${this.ax(mu - sd)} ${half}H${this.ax(mu + sd)}M${this.ax(mu - sd)} ${half - 4}v8M${this.ax(mu + sd)} ${half - 4}v8` : "");
      this._critic(d);
      if (redraw) this._ticks();
      const hasBase = d.base !== undefined;
      this.baseEl.style.display = hasBase ? "" : "none";
      if (hasBase) {
        const yb = this.dy(Math.max(-2, Math.min(44, d.base)));
        this.baseLine.setAttribute("y1", yb);
        this.baseLine.setAttribute("y2", yb);
        this.baseText.setAttribute("y", yb - 5);
        this.baseText.textContent = `baseline: ${d.base.toFixed(1)} m expected`;
      }
    }

    // A critic's estimate of how far each angle flies, and its slope at the aim: the line the aim climbs.
    _critic(d) {
      const on = !!d.critic;
      for (const e of [this.criticEl, this.criticText, this.slopeEl, this.slopeDot]) e.style.display = on ? "" : "none";
      if (!on) return;
      const c = d.critic, clamp = (m) => Math.max(-2, Math.min(44, m));
      let path = "";
      for (let a = 0; a < c.length; a++) path += `${a ? "L" : "M"}${this.ax(a).toFixed(1)} ${this.dy(clamp(c[a])).toFixed(1)}`;
      this.criticEl.setAttribute("d", path);
      this.criticText.setAttribute("x", this.ax(1));
      this.criticText.setAttribute("y", this.dy(clamp(c[0])) - 8);
      this.criticText.textContent = "the critic, q̂(a)";
      const lo = Math.floor(d.mu), f = d.mu - lo, q = c[lo] + f * ((c[Math.min(90, lo + 1)] ?? c[lo]) - c[lo]);
      const at = (a) => [this.ax(a), this.dy(clamp(q + d.slope * (a - d.mu)))];
      const [x1, y1] = at(d.mu - 9), [x2, y2] = at(d.mu + 9), up = d.slope >= 0, [tx, ty] = up ? [x2, y2] : [x1, y1], [fx, fy] = up ? [x1, y1] : [x2, y2];
      const len = Math.hypot(tx - fx, ty - fy) || 1, ux = (tx - fx) / len, uy = (ty - fy) / len;
      const head = `M${tx.toFixed(1)} ${ty.toFixed(1)}L${(tx - 9 * ux - 4.5 * uy).toFixed(1)} ${(ty - 9 * uy + 4.5 * ux).toFixed(1)}L${(tx - 9 * ux + 4.5 * uy).toFixed(1)} ${(ty - 9 * uy - 4.5 * ux).toFixed(1)}Z`;
      this.slopeEl.setAttribute("d", `M${x1.toFixed(1)} ${y1.toFixed(1)}L${x2.toFixed(1)} ${y2.toFixed(1)}${Math.abs(d.slope) > 0.05 ? head : ""}`);
      this.slopeDot.setAttribute("cx", this.ax(d.mu));
      this.slopeDot.setAttribute("cy", this.dy(clamp(q)));
    }

    // The last throws, all at once (a story shows a moment of a run without replaying them).
    recent(list) {
      this.throws = list.slice(-24);
      this._ticks();
    }

    _arm(angle) {
      const len = 20, x0 = FX - 22, y0 = GY - 32;
      this.arm.setAttribute("x2", (x0 + len * Math.cos(angle * RAD)).toFixed(1));
      this.arm.setAttribute("y2", (y0 - len * Math.sin(angle * RAD)).toFixed(1));
      this.angleText.textContent = `${angle.toFixed(1)}°`;
    }

    // The ball's flight: a parabola from the hand to where it lands (the wind included), rising more steeply the steeper
    // the throw. Heights are drawn a little flattened, so that steep throws stay inside the picture; the arm shows the
    // true angle.
    _flight(angle, land) {
      const x0 = FX - 4, y0 = GY - 32, x1 = this.fx(land), k = Math.tan(Math.max(1, Math.min(89, angle)) * RAD);
      const span = Math.max(4, x1 - x0), cy = Math.max(2 * (12 - 0.25 * y0 - 0.25 * GY), y0 - 0.7 * k * (span / 2));
      return `M${x0} ${y0}Q${(x0 + span / 2).toFixed(1)} ${cy.toFixed(1)} ${(x0 + span).toFixed(1)} ${GY}`;
    }
    _fly(ms) {
      this.flight?.cancel();
      const len = this.arc.getTotalLength(), frames = [];
      for (let i = 0; i <= 24; i++) { const q = this.arc.getPointAtLength((len * i) / 24); frames.push({ transform: `translate(${q.x}px, ${q.y}px)` }); }
      this.flight = this.ball.animate(frames, { duration: ms, easing: "cubic-bezier(.3,.1,.7,1)", fill: "forwards" });
    }
    _ballAt(x, y) { this.flight?.cancel(); this.flight = null; this.ball.style.transform = `translate(${x}px, ${y}px)`; }

    _ticks() {
      this.gTicks.replaceChildren();
      const n = this.throws.length;
      this.throws.forEach((t, i) => {
        const last = i === n - 1;
        el("line", { class: `atick${last ? " last" : ""}`, x1: this.ax(t.angle), x2: this.ax(t.angle), y1: PB, y2: PB - (last ? 14 : 8), style: `opacity:${last ? 1 : (0.15 + 0.6 * (i / n)).toFixed(2)}` }, this.gTicks);
      });
      this.gLand.replaceChildren();
      this.throws.forEach((t, i) => {
        if (i === n - 1) return;
        el("circle", { class: "landed", cx: this.fx(t.land), cy: GY, r: 3, style: `opacity:${(0.15 + 0.5 * (i / n)).toFixed(2)}` }, this.gLand);
      });
      // with a critic: each throw as a point, its angle and how far it flew, the data the critic is fitted to
      this.gPoints.replaceChildren();
      if (this.d.critic) this.throws.forEach((t, i) => {
        el("circle", { class: `pt${i === n - 1 ? " last" : ""}`, cx: this.ax(t.angle), cy: this.dy(Math.max(-2, Math.min(44, t.land))), r: i === n - 1 ? 4 : 3, style: `opacity:${i === n - 1 ? 1 : (0.25 + 0.6 * (i / n)).toFixed(2)}` }, this.gPoints);
      });
    }

    pop(x, y, text, kind = "rew") {
      const t = el("text", { class: `pop k-${kind}`, x, y }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-16px)" }], { duration: 1100, easing: "ease-out" }).onfinish = () => t.remove();
    }

    event(ev, { line = false } = {}) {
      switch (ev.type) {
        case "choose": this._arm(Math.max(-10, Math.min(100, ev.a))); return 0;
        case "move": {
          const land = ev.r, angle = ev.angle ?? ev.a;
          this.throws.push({ angle: ev.a, land });
          if (this.throws.length > 24) this.throws.shift();
          this._ticks();
          const path = this._flight(angle, land);
          this.arc.setAttribute("d", path);
          this.windText.textContent = `wind ${ev.wind >= 0 ? "+" : "−"}${Math.abs(ev.wind).toFixed(1)} m`;
          if (!RL.reducedMotion()) {
            this.arc.animate([{ strokeDashoffset: 1 }, { strokeDashoffset: 0 }], { duration: line ? 700 : 260, easing: "cubic-bezier(.3,.1,.7,1)" });
            this._fly(line ? 700 : 260);
          } else this._ballAt(this.fx(land), GY - 5);
          this.pop(this.fx(land), GY - 12, `${land >= 0 ? "" : "−"}${Math.abs(land).toFixed(1)} m`);
          return line ? 500 : 0;
        }
        case "update": {
          const c = el("circle", { class: "spark k-pol", cx: this.ax(this.d.mu), cy: PB - 10, r: 12 }, this.gFx);
          c.animate([{ opacity: 1, transform: "scale(0.3)" }, { opacity: 0, transform: "scale(1.7)" }], { duration: 600, easing: "ease-out" }).onfinish = () => c.remove();
          return 0;
        }
        default: return 0;
      }
    }

    // After a jump: the throws of the unit before (one, here) drawn at rest.
    rest(events) {
      const move = events.find((e) => e.type === "move");
      this.throws = [];
      if (move) {
        this.throws.push({ angle: move.a, land: move.r });
        this._arm(Math.max(-10, Math.min(100, move.a)));
        this.arc.setAttribute("d", this._flight(move.angle ?? move.a, move.r));
        this._ballAt(this.fx(move.r), GY - 5);
        this.windText.textContent = `wind ${move.wind >= 0 ? "+" : "−"}${Math.abs(move.wind).toFixed(1)} m`;
      } else {
        this.arc.setAttribute("d", "");
        this._ballAt(FX - 4, GY - 32);
        this.windText.textContent = "";
      }
      this._ticks();
    }

    _hover(e) {
      const box = this.svg.getBoundingClientRect(), px = ((e.clientX - box.left) / box.width) * W, py = ((e.clientY - box.top) / box.height) * H;
      if (py < PT - 30 || px < AX || px > AX + AW) { RL.tip.hide(); return; }
      const a = ((px - AX) / AW) * 90, { mu, sd, critic, deterministic } = this.d, dens = sd > 0 ? Math.exp(-0.5 * ((a - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI)) : 0;
      const q = critic ? critic[Math.max(0, Math.min(critic.length - 1, Math.round(a)))] : null;
      RL.tip.show(e.clientX, e.clientY, `<div class="head">A throw at ${a.toFixed(1)}°</div>
        <div class="row"><b>${this.env.distance(a).toFixed(1)} m</b><span>how far it flies, without wind</span></div>
        ${q !== null ? `<div class="row"><b>${q.toFixed(1)} m</b><span>what the critic expects</span></div>` : ""}
        <div class="row"><b>${(100 * dens).toFixed(2)}%</b><span>chance per degree of ${deterministic ? "a throw, the aim plus noise, going" : "the policy throwing"} here</span></div>`);
    }

    destroy() { this.flight?.cancel(); RL.tip.hide(); this.svg.remove(); }

    static options() {
      return [{ key: "truth", type: "check", label: "How far each angle flies (the learner never sees it)", value: true }];
    }

    static thumb(env, d) {
      const w = 200, h = 64, x = (a) => 4 + (a / 90) * (w - 8), y = (m) => h - 4 - (m / 42) * (h - 12);
      let reach = "";
      for (let a = 0; a <= 90; a += 3) reach += `${a ? "L" : "M"}${x(a).toFixed(1)} ${y(env.distance(a)).toFixed(1)}`;
      const top = Math.min(h - 8, 260 / Math.max(0.5, d.sd));
      let bell = `M${x(0)} ${h - 4}`;
      for (let a = 0; a <= 90; a += 1) bell += `L${x(a).toFixed(1)} ${(h - 4 - top * Math.exp(-0.5 * ((a - d.mu) / d.sd) ** 2)).toFixed(1)}`;
      return `<svg class="thumb-throw" viewBox="0 0 ${w} ${h}"><path class="t-truth" d="${reach}"/><path class="t-bell" d="${bell}L${x(90)} ${h - 4}Z"/></svg>`;
    }
  }

  RL.ThrowView = ThrowView;
  (RL.labViews = RL.labViews || {}).throw = ThrowView;
})(globalThis.RL = globalThis.RL || {});
