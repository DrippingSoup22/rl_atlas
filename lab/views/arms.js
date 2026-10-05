/* Arms view: the k slot machines of a bandit, shared by the Lab and the stories.
   Each column is one arm: its machine on top, then on the value scale the true payout (a bell curve the agent never
   sees, with its mean), the agent's estimate Q (a bar), and for UCB the optimistic bound above it. Below, how likely
   the policy is to pull the arm, and how often it has been pulled. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const NS = "http://www.w3.org/2000/svg";
  const CW = 56, L = 40, TOP = 52, PH = 156, PI = 34, H = TOP + PH + PI + 34;
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const fmt = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  const signed = (v, d = 2) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(d)}`;
  const PULL = [{ transform: "rotate(0)" }, { transform: "rotate(52deg)", offset: 0.35 }, { transform: "rotate(0)" }];
  const SHAKE = [{ transform: "translateY(0)" }, { transform: "translateY(2px)", offset: 0.3 }, { transform: "translateY(-1px)", offset: 0.6 }, { transform: "translateY(0)" }];

  class ArmsView {
    // options: truth (draw the true payouts), numbers (estimates as numbers), epsilon (for the policy of ε-greedy)
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { truth: true, numbers: false, epsilon: 0, ...options };
      this.k = env.k;
      this.lo = env.mean - 3.3;
      this.hi = env.mean + 3.3;
      this.d = { Q: new Float64Array(this.k), N: new Float64Array(this.k) };
      const W = L + this.k * CW + 8;
      this.svg = el("svg", { class: "armsview", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": env.title });
      host.appendChild(this.svg);
      this.gAxis = el("g", { class: "axis" }, this.svg);
      for (let v = Math.ceil(this.lo); v <= Math.floor(this.hi); v++) {
        el("line", { class: v === 0 ? "zero" : "grid", x1: L - 4, x2: W - 8, y1: this.y(v), y2: this.y(v) }, this.gAxis);
        el("text", { class: "tick", x: L - 8, y: this.y(v) + 4, "text-anchor": "end" }, this.gAxis).textContent = v;
      }
      el("text", { class: "axis-name", x: 4, y: TOP + PH + PI / 2 + 4 }, this.gAxis).textContent = "π";
      this.baseEl = el("g", { class: "baseline" }, this.svg);
      el("line", { x1: L, x2: W - 8 }, this.baseEl);
      el("text", { x: W - 10, "text-anchor": "end" }, this.baseEl).textContent = "baseline R̄";
      this.arms = [];
      for (let a = 0; a < this.k; a++) this._arm(a);
      this.gFx = el("g", { class: "fx" }, this.svg);
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
      this.setOptions({});
      this._truth();
    }

    y(v) { return TOP + ((this.hi - Math.max(this.lo, Math.min(this.hi, v))) / (this.hi - this.lo)) * PH; }
    cx(a) { return L + a * CW + CW / 2; }

    _arm(a) {
      const g = el("g", { class: "arm", transform: `translate(${this.cx(a)} 0)` }, this.svg), A = { g };
      A.lever = el("g", { class: "lever", transform: "translate(15 14)" }, g);
      el("line", { x1: 0, y1: 0, x2: 0, y2: -10 }, A.lever);
      el("circle", { cx: 0, cy: -11, r: 3.2 }, A.lever);
      A.box = el("g", { class: "machine" }, g);
      el("rect", { x: -16, y: 8, width: 30, height: 34, rx: 7 }, A.box);
      el("rect", { class: "window", x: -11, y: 14, width: 20, height: 13, rx: 3 }, A.box);
      el("text", { class: "arm-no", x: -1, y: 24.5 }, A.box).textContent = a + 1;
      A.violin = el("path", { class: "violin" }, g);
      A.mean = el("line", { class: "truth", x1: -12, x2: 12 }, g);
      A.bound = el("path", { class: "bound" }, g);
      A.est = el("g", { class: "est" }, g);
      el("rect", { x: -13, y: -2.5, width: 26, height: 5, rx: 2.5 }, A.est);
      A.num = el("text", { class: "est-num", x: 0, y: -7 }, A.est);
      A.pi = el("rect", { class: "pi", x: -11, width: 22, rx: 3 }, g);
      A.piText = el("text", { class: "pi-num", x: 0 }, g);
      A.count = el("text", { class: "count", x: 0, y: H - 8 }, g);
      A.star = el("text", { class: "star", x: -22, y: 16 }, g); // by the machine, clear of the bars and their labels
      A.star.textContent = "★";
      this.arms[a] = A;
    }

    setOptions(o) {
      Object.assign(this.o, o);
      this.svg.toggleAttribute("data-truth", !!this.o.truth);
      this.svg.toggleAttribute("data-numbers", !!this.o.numbers);
      this.svg.toggleAttribute("data-hide-learner", this.o.learner === false);
    }

    // Outline one arm (a < 0: none).
    mark(a) { this.arms.forEach((A, i) => A.g.classList.toggle("focus", i === a)); }

    // The true payouts: a normal curve around each arm's mean q*(a).
    _truth() {
      const q = this.env.q, best = this.env.best;
      this.arms.forEach((A, a) => {
        let right = "", left = "";
        for (let i = 0; i <= 24; i++) {
          const v = q[a] - 2.6 + (5.2 * i) / 24, w = 15 * Math.exp(-0.5 * (v - q[a]) ** 2);
          right += `${i ? "L" : "M"}${w.toFixed(1)} ${this.y(v).toFixed(1)}`;
          left = `L${(-w).toFixed(1)} ${this.y(v).toFixed(1)}${left}`;
        }
        A.violin.setAttribute("d", `${right}${left}Z`);
        A.mean.setAttribute("y1", this.y(q[a]));
        A.mean.setAttribute("y2", this.y(q[a]));
        A.g.classList.toggle("best", a === best);
      });
    }

    // What the learner knows: Q and N (action values), U (UCB's bounds), or pi and the baseline (gradient bandit).
    show(d, p = {}) {
      this.d = d;
      this.eps = p.epsilon ?? 0;
      this._truth();
      const pi = this.policy();
      this.arms.forEach((A, a) => {
        const hasQ = !!d.Q;
        A.est.style.display = hasQ ? "" : "none";
        if (hasQ) {
          A.est.style.transform = `translateY(${this.y(d.Q[a])}px)`;
          A.num.textContent = fmt(d.Q[a]);
        }
        if (d.U && Number.isFinite(d.U[a])) A.bound.setAttribute("d", `M0 ${this.y(d.Q[a]) - 3} V${this.y(d.U[a])} M-6 ${this.y(d.U[a])} H6`);
        else A.bound.setAttribute("d", d.U ? `M0 ${this.y(d.Q[a]) - 3} V${TOP} M-6 ${TOP} H6` : "");
        const h = Math.max(1, pi[a] * PI);
        A.pi.setAttribute("y", TOP + PH + PI + 4 - h);
        A.pi.setAttribute("height", h);
        A.piText.setAttribute("y", TOP + PH + PI + 1 - h);
        A.piText.textContent = pi[a] >= 0.995 ? "100%" : pi[a] >= 0.1 ? `${Math.round(pi[a] * 100)}%` : "";
        A.count.textContent = d.N ? `n ${d.N[a]}` : "";
      });
      this.baseEl.style.display = d.baseline !== undefined ? "" : "none";
      if (d.baseline !== undefined) this.baseEl.style.transform = `translateY(${this.y(d.baseline)}px)`;
    }

    // The probability the learner pulls each arm next.
    policy() {
      const d = this.d;
      if (d.pi) return Array.from(d.pi);
      if (d.U) {
        const top = Math.max(...d.U), ties = Array.from(d.U).filter((u) => u === top).length;
        return Array.from(d.U, (u) => (u === top ? 1 / ties : 0));
      }
      return lab.epsGreedyProbs(d.Q, 0, this.env, this.eps);
    }

    // ---- the Lab: one event, and the picture at rest ----
    event(ev, { line = false } = {}) {
      const A = this.arms[ev.a];
      if (!A) return 0;
      if (ev.type === "choose") {
        A.g.classList.remove("chosen");
        A.g.getBoundingClientRect();
        A.g.classList.add("chosen");
        clearTimeout(A.timer);
        A.timer = setTimeout(() => A.g.classList.remove("chosen"), 900);
        if (ev.explore) this._tag(ev.a, "explore", "explore");
        return 0;
      }
      if (ev.type === "move") {
        if (!RL.reducedMotion()) {
          A.lever.animate(PULL, { duration: 380, easing: "ease-out" });
          A.box.animate(SHAKE, { duration: 380, easing: "ease-out" });
        }
        this._tag(ev.a, signed(ev.r), ev.r >= 0 ? "rew" : "rew neg");
        const dot = el("circle", { class: "sample", cx: this.cx(ev.a), cy: this.y(ev.r), r: 4.5 }, this.gFx);
        dot.animate([{ opacity: 1, transform: "scale(1.3)" }, { opacity: 0.7, offset: 0.5 }, { opacity: 0, transform: "scale(0.8)" }], { duration: line ? 1500 : 900, easing: "ease-out" }).onfinish = () => dot.remove();
        return line ? 200 : 0;
      }
      if (ev.type === "update") {
        const c = el("circle", { class: "spark", cx: this.cx(ev.a), cy: this.d.Q ? this.y(this.d.Q[ev.a]) : TOP + PH + PI / 2, r: 14 }, this.gFx);
        c.animate([{ opacity: 1, transform: "scale(0.3)" }, { opacity: 0, transform: "scale(1.6)" }], { duration: 600, easing: "ease-out" }).onfinish = () => c.remove();
      }
      return 0;
    }

    _tag(a, text, kind) {
      const t = el("text", { class: `pop k-${kind}`, x: this.cx(a), y: 6 }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-6px)" }], { duration: 1000, easing: "ease-out" }).onfinish = () => t.remove();
    }

    rest(events) {
      const move = events.find((e) => e.type === "move");
      for (const A of this.arms) A.g.classList.remove("last");
      if (move) this.arms[move.a].g.classList.add("last");
    }

    _hover(e) {
      const box = this.svg.getBoundingClientRect(), x = ((e.clientX - box.left) / box.width) * (L + this.k * CW + 8);
      const a = Math.floor((x - L) / CW);
      if (a < 0 || a >= this.k) { RL.tip.hide(); return; }
      const d = this.d, pi = this.policy();
      const row = (v, text) => `<div class="row"><b>${v}</b><span>${text}</span></div>`;
      RL.tip.show(e.clientX, e.clientY, `<div class="head">Arm ${a + 1}${a === this.env.best ? " · the best arm" : ""}</div>` +
        row(fmt(this.env.q[a]), "true mean payout, q*(a)") +
        (d.Q ? row(fmt(d.Q[a]), "estimate Q(a)") : "") +
        (d.U ? row(Number.isFinite(d.U[a]) ? fmt(d.U[a]) : "∞", "upper bound: Q + bonus") : "") +
        (d.H ? row(fmt(d.H[a]), "preference H(a)") : "") +
        row(`${(pi[a] * 100).toFixed(1)}%`, "chance of being pulled next") +
        (d.N ? row(d.N[a], "pulls so far") : ""));
    }

    destroy() { RL.tip.hide(); this.svg.remove(); }

    static options() {
      return [
        { key: "truth", type: "check", label: "True payouts (the agent never sees them)", value: true },
        { key: "numbers", type: "check", label: "Estimates as numbers", value: false },
      ];
    }

    static legend() {
      return '<p class="arms-legend"><i class="lg-est"></i>estimate Q · <i class="lg-truth"></i>true mean · <i class="lg-pi"></i>chance of being pulled</p>';
    }

    static thumb(env, d) {
      const k = env.k, S = 15, lo = env.mean - 3, hi = env.mean + 3, Hh = 64, y = (v) => ((hi - Math.max(lo, Math.min(hi, v))) / (hi - lo)) * Hh;
      let g = `<line class="t-zero" x1="0" x2="${k * S}" y1="${y(env.mean)}" y2="${y(env.mean)}"/>`;
      for (let a = 0; a < k; a++) {
        const x = a * S + S / 2;
        g += `<line class="t-truth" x1="${x - 5}" x2="${x + 5}" y1="${y(env.q[a])}" y2="${y(env.q[a])}"/>`;
        if (d.Q) g += `<rect class="t-est" x="${x - 4.5}" y="${y(d.Q[a]) - 2}" width="9" height="4" rx="2"/>`;
        if (d.pi) g += `<rect class="t-pi" x="${x - 4}" y="${Hh + 18 - 16 * d.pi[a]}" width="8" height="${16 * d.pi[a] + 0.5}" rx="1.5"/>`;
        else if (d.N) { const n = d.N[a] / Math.max(1, ...d.N); g += `<rect class="t-pi" x="${x - 4}" y="${Hh + 18 - 16 * n}" width="8" height="${16 * n + 0.5}" rx="1.5"/>`; }
      }
      return `<svg class="thumb-arms" viewBox="0 0 ${k * S} ${Hh + 20}">${g}</svg>`;
    }
  }

  RL.ArmsView = ArmsView;
  (RL.labViews = RL.labViews || {}).bandit = ArmsView;
})(globalThis.RL = globalThis.RL || {});
