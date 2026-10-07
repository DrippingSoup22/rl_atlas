/* Car view: Mountain Car, shared by the Lab and the stories. On the left, the valley seen from the side: the car, the
   throttle it chose, and the flag at the top of the right hill. On the right, what the learner thinks of every state:
   the cost-to-go, −max_a q̂(s, a), the number of steps it expects before reaching the flag, over position and speed.
   Seen in 3D (drag to turn it) it is a landscape, as in Sutton & Barto's Figure 10.1; seen from above, a map. The
   states of the current episode are drawn on it: an episode that swings back and forth spirals outward. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 330, H = 230, M = 22, N = 31; // the hill drawing, and the grid of states the surface is computed on
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const nice = (v) => { for (const n of [10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000]) if (v <= n) return n; return Math.ceil(v / 1000) * 1000; };

  // The cost-to-go of every state of an N × N grid over position and speed, from the weights.
  function costs(env, d) {
    const Z = new Float64Array(N * N), F = d.F, n = F.n;
    let top = 0;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const s = [env.xMin + ((env.xMax - env.xMin) * i) / (N - 1), -env.vMax + (2 * env.vMax * j) / (N - 1)], x = F.of(s);
      // action values: the best one; a policy method: its critic's value (one copy of the weights, not one per action)
      let best = -Infinity;
      if (d.theta) best = lab().dot(d.w, x);
      else for (let a = 0; a < env.nA; a++) best = Math.max(best, lab().dot(d.w, x, a * n));
      // a learner whose step size is too large diverges: its runaway estimates are drawn at the top of a tall scale
      Z[j * N + i] = Number.isFinite(best) ? Math.min(1e6, Math.max(0, -best)) : 1e6;
      top = Math.max(top, Z[j * N + i]);
    }
    return { Z, top };
  }
  const lab = () => RL.lab;

  class CarView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { look: "3d", path: true, ...options };
      this.box = RL.h('<div class="carview"><div class="hill-pane"></div><div class="map-pane"><div class="map-head"><b>Cost-to-go</b><span>steps the learner expects before the flag</span></div><div class="map-host"></div></div></div>');
      host.appendChild(this.box);
      this.svg = el("svg", { class: "hill", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "The valley and the car" }, this.box.querySelector(".hill-pane"));
      // the ground
      let d = "";
      for (let k = 0; k <= 80; k++) { const x = env.xMin + ((env.xMax - env.xMin) * k) / 80; d += `${k ? "L" : "M"}${this.hx(x).toFixed(1)} ${this.hy(x).toFixed(1)}`; }
      el("path", { class: "ground", d: `${d}L${W - M} ${H}L${M} ${H}Z` }, this.svg);
      el("path", { class: "ground-line", d }, this.svg);
      const fx = this.hx(env.xMax), fy = this.hy(env.xMax);
      el("line", { class: "pole", x1: fx, x2: fx, y1: fy, y2: fy - 30 }, this.svg);
      el("path", { class: "flag", d: `M${fx} ${fy - 30}l16 5l-16 5z` }, this.svg);
      el("text", { class: "hill-note", x: M, y: H - 8 }, this.svg).textContent = "gravity is stronger than the engine: back up first";
      this.car = el("g", { class: "car" }, this.svg);
      this.body = el("g", { class: "car-body" }, this.car);
      el("rect", { class: "car-shell", x: -13, y: -15, width: 26, height: 10, rx: 4 }, this.body);
      el("rect", { class: "car-top", x: -7, y: -21, width: 13, height: 7, rx: 3 }, this.body);
      el("circle", { class: "wheel", cx: -8, cy: -4, r: 4 }, this.body);
      el("circle", { class: "wheel", cx: 8, cy: -4, r: 4 }, this.body);
      this.throttle = el("path", { class: "throttle" }, this.body);
      this.speedo = el("text", { class: "speedo", x: W - M, y: 18, "text-anchor": "end" }, this.svg);
      this.gFx = el("g", { class: "fx" }, this.svg);
      this.mapHost = this.box.querySelector(".map-host");
      this.top = null;
      this.points = [];
      this.setOptions({});
      this.place(env.reset(null, [-0.5, 0]), true);
    }

    hx(x) { return M + ((x - this.env.xMin) / (this.env.xMax - this.env.xMin)) * (W - 2 * M); }
    hy(x) { return 70 + (1 - this.env.height(x)) * 0.5 * (H - 110); }

    setOptions(o) {
      Object.assign(this.o, o);
      if (this.look !== this.o.look) {
        this.look = this.o.look;
        this.surface?.destroy();
        this.surface = null;
        this.mapHost.replaceChildren();
        if (this.look === "3d") {
          this.surface = new RL.Surface3D(this.mapHost, {
            x: [this.env.xMin, this.env.xMax], y: [-this.env.vMax, this.env.vMax], xLabel: "position", yLabel: "speed", zLabel: "steps to go",
            xTicks: [-1.2, 0.5], yTicks: [-0.07, 0.07], fy: (v) => v.toFixed(2),
          });
        } else this._mapSvg();
        if (this.d) this.show(this.d);
      }
      this._drawPath();
    }

    // The map seen from above: a heat map of the cost-to-go, with the path over it.
    _mapSvg() {
      const S = 260, P = 34;
      this.map = el("svg", { class: "phase", viewBox: `0 0 ${S + P + 8} ${S + P}`, role: "img", "aria-label": "Cost-to-go over position and speed" }, this.mapHost);
      this.cells = el("g", {}, this.map);
      const c = S / N;
      this.cellEls = [];
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) this.cellEls.push(el("rect", { x: P + i * c, y: (N - 1 - j) * c, width: c + 0.4, height: c + 0.4 }, this.cells));
      el("text", { class: "tick", x: P, y: S + 14 }, this.map).textContent = "−1.2";
      el("text", { class: "tick", x: P + S, y: S + 14, "text-anchor": "end" }, this.map).textContent = "0.5";
      el("text", { class: "axis-name", x: P + S / 2, y: S + 28, "text-anchor": "middle" }, this.map).textContent = "position";
      el("text", { class: "tick", x: P - 4, y: 10, "text-anchor": "end" }, this.map).textContent = "0.07";
      el("text", { class: "tick", x: P - 4, y: S, "text-anchor": "end" }, this.map).textContent = "−0.07";
      el("text", { class: "axis-name", transform: `translate(12 ${S / 2}) rotate(-90)`, "text-anchor": "middle" }, this.map).textContent = "speed";
      el("line", { class: "zero", x1: P, x2: P + S, y1: S / 2, y2: S / 2 }, this.map);
      this.mapPath = el("path", { class: "phase-path" }, this.map);
      this.mapDot = el("circle", { class: "phase-dot", r: 4.5 }, this.map);
      this.mapXY = (s) => [P + ((s[0] - this.env.xMin) / (this.env.xMax - this.env.xMin)) * S, S / 2 - (s[1] / this.env.vMax) * (S / 2)];
    }

    show(d) {
      if (!d.w) return;
      this.d = d;
      const { Z, top } = costs(this.env, d), zMax = nice(Math.max(10, top));
      this.Z = Z;
      this.zMax = zMax;
      if (this.surface) this.surface.set(Z, N, N, zMax);
      else if (this.cellEls) {
        const lo = getComputedStyle(this.box).getPropertyValue("--v-mid"), hi = getComputedStyle(this.box).getPropertyValue("--v-neg");
        for (let k = 0; k < Z.length; k++) this.cellEls[k].setAttribute("fill", `color-mix(in oklab, ${hi} ${Math.round(100 * Math.min(1, Z[k] / zMax) ** 0.85)}%, ${lo})`);
      }
      this.box.querySelector(".map-head span").textContent = `steps the learner expects before the flag (top of the scale: ${zMax})`;
    }

    place(s, instant = false) {
      const x = s[0], ang = Math.atan2(-(this.hy(x + 0.01) - this.hy(x - 0.01)), this.hx(x + 0.01) - this.hx(x - 0.01));
      if (instant) this.car.classList.add("instant");
      this.car.style.transform = `translate(${this.hx(x)}px, ${this.hy(x)}px) rotate(${(-ang * 180) / Math.PI}deg)`;
      if (instant) { this.car.getBoundingClientRect(); this.car.classList.remove("instant"); }
      this.speedo.textContent = `position ${s[0] < 0 ? "−" : ""}${Math.abs(s[0]).toFixed(2)} · speed ${s[1] >= 0 ? "+" : "−"}${Math.abs(s[1]).toFixed(3)}`;
      this.at = s;
    }

    // The throttle as an arrow on the car: backward, none (a dot), forward.
    gas(a) {
      this.throttle.setAttribute("d", a === 1 ? "M-2 -10a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" : a === 2 ? "M14 -10h9m-4 -4l4 4l-4 4" : "M-14 -10h-9m4 -4l-4 4l4 4");
    }

    _drawPath() {
      const pts = this.o.path ? this.points.map((s) => [s[0], s[1]]) : [];
      if (this.surface) { this.surface.path(pts); this.surface.dot(this.at ? [this.at[0], this.at[1]] : null); }
      else if (this.mapPath) {
        this.mapPath.setAttribute("d", pts.map((s, i) => `${i ? "L" : "M"}${this.mapXY(s).map((v) => v.toFixed(1)).join(" ")}`).join(""));
        if (this.at) { const [x, y] = this.mapXY(this.at); this.mapDot.setAttribute("cx", x); this.mapDot.setAttribute("cy", y); }
      }
    }

    event(ev) {
      switch (ev.type) {
        case "start": this.points = [ev.s]; this.place(ev.s, true); this._drawPath(); return 0;
        case "choose": this.gas(ev.a); return 0;
        case "move":
          this.place(ev.s2);
          this.points.push(ev.s2);
          this._drawPath();
          if (this.env.terminal(ev.s2)) { this.pop("the flag!"); return 500; }
          return 0;
        default: return 0;
      }
    }

    pop(text) {
      const t = el("text", { class: "pop", x: this.hx(this.env.xMax) - 6, y: this.hy(this.env.xMax) - 40 }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0 }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-14px)" }], { duration: 1100, easing: "ease-out" }).onfinish = () => t.remove();
    }

    rest(events) {
      const moves = events.filter((e) => e.type === "move"), start = events.find((e) => e.type === "start");
      this.points = start ? [start.s, ...moves.map((e) => e.s2)] : [];
      this.place(moves.length ? moves[moves.length - 1].s2 : [-0.5, 0], true);
      this.gas(1);
      this._drawPath();
    }

    destroy() { this.surface?.destroy(); this.box.remove(); }

    static options() {
      return [
        { key: "look", type: "seg", label: "How to draw the cost-to-go", choices: [["3d", "Landscape (3D)"], ["map", "Map (from above)"]], value: "3d" },
        { key: "path", type: "check", label: "The states of this episode", value: true },
      ];
    }

    static thumb(env, d) {
      const { Z, top } = costs(env, d);
      return RL.Surface3D.svg(Z, N, N, nice(Math.max(10, top)), { w: 200, h: 120 });
    }
  }

  RL.CarView = CarView;
  (RL.labViews = RL.labViews || {}).car = CarView;
})(globalThis.RL = globalThis.RL || {});
