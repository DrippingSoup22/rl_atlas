/* CartPole view, for recorded runs. On the left, the cart on its track and the pole on the cart: the push the network
   chose, and its numbers for each push (action values for DQN, probabilities for a policy) as bars. On the right, what
   the network thinks of the pole's situations: a map over the pole's angle and spin (the cart at rest in the middle),
   colored by value, with the push it prefers in each region, and the path of the test episode drawn over it. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 330, H = 230, TY = 150, SCALE = 60; // the track's height on the drawing, and pixels per meter
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const deg = (r) => (r * 180) / Math.PI;
  const sgn = (v, d = 1) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(d)}`;

  // The colors of a map, over the range its values span, so that small differences show even when every state is worth
  // about the same: the lowest gray (orange below zero), the highest blue (gray below zero).
  const span = (v) => { let lo = Infinity, hi = -Infinity; for (const x of v) { lo = Math.min(lo, x); hi = Math.max(hi, x); } return [lo, hi]; };
  function rangeColor(box, lo, hi) {
    const css = getComputedStyle(box), mid = css.getPropertyValue("--v-mid"), low = lo < 0 ? css.getPropertyValue("--v-neg") : mid, high = hi > 0 ? css.getPropertyValue("--v-pos") : mid;
    return (v) => `color-mix(in oklab, ${high} ${Math.round(100 * Math.max(0, Math.min(1, (v - lo) / Math.max(1e-9, hi - lo))))}%, ${low})`;
  }
  const num = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)).replace("-", "−");

  class CartPoleView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { path: true, map: "value", ...options };
      this.box = RL.h('<div class="cartview"><div class="track-pane"></div><div class="map-pane"><div class="map-head"><b>What the network thinks</b><span></span></div><div class="map-host"></div></div></div>');
      host.appendChild(this.box);
      const svg = (this.svg = el("svg", { class: "track", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "The cart and the pole" }, this.box.querySelector(".track-pane")));
      // the track, with the limits that end an episode
      const x0 = this.tx(-env.xMax), x1 = this.tx(env.xMax);
      el("line", { class: "rail", x1: x0 - 8, x2: x1 + 8, y1: TY, y2: TY }, svg);
      for (const x of [x0, x1]) el("line", { class: "stop", x1: x, x2: x, y1: TY - 14, y2: TY + 6 }, svg);
      el("text", { class: "track-note", x: x0, y: TY + 20 }, svg).textContent = "−2.4 m";
      el("text", { class: "track-note", x: x1, y: TY + 20, "text-anchor": "end" }, svg).textContent = "+2.4 m";
      // the 12° beyond which the pole counts as fallen, drawn as a fan above the cart
      this.fan = el("path", { class: "fan" }, svg);
      this.cart = el("g", { class: "cart" }, svg);
      el("rect", { class: "cart-body", x: -22, y: -16, width: 44, height: 16, rx: 4 }, this.cart);
      el("circle", { class: "wheel", cx: -13, cy: 1, r: 4.5 }, this.cart);
      el("circle", { class: "wheel", cx: 13, cy: 1, r: 4.5 }, this.cart);
      this.pole = el("line", { class: "pole-rod", x1: 0, y1: -16, x2: 0, y2: -16 - SCALE * env.poleLength }, this.cart);
      el("circle", { class: "pivot", cx: 0, cy: -16, r: 3 }, this.cart);
      this.push = el("path", { class: "push" }, this.cart);
      this.readout = el("text", { class: "readout", x: W - 6, y: 16, "text-anchor": "end" }, svg);
      // the network's numbers for each push, as two bars under the track
      this.bars = [0, 1].map((a) => {
        const g = el("g", { class: "bar-row" }, svg), y = TY + 34 + a * 18;
        el("text", { class: "bar-name", x: 8, y: y + 9 }, g).textContent = env.actionNames[a];
        el("rect", { class: "bar-bg", x: 88, y, width: 190, height: 11, rx: 3 }, g);
        const fill = el("rect", { class: "bar-fill", x: 88, y, width: 0, height: 11, rx: 3 }, g);
        const num = el("text", { class: "bar-num", x: 284, y: y + 9 }, g);
        return { g, fill, num };
      });
      this.gFx = el("g", { class: "fx" }, svg);
      this.mapHost = this.box.querySelector(".map-host");
      this._map();
      this.points = [];
      this.place([0, 0, 0, 0]);
      this.setOptions({});
    }

    tx(x) { return W / 2 + x * SCALE; }

    setOptions(o) {
      Object.assign(this.o, o);
      this.box.dataset.map = this.o.map;
      if (this.d) this.show(this.d);
      this._drawPath();
    }

    // The map: a heat map over angle × angular speed, chevrons for the preferred push, the episode's path.
    _map() {
      const S = 250, P = 36, N = 31;
      this.map = el("svg", { class: "phase", viewBox: `0 0 ${S + P + 8} ${S + P}`, role: "img", "aria-label": "Values and choices over the pole's angle and spin" }, this.mapHost);
      this.cellEls = [];
      const c = S / N, cells = el("g", {}, this.map);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) this.cellEls.push(el("rect", { x: P + i * c, y: (N - 1 - j) * c, width: c + 0.4, height: c + 0.4 }, cells));
      // chevrons on every third cell: the push the network prefers there
      this.chev = [];
      const marks = el("g", { class: "chevrons" }, this.map);
      for (let j = 1; j < N; j += 3) for (let i = 1; i < N; i += 3) this.chev.push({ k: j * N + i, e: el("path", { class: "chev", transform: `translate(${P + (i + 0.5) * c} ${(N - j - 0.5) * c})` }, marks) });
      el("line", { class: "zero", x1: P + S / 2, x2: P + S / 2, y1: 0, y2: S }, this.map);
      el("line", { class: "zero", x1: P, x2: P + S, y1: S / 2, y2: S / 2 }, this.map);
      el("text", { class: "tick", x: P, y: S + 14 }, this.map).textContent = "−12°";
      el("text", { class: "tick", x: P + S, y: S + 14, "text-anchor": "end" }, this.map).textContent = "+12°";
      el("text", { class: "axis-name", x: P + S / 2, y: S + 28, "text-anchor": "middle" }, this.map).textContent = "the pole's angle (right is positive)";
      el("text", { class: "tick", x: P - 4, y: 10, "text-anchor": "end" }, this.map).textContent = "+2";
      el("text", { class: "tick", x: P - 4, y: S, "text-anchor": "end" }, this.map).textContent = "−2";
      el("text", { class: "axis-name", transform: `translate(12 ${S / 2}) rotate(-90)`, "text-anchor": "middle" }, this.map).textContent = "spin (rad/s)";
      this.mapPath = el("path", { class: "phase-path" }, this.map);
      this.mapDot = el("circle", { class: "phase-dot", r: 4.5 }, this.map);
      this.mapXY = (s) => [P + ((Math.max(-0.21, Math.min(0.21, s[2])) + 0.21) / 0.42) * S, S / 2 - (Math.max(-2, Math.min(2, s[3])) / 2) * (S / 2)];
    }

    // What a snapshot knows: values over the grid (DQN: the larger action value; a policy method: its critic), and the
    // push it prefers (DQN: the larger value; a policy: its probabilities, drawn fainter where it is unsure).
    show(d) {
      if (!d || !d.v) return;
      this.d = d;
      this.kind = d.kind;
      const [lo, hi] = span(d.v), color = rangeColor(this.box, lo, hi), valueMode = this.o.map !== "action";
      for (let k = 0; k < this.cellEls.length; k++) {
        const pr = d.act ? d.act[k] : d.mean ? d.mean[k] : 0.5; // 1 or 0: right or left (DQN); P(right) (a policy)
        this.cellEls[k].setAttribute("fill", valueMode ? color(d.v[k]) : `color-mix(in oklab, var(--pol) ${Math.round(Math.abs(pr - 0.5) * 140)}%, var(--v-mid))`);
      }
      for (const { k, e } of this.chev) {
        const pr = d.act ? d.act[k] : d.mean ? d.mean[k] : 0.5, right = pr >= 0.5;
        e.setAttribute("d", right ? "M-3 -4.5l4.5 4.5l-4.5 4.5" : "M3 -4.5l-4.5 4.5l4.5 4.5");
        e.style.opacity = (0.25 + 1.5 * Math.abs(pr - 0.5)).toFixed(2);
      }
      this.box.querySelector(".map-head span").textContent = valueMode
        ? `color: the value of each angle and spin, from ${num(lo)} (${lo < 0 ? "orange" : "gray"}) to ${num(hi)} (${hi > 0 ? "blue" : "gray"}), the cart at rest in the middle; chevrons: the push it prefers`
        : "color and chevrons: the push it prefers, paler where it is unsure; the cart at rest in the middle";
    }

    place(s) {
      const L = SCALE * this.env.poleLength;
      this.cart.setAttribute("transform", `translate(${this.tx(s[0]).toFixed(1)} ${TY - 6})`);
      this.pole.setAttribute("x2", (L * Math.sin(s[2])).toFixed(1));
      this.pole.setAttribute("y2", (-16 - L * Math.cos(s[2])).toFixed(1));
      const lean = Math.abs(s[2]) / this.env.thetaMax;
      this.box.style.setProperty("--lean", Math.min(1, lean).toFixed(2));
      this.fan.setAttribute("d", this._fan(s[0]));
      this.at = s;
    }

    _fan(x) {
      const cx = this.tx(x), cy = TY - 22, r = SCALE * this.env.poleLength + 6, a = this.env.thetaMax;
      return `M${cx} ${cy}L${(cx + r * Math.sin(-a)).toFixed(1)} ${(cy - r * Math.cos(a)).toFixed(1)}A${r} ${r} 0 0 1 ${(cx + r * Math.sin(a)).toFixed(1)} ${(cy - r * Math.cos(a)).toFixed(1)}Z`;
    }

    // The network's numbers for this state: two values (bars scaled to the larger), or two probabilities.
    numbers(x, a, kind) {
      const probs = kind === "pg", vals = probs ? [1 - x[0], x[0]] : [x[0], x[1]];
      const top = probs ? 1 : Math.max(1e-9, Math.abs(vals[0]), Math.abs(vals[1]));
      this.bars.forEach((b, i) => {
        const v = vals[i];
        b.fill.setAttribute("width", (190 * Math.max(0, Math.min(1, Math.abs(v) / top))).toFixed(1));
        b.g.classList.toggle("chosen", i === a);
        b.num.textContent = probs ? `${Math.round(100 * v)}%` : v.toFixed(1);
      });
    }

    pushArrow(a) {
      this.push.setAttribute("d", a ? "M24 -8h16m-5 -5l5 5l-5 5" : "M-24 -8h-16m5 -5l-5 5l5 5");
    }

    _drawPath() {
      const pts = this.o.path ? this.points : [];
      this.mapPath.setAttribute("d", pts.map((s, i) => `${i ? "L" : "M"}${this.mapXY(s).map((v) => v.toFixed(1)).join(" ")}`).join(""));
      if (this.at) { const [x, y] = this.mapXY(this.at); this.mapDot.setAttribute("cx", x); this.mapDot.setAttribute("cy", y); }
    }

    event(ev, { p } = {}) {
      switch (ev.type) {
        case "start":
          this.points = [ev.s];
          this.place(ev.s);
          this.readout.textContent = "step 0";
          this._drawPath();
          return 0;
        case "choose":
          this.pushArrow(ev.a);
          this.numbers(ev.x, ev.a, this.kind);
          return 0;
        case "move":
          this.place(ev.s2);
          this.points.push(ev.s2);
          this._drawPath();
          this.readout.textContent = `step ${ev.k + 1} · angle ${sgn(deg(ev.s2[2]))}°`;
          if (ev.end) {
            // judged by length: a recording clamps its states to the track and to 12°, so the limits are never passed
            const lasted = ev.k + 1 >= this.env.maxSteps, edge = Math.abs(ev.s2[0]) >= this.env.xMax - 0.05;
            this.pop(lasted ? "500 steps: balanced!" : edge ? "off the track" : "the pole fell");
            return 600;
          }
          return 0;
        default:
          return 0;
      }
    }

    pop(text) {
      const t = el("text", { class: "pop", x: W / 2, y: 40, "text-anchor": "middle" }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0 }, { opacity: 1, offset: 0.15 }, { opacity: 0, transform: "translateY(-12px)" }], { duration: 1400, easing: "ease-out" }).onfinish = () => t.remove();
    }

    // At rest after a unit: the last state of its test episode, and its whole path on the map.
    rest(events) {
      const moves = events.filter((e) => e.type === "move"), start = events.find((e) => e.type === "start");
      this.points = start ? [start.s, ...moves.map((e) => e.s2)] : [];
      const last = moves[moves.length - 1];
      this.place(last ? last.s2 : [0, 0, 0, 0]);
      if (last) { this.pushArrow(last.a); this.numbers(last.x, last.a, this.kind); }
      this.readout.textContent = last ? `the test episode lasted ${moves.length} steps` : "";
      this._drawPath();
    }

    destroy() { this.box.remove(); }

    static options() {
      return [
        { key: "map", type: "seg", label: "What the map shows", choices: [["value", "Values"], ["action", "Choices"]], value: "value" },
        { key: "path", type: "check", label: "The path of the test episode", value: true },
      ];
    }

    // A filmstrip frame: the value map, small.
    static thumb(env, d) {
      if (!d?.v) return "";
      const N = 31, c = 4;
      const [lo, hi] = span(d.v);
      let cells = "";
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const t = Math.max(0, Math.min(1, (d.v[j * N + i] - lo) / Math.max(1e-9, hi - lo)));
        cells += `<rect x="${i * c}" y="${(N - 1 - j) * c}" width="${c + 0.3}" height="${c + 0.3}" fill="color-mix(in oklab, var(--v-pos) ${Math.round(100 * t ** 0.8)}%, var(--v-mid))"/>`;
      }
      return `<svg viewBox="0 0 ${N * c} ${N * c}" class="cart-thumb" role="img" aria-label="Values over angle and spin">${cells}</svg><span class="thumb-note">test: ${d.steps} steps</span>`;
    }
  }

  RL.CartPoleView = CartPoleView;
  (RL.labViews = RL.labViews || {}).cartpole = CartPoleView;
})(globalThis.RL = globalThis.RL || {});
