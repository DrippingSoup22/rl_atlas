/* Taxi view: the 5 × 5 city with its walls and four stands, the taxi, the passenger and where the passenger wants to
   go. A table for Taxi has 500 rows, one for every taxi tile, passenger place and destination; the tiles show the 25 of
   them that belong to the trip under way (passenger here, going there): their value as a color, and the action the
   learner thinks best as an arrow, or as "pick up" and "drop off" badges. When the passenger gets in, the tiles switch
   to the values of the second half of the trip. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const T = 72, PAD = 8, N = 5, W = N * T, CAP = 30;
  const ARROWS = ["↓", "↑", "→", "←"];
  const HOP = [{ transform: "scale(1)" }, { transform: "scale(1.12)" }, { transform: "scale(1)" }];
  const SHAKE = [{ transform: "translateX(0)" }, { transform: "translateX(-5px)" }, { transform: "translateX(5px)" }, { transform: "translateX(0)" }];
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
  const signed = (v, d = 1) => `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(v).toFixed(d)}`;
  const cx = (c) => c * T + T / 2, cy = (r) => r * T + T / 2;

  class TaxiView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { arrows: true, numbers: false, ...options };
      this.Q = new Float64Array(env.nS * env.nA);
      this.trip = { pass: 0, dest: 3 };
      this.svg = el("svg", { class: "taxiview", viewBox: `${-PAD} ${-PAD} ${W + 2 * PAD} ${W + CAP + 2 * PAD}`, role: "img", "aria-label": env.title });
      this.svg.style.maxWidth = "460px";
      host.appendChild(this.svg);
      const g = (cls) => el("g", { class: cls }, this.svg);
      this.gTiles = g("tiles");
      this.gStands = g("stands");
      this.gWalls = g("walls");
      this.gArrows = g("arrows");
      this.gNums = g("nums");
      this.gCar = g("car-layer");
      this.gFx = g("fx");
      this.tiles = [];
      this.marks = [];
      this.nums = [];
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        this.tiles.push(el("rect", { class: "tile", x: c * T + 1, y: r * T + 1, width: T - 2, height: T - 2, rx: 6 }, this.gTiles));
        this.marks.push(el("text", { class: "best", x: cx(c), y: cy(r) + 7 }, this.gArrows));
        this.nums.push(el("text", { class: "num", x: cx(c), y: r * T + T - 8 }, this.gNums));
      }
      // the stands: a colored corner and its letter
      env.stands.forEach(([r, c], i) => {
        const s = el("g", { class: `stand k${i}` }, this.gStands);
        el("rect", { x: c * T + 3, y: r * T + 3, width: 22, height: 22, rx: 5 }, s);
        el("text", { x: c * T + 14, y: r * T + 19 }, s).textContent = env.standNames[i];
      });
      // the walls, read off Gymnasium's map: "|" between two tiles
      el("rect", { class: "edge", x: 0, y: 0, width: W, height: W, rx: 4 }, this.gWalls);
      for (let r = 0; r < N; r++) for (let c = 0; c < N - 1; c++) {
        if (!env.openEast(r, c)) el("line", { class: "wall", x1: (c + 1) * T, x2: (c + 1) * T, y1: r * T, y2: (r + 1) * T }, this.gWalls);
      }
      // the destination flag, the passenger, the taxi
      this.flag = el("g", { class: "flag" }, this.gCar);
      el("line", { x1: 0, y1: 10, x2: 0, y2: -16 }, this.flag);
      el("path", { d: "M0 -16 L14 -11 L0 -6 Z" }, this.flag);
      this.car = el("g", { class: "car" }, this.gCar);
      this.carBody = el("g", { class: "car-body" }, this.car);
      el("rect", { class: "shell", x: -20, y: -11, width: 40, height: 20, rx: 7 }, this.carBody);
      el("rect", { class: "roof", x: -9, y: -17, width: 18, height: 8, rx: 3 }, this.carBody);
      el("circle", { class: "wheel", cx: -11, cy: 10, r: 4 }, this.carBody);
      el("circle", { class: "wheel", cx: 11, cy: 10, r: 4 }, this.carBody);
      this.rider = el("g", { class: "rider" }, this.gCar);
      el("circle", { cx: 0, cy: -8, r: 5 }, this.rider);
      el("path", { d: "M-7 8 Q0 -4 7 8 Z" }, this.rider);
      this.caption = el("text", { class: "trip", x: W / 2, y: W + 22 }, this.svg);
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
      this.setOptions({});
      this.place(env.encode(2, 2, 0, 3), true);
    }

    setOptions(o) {
      Object.assign(this.o, o);
      this.gArrows.style.display = this.o.arrows ? "" : "none";
      this.gNums.style.display = this.o.numbers ? "" : "none";
      this.car.classList.toggle("gone", this.o.agent === false);
      this._draw();
    }

    // What the learner knows: its action values (or its state values and policy).
    show(d) {
      if (d.Q) this.Q = d.Q;
      this.V = d.V || null;
      this.P = d.P || null;
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
      const nA = this.env.nA, src = this.P || this.Q;
      let b = 0;
      for (let a = 1; a < nA; a++) if (src[s * nA + a] > src[s * nA + b] + 1e-12) b = a;
      const flat = !this.P && this.Q.subarray(s * nA, s * nA + nA).every((q) => q === this.Q[s * nA]);
      return flat ? -1 : b;
    }

    // The 25 tiles of the trip under way.
    _draw() {
      const { pass, dest } = this.trip, env = this.env, range = env.valueRange;
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const i = r * N + c, s = env.encode(r, c, pass, dest), v = this.value(s), b = this.best(s);
        this.tiles[i].style.fill = valueColor(v, range);
        const m = this.marks[i];
        m.textContent = b < 0 ? "" : b < 4 ? ARROWS[b] : b === 4 ? "pick up" : "drop off";
        m.classList.toggle("word", b >= 4);
        this.nums[i].textContent = signed(v);
      }
      const name = env.standNames;
      this.caption.textContent = this.trip.delivered ? `Delivered at ${name[dest]} · tiles: the ride to ${name[dest]}` : pass === env.IN_TAXI ? `Passenger aboard, going to ${name[dest]}` : `Passenger waiting at ${name[pass]}, going to ${name[dest]}`;
    }

    // Put everything where state s says, and switch the tiles to its trip.
    place(s, instant = false) {
      const d = this.env.decode(s), [dr, dc] = this.env.stands[d.dest];
      if (instant) this.car.classList.add("instant");
      this.car.style.transform = `translate(${cx(d.col)}px, ${cy(d.row) + 4}px)`;
      if (instant) { this.car.getBoundingClientRect(); this.car.classList.remove("instant"); }
      this.flag.style.transform = `translate(${dc * T + T - 16}px, ${dr * T + 24}px)`;
      const aboard = d.pass === this.env.IN_TAXI, delivered = this.env.terminal(s);
      this.rider.classList.toggle("delivered", delivered);
      this.rider.classList.toggle("aboard", aboard);
      if (aboard) this.rider.style.transform = `translate(${cx(d.col)}px, ${cy(d.row) - 18}px) scale(0.75)`;
      else { const [pr, pc] = this.env.stands[d.pass]; this.rider.style.transform = `translate(${pc * T + 44}px, ${pr * T + 30}px)`; }
      // once delivered, the tiles keep showing the last half of the trip: the delivered states are all worth 0
      const trip = { pass: delivered ? this.env.IN_TAXI : d.pass, dest: d.dest, delivered };
      if (trip.pass !== this.trip.pass || trip.dest !== this.trip.dest || trip.delivered !== this.trip.delivered) { this.trip = trip; this._draw(); }
      this.at = s;
    }

    pop(s, text, kind = "rew") {
      const d = this.env.decode(s), t = el("text", { class: `pop k-${kind}`, x: cx(d.col), y: cy(d.row) - 24 }, this.gFx);
      t.textContent = text;
      t.animate([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-16px)" }], { duration: 950, easing: "ease-out" }).onfinish = () => t.remove();
    }
    spark(s) {
      const d = this.env.decode(s), c = el("circle", { class: "spark", cx: cx(d.col), cy: cy(d.row), r: 16 }, this.gFx);
      c.animate([{ opacity: 1, transform: "scale(0.3)" }, { opacity: 0, transform: "scale(1.6)" }], { duration: 600, easing: "ease-out" }).onfinish = () => c.remove();
    }

    event(ev, { line = false } = {}) {
      const motion = !RL.reducedMotion();
      switch (ev.type) {
        case "start": this.place(ev.s, true); return 0;
        case "move": {
          const a = ev.a;
          this.place(ev.s2);
          if (a >= 4) {
            if (ev.r === 20) { this.pop(ev.s2, "+20", "rew"); if (motion) this.carBody.animate(HOP, { duration: 320 }); return 300; }
            if (ev.r <= -10) { this.pop(ev.s, "−10", "err"); if (motion) this.carBody.animate(SHAKE, { duration: 280 }); return 200; }
            if (motion) this.rider.animate(HOP, { duration: 260 });
            return 120;
          }
          if (ev.s2 === ev.s && motion) this.carBody.animate(SHAKE, { duration: 220 }); // a wall, or the edge of the city
          return 0;
        }
        case "update": this.spark(ev.s ?? this.at); return line ? 120 : 0;
        default: return 0;
      }
    }

    // After a jump to another moment: the taxi where the last episode ended.
    rest(events) {
      const moves = events.filter((e) => e.type === "move");
      const first = events.find((e) => e.type === "start");
      if (moves.length) this.place(moves[moves.length - 1].s2, true);
      else if (first) this.place(first.s, true);
    }

    _hover(e) {
      const box = this.svg.getBoundingClientRect(), scale = (W + 2 * PAD) / box.width;
      const x = (e.clientX - box.left) * scale - PAD, y = (e.clientY - box.top) * scale - PAD;
      const r = Math.floor(y / T), c = Math.floor(x / T);
      if (r < 0 || c < 0 || r >= N || c >= N) { RL.tip.hide(); return; }
      const s = this.env.encode(r, c, this.trip.pass, this.trip.dest), nA = this.env.nA;
      const rows = this.env.actionNames.map((name, a) => `<div class="row"><b>${signed(this.Q[s * nA + a], 2)}</b><span>${name}</span></div>`).join("");
      RL.tip.show(e.clientX, e.clientY, `<div class="head">Row ${r + 1}, column ${c + 1}, this trip</div>${rows}`);
    }

    destroy() { RL.tip.hide(); this.svg.remove(); }

    static options() {
      return [
        { key: "arrows", type: "check", label: "The best action in each tile", value: true },
        { key: "numbers", type: "check", label: "Values as numbers", value: false },
      ];
    }

    static thumb(env, d) {
      const w = 5 * 12, cells = [];
      const nA = env.nA, val = (s) => (d.V ? d.V[s] : Math.max(...Array.from({ length: nA }, (_, a) => d.Q ? d.Q[s * nA + a] : 0)));
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) cells.push(`<rect x="${c * 12}" y="${r * 12}" width="11" height="11" rx="2" style="fill:${valueColor(val(env.encode(r, c, 0, 3)), env.valueRange)}"/>`);
      return `<svg class="thumb-taxi" viewBox="0 0 ${w} ${w}">${cells.join("")}</svg>`;
    }
  }

  RL.TaxiView = TaxiView;
  (RL.labViews = RL.labViews || {}).taxi = TaxiView;
})(globalThis.RL = globalThis.RL || {});
