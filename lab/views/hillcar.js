/* Continuous Mountain Car view, for recorded runs. On the left, the valley seen from the side: the car, the force the
   policy chose as an arrow on it, and as bars the force and where the policy aims (its mean, with its spread). On the
   right, what the critic thinks of the car's states: a map over position and speed, colored by value or by the
   policy's push, with the path of the test episode drawn over it. An episode that swings back and forth spirals
   outward on the map, until it leaves on the right, at the flag. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 330, H = 168, M = 16; // the drawing and its margin (pixels)
  const X0 = -1.2, X1 = 0.6, GOAL = 0.45, V = 0.07; // Gymnasium's track, the flag, the top speed
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const height = (x) => Math.sin(3 * x) * 0.45 + 0.55; // the hill as Gymnasium draws it
  const hx = (x) => M + ((x - X0) / (X1 - X0)) * (W - 2 * M);
  const hy = (x) => 28 + (1 - height(x)) * (H - 52);
  const sgn = (v, d = 2) => { const t = Math.abs(v).toFixed(d); return `${v < 0 && +t ? "−" : "+"}${t}`; };
  const num = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)).replace("-", "−");
  const span = (v) => { let lo = Infinity, hi = -Infinity; for (const x of v) { lo = Math.min(lo, x); hi = Math.max(hi, x); } return [lo, hi]; };
  // The colors of a value map over the range its values span: the worst states orange, the best blue (or gray at 0).
  function rangeColor(lo, hi) {
    const mid = RL.token("--v-mid"), low = lo < 0 ? RL.token("--v-neg") : mid, high = hi > 0 ? RL.token("--v-pos") : mid;
    return (v) => `color-mix(in oklab, ${high} ${Math.round(100 * Math.max(0, Math.min(1, (v - lo) / Math.max(1e-9, hi - lo))))}%, ${low})`;
  }
  const ends = (lo, hi) => [lo < 0 ? "orange" : "gray", hi > 0 ? "blue" : "gray"];

  class HillCarView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { path: true, map: "value", ...options };
      this.box = RL.h('<div class="cartview hillview"><div class="track-pane"></div><div class="map-pane"><div class="map-head"><b>What the critic thinks</b><span></span></div><div class="map-host"></div></div></div>');
      host.appendChild(this.box);
      const pane = this.box.querySelector(".track-pane"), { caption, draw } = RL.readouts.frame(pane);
      this.readout = caption;
      this.draw = draw;
      const svg = (this.svg = el("svg", { class: "track", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "The valley and the car" }, draw));
      let d = "";
      for (let k = 0; k <= 90; k++) { const x = X0 + ((X1 - X0) * k) / 90; d += `${k ? "L" : "M"}${hx(x).toFixed(1)} ${hy(x).toFixed(1)}`; }
      el("path", { class: "ground", d: `${d}L${W - M} ${H}L${M} ${H}Z` }, svg);
      el("path", { class: "ground-line", d }, svg);
      const fx = hx(GOAL), fy = hy(GOAL);
      el("line", { class: "pole", x1: fx, x2: fx, y1: fy, y2: fy - 26 }, svg);
      el("path", { class: "flag", d: `M${fx} ${fy - 26}l14 4.5l-14 4.5z` }, svg);
      el("text", { class: "track-note", x: M, y: 12 }, svg).textContent = "the engine is too weak: swing";
      this.car = el("g", { class: "car" }, svg);
      el("rect", { class: "car-shell", x: -11, y: -13, width: 22, height: 8, rx: 3.5 }, this.car);
      el("rect", { class: "car-top", x: -6, y: -18, width: 11, height: 6, rx: 2.5 }, this.car);
      el("circle", { class: "wheel", cx: -6.5, cy: -3.5, r: 3.5 }, this.car);
      el("circle", { class: "wheel", cx: 6.5, cy: -3.5, r: 3.5 }, this.car);
      this.force = el("path", { class: "force" }, this.car); // drawn along the car: forward to the right
      // the force drawn this step, and the policy's aim with its spread: bars from the middle, −1 to +1 (and the
      // network strip under them)
      ({ grid: this.grid, rows: [this.drawn, this.aim] } = RL.readouts.bars(pane, ["force", "aim μ ± σ"], { centered: true }));
      this.drawn.chosen(true);
      this.aim.chosen(true);
      this._map();
      this.points = [];
      this.place([-0.5, 0]);
      this.setOptions({});
    }

    setOptions(o) {
      Object.assign(this.o, o);
      this.box.dataset.map = this.o.map;
      if (this.d) this.show(this.d);
      this._drawPath();
    }

    // The map: position × speed, colored by the critic's value or by the policy's push, and the episode's path.
    _map() {
      const S = 250, P = 36, N = 31;
      this.map = el("svg", { class: "phase", viewBox: `0 0 ${S + P + 8} ${S + P}`, role: "img", "aria-label": "Values and pushes over the car's position and speed" }, this.box.querySelector(".map-host"));
      const c = S / N;
      this.cells = el("g", { transform: `translate(${P} 0) scale(${c})` }, this.map); // cells one unit wide, one path per color
      this.cellsN = N;
      this.chev = [];
      const marks = el("g", { class: "chevrons" }, this.map);
      for (let j = 1; j < N; j += 3) for (let i = 1; i < N; i += 3) this.chev.push({ k: j * N + i, e: el("path", { class: "chev", transform: `translate(${P + (i + 0.5) * c} ${(N - j - 0.5) * c})` }, marks) });
      const gx = P + ((GOAL - X0) / (X1 - X0)) * S;
      el("line", { class: "zero", x1: gx, x2: gx, y1: 0, y2: S }, this.map);
      el("line", { class: "zero", x1: P, x2: P + S, y1: S / 2, y2: S / 2 }, this.map);
      el("text", { class: "tick", x: P, y: S + 14 }, this.map).textContent = "−1.2";
      el("text", { class: "tick", x: gx, y: S + 14, "text-anchor": "middle" }, this.map).textContent = "flag 0.45";
      el("text", { class: "axis-name", x: P + S / 2 - 30, y: S + 28, "text-anchor": "middle" }, this.map).textContent = "position";
      el("text", { class: "tick", x: P - 4, y: 10, "text-anchor": "end" }, this.map).textContent = "+.07";
      el("text", { class: "tick", x: P - 4, y: S, "text-anchor": "end" }, this.map).textContent = "−.07";
      el("text", { class: "axis-name", transform: `translate(12 ${S / 2}) rotate(-90)`, "text-anchor": "middle" }, this.map).textContent = "speed";
      this.mapPath = el("path", { class: "phase-path" }, this.map);
      this.mapDot = el("circle", { class: "phase-dot", r: 4.5 }, this.map);
      this.mapXY = (s) => [P + ((Math.max(X0, Math.min(X1, s[0])) - X0) / (X1 - X0)) * S, S / 2 - (Math.max(-V, Math.min(V, s[1])) / V) * (S / 2)];
    }

    // What a snapshot knows: the critic's values over the grid, and the policy's mean force (its push).
    show(d) {
      if (!d || !d.v) return;
      this.d = d;
      RL.netStrip.draw(this.grid, d.net, d.units); // the network strip under the bars (runs of lab/deep/)
      const [lo, hi] = span(d.v), color = rangeColor(lo, hi), valueMode = this.o.map !== "action";
      this.cells.innerHTML = RL.Film.cells(this.cellsN, 1, (k) => {
        const mu = d.mean ? d.mean[k] : 0;
        return valueMode ? color(d.v[k]) : `color-mix(in oklab, var(--pol) ${Math.round(Math.min(1, Math.abs(mu)) * 70)}%, var(--v-mid))`;
      });
      for (const { k, e } of this.chev) {
        const mu = d.mean ? d.mean[k] : 0;
        e.setAttribute("d", mu >= 0 ? "M-3 -4.5l4.5 4.5l-4.5 4.5" : "M3 -4.5l-4.5 4.5l4.5 4.5");
        e.style.opacity = (0.2 + 0.6 * Math.min(1, Math.abs(mu))).toFixed(2);
      }
      this.box.querySelector(".map-head span").textContent = valueMode
        ? `color: the value of each position and speed, from ${num(lo)} (${ends(lo, hi)[0]}) to ${num(hi)} (${ends(lo, hi)[1]}); chevrons: the way the policy pushes, darker the harder`
        : "color and chevrons: the way the policy pushes, darker the harder";
      this.box.querySelector(".map-head").title = this.box.querySelector(".map-head span").textContent; // narrow panes hide it
    }

    place(s) {
      const x = s[0], ang = Math.atan2(hy(x + 0.01) - hy(x - 0.01), hx(x + 0.01) - hx(x - 0.01));
      this.car.setAttribute("transform", `translate(${hx(x).toFixed(1)} ${hy(x).toFixed(1)}) rotate(${((ang * 180) / Math.PI).toFixed(1)})`);
      this.at = s;
    }

    // The force as an arrow along the car, longer the harder: to the right for a positive one.
    push(f) {
      const L = 22 * Math.max(-1, Math.min(1, f));
      if (Math.abs(L) < 1.5) { this.force.setAttribute("d", ""); return; }
      const x0 = Math.sign(L) * 12, x1 = x0 + L, dir = Math.sign(L);
      this.force.setAttribute("d", `M${x0} -9H${x1.toFixed(1)}m${-5 * dir} -4l${5 * dir} 4l${-5 * dir} 4`);
    }

    // The policy's numbers for this state: the force it drew, and its mean and spread.
    numbers(x, f) {
      const at = (v) => 0.5 + Math.max(-1, Math.min(1, v)) / 2; // a force, as a share of the bar's width
      this.drawn.span(0.5, at(f));
      this.drawn.num(sgn(f));
      const [mu, sd] = x;
      this.aim.band(at(mu - sd), at(mu + sd));
      this.aim.span(0.5, at(mu));
      this.aim.num(`${sgn(mu)} ± ${sd.toFixed(2)}`);
    }

    _drawPath() {
      const pts = this.o.path ? this.points : [];
      this.mapPath.setAttribute("d", pts.map((s, i) => `${i ? "L" : "M"}${this.mapXY(s).map((v) => v.toFixed(1)).join(" ")}`).join(""));
      if (this.at) { const [x, y] = this.mapXY(this.at); this.mapDot.setAttribute("cx", x); this.mapDot.setAttribute("cy", y); }
    }

    event(ev) {
      switch (ev.type) {
        case "start":
          this.points = [ev.s];
          this.place(ev.s);
          this.readout.textContent = this.o.count ? "step 0" : `position ${sgn(ev.s[0])} · speed ${sgn(ev.s[1], 3)}`;
          this._drawPath();
          return 0;
        case "choose":
          this.push(ev.a);
          this.numbers(ev.x, ev.a);
          return 0;
        case "move":
          this.place(ev.s2);
          this.points.push(ev.s2);
          this._drawPath();
          // the step too in a story; the Lab counts the steps in the line under the pane
          this.readout.textContent = `${this.o.count ? `step ${ev.k + 1} · ` : ""}position ${sgn(ev.s2[0])} · speed ${sgn(ev.s2[1], 3)}`;
          if (ev.end) {
            this.pop(ev.k + 1 < this.env.maxSteps ? `the flag, in ${ev.k + 1} steps` : `${this.env.maxSteps} steps: no flag`);
            return 600;
          }
          return 0;
        default:
          return 0;
      }
    }

    pop(text) { RL.readouts.pop(this.draw, text, 0.3); }

    // At rest after a unit: the last state of its test episode, and its whole path on the map.
    rest(events) {
      const moves = events.filter((e) => e.type === "move"), start = events.find((e) => e.type === "start");
      this.points = start ? [start.s, ...moves.map((e) => e.s2)] : [];
      const last = moves[moves.length - 1];
      this.place(last ? last.s2 : [-0.5, 0]);
      if (last) { this.push(last.a); this.numbers(last.x, last.a); } else this.push(0);
      this.readout.textContent = last ? (moves.length < this.env.maxSteps ? `the test episode reached the flag in ${moves.length} steps` : `the test episode ran out of time at ${sgn(last.s2[0])}`) : "";
      this._drawPath();
    }

    destroy() { this.box.remove(); }

    // Narrow panes stay legible down to this width (the panels stack): four racers can sit side by side.
    static get minWidth() { return 170; }

    static options() {
      return [
        { key: "map", type: "seg", label: "What the map shows", choices: [["value", "Values"], ["action", "Pushes"]], value: "value" },
        { key: "path", type: "check", label: "The path of the test episode", value: true },
      ];
    }

    // A filmstrip frame: the value map, small.
    static thumb(env, d) {
      if (!d?.v) return "";
      const N = 31, c = 4, [lo, hi] = span(d.v);
      const cells = RL.Film.cells(N, c, (k) => {
        const t = Math.max(0, Math.min(1, (d.v[k] - lo) / Math.max(1e-9, hi - lo)));
        return `color-mix(in oklab, var(${hi > 0 ? "--v-pos" : "--v-mid"}) ${Math.round(100 * t)}%, var(${lo < 0 ? "--v-neg" : "--v-mid"}))`;
      });
      return `<svg viewBox="0 0 ${N * c} ${N * c}" class="cart-thumb" role="img" aria-label="Values over position and speed">${cells}</svg><span class="thumb-note">test: ${num(d.ret)}</span>`;
    }
  }
  RL.labViews = RL.labViews || {};
  RL.labViews.hillcar = HillCarView;
})(globalThis.RL = globalThis.RL || {});
