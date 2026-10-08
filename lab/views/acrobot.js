/* Acrobot view, for runs trained live. On the left, the two links hanging from the pivot, the motor at the joint with
   the torque it gives, the line the tip must rise above, and the learner's numbers for each torque as bars. On the
   right, what the learner thinks of each pose: a map over the two angles (both links at rest), colored by value or by
   the torque it prefers, with the episode's path drawn over it. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 330, H = 210, PX = W / 2, PY = 104, L = 48; // the pivot, and pixels per link
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const deg = (r) => Math.round((r * 180) / Math.PI);
  const sgn = (v, d = 2) => { const t = Math.abs(v).toFixed(d); return `${v < 0 && +t ? "−" : "+"}${t}`; }; // no "−0.00"
  const span = (v) => { let lo = Infinity, hi = -Infinity; for (const x of v) { lo = Math.min(lo, x); hi = Math.max(hi, x); } return [lo, hi]; };
  function rangeColor(box, lo, hi) {
    const css = getComputedStyle(box), mid = css.getPropertyValue("--v-mid"), low = lo < 0 ? css.getPropertyValue("--v-neg") : mid, high = hi > 0 ? css.getPropertyValue("--v-pos") : mid;
    return (v) => `color-mix(in oklab, ${high} ${Math.round(100 * Math.max(0, Math.min(1, (v - lo) / Math.max(1e-9, hi - lo))))}%, ${low})`;
  }
  const num = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)).replace("-", "−");
  const GLYPH = ["↻", "·", "↺"]; // torque −1 (clockwise on screen), none, +1

  class AcrobotView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { path: true, map: "value", ...options };
      this.box = RL.h('<div class="cartview acroview"><div class="track-pane"></div><div class="map-pane"><div class="map-head"><b>What the learner thinks</b><span></span></div><div class="map-host"></div></div></div>');
      host.appendChild(this.box);
      const pane = this.box.querySelector(".track-pane"), { caption, draw } = RL.readouts.frame(pane);
      this.readout = caption;
      this.draw = draw;
      const svg = (this.svg = el("svg", { class: "track", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "The acrobot's two links" }, draw));
      // the line the tip must rise above: one link length above the pivot
      el("line", { class: "goal-line", x1: 24, x2: W - 24, y1: PY - L, y2: PY - L }, svg);
      el("text", { class: "track-note", x: W - 24, y: PY - L - 6, "text-anchor": "end" }, svg).textContent = "goal: the tip above this line";
      this.trail = el("path", { class: "tip-trail" }, svg);
      this.link1 = el("line", { class: "link", x1: PX, y1: PY }, svg);
      this.link2 = el("line", { class: "link two" }, svg);
      el("circle", { class: "pivot", cx: PX, cy: PY, r: 4 }, svg);
      this.joint = el("circle", { class: "joint", r: 6 }, svg);
      this.torque = el("text", { class: "torque", "text-anchor": "middle" }, svg);
      this.tip = el("circle", { class: "tip", r: 4.5 }, svg);
      // the learner's numbers for each torque, as three bars under the links
      ({ rows: this.bars } = RL.readouts.bars(pane, env.actionNames.slice(0, 3)));
      this.mapHost = this.box.querySelector(".map-host");
      this._map();
      this.points = [];
      this.k = 0;
      this.place([0, 0, 0, 0]);
      this.setOptions({});
    }

    setOptions(o) {
      Object.assign(this.o, o);
      if (this.d) this.show(this.d.live);
      this._drawPath();
    }

    // The map: a heat map over the two angles, glyphs for the preferred torque, the episode's path.
    _map() {
      const S = 250, P = 36, N = this.env.phase.N;
      this.map = el("svg", { class: "phase", viewBox: `0 0 ${S + P + 8} ${S + P}`, role: "img", "aria-label": "Values and choices over the two angles" }, this.mapHost);
      this.cellEls = [];
      const c = S / N, cells = el("g", {}, this.map);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) this.cellEls.push(el("rect", { x: P + i * c, y: (N - 1 - j) * c, width: c + 0.4, height: c + 0.4 }, cells));
      this.marks = [];
      const marks = el("g", { class: "torques" }, this.map);
      for (let j = 1; j < N; j += 3) for (let i = 1; i < N; i += 3) this.marks.push({ k: j * N + i, e: el("text", { class: "tq", x: P + (i + 0.5) * c, y: (N - j - 0.5) * c + 4, "text-anchor": "middle" }, marks) });
      el("line", { class: "zero", x1: P + S / 2, x2: P + S / 2, y1: 0, y2: S }, this.map);
      el("line", { class: "zero", x1: P, x2: P + S, y1: S / 2, y2: S / 2 }, this.map);
      el("text", { class: "tick", x: P, y: S + 14 }, this.map).textContent = "−180°";
      el("text", { class: "tick", x: P + S, y: S + 14, "text-anchor": "end" }, this.map).textContent = "+180°";
      el("text", { class: "axis-name", x: P + S / 2, y: S + 28, "text-anchor": "middle" }, this.map).textContent = "the first link's angle (0: hanging down)";
      el("text", { class: "tick", x: P - 4, y: 10, "text-anchor": "end" }, this.map).textContent = "+180°";
      el("text", { class: "tick", x: P - 4, y: S, "text-anchor": "end" }, this.map).textContent = "−180°";
      el("text", { class: "axis-name", transform: `translate(12 ${S / 2}) rotate(-90)`, "text-anchor": "middle" }, this.map).textContent = "the second link, bent";
      this.mapPath = el("path", { class: "phase-path" }, this.map);
      this.mapDot = el("circle", { class: "phase-dot", r: 4.5 }, this.map);
      this.mapXY = (s) => [P + ((s[0] + Math.PI) / (2 * Math.PI)) * S, S / 2 - (s[1] / Math.PI) * (S / 2)];
    }

    show(d) {
      if (!d || !(d.w || d.theta)) return;
      const m = RL.lab.controlMap(this.env, d);
      this.d = { ...m, live: d };
      const valueMode = this.o.map !== "action" && !m.noValue, [lo, hi] = span(m.v), color = rangeColor(this.box, lo, hi);
      const css = getComputedStyle(this.box), tint = [css.getPropertyValue("--v-neg"), css.getPropertyValue("--v-mid"), css.getPropertyValue("--v-pos")];
      for (let k = 0; k < this.cellEls.length; k++) {
        const a = m.act[k], sure = m.kind === "pg" ? Math.max(0, (m.pr[k] - 1 / 3) * 1.5) : 0.75;
        this.cellEls[k].setAttribute("fill", valueMode ? color(m.v[k]) : `color-mix(in oklab, ${tint[a]} ${Math.round(100 * Math.min(1, sure))}%, ${tint[1]})`);
      }
      for (const { k, e } of this.marks) {
        e.textContent = GLYPH[m.act[k]];
        e.style.opacity = m.kind === "pg" ? (0.3 + Math.min(0.7, m.pr[k] - 1 / 3)).toFixed(2) : "0.75";
      }
      this.box.querySelector(".map-head span").textContent = valueMode
        ? `color: the value of each pose, from ${num(lo)} (${lo < 0 ? "orange" : "gray"}) to ${num(hi)} (${hi > 0 ? "blue" : "gray"}), both links at rest; marks: the torque it prefers (↺ +1, ↻ −1, · none)`
        : `color and marks: the torque it prefers in each pose (↺ +1 blue, ↻ −1 orange, · none)${m.kind === "pg" ? ", paler where it is unsure" : ""}${m.noValue ? "; no critic to give values" : ""}`;
    }

    // The links for state s: the elbow and the tip on the drawing.
    pose(s) {
      const ex = PX + L * Math.sin(s[0]), ey = PY + L * Math.cos(s[0]);
      return [ex, ey, ex + L * Math.sin(s[0] + s[1]), ey + L * Math.cos(s[0] + s[1])];
    }

    place(s) {
      const [ex, ey, tx, ty] = this.pose(s);
      for (const [e, a] of [[this.link1, { x2: ex, y2: ey }], [this.link2, { x1: ex, y1: ey, x2: tx, y2: ty }], [this.joint, { cx: ex, cy: ey }], [this.tip, { cx: tx, cy: ty }]])
        for (const k in a) e.setAttribute(k, a[k].toFixed(1));
      this.torque.setAttribute("x", ex.toFixed(1));
      this.torque.setAttribute("y", (ey - 10).toFixed(1));
      this.box.style.setProperty("--lift", Math.max(0, Math.min(1, (this.env.tipHeight(s) + 2) / 3)).toFixed(2));
      this.at = s;
    }

    gas(a) { this.torque.textContent = a === 1 ? "" : GLYPH[a]; }

    numbers(vals, a, kind) {
      const probs = kind === "pg", top = probs ? 1 : Math.max(1e-9, ...vals.map(Math.abs));
      this.bars.forEach((b, i) => {
        const v = vals[i];
        b.set(Math.abs(v) / top);
        b.chosen(i === a);
        b.num(probs ? `${Math.round(100 * v)}%` : num(v));
      });
    }

    _numbersAt(ev) {
      const live = this.d?.live;
      if (!live) return;
      const { kind, vals } = RL.lab.controlNumbers(this.env, live, ev.s);
      this.numbers(vals, ev.a, kind);
    }

    _drawPath() {
      const pts = this.o.path ? this.points : [];
      // the angles wrap at ±180°: the path breaks there instead of crossing the map
      let d = "", prev = null;
      for (const s of pts) {
        const [x, y] = this.mapXY(s), jump = prev && (Math.abs(s[0] - prev[0]) > Math.PI || Math.abs(s[1] - prev[1]) > Math.PI);
        d += `${!prev || jump ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
        prev = s;
      }
      this.mapPath.setAttribute("d", d);
      if (this.at) { const [x, y] = this.mapXY(this.at); this.mapDot.setAttribute("cx", x); this.mapDot.setAttribute("cy", y); }
      // the tip's last dozen positions, on the drawing
      const tail = this.o.path ? pts.slice(-12) : [];
      this.trail.setAttribute("d", tail.map((s, i) => { const p = this.pose(s); return `${i ? "L" : "M"}${p[2].toFixed(1)} ${p[3].toFixed(1)}`; }).join(""));
    }

    event(ev) {
      if (ev.w) return 0; // several workers: the view follows the first
      switch (ev.type) {
        case "start":
          this.k = 0;
          this.points = [ev.s];
          this.place(ev.s);
          this.gas(1);
          this.readout.textContent = "step 0";
          this._drawPath();
          return 0;
        case "choose":
          this.gas(ev.a);
          this._numbersAt(ev);
          return 0;
        case "move": {
          this.k += 1;
          this.place(ev.s2);
          this.points.push(ev.s2);
          this._drawPath();
          this.readout.textContent = `step ${this.k} · tip ${sgn(this.env.tipHeight(ev.s2))}`;
          if (this.env.terminal(ev.s2)) { this.pop(`over the line in ${this.k} steps!`); return 600; }
          if (this.k >= this.env.maxSteps) { this.pop(`${this.env.maxSteps} steps: cut short`); return 600; }
          return 0;
        }
        default:
          return 0;
      }
    }

    pop(text) { RL.readouts.pop(this.draw, text, 0.08); }

    // At rest after a unit: the episode's last pose, and its whole path on the map.
    rest(events) {
      const moves = events.filter((e) => e.type === "move" && !e.w), start = events.find((e) => e.type === "start" && !e.w);
      this.points = start ? [start.s, ...moves.map((e) => e.s2)] : [];
      const last = moves[moves.length - 1];
      this.place(last ? last.s2 : [0, 0, 0, 0]);
      this.gas(last ? last.a : 1);
      if (last) this._numbersAt(last);
      this.readout.textContent = last ? `this episode lasted ${moves.length} steps` : "";
      this._drawPath();
    }

    destroy() { this.box.remove(); }

    static options() {
      return [
        { key: "map", type: "seg", label: "What the map shows", choices: [["value", "Values"], ["action", "Choices"]], value: "value" },
        { key: "path", type: "check", label: "The path of the episode", value: true },
      ];
    }

    // A filmstrip frame: the value map, small (the preferred torques, for a policy without a critic).
    static thumb(env, d) {
      if (!d || !(d.w || d.theta)) return "";
      const m = RL.lab.controlMap(env, d), N = m.N, c = 4, vals = m.noValue ? Array.from(m.act, (a) => a - 1) : m.v, [lo, hi] = span(vals);
      let cells = "";
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const t = Math.max(0, Math.min(1, (vals[j * N + i] - lo) / Math.max(1e-9, hi - lo)));
        cells += `<rect x="${i * c}" y="${(N - 1 - j) * c}" width="${c + 0.3}" height="${c + 0.3}" fill="color-mix(in oklab, var(--v-mid) ${Math.round(100 * t ** 0.8)}%, var(--v-neg))"/>`;
      }
      return `<svg viewBox="0 0 ${N * c} ${N * c}" class="cart-thumb" role="img" aria-label="Values over the two angles">${cells}</svg>`;
    }
  }

  RL.AcrobotView = AcrobotView;
  (RL.labViews = RL.labViews || {}).acrobot = AcrobotView;
})(globalThis.RL = globalThis.RL || {});
