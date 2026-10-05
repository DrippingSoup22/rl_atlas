/* Pendulum view, for recorded runs. On the left, the pendulum on its pivot, 0° straight up: the torque the policy drew
   as a curved arrow, and as bars the torque and where the policy aims (its mean, with its spread). On the right, what
   the critic thinks of the pendulum's states: a map over the angle and the spin, colored by value, with the way the
   policy pushes in each region, and the path of the test episode drawn over it. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const W = 330, H = 230, CX = 165, CY = 92, R = 62, BAR = 78; // pivot, rod length, half a torque bar (pixels)
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const deg = (r) => (r * 180) / Math.PI;
  const sgn = (v, d = 1) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(d)}`;
  const num = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)).replace("-", "−");
  const ends = (lo, hi) => [lo < 0 ? "orange" : "gray", hi > 0 ? "blue" : "gray"]; // the colors rangeColor gives them
  const span = (v) => { let lo = Infinity, hi = -Infinity; for (const x of v) { lo = Math.min(lo, x); hi = Math.max(hi, x); } return [lo, hi]; };
  // The colors of a map over the range its values span: the costliest states orange, the best gray (all are below 0).
  function rangeColor(box, lo, hi) {
    const css = getComputedStyle(box), mid = css.getPropertyValue("--v-mid"), low = lo < 0 ? css.getPropertyValue("--v-neg") : mid, high = hi > 0 ? css.getPropertyValue("--v-pos") : mid;
    return (v) => `color-mix(in oklab, ${high} ${Math.round(100 * Math.max(0, Math.min(1, (v - lo) / Math.max(1e-9, hi - lo))))}%, ${low})`;
  }

  class PendulumView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { path: true, map: "value", ...options };
      this.box = RL.h('<div class="cartview pendview"><div class="track-pane"></div><div class="map-pane"><div class="map-head"><b>What the critic thinks</b><span></span></div><div class="map-host"></div></div></div>');
      host.appendChild(this.box);
      const svg = el("svg", { class: "track", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "The pendulum" }, this.box.querySelector(".track-pane"));
      el("circle", { class: "guide", cx: CX, cy: CY, r: R }, svg);
      el("line", { class: "goal", x1: CX, x2: CX, y1: CY - R - 12, y2: CY - R + 6 }, svg);
      el("text", { class: "track-note", x: CX - 8, y: CY - R - 4, "text-anchor": "end" }, svg).textContent = "upright: 0°";
      this.torque = el("path", { class: "torque" }, svg);
      this.rod = el("line", { class: "pole-rod", x1: CX, y1: CY, x2: CX, y2: CY - R }, svg);
      this.bob = el("circle", { class: "bob", r: 9, cx: CX, cy: CY - R }, svg);
      el("circle", { class: "pivot", cx: CX, cy: CY, r: 3.5 }, svg);
      this.readout = el("text", { class: "readout", x: W - 6, y: 16, "text-anchor": "end" }, svg);
      // the torque drawn this step, and the policy's aim with its spread: bars from the middle, −2 to +2
      const row = (y, name) => {
        const g = el("g", { class: "bar-row chosen" }, svg);
        el("text", { class: "bar-name", x: 8, y: y + 9 }, g).textContent = name;
        el("rect", { class: "bar-bg", x: 88, y, width: 2 * BAR, height: 11, rx: 3 }, g);
        const band = el("rect", { class: "band", x: 88 + BAR, y, width: 0, height: 11 }, g);
        const fill = el("rect", { class: "bar-fill", x: 88 + BAR, y, width: 0, height: 11 }, g);
        el("line", { class: "zero", x1: 88 + BAR, x2: 88 + BAR, y1: y - 2, y2: y + 13 }, g);
        return { band, fill, num: el("text", { class: "bar-num", x: 88 + 2 * BAR + 6, y: y + 9 }, g) };
      };
      this.drawn = row(CY + R + 30, "torque");
      this.aim = row(CY + R + 50, "aim μ ± σ");
      this.gFx = el("g", { class: "fx" }, svg);
      this._map();
      this.points = [];
      this.place([Math.PI, 0]);
      this.setOptions({});
    }

    setOptions(o) {
      Object.assign(this.o, o);
      this.box.dataset.map = this.o.map;
      if (this.d) this.show(this.d);
      this._drawPath();
    }

    // The map: angle × spin, colored by the critic's value or by the policy's push, and the episode's path.
    _map() {
      const S = 250, P = 36, N = 31;
      this.map = el("svg", { class: "phase", viewBox: `0 0 ${S + P + 8} ${S + P}`, role: "img", "aria-label": "Values and pushes over the pendulum's angle and spin" }, this.box.querySelector(".map-host"));
      this.cellEls = [];
      const c = S / N, cells = el("g", {}, this.map);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) this.cellEls.push(el("rect", { x: P + i * c, y: (N - 1 - j) * c, width: c + 0.4, height: c + 0.4 }, cells));
      this.chev = [];
      const marks = el("g", { class: "chevrons" }, this.map);
      for (let j = 1; j < N; j += 3) for (let i = 1; i < N; i += 3) this.chev.push({ k: j * N + i, e: el("path", { class: "chev", transform: `translate(${P + (i + 0.5) * c} ${(N - j - 0.5) * c})` }, marks) });
      el("line", { class: "zero", x1: P + S / 2, x2: P + S / 2, y1: 0, y2: S }, this.map);
      el("line", { class: "zero", x1: P, x2: P + S, y1: S / 2, y2: S / 2 }, this.map);
      el("text", { class: "tick", x: P, y: S + 14 }, this.map).textContent = "−180°";
      el("text", { class: "tick", x: P + S / 2, y: S + 14, "text-anchor": "middle" }, this.map).textContent = "0° (up)";
      el("text", { class: "tick", x: P + S, y: S + 14, "text-anchor": "end" }, this.map).textContent = "+180°";
      el("text", { class: "axis-name", x: P + S / 2, y: S + 28, "text-anchor": "middle" }, this.map).textContent = "the angle (both ends: hanging down)";
      el("text", { class: "tick", x: P - 4, y: 10, "text-anchor": "end" }, this.map).textContent = "+8";
      el("text", { class: "tick", x: P - 4, y: S, "text-anchor": "end" }, this.map).textContent = "−8";
      el("text", { class: "axis-name", transform: `translate(12 ${S / 2}) rotate(-90)`, "text-anchor": "middle" }, this.map).textContent = "spin (rad/s)";
      this.mapPath = el("path", { class: "phase-path" }, this.map);
      this.mapDot = el("circle", { class: "phase-dot", r: 4.5 }, this.map);
      this.mapXY = (s) => [P + ((Math.max(-Math.PI, Math.min(Math.PI, s[0])) + Math.PI) / (2 * Math.PI)) * S, S / 2 - (Math.max(-8, Math.min(8, s[1])) / 8) * (S / 2)];
    }

    // What a snapshot knows: the critic's values over the grid, and the policy's mean torque (its push).
    show(d) {
      if (!d || !d.v) return;
      this.d = d;
      const [lo, hi] = span(d.v), color = rangeColor(this.box, lo, hi), valueMode = this.o.map !== "action";
      for (let k = 0; k < this.cellEls.length; k++) {
        const mu = d.mean ? d.mean[k] : 0;
        this.cellEls[k].setAttribute("fill", valueMode ? color(d.v[k]) : `color-mix(in oklab, var(--pol) ${Math.round(Math.min(1, Math.abs(mu) / 2) * 70)}%, var(--v-mid))`);
      }
      for (const { k, e } of this.chev) {
        const mu = d.mean ? d.mean[k] : 0;
        e.setAttribute("d", mu >= 0 ? "M-3 -4.5l4.5 4.5l-4.5 4.5" : "M3 -4.5l-4.5 4.5l4.5 4.5");
        e.style.opacity = (0.2 + 0.4 * Math.min(2, Math.abs(mu))).toFixed(2);
      }
      this.box.querySelector(".map-head span").textContent = valueMode
        ? `color: the value of each angle and spin, from ${num(lo)} (${ends(lo, hi)[0]}) to ${num(hi)} (${ends(lo, hi)[1]}); chevrons: the way the policy pushes, darker the harder`
        : "color and chevrons: the way the policy pushes, darker the harder";
    }

    place(s) {
      const x = CX + R * Math.sin(s[0]), y = CY - R * Math.cos(s[0]);
      this.rod.setAttribute("x2", x.toFixed(1));
      this.rod.setAttribute("y2", y.toFixed(1));
      this.bob.setAttribute("cx", x.toFixed(1));
      this.bob.setAttribute("cy", y.toFixed(1));
      this.at = s;
    }

    // The torque as an arc around the pivot: clockwise for a positive one (it turns the angle up), longer the harder.
    torqueArrow(u) {
      const r = 24, a0 = this.at ? this.at[0] : 0, sweep = (Math.max(-2, Math.min(2, u)) / 2) * 2.4;
      if (Math.abs(sweep) < 0.05) { this.torque.setAttribute("d", ""); return; }
      const p = (a) => [CX + r * Math.sin(a), CY - r * Math.cos(a)], [x0, y0] = p(a0), [x1, y1] = p(a0 + sweep), dir = Math.sign(sweep);
      const tx = Math.cos(a0 + sweep) * dir, ty = Math.sin(a0 + sweep) * dir; // the direction the arc runs at its end
      const head = `M${(x1 - 6 * tx - 4 * ty).toFixed(1)} ${(y1 - 6 * ty + 4 * tx).toFixed(1)}L${x1.toFixed(1)} ${y1.toFixed(1)}L${(x1 - 6 * tx + 4 * ty).toFixed(1)} ${(y1 - 6 * ty - 4 * tx).toFixed(1)}`;
      this.torque.setAttribute("d", `M${x0.toFixed(1)} ${y0.toFixed(1)}A${r} ${r} 0 ${Math.abs(sweep) > Math.PI ? 1 : 0} ${sweep > 0 ? 1 : 0} ${x1.toFixed(1)} ${y1.toFixed(1)}${head}`);
    }

    // The policy's numbers for this state: the torque it drew, and its mean and spread.
    numbers(x, u) {
      const at = (v) => (Math.max(-2, Math.min(2, v)) / 2) * BAR, bar = (b, from, to) => { b.setAttribute("x", (88 + BAR + Math.min(from, to)).toFixed(1)); b.setAttribute("width", Math.abs(to - from).toFixed(1)); };
      bar(this.drawn.fill, 0, at(u));
      this.drawn.num.textContent = sgn(u, 2);
      const [mu, sd] = x;
      bar(this.aim.band, at(mu - sd), at(mu + sd));
      bar(this.aim.fill, 0, at(mu));
      this.aim.num.textContent = `${sgn(mu, 2)} ± ${sd.toFixed(2)}`;
    }

    _drawPath() {
      const pts = this.o.path ? this.points : [];
      let d = "";
      pts.forEach((s, i) => { const [x, y] = this.mapXY(s); d += `${i && Math.abs(s[0] - pts[i - 1][0]) < Math.PI ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`; }); // over ±180° the path jumps
      this.mapPath.setAttribute("d", d);
      if (this.at) { const [x, y] = this.mapXY(this.at); this.mapDot.setAttribute("cx", x); this.mapDot.setAttribute("cy", y); }
    }

    event(ev) {
      switch (ev.type) {
        case "start":
          this.points = [ev.s];
          this.place(ev.s);
          this.readout.textContent = "step 0";
          this._drawPath();
          return 0;
        case "choose":
          this.torqueArrow(ev.a);
          this.numbers(ev.x, ev.a);
          return 0;
        case "move":
          this.place(ev.s2);
          this.points.push(ev.s2);
          this._drawPath();
          this.readout.textContent = `step ${ev.k + 1} · angle ${sgn(deg(ev.s2[0]))}°`;
          if (ev.end) {
            this.pop(Math.abs(ev.s2[0]) < 0.2 ? "held upright" : "200 steps: still swinging");
            return 600;
          }
          return 0;
        default:
          return 0;
      }
    }

    pop(text) {
      const t = el("text", { class: "pop", x: CX, y: CY + R + 21, "text-anchor": "middle" }, this.gFx); // under the swing, clear of the labels
      t.textContent = text;
      t.animate([{ opacity: 0 }, { opacity: 1, offset: 0.15 }, { opacity: 0, transform: "translateY(-12px)" }], { duration: 1400, easing: "ease-out" }).onfinish = () => t.remove();
    }

    // At rest after a unit: the last state of its test episode, and its whole path on the map.
    rest(events) {
      const moves = events.filter((e) => e.type === "move"), start = events.find((e) => e.type === "start");
      this.points = start ? [start.s, ...moves.map((e) => e.s2)] : [];
      const last = moves[moves.length - 1];
      this.place(last ? last.s2 : [Math.PI, 0]);
      if (last) { this.torqueArrow(last.a); this.numbers(last.x, last.a); } else this.torqueArrow(0);
      this.readout.textContent = last ? `the test episode ended at ${sgn(deg(last.s2[0]))}°` : "";
      this._drawPath();
    }

    destroy() { this.box.remove(); }

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
      let cells = "";
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const t = Math.max(0, Math.min(1, (d.v[j * N + i] - lo) / Math.max(1e-9, hi - lo)));
        cells += `<rect x="${i * c}" y="${(N - 1 - j) * c}" width="${c + 0.3}" height="${c + 0.3}" fill="color-mix(in oklab, var(${hi > 0 ? "--v-pos" : "--v-mid"}) ${Math.round(100 * t)}%, var(${lo < 0 ? "--v-neg" : "--v-mid"}))"/>`;
      }
      return `<svg viewBox="0 0 ${N * c} ${N * c}" class="cart-thumb" role="img" aria-label="Values over angle and spin">${cells}</svg><span class="thumb-note">test: ${num(d.ret)}</span>`;
    }
  }
  RL.labViews = RL.labViews || {};
  RL.labViews.pendulum = PendulumView;
})(globalThis.RL = globalThis.RL || {});
