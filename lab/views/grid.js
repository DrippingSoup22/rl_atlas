/* Grid view: one SVG for a grid world, shared by the Lab and the stories.
   Tiles show values (four triangles for Q, or one color for V), arrows show the policy,
   and the agent walks, slides, falls and sparks so you can see each update happen. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const T = 60, HALF = T / 2, PAD = 6;
  const NS = "http://www.w3.org/2000/svg";
  const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // screen direction of up, right, down, left
  const ARROW_TEXT = ["↑", "→", "↓", "←"];
  const MARKS = { S: "S", A: "A", B: "B", a: "A′", b: "B′" };
  const KINDS = { ".": "free", "#": "wall", C: "C", G: "G", T: "exit", H: "hole" };
  let uid = 0;
  const TRIANGLES = [
    [[2, 2], [T - 2, 2]],
    [[T - 2, 2], [T - 2, T - 2]],
    [[T - 2, T - 2], [2, T - 2]],
    [[2, T - 2], [2, 2]],
  ];
  const FALL = [{ transform: "scale(1) rotate(0deg)", opacity: 1 }, { transform: "scale(0.1) rotate(200deg)", opacity: 0 }];
  const SINK = [{ transform: "scale(1)", opacity: 1 }, { transform: "scale(0.2) translateY(6px)", opacity: 0 }];
  const RESPAWN = [{ transform: "scale(0)" }, { transform: "scale(1)" }];
  const HOP = [{ transform: "translateY(0)" }, { transform: "translateY(-6px) scale(1.06, 0.94)" }, { transform: "translateY(0)" }];
  const WOBBLE = [{ transform: "rotate(0)" }, { transform: "rotate(-14deg)" }, { transform: "rotate(10deg)" }, { transform: "rotate(0)" }];

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // Diverging value color: orange below zero, neutral gray at zero, blue above.
  // CSS color-mix does the blending, so the colors follow the theme by themselves.
  function valueColor(v, range) {
    const t = Math.max(-1, Math.min(1, v / range));
    const pct = Math.round(Math.abs(t) * 100);
    return `color-mix(in oklab, var(${t < 0 ? "--v-neg" : "--v-pos"}) ${pct}%, var(--v-mid))`;
  }
  const fmt = (v, digits = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(digits);
  const signed = (v, digits = 0) => `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(v).toFixed(digits)}`;

  class GridView {
    // options: tiles "q" | "v" | "none", arrows, numbers, trail, agent, digits, epsilon (for the arrows of ε-greedy), range
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { tiles: "q", arrows: false, numbers: false, trail: true, agent: true, digits: 2, epsilon: 0.1, ...options };
      this.Q = new Float64Array(env.nS * env.nA);
      this.hasQ = true; // false when an algorithm only knows state values
      this.V = null; // state values to show instead of the best Q
      this.P = null; // a policy to draw instead of ε-greedy on Q
      this.Z = null; // eligibility traces (per tile, or per move), when the algorithm keeps them
      this.model = null; // a learned model: where each move led (−1: never tried)
      this.path_ = [];
      const W = env.cols * T, H = env.rows * T;
      this.svg = el("svg", { class: `gridview${env.ice ? " ice" : ""}`, viewBox: `${-PAD} ${-PAD} ${W + 2 * PAD} ${H + 2 * PAD}`, role: "img", "aria-label": env.title });
      this.svg.style.maxWidth = `${env.cols * 100}px`; // small worlds stay a comfortable size instead of filling the page
      host.appendChild(this.svg);
      const layer = (cls) => el("g", { class: cls }, this.svg);
      this.gTiles = layer("tiles");
      this.gTrace = layer("trace-layer"); // eligibility traces, glowing where credit will flow
      this.gFog = layer("fog-layer"); // tiles the agent's model has never seen
      this.gJumps = layer("jumps");
      this.gTrail = layer("trail-layer");
      this.gPath = layer("path-layer");
      this.gArrows = layer("arrows");
      this.gNums = layer("nums"); // numbers above the arrows and paths, so they always read
      this.gAgent = layer("agent-layer");
      this.gFx = layer("fx");
      this.tiles = [];
      this.tris = [];
      this.arrows = [];
      this.nums = [];
      for (let s = 0; s < env.nS; s++) this._tile(s);
      if (env.jumps?.length) this._jumps();
      this._agent();
      this.trailEl = el("path", { class: "trail", pathLength: 1 }, this.gTrail);
      this.pathEl = el("path", { class: "greedy", pathLength: 1 }, this.gPath);
      this.svg.addEventListener("pointermove", (e) => this._hover(e));
      this.svg.addEventListener("pointerleave", () => RL.tip.hide());
      this.setOptions({});
      this.place(env.start, 1, true);
    }

    origin(s) { const [r, c] = this.env.rc(s); return [c * T, r * T]; }
    center(s) { const [x, y] = this.origin(s); return [x + HALF, y + HALF]; }
    // The tile under a pointer event, or -1.
    tileAt(e) {
      const m = this.svg.getScreenCTM();
      if (!m) return -1;
      const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
      const c = Math.floor(p.x / T), r = Math.floor(p.y / T);
      return r < 0 || c < 0 || r >= this.env.rows || c >= this.env.cols ? -1 : r * this.env.cols + c;
    }

    _tile(s) {
      const kind = this.env.tile(s), [x, y] = this.origin(s);
      const g = el("g", { class: `tile k-${KINDS[kind] || kind}`, transform: `translate(${x} ${y})` }, this.gTiles);
      g.dataset.kind = kind;
      this.tiles[s] = g;
      g.bg = el("rect", { class: "bg", x: 1.5, y: 1.5, width: T - 3, height: T - 3, rx: 7 }, g);
      if (kind === "C") {
        el("path", { class: "rocks", d: "M4 15 l7 -6 l6 5 l8 -7 l7 6 l8 -5 l7 6 l9 -4" }, g);
        return;
      }
      if (kind === "#") {
        el("path", { class: "bricks", d: "M2 20 H58 M2 40 H58 M20 2 V20 M42 20 V40 M24 40 V58" }, g);
        return;
      }
      if (kind === "H") {
        el("ellipse", { class: "hole", cx: HALF, cy: HALF + 2, rx: 19, ry: 13 }, g);
        el("path", { class: "crack", d: "M10 12 l8 6 l-3 7 M50 48 l-7 -4 l2 -8" }, g);
        return;
      }
      if (kind === "G") {
        const gem = el("g", { class: "gem" }, g);
        el("polygon", { points: `${HALF},${HALF - 15} ${HALF + 14},${HALF - 3} ${HALF},${HALF + 16} ${HALF - 14},${HALF - 3}` }, gem);
        el("polygon", { class: "shine", points: `${HALF},${HALF - 15} ${HALF + 6},${HALF - 3} ${HALF},${HALF + 2} ${HALF - 6},${HALF - 3}` }, gem);
        return;
      }
      if (kind === "T") {
        el("text", { class: "exit-mark", x: HALF, y: HALF + 5 }, g).textContent = "end";
        return;
      }
      this.tris[s] = TRIANGLES.map(([p, q]) => el("polygon", { class: "tri", points: `${p} ${q} ${HALF},${HALF}` }, g));
      if (MARKS[kind]) el("text", { class: "mark", x: 7, y: 15 }, g).textContent = MARKS[kind];
      this.nums[s] = el("text", { class: "num", x: x + HALF, y: y + HALF + 4 }, this.gNums);
      const set = el("g", { class: "arrow-set", transform: `translate(${x + HALF} ${y + HALF})` }, this.gArrows);
      this.arrows[s] = DIRS.map(() => {
        const a = el("g", { class: "arrow" }, set);
        a.halo = el("path", { class: "halo" }, a);
        a.shaft = el("path", { class: "shaft" }, a);
        return a;
      });
    }

    // Tiles that jump elsewhere: a curved arrow to where they land, with the reward on it.
    _jumps() {
      const id = `jump-tip-${++uid}`;
      const marker = el("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, el("defs", {}, this.svg));
      el("path", { class: "jump-tip", d: "M0 0 L10 5 L0 10 z" }, marker);
      for (const j of this.env.jumps) {
        const [x1, y1] = this.center(j.from), [x2, y2] = this.center(j.to);
        const p0 = [x1 + 12, y1 + 10], p1 = [Math.max(x1, x2) + 46, (y1 + y2) / 2], p2 = [x2 + 12, y2 - 12];
        el("path", { class: "jump", d: `M${p0} Q${p1} ${p2}`, "marker-end": `url(#${id})` }, this.gJumps);
        // The reward sits by the arc, early on its way, where it clears the numbers in the middle of the tiles.
        const at = (k) => 0.49 * p0[k] + 0.42 * p1[k] + 0.09 * p2[k];
        el("text", { class: "jump-r", x: at(0) + 6, y: at(1) + 4 }, this.gJumps).textContent = `+${j.reward}`;
      }
    }

    _agent() {
      this.bot = el("g", { class: "bot" }, this.gAgent);
      this.body = el("g", { class: "bot-body" }, this.bot);
      el("ellipse", { class: "bot-shadow", cx: 0, cy: 15, rx: 11, ry: 3.5 }, this.body);
      el("circle", { class: "bot-head", r: 14 }, this.body);
      this.eyes = el("g", { class: "bot-eyes" }, this.body);
      el("circle", { cx: -5, cy: -2, r: 2.8 }, this.eyes);
      el("circle", { cx: 5, cy: -2, r: 2.8 }, this.eyes);
    }

    // ---- what is shown ----
    setOptions(o) {
      Object.assign(this.o, o);
      this.svg.dataset.tiles = this._mode();
      this.svg.toggleAttribute("data-arrows", !!this.o.arrows);
      this.svg.toggleAttribute("data-numbers", !!this.o.numbers);
      this.showAgent(this.o.agent !== false);
      if (!this.o.trail) this.trail(null);
      this._paintAll();
    }
    _mode() { return this.o.tiles === "q" && !this.hasQ ? "v" : this.o.tiles; }

    // What an algorithm knows (its `show`): Q, or V, and maybe a policy P; greedy: draw the greedy policy of Q.
    show(d, p = {}) {
      this.hasQ = !!d.Q;
      if (d.Q) this.Q.set(d.Q);
      else this.Q.fill(0);
      this.V = d.V ? Float64Array.from(d.V) : null;
      this.P = d.P ? Float64Array.from(d.P) : null;
      this.Z = d.Z ? Float64Array.from(d.Z) : null;
      this.model = d.model ? Float64Array.from(d.model) : null;
      this.o.epsilon = d.greedy ? 0 : p.epsilon ?? this.o.epsilon;
      if (d.t !== undefined && this.env.setTime) { this.env.setTime(d.t); this.retile(); }
      this.svg.dataset.tiles = this._mode();
      this._paintAll();
    }

    // A world whose walls moved: draw the tiles that changed again.
    retile() {
      for (let s = 0; s < this.env.nS; s++) {
        if (this.tiles[s].dataset.kind === this.env.tile(s)) continue;
        this.tiles[s].remove();
        this.nums[s]?.remove();
        this.arrows[s]?.[0].parentNode.remove();
        this.traces?.[s]?.remove();
        this.fogs?.[s]?.remove();
        this.tris[s] = this.nums[s] = this.arrows[s] = undefined;
        if (this.traces) this.traces[s] = undefined;
        if (this.fogs) this.fogs[s] = undefined;
        this._tile(s);
        this.tiles[s].bg.animate?.([{ opacity: 0.2 }, { opacity: 1 }], { duration: 500, easing: "ease-out" });
      }
      this._dist = null;
    }

    setQ(Q, { ripple = false } = {}) {
      this.Q.set(Q);
      if (ripple && !RL.reducedMotion()) {
        const dist = this._distances();
        for (let s = 0; s < this.env.nS; s++) this._delay(s, `${dist[s] * 55}ms`);
        clearTimeout(this._rippleTimer);
        this._rippleTimer = setTimeout(() => { for (let s = 0; s < this.env.nS; s++) this._delay(s, ""); }, 1600);
      }
      this._paintAll();
    }

    setCell(s, a, v) {
      this.Q[s * this.env.nA + a] = v;
      this._paint(s);
    }

    // Show these state values on the tiles (null: the best Q), and draw this policy (null: ε-greedy on Q).
    setV(V) { this.V = V ? Float64Array.from(V) : null; this._paintAll(); }
    setPolicy(P) { this.P = P ? Float64Array.from(P) : null; this._paintAll(); }

    // Outline tiles: "focus" for the tile being explained, "next" for where its moves lead.
    mark(states, cls = "focus") {
      for (const t of this.svg.querySelectorAll(`.tile.${cls}`)) t.classList.remove(cls);
      for (const s of states || []) this.tiles[s]?.classList.add(cls);
    }

    _probs(s) {
      const { env } = this;
      return this.P ? Array.from(this.P.subarray(s * env.nA, s * env.nA + env.nA)) : lab.epsGreedyProbs(this.Q, s, env, this.o.epsilon);
    }
    _value(s) { return this.V ? this.V[s] : lab.maxQ(this.Q, s, this.env); }

    _delay(s, d) {
      this.tiles[s].bg.style.transitionDelay = d;
      for (const t of this.tris[s] || []) t.style.transitionDelay = d;
    }

    _paintAll() { for (let s = 0; s < this.env.nS; s++) this._paint(s); }

    _paint(s) {
      const tris = this.tris[s];
      if (!tris) return;
      const { nA } = this.env, range = this.o.range || this.env.valueRange, Q = this.Q, mode = this._mode();
      const v = this._value(s);
      for (let a = 0; a < nA; a++) tris[a].style.fill = mode === "q" ? valueColor(Q[s * nA + a], range) : "";
      this.tiles[s].bg.style.fill = mode === "v" ? valueColor(v, range) : "";
      this._trace(s);
      this._fog(s);
      const num = this.nums[s];
      num.textContent = this.o.numbers ? fmt(v, this.o.digits) : "";
      if (!this.o.arrows) return;
      // With numbers on, the number owns the middle of the tile and the arrows become chevrons by its edges.
      const p = this._probs(s), edge = this.o.numbers;
      for (let a = 0; a < nA; a++) {
        const arrow = this.arrows[s][a], [dx, dy] = DIRS[a];
        if (p[a] < 0.06) { arrow.style.opacity = 0; continue; }
        const r0 = 6, r1 = edge ? 27 : 8 + 18 * p[a], hs = edge ? 2.6 + 3.4 * p[a] : 4 + 2.5 * p[a];
        const bx = dx * (r1 - hs), by = dy * (r1 - hs), nx = -dy * hs * 0.85, ny = dx * hs * 0.85;
        const head = `M${bx + nx} ${by + ny}L${dx * r1} ${dy * r1}L${bx - nx} ${by - ny}`;
        const d = edge ? head : `M${dx * r0} ${dy * r0}L${dx * r1} ${dy * r1}${head}`;
        arrow.halo.setAttribute("d", d);
        arrow.shaft.setAttribute("d", d);
        arrow.style.opacity = 1;
      }
    }

    // The glow of a trace: one shape per tile (traces of states) or per move (traces of state–action pairs).
    // Brightness grows with the trace and saturates at 1, so a fading trail stays visible for a while.
    _trace(s) {
      const Z = this.o.traces !== false ? this.Z : null, nA = this.env.nA;
      if (!Z && !this.traces?.[s]) return;
      this.traces ||= [];
      let g = this.traces[s];
      if (!g) {
        const [x, y] = this.origin(s), perMove = Z.length === this.env.nS * nA;
        g = this.traces[s] = el("g", { class: "trace", transform: `translate(${x} ${y})` }, this.gTrace);
        g.parts = perMove ? TRIANGLES.map(([p, q]) => el("polygon", { points: `${p} ${q} ${HALF},${HALF}` }, g)) : [el("rect", { x: 3, y: 3, width: T - 6, height: T - 6, rx: 7 }, g)];
      }
      g.parts.forEach((part, a) => {
        const z = Z ? (g.parts.length > 1 ? Z[s * nA + a] : Z[s]) : 0;
        part.style.opacity = z > 0.005 ? (0.18 + 0.8 * Math.min(1, z) ** 0.7).toFixed(3) : 0;
      });
    }

    // Fog over the tiles the model knows nothing about yet: planning can only use what the agent has seen.
    _fog(s) {
      const M = this.o.fog !== false ? this.model : null, nA = this.env.nA;
      if (!M && !this.fogs?.[s]) return;
      this.fogs ||= [];
      if (!this.fogs[s]) { const [x, y] = this.origin(s); this.fogs[s] = el("rect", { class: "fog", x: x + 1.5, y: y + 1.5, width: T - 3, height: T - 3, rx: 7 }, this.gFog); }
      let known = false;
      if (M) for (let a = 0; a < nA; a++) if (M[s * nA + a] >= 0) { known = true; break; }
      this.fogs[s].style.opacity = M && !known && this.env.tile(s) !== "G" ? 1 : 0;
    }

    // ---- the Lab: one event of a run, and the picture at rest after a jump ----
    // Returns how long to pause after the event, in milliseconds.
    event(ev, { line = false } = {}) {
      const env = this.env;
      switch (ev.type) {
        case "start":
          this.mark([], "focus");
          this.place(ev.s, -1, true);
          this.body.animate?.(RESPAWN, { duration: 300, easing: "cubic-bezier(.2,.9,.25,1.25)" });
          this.path_ = [ev.s];
          this.trail(this.o.trail ? this.path_ : null);
          return 0;
        case "choose":
          if (ev.a >= 0) { this.hint(ev.s, ev.a); this.look(ev.a); }
          return 0;
        case "move": {
          let wait = 0;
          if (ev.fell !== undefined) {
            this.fall(ev.fell, ev.s2);
            this.pop(ev.fell, signed(ev.r));
            this.path_.push(ev.fell, -1, ev.s2);
            wait = 650;
          } else if (env.jumps.some((j) => j.from === ev.s)) {
            this.spark(ev.s, -1);
            this.place(ev.s2, ev.a, true);
            this.spark(ev.s2, -1);
            this.pop(ev.s2, signed(ev.r));
            this.path_.push(-1, ev.s2);
            wait = 300;
          } else {
            this.move(ev.s2, ev.a);
            if (ev.slid !== undefined) this.slide(ev.s2);
            const k = env.tile(ev.s2);
            if (k === "H") { this.sink(); wait = 500; }
            else if (k === "G" && ev.r) { this.pop(ev.s2, signed(ev.r, Number.isInteger(ev.r) ? 0 : 2)); wait = 300; }
            else if (line && ev.r) this.pop(ev.s2, signed(ev.r, Number.isInteger(ev.r) ? 0 : 2));
            this.path_.push(ev.s2);
          }
          this.trail(this.o.trail ? this.path_ : null);
          return wait;
        }
        case "update":
          this.spark(ev.s, ev.a ?? -1);
          if (ev.states && line) this.mark([], "next");
          return 0;
        case "window": // n-step: the stretch of the episode whose rewards make the target
          this.mark([ev.s], "focus");
          this.mark(ev.states.slice(1), "next");
          return 0;
        case "trace":
          this.spark(ev.s, ev.a ?? -1, "trc");
          return 0;
        case "plan": { // planning: remembered moves replayed and learned from, in the order they were replayed
          const list = ev.list.slice(0, 60), gap = ev.ordered ? 45 : 600 / Math.max(10, list.length);
          list.forEach((u, k) => this.spark(u.s, u.a, "dream", k * gap));
          if (line) this.pop(ev.s, `${ev.list.length} planning ${ev.list.length === 1 ? "update" : "updates"}`, "plan");
          return ev.ordered ? Math.min(900, list.length * gap) : 0;
        }
        case "world":
          this.retile();
          this.pop(ev.s, "the walls moved!", "plan");
          return 900;
        case "sweep":
          this.mark([ev.s], "focus");
          this.mark(line ? this._successors(ev.s) : [], "next");
          return 0;
        case "improve":
          if (ev.changed) this.spark(ev.s, -1);
          return 0;
        case "return":
          this.mark([ev.s], "focus");
          this.pop(ev.s, `G ${signed(ev.G, Number.isInteger(ev.G) ? 0 : 1)}`, "ret");
          return 0;
        case "skip":
          this.pop(ev.s, "seen", "skip");
          return 0;
        case "cut":
          this.pop(ev.s, "✕ stop", "cut");
          return 450;
        default:
          return 0;
      }
    }

    // After a jump to the start of a unit: the path of the unit before it, and the agent where it ended.
    rest(events, draw = false) {
      this.mark([], "focus");
      this.mark([], "next");
      const path = [];
      for (const ev of events) {
        if (ev.type === "start") path.push(ev.s);
        else if (ev.type === "move") { if (ev.fell !== undefined) path.push(ev.fell, -1, ev.s2); else if (this.env.jumps.some((j) => j.from === ev.s)) path.push(-1, ev.s2); else path.push(ev.s2); }
      }
      this.trail(this.o.trail && path.length ? path : null, draw);
      this.bot.classList.remove("sunk");
      if (path.length) this.place(path[path.length - 1], -1, true);
      else this.place(this.env.start, -1, true);
    }

    _successors(s) {
      const out = new Set();
      for (const a of this.env.acts(s)) for (const { s2 } of this.env.model(s, a)) if (s2 !== s) out.add(s2);
      return [...out];
    }

    // ---- the agent ----
    place(s, look = -1, instant = false) {
      this._stopFall();
      this.bot.classList.remove("sunk");
      this._put(s, instant);
      if (look >= 0) this.look(look);
    }

    _put(s, instant) {
      const [x, y] = this.center(s);
      if (instant) this.bot.classList.add("instant");
      this.bot.style.transform = `translate(${x}px, ${y}px)`;
      if (instant) { this.bot.getBoundingClientRect(); this.bot.classList.remove("instant"); }
      this.at = s;
    }

    look(a) {
      const [dx, dy] = DIRS[a];
      this.eyes.style.transform = `translate(${dx * 3}px, ${dy * 3}px)`;
    }

    showAgent(on) { this.bot.classList.toggle("gone", !on); }

    move(s, a) {
      this.place(s, a);
      if (!RL.reducedMotion()) this.body.animate(HOP, { duration: 220, easing: "ease-out" });
    }

    // The ice gave way under the agent: it went somewhere it did not choose.
    slide(s) {
      if (!RL.reducedMotion()) this.body.animate(WOBBLE, { duration: 360, easing: "ease-out" });
      this.pop(s, "slip", "slip");
    }

    // Through the ice: the agent sinks and stays gone until the next episode.
    sink() {
      if (RL.reducedMotion()) { this.bot.classList.add("sunk"); return; }
      this._fallTimer = setTimeout(() => {
        this._fallAnim = this.body.animate(SINK, { duration: 380, easing: "ease-in", fill: "forwards" });
        this._fallAnim.onfinish = () => { this._stopFall(); this.bot.classList.add("sunk"); };
      }, 200);
    }

    // Walk into the cliff tile, tumble down, and reappear at `back`.
    fall(cell, back) {
      this.place(cell, 2);
      this._fallTimer = setTimeout(() => {
        this.spark(cell, -1);
        this._fallAnim = this.body.animate(FALL, { duration: 420, easing: "ease-in", fill: "forwards" });
        this._fallAnim.onfinish = () => {
          this._stopFall();
          this._put(back, true);
          this.body.animate(RESPAWN, { duration: 380, easing: "cubic-bezier(.2,.9,.25,1.25)" });
        };
      }, 190);
    }

    _stopFall() {
      clearTimeout(this._fallTimer);
      if (this._fallAnim) {
        this._fallAnim.onfinish = null;
        this._fallAnim.cancel();
        this._fallAnim = null;
      }
    }

    // The sandbox paints a tile before the world is rebuilt: a quick overlay in the new tile's color.
    preview(s, tile) {
      const [x, y] = this.origin(s);
      this.gFx.querySelector(`.paint[data-s="${s}"]`)?.remove();
      el("rect", { class: `paint k-${KINDS[tile] || tile}`, "data-s": s, x: x + 1.5, y: y + 1.5, width: T - 3, height: T - 3, rx: 7 }, this.gFx);
    }

    // ---- effects ----
    pop(s, text, kind = "rew") {
      const [x, y] = this.center(s);
      const t = el("text", { class: `pop k-${kind}`, x, y: y - 12 }, this.gFx);
      t.textContent = text;
      t.animate(
        [{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-22px)" }],
        { duration: 950, easing: "ease-out" },
      ).onfinish = () => t.remove();
    }

    // A ring where an update just happened: on the triangle of action a, or on the whole tile when a < 0.
    // kind: "trc" for a trace being laid down, "dream" for a planning update; delay in milliseconds.
    spark(s, a, kind = "", delay = 0) {
      let [x, y] = this.center(s);
      if (a >= 0) { x += DIRS[a][0] * 17; y += DIRS[a][1] * 17; }
      const c = el("circle", { class: `spark${kind ? ` k-${kind}` : ""}`, cx: x, cy: y, r: a >= 0 ? 9 : 20, opacity: 0 }, this.gFx);
      c.animate([{ opacity: 1, transform: "scale(0.3)" }, { opacity: 0, transform: "scale(1.7)" }], { duration: 600, delay, easing: "ease-out" })
        .onfinish = () => c.remove();
    }

    // Briefly mark the action being chosen.
    hint(s, a) {
      const t = this.tris[s]?.[a];
      if (!t) return;
      t.classList.remove("hinted");
      t.getBoundingClientRect();
      t.classList.add("hinted");
      clearTimeout(t.hintTimer);
      t.hintTimer = setTimeout(() => t.classList.remove("hinted"), 500);
    }

    // Keep these (state, action) triangles glowing until the next call.
    glow(list) {
      for (const t of this.svg.querySelectorAll(".tri.glow")) t.classList.remove("glow");
      for (const [s, a] of list) this.tris[s]?.[a]?.classList.add("glow");
    }

    // The path of an episode (thin, gray) and the greedy path (thick, policy color); both can draw themselves in.
    trail(states, animate = false) { this._draw(this.trailEl, states, animate); }
    path(states, animate = false) { this._draw(this.pathEl, states, animate); }

    _draw(pathEl, states, animate) {
      pathEl.setAttribute("d", states ? this._line(states) : "");
      pathEl.classList.remove("draw");
      if (states && animate && !RL.reducedMotion()) { pathEl.getBoundingClientRect(); pathEl.classList.add("draw"); }
    }

    _line(states) {
      let d = "", jump = true;
      for (const s of states) {
        if (s < 0) { jump = true; continue; }
        const [x, y] = this.center(s);
        d += `${jump ? "M" : "L"}${x} ${y}`;
        jump = false;
      }
      return d;
    }

    // Steps from each tile to the nearest goal (for the ripple).
    _distances() {
      if (this._dist) return this._dist;
      const { env } = this, dist = new Array(env.nS).fill(Infinity), queue = [];
      for (let s = 0; s < env.nS; s++) if ("GT".includes(env.tile(s))) { dist[s] = 0; queue.push(s); }
      while (queue.length) {
        const s = queue.shift(), [r, c] = env.rc(s);
        for (const [dc, dr] of DIRS) {
          const r2 = r + dr, c2 = c + dc, s2 = r2 * env.cols + c2;
          if (r2 < 0 || c2 < 0 || r2 >= env.rows || c2 >= env.cols || env.blocked(s2) || dist[s2] <= dist[s] + 1) continue;
          dist[s2] = dist[s] + 1;
          queue.push(s2);
        }
      }
      return (this._dist = dist.map((d) => (Number.isFinite(d) ? d : 0)));
    }

    _hover(e) {
      const s = this.tileAt(e);
      if (s < 0 || !this.tris[s] || this.o.editing) { RL.tip.hide(); return; }
      const [r, c] = this.env.rc(s);
      const { nA } = this.env, Q = this.Q, probs = this._probs(s), d = this.o.digits;
      const head = `<div class="head">Tile row ${r + 1}, column ${c + 1}</div>`;
      if (this.V || !this.hasQ) {
        RL.tip.show(e.clientX, e.clientY, `${head}<div class="row"><b>${fmt(this._value(s), d)}</b><span>V, the value of this tile</span></div>` +
          (this.Z?.length === this.env.nS ? `<div class="row"><b>${this.Z[s].toFixed(2)}</b><span>z, its eligibility trace</span></div>` : "") +
          probs.map((p, a) => (p ? `<div class="row"><span>${ARROW_TEXT[a]}</span><span class="faint">π ${Math.round(p * 100)}%</span></div>` : "")).join(""));
        return;
      }
      const rows = Q.subarray(s * nA, s * nA + nA), Z = this.Z?.length === Q.length ? this.Z : null;
      RL.tip.show(e.clientX, e.clientY, `${head}<div class="row"><b>${fmt(lab.maxQ(Q, s, this.env), d)}</b><span>V = best Q</span></div>` +
        Array.from(rows, (q, a) => `<div class="row"><span>${ARROW_TEXT[a]}</span><b>${fmt(q, d)}</b><span class="faint">π ${Math.round(probs[a] * 100)}%${Z && Z[s * nA + a] > 0.005 ? ` · trace ${Z[s * nA + a].toFixed(2)}` : ""}</span></div>`).join(""));
    }

    destroy() {
      this._stopFall();
      clearTimeout(this._rippleTimer);
      RL.tip.hide();
      this.svg.remove();
    }

    // ---- for the Lab: the choices in its "Show" panel, a legend, and a small drawing for the filmstrip ----
    static options(env, displays) {
      const hasQ = displays.some((d) => d.Q), episodes = displays.some((d) => !d.sweeps);
      const traces = displays.some((d) => d.Z), model = displays.some((d) => d.model);
      return [
        { key: "tiles", type: "seg", label: "Color the tiles by", value: hasQ ? "q" : "v",
          choices: hasQ ? [["q", "Q per move"], ["v", "V per tile"], ["none", "Off"]] : [["v", "V per tile"], ["none", "Off"]] },
        { key: "arrows", type: "check", label: "Policy arrows", swatch: "pol", value: true },
        ...(episodes ? [{ key: "trail", type: "check", label: "Path of the episode", swatch: "trail", value: true }] : []),
        { key: "numbers", type: "check", label: "Numbers on tiles", value: env.nS <= 16 },
        ...(traces ? [{ key: "traces", type: "check", label: "Eligibility traces", swatch: "trc", value: true }] : []),
        ...(model ? [{ key: "fog", type: "check", label: "Fog where the model knows nothing", swatch: "fog", value: true }] : []),
      ];
    }

    static legend() {
      return '<div class="scale"><span>worse</span><i></i><span>0</span><i class="up"></i><span>better</span></div>';
    }

    static thumb(env, d, p = {}) {
      const S = Math.max(9, Math.min(16, Math.floor(176 / env.cols))), range = env.valueRange, nA = env.nA;
      const eps = d.greedy ? 0 : p.epsilon ?? 0;
      if (d.t !== undefined) env.setTime?.(d.t); // a maze whose walls move: draw them as they were then
      let g = "";
      for (let s = 0; s < env.nS; s++) {
        const [r, c] = env.rc(s), k = env.tile(s), x = c * S, y = r * S;
        const fill = k === "#" ? "var(--ink-3)" : k === "C" || k === "H" ? "var(--chasm)" : k === "G" ? "var(--rew)" : k === "T" ? "var(--surface-3)"
          : valueColor(d.V ? d.V[s] : d.Q ? lab.maxQ(d.Q, s, env) : 0, range);
        g += `<rect x="${x + 0.5}" y="${y + 0.5}" width="${S - 1}" height="${S - 1}" rx="2" style="fill:${fill}"/>`;
        if (env.terminal(s) || env.blocked(s) || (!d.Q && !d.P)) continue;
        const probs = d.P ? Array.from(d.P.subarray(s * nA, s * nA + nA)) : lab.epsGreedyProbs(d.Q, s, env, eps);
        const best = probs.indexOf(Math.max(...probs));
        if (probs.filter((q) => q === probs[best]).length === nA) continue; // no preference yet
        const [dx, dy] = DIRS[best], cx = x + S / 2, cy = y + S / 2;
        g += `<path class="t-arrow" d="M${cx - dx * S * 0.28} ${cy - dy * S * 0.28} L${cx + dx * S * 0.32} ${cy + dy * S * 0.32}"/>`;
      }
      return `<svg class="thumb-grid" viewBox="0 0 ${env.cols * S} ${env.rows * S}">${g}</svg>`;
    }
  }

  GridView.valueColor = valueColor;
  RL.GridView = GridView;
  (RL.labViews = RL.labViews || {}).grid = GridView;
})(globalThis.RL = globalThis.RL || {});
