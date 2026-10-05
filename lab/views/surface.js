/* A small 3D surface plot on a canvas, for functions of two numbers such as Mountain Car's cost-to-go over position
   and speed. Drag to turn it, double-click to put it back. It draws the surface as shaded tiles from back to front
   (the painter's way), with a path on it (the states of an episode) and a dot (where the agent is now). The same
   projection also writes a still SVG of the surface, for filmstrip frames and figures.
   No library: a few hundred tiles need nothing more. */
(function (RL) {
  "use strict";
  const hex = (c) => {
    c = c.trim();
    if (c.startsWith("#")) {
      const n = c.length === 4 ? c.slice(1).split("").map((d) => d + d).join("") : c.slice(1);
      return [0, 2, 4].map((i) => parseInt(n.slice(i, i + 2), 16));
    }
    const m = c.match(/[\d.]+/g);
    return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
  };
  const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const rgb = (c) => `rgb(${c.map((v) => Math.round(Math.max(0, Math.min(255, v)))).join(" ")})`;

  // The colors the theme gives the low and high ends of the surface, read from the page each time it is drawn.
  function palette(from = document.documentElement, high = "--v-neg") {
    const cs = getComputedStyle(from), get = (k) => hex(cs.getPropertyValue(k) || "#888");
    return { lo: get("--v-mid"), hi: get(high), ink: cs.getPropertyValue("--ink").trim(), ink3: cs.getPropertyValue("--ink-3").trim(), line: cs.getPropertyValue("--line-2").trim(), surface: cs.getPropertyValue("--surface").trim() };
  }

  // The camera: azimuth turns the floor, elevation tilts it toward the viewer. Points are in the unit box
  // (x, y in [−½, ½], z in [0, zs]); project returns screen x, screen y (up) and depth (bigger is farther).
  function camera(az, elev) {
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(elev), se = Math.sin(elev);
    return (X, Y, Z) => {
      const x1 = X * ca - Y * sa, y1 = X * sa + Y * ca;
      return [x1, y1 * se + Z * ce, y1 * ce - Z * se];
    };
  }

  // The tiles of the surface, back to front, each with its four projected corners, depth and color.
  function tiles(Z, nx, ny, zMax, project, pal, zs) {
    const out = [], P = new Array(nx * ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      P[j * nx + i] = project(i / (nx - 1) - 0.5, j / (ny - 1) - 0.5, (Math.min(zMax, Z[j * nx + i]) / zMax) * zs);
    }
    const light = [-0.45, -0.55, 0.7], ll = Math.hypot(...light);
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i, b = a + 1, c = a + nx + 1, d = a + nx;
      const z = (Z[a] + Z[b] + Z[c] + Z[d]) / 4;
      // the tile's slope, for shading: a tile facing the light is brighter
      const dzx = ((Z[b] - Z[a] + Z[c] - Z[d]) / 2 / zMax) * zs * (nx - 1), dzy = ((Z[d] - Z[a] + Z[c] - Z[b]) / 2 / zMax) * zs * (ny - 1);
      const lam = Math.max(0, (-dzx * light[0] - dzy * light[1] + light[2]) / (Math.hypot(dzx, dzy, 1) * ll));
      const base = mix(pal.lo, pal.hi, Math.max(0, Math.min(1, z / zMax)) ** 0.85);
      out.push({ pts: [P[a], P[b], P[c], P[d]], depth: (P[a][2] + P[b][2] + P[c][2] + P[d][2]) / 4, color: rgb(base.map((v) => v * (0.72 + 0.34 * lam))) });
    }
    return out;
  }

  // Scale and center so the box fits whatever the azimuth: the floor's corners reach at most √½ from its center.
  // pad: room kept for the labels above the box.
  function fit(w, h, elev, zs, m = 18, pad = 26) {
    const r = Math.SQRT1_2, lo = -r * Math.sin(elev), hi = r * Math.sin(elev) + zs * Math.cos(elev);
    const S = Math.min((w - 2 * m) / (2 * r), (h - 2 * m - pad) / (hi - lo));
    return { S, cx: w / 2, cy: m + pad * 0.7 + hi * S };
  }

  class Surface3D {
    // opts: { x: [lo, hi], y: [lo, hi], xLabel, yLabel, zLabel, xTicks, yTicks, fx, fy, high (the token for high values) }
    constructor(host, opts) {
      this.o = { az: -0.62, elev: 0.5, zs: 0.62, ...opts };
      this.az = this.o.az;
      this.elev = this.o.elev;
      this.box = RL.h('<div class="surface3d"><canvas></canvas><span class="surface-hint">drag to turn</span></div>');
      host.appendChild(this.box);
      this.canvas = this.box.querySelector("canvas");
      this.ctx = this.canvas.getContext("2d");
      this.Z = null;
      this.pathPts = [];
      this.dotAt = null;
      let drag = null;
      this.canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, az: this.az, elev: this.elev }; this.canvas.setPointerCapture(e.pointerId); this.box.classList.add("turned"); });
      this.canvas.addEventListener("pointermove", (e) => {
        if (!drag) return;
        this.az = drag.az + (e.clientX - drag.x) * 0.01;
        this.elev = Math.max(0.05, Math.min(1.45, drag.elev + (e.clientY - drag.y) * 0.008));
        this.draw();
      });
      const end = () => { drag = null; };
      this.canvas.addEventListener("pointerup", end);
      this.canvas.addEventListener("pointercancel", end);
      this.canvas.addEventListener("dblclick", () => { this.az = this.o.az; this.elev = this.o.elev; this.draw(); });
      this.ro = new ResizeObserver(() => this.draw());
      this.ro.observe(this.box);
    }

    // The heights: Z[j·nx + i] at x = x-range at i/(nx−1), y at j/(ny−1). zMax: the top of the scale.
    set(Z, nx, ny, zMax) { this.Z = Z; this.nx = nx; this.ny = ny; this.zMax = zMax; this.draw(); }
    // A path of points [x, y] in the data's units; its heights are read off the surface.
    path(pts) { this.pathPts = pts || []; this.draw(); }
    dot(p) { this.dotAt = p; this.draw(); }

    // Height of the surface at a point in data units, by bilinear interpolation.
    heightAt(x, y) {
      const { nx, ny, Z, o } = this;
      const u = Math.max(0, Math.min(nx - 1.001, ((x - o.x[0]) / (o.x[1] - o.x[0])) * (nx - 1)));
      const v = Math.max(0, Math.min(ny - 1.001, ((y - o.y[0]) / (o.y[1] - o.y[0])) * (ny - 1)));
      const i = Math.floor(u), j = Math.floor(v), fu = u - i, fv = v - j, at = (a, b) => Z[b * nx + a];
      return (at(i, j) * (1 - fu) + at(i + 1, j) * fu) * (1 - fv) + (at(i, j + 1) * (1 - fu) + at(i + 1, j + 1) * fu) * fv;
    }

    draw() {
      const { canvas, ctx, o } = this;
      const w = this.box.clientWidth, h = Math.round(w * 0.78);
      if (!w) return;
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.height = `${h}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!this.Z) return;
      const pal = palette(this.box, o.high), project = camera(this.az, this.elev), { S, cx, cy } = fit(w, h, this.elev, o.zs);
      const scr = (p) => [cx + p[0] * S, cy - p[1] * S];
      const items = tiles(this.Z, this.nx, this.ny, this.zMax, project, pal, o.zs).map((t) => ({ ...t, kind: "tile" }));
      // the path, lifted a hair above the surface, sorted in with the tiles so hills can hide it
      const toUnit = (x, y) => [(x - o.x[0]) / (o.x[1] - o.x[0]) - 0.5, (y - o.y[0]) / (o.y[1] - o.y[0]) - 0.5];
      const lift = (x, y) => (Math.min(this.zMax, this.heightAt(x, y)) / this.zMax) * o.zs + 0.012;
      for (let k = 1; k < this.pathPts.length; k++) {
        const [x0, y0] = this.pathPts[k - 1], [x1, y1] = this.pathPts[k], u0 = toUnit(x0, y0), u1 = toUnit(x1, y1);
        const a = project(u0[0], u0[1], lift(x0, y0)), b = project(u1[0], u1[1], lift(x1, y1));
        items.push({ kind: "seg", a, b, depth: (a[2] + b[2]) / 2 - 0.01 });
      }
      items.sort((p, q) => q.depth - p.depth);
      // the floor and its axes first
      const floor = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([X, Y]) => scr(project(X, Y, 0)));
      ctx.strokeStyle = pal.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      floor.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
      ctx.closePath();
      ctx.stroke();
      this._zAxis(ctx, project, scr, pal);
      for (const it of items) {
        if (it.kind === "tile") {
          ctx.fillStyle = it.color;
          ctx.strokeStyle = it.color;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          it.pts.forEach((p, i) => (i ? ctx.lineTo(...scr(p)) : ctx.moveTo(...scr(p))));
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        } else {
          ctx.strokeStyle = pal.ink;
          ctx.lineWidth = 2;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(...scr(it.a));
          ctx.lineTo(...scr(it.b));
          ctx.stroke();
        }
      }
      if (this.dotAt) {
        const u = toUnit(...this.dotAt), p = scr(project(u[0], u[1], lift(...this.dotAt)));
        ctx.fillStyle = pal.ink;
        ctx.strokeStyle = pal.surface;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
      this._axes(ctx, project, scr, pal);
    }

    // Labels on the two floor edges nearest the viewer, and the height scale on the farthest corner.
    _axes(ctx, project, scr, pal) {
      const o = this.o;
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = pal.ink3;
      ctx.textAlign = "center";
      const edgeX = project(0, -0.5, 0)[2] < project(0, 0.5, 0)[2] ? -0.5 : 0.5; // the near edge along x
      const edgeY = project(-0.5, 0, 0)[2] < project(0.5, 0, 0)[2] ? -0.5 : 0.5;
      const label = (X, Y, text, dx = 0, dy = 0) => { const [x, y] = scr(project(X, Y, 0)); ctx.fillText(text, x + dx, y + dy); };
      const out = (v) => v * 1.16;
      for (const t of o.xTicks || []) label((t - o.x[0]) / (o.x[1] - o.x[0]) - 0.5, out(edgeX), o.fx ? o.fx(t) : String(t), 0, 4);
      for (const t of o.yTicks || []) label(out(edgeY), (t - o.y[0]) / (o.y[1] - o.y[0]) - 0.5, o.fy ? o.fy(t) : String(t), 0, 4);
      ctx.fillStyle = pal.ink;
      ctx.font = "600 11.5px system-ui, sans-serif";
      label(0, out(out(edgeX)), o.xLabel || "", 0, 10);
      label(out(out(edgeY)), 0, o.yLabel || "", 0, 10);
    }

    // The height scale, on the corner of the floor farthest from the viewer: drawn first, so the surface hides it.
    _zAxis(ctx, project, scr, pal) {
      const o = this.o, corners = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
      const far = corners.reduce((a, b) => (project(...b, 0)[2] > project(...a, 0)[2] ? b : a));
      const base = scr(project(...far, 0)), top = scr(project(...far, o.zs));
      ctx.strokeStyle = pal.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(...base);
      ctx.lineTo(...top);
      ctx.stroke();
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = pal.ink3;
      ctx.textAlign = "right";
      ctx.fillText(String(Math.round(this.zMax)), top[0] - 5, top[1] + 4);
      ctx.fillStyle = pal.ink;
      ctx.textAlign = "left";
      ctx.font = "600 11.5px system-ui, sans-serif";
      ctx.fillText(o.zLabel || "", top[0] + 6, top[1] - 6);
    }

    destroy() { this.ro.disconnect(); this.box.remove(); }

    // A still drawing of a surface as SVG, from the default angle: for filmstrip frames and figures.
    static svg(Z, nx, ny, zMax, { w = 200, h = 130, az = -0.62, elev = 0.5, zs = 0.62, high = "--v-neg", cls = "thumb-surface" } = {}) {
      const pal = palette(document.documentElement, high), project = camera(az, elev), { S, cx, cy } = fit(w, h, elev, zs, 3, 0);
      const list = tiles(Z, nx, ny, zMax, project, pal, zs).sort((p, q) => q.depth - p.depth);
      const pt = (p) => `${(cx + p[0] * S).toFixed(1)},${(cy - p[1] * S).toFixed(1)}`;
      return `<svg class="${cls}" viewBox="0 0 ${w} ${h}">${list.map((t) => `<polygon points="${t.pts.map(pt).join(" ")}" fill="${t.color}" stroke="${t.color}" stroke-width="0.4"/>`).join("")}</svg>`;
    }
  }

  RL.Surface3D = Surface3D;
})(globalThis.RL = globalThis.RL || {});
