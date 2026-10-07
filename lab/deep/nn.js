/* The deep trainer's networks: a seeded generator, a fully connected network (its forward pass, its gradients, the
   gradient with respect to its input) and Adam. Plain JavaScript on Float64Arrays, with the deterministic math of
   dmath.js, so the same seed trains the same network in Node and in any browser.

   Every sum adds its terms in one fixed order (the bias first, then the inputs in turn), whatever the batch size and
   whichever loop computes it: a state's values are the same alone or inside a batch. The loops work on two rows and
   four units at a time, which keeps the running sums in registers: about 1.7 times faster than one at a time. */
(function (RL) {
  "use strict";
  const D = RL.dmath;
  const deep = (RL.deep = RL.deep || {});

  // mulberry32 (the Lab's own generator) with normals by Box–Muller, on deterministic math.
  deep.rng = function (seed) {
    let a = seed >>> 0, spare = null;
    const next = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return {
      next,
      int: (n) => Math.floor(next() * n),
      normal() {
        if (spare !== null) { const s = spare; spare = null; return s; }
        let u = 0;
        while (u === 0) u = next();
        const v = next(), m = Math.sqrt(-2 * D.log(u));
        spare = m * D.sin(2 * Math.PI * v);
        return m * D.cos(2 * Math.PI * v);
      },
    };
  };

  // out (B × n) = x (B × m) · W (m × n) + b, row-major. Each sum starts from b[j] and adds k = 0, 1, … in turn.
  function affine(x, W, b, out, B, m, n) {
    const n4 = n - (n % 4);
    let r = 0;
    for (; r + 1 < B; r += 2) {
      const xa = r * m, xb = xa + m, oa = r * n, ob = oa + n;
      for (let j = 0; j < n4; j += 4) {
        let a0 = b[j], a1 = b[j + 1], a2 = b[j + 2], a3 = b[j + 3], c0 = a0, c1 = a1, c2 = a2, c3 = a3;
        for (let k = 0; k < m; k++) {
          const u = x[xa + k], v = x[xb + k], w = k * n + j, w0 = W[w], w1 = W[w + 1], w2 = W[w + 2], w3 = W[w + 3];
          a0 += u * w0; a1 += u * w1; a2 += u * w2; a3 += u * w3;
          c0 += v * w0; c1 += v * w1; c2 += v * w2; c3 += v * w3;
        }
        out[oa + j] = a0; out[oa + j + 1] = a1; out[oa + j + 2] = a2; out[oa + j + 3] = a3;
        out[ob + j] = c0; out[ob + j + 1] = c1; out[ob + j + 2] = c2; out[ob + j + 3] = c3;
      }
      for (let j = n4; j < n; j++) {
        let a = b[j], c = b[j];
        for (let k = 0; k < m; k++) { const w = W[k * n + j]; a += x[xa + k] * w; c += x[xb + k] * w; }
        out[oa + j] = a; out[ob + j] = c;
      }
    }
    for (; r < B; r++) {
      const xa = r * m, oa = r * n;
      for (let j = 0; j < n; j++) {
        let a = b[j];
        for (let k = 0; k < m; k++) a += x[xa + k] * W[k * n + j];
        out[oa + j] = a;
      }
    }
  }

  // dW (m × n) = xᵀ · g and db = the column sums of g, each sum over the rows r = 0, 1, … in turn.
  function weightGrads(x, g, dW, db, B, m, n) {
    const n4 = n - (n % 4);
    let k = 0;
    for (; k + 1 < m; k += 2) {
      for (let j = 0; j < n4; j += 4) {
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (let r = 0; r < B; r++) {
          const u = x[r * m + k], v = x[r * m + k + 1], q = r * n + j, g0 = g[q], g1 = g[q + 1], g2 = g[q + 2], g3 = g[q + 3];
          a0 += u * g0; a1 += u * g1; a2 += u * g2; a3 += u * g3;
          c0 += v * g0; c1 += v * g1; c2 += v * g2; c3 += v * g3;
        }
        const wa = k * n + j, wb = wa + n;
        dW[wa] = a0; dW[wa + 1] = a1; dW[wa + 2] = a2; dW[wa + 3] = a3;
        dW[wb] = c0; dW[wb + 1] = c1; dW[wb + 2] = c2; dW[wb + 3] = c3;
      }
      for (let j = n4; j < n; j++) {
        let a = 0, c = 0;
        for (let r = 0; r < B; r++) { const q = g[r * n + j]; a += x[r * m + k] * q; c += x[r * m + k + 1] * q; }
        dW[k * n + j] = a; dW[(k + 1) * n + j] = c;
      }
    }
    for (; k < m; k++) for (let j = 0; j < n; j++) { let a = 0; for (let r = 0; r < B; r++) a += x[r * m + k] * g[r * n + j]; dW[k * n + j] = a; }
    for (let j = 0; j < n; j++) { let a = 0; for (let r = 0; r < B; r++) a += g[r * n + j]; db[j] = a; }
  }

  // gx (B × m) = g (B × n) · Wᵀ, each sum over the units j = 0, 1, … in turn.
  function inputGrads(g, W, gx, B, m, n) {
    const m4 = m - (m % 4);
    let r = 0;
    for (; r + 1 < B; r += 2) {
      const ga = r * n, gb = ga + n, oa = r * m, ob = oa + m;
      for (let k = 0; k < m4; k += 4) {
        const w0 = k * n, w1 = w0 + n, w2 = w1 + n, w3 = w2 + n;
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (let j = 0; j < n; j++) {
          const u = g[ga + j], v = g[gb + j], p0 = W[w0 + j], p1 = W[w1 + j], p2 = W[w2 + j], p3 = W[w3 + j];
          a0 += u * p0; a1 += u * p1; a2 += u * p2; a3 += u * p3;
          c0 += v * p0; c1 += v * p1; c2 += v * p2; c3 += v * p3;
        }
        gx[oa + k] = a0; gx[oa + k + 1] = a1; gx[oa + k + 2] = a2; gx[oa + k + 3] = a3;
        gx[ob + k] = c0; gx[ob + k + 1] = c1; gx[ob + k + 2] = c2; gx[ob + k + 3] = c3;
      }
      for (let k = m4; k < m; k++) {
        let a = 0, c = 0;
        for (let j = 0; j < n; j++) { const p = W[k * n + j]; a += g[ga + j] * p; c += g[gb + j] * p; }
        gx[oa + k] = a; gx[ob + k] = c;
      }
    }
    for (; r < B; r++) for (let k = 0; k < m; k++) { let a = 0; for (let j = 0; j < n; j++) a += g[r * n + j] * W[k * n + j]; gx[r * m + k] = a; }
  }

  // x → hidden layers (ReLU or tanh) → a linear output; He initialization for ReLU, Glorot for tanh, as the NumPy
  // recorder did. The output layer can start small (outScale), for a policy close to uniform.
  class MLP {
    constructor(sizes, rng, act = "relu", outScale = 1) {
      this.sizes = sizes; this.act = act; this.L = sizes.length - 1;
      this.W = []; this.b = [];
      for (let i = 0; i < this.L; i++) {
        const m = sizes[i], n = sizes[i + 1], scale = (act === "relu" ? Math.sqrt(2 / m) : Math.sqrt(1 / m)) * (i === this.L - 1 ? outScale : 1);
        const W = new Float64Array(m * n);
        for (let k = 0; k < W.length; k++) W[k] = rng.normal() * scale;
        this.W.push(W); this.b.push(new Float64Array(n));
      }
      this.params = [...this.W, ...this.b];
      this.grads = this.params.map((p) => new Float64Array(p.length));
      this.cap = 0;
    }
    get size() { return this.params.reduce((s, p) => s + p.length, 0); }
    copyFrom(o) { this.params.forEach((p, i) => p.set(o.params[i])); }
    softUpdate(o, tau) { this.params.forEach((p, i) => { const q = o.params[i]; for (let k = 0; k < p.length; k++) p[k] = (1 - tau) * p[k] + tau * q[k]; }); }
    _room(B) { // buffers for a batch of B: each layer's output, and the gradients flowing back
      if (B <= this.cap) return;
      this.cap = B;
      this.h = this.sizes.slice(1).map((n) => new Float64Array(B * n));
      this.g = this.sizes.map((n) => new Float64Array(B * n));
    }
    // x: B × sizes[0]. Returns the output (a view, valid until the next call) and keeps each layer's input.
    forward(x, B) {
      this._room(B);
      this.x = x; this.B = B;
      let h = x;
      for (let i = 0; i < this.L; i++) {
        const m = this.sizes[i], n = this.sizes[i + 1], out = this.h[i];
        affine(h, this.W[i], this.b[i], out, B, m, n);
        if (i < this.L - 1) {
          const len = B * n;
          if (this.act === "relu") { for (let k = 0; k < len; k++) if (out[k] < 0) out[k] = 0; }
          else for (let k = 0; k < len; k++) out[k] = D.tanh(out[k]);
        }
        h = out;
      }
      return h.subarray(0, B * this.sizes[this.L]);
    }
    // dy: B × sizes[L], the loss's gradient at the output of the last forward pass. Fills this.grads (the order of
    // params: every W, then every b) and returns the gradient with respect to the input.
    backward(dy) {
      const B = this.B;
      let g = dy;
      for (let i = this.L - 1; i >= 0; i--) {
        const m = this.sizes[i], n = this.sizes[i + 1], x = i ? this.h[i - 1] : this.x;
        weightGrads(x, g, this.grads[i], this.grads[this.L + i], B, m, n);
        const gx = this.g[i];
        inputGrads(g, this.W[i], gx, B, m, n);
        if (i > 0) { // through the activation that produced layer i's input
          const len = B * m;
          if (this.act === "relu") { for (let k = 0; k < len; k++) if (x[k] <= 0) gx[k] = 0; }
          else for (let k = 0; k < len; k++) gx[k] *= 1 - x[k] * x[k];
        }
        g = gx;
      }
      return g.subarray(0, B * this.sizes[0]);
    }
  }
  deep.MLP = MLP;

  // Adam (Kingma & Ba, 2015) with an optional limit on the gradient's norm. The bias corrections keep running
  // products, not powers: Math.pow may round differently from one engine to the next.
  class Adam {
    constructor(params, lr, clip = 0) {
      this.params = params; this.lr = lr; this.clip = clip;
      this.m = params.map((p) => new Float64Array(p.length)); this.v = params.map((p) => new Float64Array(p.length));
      this.b1 = 1; this.b2 = 1;
    }
    step(grads, lr = this.lr) {
      let scale = 1;
      if (this.clip) {
        let s = 0;
        for (const g of grads) for (let k = 0; k < g.length; k++) s += g[k] * g[k];
        const norm = Math.sqrt(s);
        if (norm > this.clip) scale = this.clip / norm;
      }
      this.b1 *= 0.9; this.b2 *= 0.999;
      const c1 = 1 - this.b1, c2 = 1 - this.b2;
      for (let i = 0; i < this.params.length; i++) {
        const p = this.params[i], g = grads[i], m = this.m[i], v = this.v[i];
        for (let k = 0; k < p.length; k++) {
          const gk = g[k] * scale;
          m[k] = 0.9 * m[k] + 0.1 * gk;
          v[k] = 0.999 * v[k] + 0.001 * gk * gk;
          p[k] -= (lr * (m[k] / c1)) / (Math.sqrt(v[k] / c2) + 1e-8);
        }
      }
    }
  }
  deep.Adam = Adam;
})(globalThis.RL = globalThis.RL || {});
