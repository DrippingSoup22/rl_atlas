/* Policy-gradient learners with networks: A2C, PPO and TRPO, for a softmax policy over a few actions or a Gaussian one
   over a real-valued action. Several workers play at once; GAE judges every step; the critic is a separate network.
   The JavaScript port of recorder/pg.py, same settings and same defaults. */
(function (RL) {
  "use strict";
  const D = RL.dmath, deep = (RL.deep = RL.deep || {});

  deep.PG_DEFAULTS = {
    algo: "ppo", workers: 8, steps: 128, gamma: 0.99, lam: 0.95, lr: 3e-4, lr_v: 1e-3, epochs: 10, minibatch: 64, clip: 0.2,
    ent: 0, delta: 0.01, v_epochs: 10, depth: 2, width: 64, norm_adv: true, log_std0: 0, reward_scale: 1,
  };
  const LOG_2PI = 1.8378770664093453;

  // A random order of 0 … n−1 (Fisher–Yates).
  const permutation = (n, rng) => { const p = Int32Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = rng.int(i + 1), t = p[i]; p[i] = p[j]; p[j] = t; } return p; };
  const rows = (x, dim, idx) => { const out = new Float64Array(idx.length * dim); for (let k = 0; k < idx.length; k++) for (let c = 0; c < dim; c++) out[k * dim + c] = x[idx[k] * dim + c]; return out; };

  // π(a | s, θ): a softmax over logits, or a Gaussian with a learned mean and a log spread that is a free parameter.
  class Policy {
    constructor(dim, world, cfg, rng) {
      this.discrete = !world.act; this.dim = dim;
      this.out = this.discrete ? world.nA : world.act;
      this.net = new deep.MLP([dim, ...deep.hiddenOf(cfg), this.out], rng, "tanh", 0.01);
      this.logStd = this.discrete ? null : new Float64Array(this.out).fill(cfg.log_std0);
      this.lo = world.lo; this.hi = world.hi;
      this.params = [...this.net.params, ...(this.discrete ? [] : [this.logStd])];
    }
    // probabilities (B × nA), or means (B × out) and spreads (out)
    dist(x, B) {
      const y = this.net.forward(x, B), n = this.out;
      if (!this.discrete) return { mu: y.slice(), sd: Float64Array.from(this.logStd, (l) => D.exp(l)) };
      const p = new Float64Array(B * n);
      for (let r = 0; r < B; r++) {
        let m = -Infinity, z = 0;
        for (let a = 0; a < n; a++) m = Math.max(m, y[r * n + a]);
        for (let a = 0; a < n; a++) { p[r * n + a] = D.exp(y[r * n + a] - m); z += p[r * n + a]; }
        for (let a = 0; a < n; a++) p[r * n + a] /= z;
      }
      return { p };
    }
    // one action per state: an index, or a vector of out numbers
    sample(x, B, rng) {
      const d = this.dist(x, B), n = this.out, out = [];
      for (let r = 0; r < B; r++) {
        if (this.discrete) { // the first cumulative probability above a uniform draw (never past the last action)
          const u = rng.next(); let c = 0, a = 0;
          for (; a < n; a++) { c += d.p[r * n + a]; if (!(u > c)) break; }
          out.push(Math.min(a, n - 1));
        } else out.push(Array.from({ length: n }, (_, j) => d.mu[r * n + j] + d.sd[j] * rng.normal()));
      }
      return out;
    }
    logProb(x, a, B) {
      const d = this.dist(x, B), n = this.out, lp = new Float64Array(B);
      for (let r = 0; r < B; r++) {
        if (this.discrete) { lp[r] = D.log(d.p[r * n + a[r]] + 1e-12); continue; }
        let s = 0;
        for (let j = 0; j < n; j++) { const z = (a[r][j] - d.mu[r * n + j]) / d.sd[j]; s += -0.5 * z * z - this.logStd[j] - 0.5 * LOG_2PI; }
        lp[r] = s;
      }
      return lp;
    }
    // The gradient of Σ_i w_i ln π(a_i | s_i) + β Σ_i H(π(· | s_i)), one array per parameter (copies).
    grad(x, a, w, B, beta = 0) {
      const y = this.net.forward(x, B), n = this.out, d = new Float64Array(B * n);
      if (this.discrete) {
        for (let r = 0; r < B; r++) {
          let m = -Infinity, z = 0;
          const p = new Float64Array(n);
          for (let j = 0; j < n; j++) m = Math.max(m, y[r * n + j]);
          for (let j = 0; j < n; j++) { p[j] = D.exp(y[r * n + j] - m); z += p[j]; }
          for (let j = 0; j < n; j++) p[j] /= z;
          for (let j = 0; j < n; j++) d[r * n + j] = -p[j] * w[r];
          d[r * n + a[r]] += w[r];
          if (beta) {
            let H = 0; const lp = Float64Array.from(p, (q) => D.log(q + 1e-12));
            for (let j = 0; j < n; j++) H -= p[j] * lp[j];
            for (let j = 0; j < n; j++) d[r * n + j] += beta * (-p[j] * (lp[j] + H));
          }
        }
        this.net.backward(d);
        return this.net.grads.map((g) => g.slice());
      }
      const sd = Float64Array.from(this.logStd, (l) => D.exp(l)), dls = new Float64Array(n);
      for (let r = 0; r < B; r++) for (let j = 0; j < n; j++) {
        const z = (a[r][j] - y[r * n + j]) / sd[j];
        d[r * n + j] = (w[r] * z) / sd[j];
        dls[j] += w[r] * (z * z - 1);
      }
      for (let j = 0; j < n; j++) dls[j] += beta * B; // ∂H/∂ln σ = 1 per sample and dimension
      this.net.backward(d);
      return [...this.net.grads.map((g) => g.slice()), dls];
    }
    // The average KL(π_old ‖ π) over states x; old is what dist returned before the update.
    klFrom(x, B, old) {
      const d = this.dist(x, B), n = this.out;
      let s = 0;
      for (let r = 0; r < B; r++) for (let j = 0; j < n; j++) {
        if (this.discrete) { const q = old.p[r * n + j]; s += q * (D.log(q + 1e-12) - D.log(d.p[r * n + j] + 1e-12)); }
        else { const m0 = old.mu[r * n + j], s0 = old.sd[j], m = d.mu[r * n + j], s1 = d.sd[j]; s += D.log(s1 / s0) + (s0 * s0 + (m0 - m) * (m0 - m)) / (2 * s1 * s1) - 0.5; }
      }
      return s / B;
    }
    // F v, with F the Fisher information of the policy averaged over states x: the network's Jacobian carries v to its
    // outputs (a central difference), the distribution's Fisher matrix acts there, and backward carries it back.
    fisherVector(x, B, v, eps = 1e-5) {
      const P = this.net.params, n = this.out;
      const shift = (f) => P.forEach((p, i) => { const d = v[i]; for (let k = 0; k < p.length; k++) p[k] += f * d[k]; });
      shift(eps); const up = this.net.forward(x, B).slice();
      shift(-2 * eps); const down = this.net.forward(x, B).slice();
      shift(eps);
      const y = this.net.forward(x, B), g = new Float64Array(B * n);
      if (this.discrete) {
        for (let r = 0; r < B; r++) {
          let m = -Infinity, z = 0, pj = 0;
          const p = new Float64Array(n), jv = new Float64Array(n);
          for (let j = 0; j < n; j++) m = Math.max(m, y[r * n + j]);
          for (let j = 0; j < n; j++) { p[j] = D.exp(y[r * n + j] - m); z += p[j]; jv[j] = (up[r * n + j] - down[r * n + j]) / (2 * eps); }
          for (let j = 0; j < n; j++) { p[j] /= z; }
          for (let j = 0; j < n; j++) pj += p[j] * jv[j];
          for (let j = 0; j < n; j++) g[r * n + j] = (p[j] * jv[j] - p[j] * pj) / B; // (diag π − π πᵀ) J v
        }
        this.net.backward(g);
        return this.net.grads.map((q) => q.slice());
      }
      for (let r = 0; r < B; r++) for (let j = 0; j < n; j++) g[r * n + j] = (up[r * n + j] - down[r * n + j]) / (2 * eps) / D.exp(2 * this.logStd[j]) / B;
      this.net.backward(g);
      return [...this.net.grads.map((q) => q.slice()), Float64Array.from(v[v.length - 1], (q) => 2 * q)];
    }
    forEnv(a) { return this.discrete ? a : a.map((v) => Math.max(this.lo, Math.min(this.hi, v))); }
  }

  // Advantages and λ-returns for T steps of N workers (T × N, row-major). done: the episode ended (no bootstrap); cut:
  // cut by the time limit (bootstrap from the value of the last state, in vNext).
  deep.gae = function (r, v, vNext, done, cut, T, N, gamma, lam) {
    const adv = new Float64Array(T * N), a = new Float64Array(N);
    for (let t = T - 1; t >= 0; t--) for (let i = 0; i < N; i++) {
      const k = t * N + i, delta = r[k] + gamma * vNext[k] * (1 - done[k]) - v[k];
      a[i] = delta + gamma * lam * (1 - done[k]) * (1 - cut[k]) * a[i];
      adv[k] = a[i];
    }
    return { adv, ret: Float64Array.from(adv, (x, k) => x + v[k]) };
  };

  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) for (let k = 0; k < a[i].length; k++) s += a[i][k] * b[i][k]; return s; };
  // Solve F x = b approximately, with F available only through products F v.
  function conjugateGradient(fvp, b, iters = 10) {
    const x = b.map((g) => new Float64Array(g.length)), r = b.map((g) => g.slice()), p = b.map((g) => g.slice());
    let rr = dot(r, r);
    for (let it = 0; it < iters; it++) {
      const fp = fvp(p), alpha = rr / (dot(p, fp) + 1e-10);
      for (let i = 0; i < x.length; i++) for (let k = 0; k < x[i].length; k++) { x[i][k] += alpha * p[i][k]; r[i][k] -= alpha * fp[i][k]; }
      const rrNew = dot(r, r);
      if (rrNew < 1e-10) break;
      for (let i = 0; i < p.length; i++) for (let k = 0; k < p[i].length; k++) p[i][k] = p[i][k] * (rrNew / rr) + r[i][k];
      rr = rrNew;
    }
    return x;
  }

  // A2C, PPO or TRPO with a policy network, a critic network and N workers.
  class PG {
    constructor(dim, world, config, rng) {
      const cfg = (this.cfg = { ...deep.PG_DEFAULTS, ...config });
      this.rng = rng; this.dim = dim;
      this.pi = new Policy(dim, world, cfg, rng);
      this.v = new deep.MLP([dim, ...deep.hiddenOf(cfg), 1], rng, "tanh");
      this.optPi = new deep.Adam(this.pi.params, cfg.lr, 0.5);
      this.optV = new deep.Adam(this.v.params, cfg.lr_v, 0.5);
      this.stats = {};
    }
    value(x, B) { return this.v.forward(x, B).slice(); }
    // One update from a batch: states (n × dim), actions, advantages and λ-returns, over steps and workers.
    update(s, a, adv, ret) {
      const cfg = this.cfg, n = a.length, pi = this.pi, dim = this.dim;
      if (cfg.norm_adv) {
        let m = 0, v = 0;
        for (const x of adv) m += x;
        m /= n;
        for (const x of adv) v += (x - m) * (x - m);
        const sd = Math.sqrt(v / n) + 1e-8;
        adv = Float64Array.from(adv, (x) => (x - m) / sd);
      }
      const old = pi.dist(s, n), logpOld = pi.logProb(s, a, n);
      const neg = (g) => g.map((x) => Float64Array.from(x, (y) => -y));
      if (cfg.algo === "a2c") {
        this.optPi.step(neg(pi.grad(s, a, Float64Array.from(adv, (x) => x / n), n, cfg.ent / n)));
        this.stats.clipped = 0;
      } else if (cfg.algo === "ppo") {
        let clipped = 0;
        for (let e = 0; e < cfg.epochs; e++) {
          const order = permutation(n, this.rng);
          clipped = 0;
          for (let k = 0; k < n; k += cfg.minibatch) {
            const idx = order.subarray(k, k + cfg.minibatch), B = idx.length, xs = rows(s, dim, idx), as = Array.from(idx, (i) => a[i]);
            const lp = pi.logProb(xs, as, B), w = new Float64Array(B);
            for (let j = 0; j < B; j++) {
              const i = idx[j], r = D.exp(lp[j] - logpOld[i]);
              // a sample pushes only while its ratio is inside 1 ± ε in the direction its advantage favors
              const out = cfg.clip ? (adv[i] > 0 && r > 1 + cfg.clip) || (adv[i] < 0 && r < 1 - cfg.clip) : false;
              if (out) clipped++;
              w[j] = out ? 0 : (adv[i] * r) / B;
            }
            this.optPi.step(neg(pi.grad(xs, as, w, B, cfg.ent / B)));
          }
        }
        this.stats.clipped = clipped / n;
      } else { // TRPO
        const g = pi.grad(s, a, Float64Array.from(adv, (x) => x / n), n);
        const step = conjugateGradient((v) => pi.fisherVector(s, n, v).map((f, i) => Float64Array.from(f, (y, k) => y + 0.1 * v[i][k])), g);
        const shs = dot(step, pi.fisherVector(s, n, step)), scale = Math.sqrt((2 * cfg.delta) / (shs + 1e-10));
        const before = pi.params.map((p) => p.slice());
        let surr0 = 0;
        for (const x of adv) surr0 += x;
        surr0 /= n; // the surrogate at the old policy: ratio 1 times the advantage, on average
        let accepted = 0, frac = 1;
        for (let it = 0; it < 10; it++, frac /= 2) {
          pi.params.forEach((p, i) => { for (let k = 0; k < p.length; k++) p[k] = before[i][k] + frac * scale * step[i][k]; });
          const lp = pi.logProb(s, a, n);
          let surr = 0;
          for (let i = 0; i < n; i++) surr += D.exp(lp[i] - logpOld[i]) * adv[i];
          if (pi.klFrom(s, n, old) <= cfg.delta && surr / n > surr0) { accepted = frac; break; }
        }
        if (!accepted) pi.params.forEach((p, i) => p.set(before[i]));
        this.stats.step = accepted;
      }
      this.stats.kl = pi.klFrom(s, n, old);
      // the critic: a few passes of minibatch steps toward the λ-returns
      for (let e = 0; e < cfg.v_epochs; e++) {
        const order = permutation(n, this.rng);
        for (let k = 0; k < n; k += cfg.minibatch) {
          const idx = order.subarray(k, k + cfg.minibatch), B = idx.length, out = this.v.forward(rows(s, dim, idx), B), d = new Float64Array(B);
          for (let j = 0; j < B; j++) d[j] = (out[j] - ret[idx[j]]) / B;
          this.v.backward(d);
          this.optV.step(this.v.grads);
        }
      }
    }
  }
  deep.PG = PG;
})(globalThis.RL = globalThis.RL || {});
