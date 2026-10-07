/* DQN and its parts, each of which can be switched off or upgraded, so one implementation trains every comparison of
   Part 9: experience replay or only the latest experience, a target network or none, double Q-learning, a dueling
   network, prioritized replay; a squared or a Huber loss. The JavaScript port of recorder/dqn.py, same settings and
   same defaults. */
(function (RL) {
  "use strict";
  const D = RL.dmath, deep = (RL.deep = RL.deep || {});

  deep.DQN_DEFAULTS = {
    lr: 1e-3, gamma: 0.99, batch: 64, buffer: 50000, start: 1000, train_every: 1, target_every: 500, tau: 1, grad_steps: 1,
    eps: [1, 0.05, 10000], double: false, dueling: false, prioritized: false, alpha: 0.6, beta: [0.4, 1],
    depth: 2, width: 64, act: "relu", clip: 10, huber: true, // the network: depth hidden layers of width units
  };
  // The hidden layers of a configuration: its own list, or depth layers of width units.
  deep.hiddenOf = (cfg) => cfg.hidden || Array.from({ length: cfg.depth }, () => cfg.width);

  // A circular buffer of transitions, sampled uniformly. With capacity = batch it is just the latest experience.
  class Replay {
    constructor(capacity, dim) {
      this.capacity = capacity; this.dim = dim; this.next = 0; this.size = 0;
      this.s = new Float64Array(capacity * dim); this.s2 = new Float64Array(capacity * dim);
      this.a = new Int32Array(capacity); this.r = new Float64Array(capacity); this.done = new Float64Array(capacity);
    }
    add(s, a, r, s2, done) {
      const i = this.next;
      this.s.set(s, i * this.dim); this.s2.set(s2, i * this.dim); this.a[i] = a; this.r[i] = r; this.done[i] = done ? 1 : 0;
      this.next = (i + 1) % this.capacity; this.size = Math.min(this.size + 1, this.capacity);
      return i;
    }
    // indices and importance weights (all 1 here); the whole buffer while it holds no more than n
    sample(n, rng) {
      const k = Math.min(n, this.size), idx = new Int32Array(k), w = new Float64Array(k).fill(1);
      if (this.size <= n) for (let j = 0; j < k; j++) idx[j] = j;
      else for (let j = 0; j < k; j++) idx[j] = rng.int(this.size);
      return { idx, w };
    }
    update() {}
  }

  // Proportional prioritized replay (Schaul et al., 2016): a transition is drawn with probability ∝ (|δ| + 10⁻³)^α from
  // a sum tree, one draw from each of n equal slices of the total; the weights (N P(i))^−β, over their largest, undo
  // the bias.
  class PrioritizedReplay extends Replay {
    constructor(capacity, dim, alpha) {
      super(capacity, dim);
      this.alpha = alpha; this.top = 1; this.beta = 0.4; this.tree = new Float64Array(2 * capacity);
    }
    _set(i, p) { let j = i + this.capacity; const d = p - this.tree[j]; while (j >= 1) { this.tree[j] += d; j = j >> 1; } }
    add(s, a, r, s2, done) { const i = super.add(s, a, r, s2, done); this._set(i, this.top); return i; }
    sample(n, rng) {
      const total = this.tree[1], idx = new Int32Array(n), w = new Float64Array(n), C = this.capacity;
      let wMax = 0;
      for (let k = 0; k < n; k++) {
        let u = ((k + rng.next()) * total) / n, j = 1;
        while (j < C) { j *= 2; if (u > this.tree[j]) { u -= this.tree[j]; j += 1; } }
        idx[k] = Math.min(j - C, this.size - 1);
        w[k] = D.pow((this.size * this.tree[idx[k] + C]) / total, -this.beta);
        if (w[k] > wMax) wMax = w[k];
      }
      for (let k = 0; k < n; k++) w[k] /= wMax;
      return { idx, w };
    }
    update(idx, td) {
      for (let k = 0; k < idx.length; k++) {
        const p = D.pow(Math.abs(td[k]) + 1e-3, this.alpha);
        if (p > this.top) this.top = p;
        this._set(idx[k], p);
      }
    }
  }

  // Action values from a network: plain, or dueling (a value V(s) and one advantage per action, Q = V + A − mean A).
  class QNet {
    constructor(dim, nA, cfg, rng) {
      this.nA = nA; this.dueling = cfg.dueling;
      this.net = new deep.MLP([dim, ...deep.hiddenOf(cfg), nA + (cfg.dueling ? 1 : 0)], rng, cfg.act);
      this.out = new Float64Array(0); this.dout = new Float64Array(0);
    }
    copyFrom(o) { this.net.copyFrom(o.net); }
    forward(x, B) {
      const y = this.net.forward(x, B);
      if (!this.dueling) return y;
      const nA = this.nA, w = nA + 1;
      if (this.out.length < B * nA) this.out = new Float64Array(B * nA);
      for (let r = 0; r < B; r++) {
        let mean = 0;
        for (let a = 0; a < nA; a++) mean += y[r * w + 1 + a];
        mean /= nA;
        for (let a = 0; a < nA; a++) this.out[r * nA + a] = y[r * w] + y[r * w + 1 + a] - mean;
      }
      return this.out.subarray(0, B * nA);
    }
    backward(dq, B) {
      if (!this.dueling) return this.net.backward(dq);
      const nA = this.nA, w = nA + 1;
      if (this.dout.length < B * w) this.dout = new Float64Array(B * w);
      for (let r = 0; r < B; r++) {
        let sum = 0;
        for (let a = 0; a < nA; a++) sum += dq[r * nA + a];
        this.dout[r * w] = sum;
        for (let a = 0; a < nA; a++) this.dout[r * w + 1 + a] = dq[r * nA + a] - sum / nA;
      }
      return this.net.backward(this.dout.subarray(0, B * w));
    }
  }

  // The learner: ε-greedy acting, a replay memory, a target network, a gradient step every train_every steps.
  class DQN {
    constructor(dim, nA, config, rng, total) {
      const cfg = (this.cfg = { ...deep.DQN_DEFAULTS, ...config });
      this.dim = dim; this.nA = nA; this.rng = rng; this.total = total; this.t = 0;
      this.q = new QNet(dim, nA, cfg, rng);
      this.target = new QNet(dim, nA, cfg, rng);
      this.target.copyFrom(this.q);
      this.opt = new deep.Adam(this.q.net.params, cfg.lr, cfg.clip);
      this.memory = cfg.prioritized ? new PrioritizedReplay(cfg.buffer, dim, cfg.alpha) : new Replay(cfg.buffer, dim);
      const B = cfg.batch;
      this.xb = new Float64Array(B * dim); this.x2b = new Float64Array(B * dim); this.dq = new Float64Array(B * nA);
      this.one = new Float64Array(dim);
      this.log = { loss: [], td: [], q: [] };
    }
    epsilon() { const [hi, lo, over] = this.cfg.eps; return lo + (hi - lo) * Math.max(0, 1 - this.t / over); }
    values(s) { this.one.set(s); return this.q.forward(this.one, 1).slice(); }
    // the values of many states at once (a snapshot's grid), nA per state
    valuesOf(states) {
      const out = new Float64Array(states.length * this.nA), x = new Float64Array(64 * this.dim);
      for (let i = 0; i < states.length; i += 64) {
        const B = Math.min(64, states.length - i);
        for (let r = 0; r < B; r++) x.set(states[i + r], r * this.dim);
        out.set(this.q.forward(x, B), i * this.nA);
      }
      return out;
    }
    act(s, greedy = false) {
      if (!greedy && this.rng.next() < this.epsilon()) return this.rng.int(this.nA);
      const q = this.values(s);
      let best = 0;
      for (let a = 1; a < this.nA; a++) if (q[a] > q[best]) best = a;
      return best;
    }
    observe(s, a, r, s2, done) {
      const cfg = this.cfg;
      this.memory.add(s, a, r, s2, done);
      this.t++;
      if (cfg.prioritized) this.memory.beta = cfg.beta[0] + (cfg.beta[1] - cfg.beta[0]) * Math.min(1, this.t / this.total);
      if (this.t >= cfg.start && this.t % cfg.train_every === 0) for (let k = 0; k < cfg.grad_steps; k++) this.learn();
      if (cfg.target_every && this.t % cfg.target_every === 0) {
        if (cfg.tau < 1) this.target.net.softUpdate(this.q.net, cfg.tau);
        else this.target.copyFrom(this.q);
      }
    }
    learn() {
      const cfg = this.cfg, m = this.memory, nA = this.nA, dim = this.dim;
      const { idx, w } = m.sample(cfg.batch, this.rng), B = idx.length;
      for (let k = 0; k < B; k++) {
        const j = idx[k];
        for (let c = 0; c < dim; c++) { this.xb[k * dim + c] = m.s[j * dim + c]; this.x2b[k * dim + c] = m.s2[j * dim + c]; }
      }
      // what the next states are worth: the target network's values (or, without one, the network being trained)
      const boot = new Float64Array(B), q2 = (cfg.target_every ? this.target : this.q).forward(this.x2b, B);
      if (cfg.double) { // double DQN: the online network picks the next action, the target network values it
        const nxt = q2.slice(), qo = this.q.forward(this.x2b, B);
        for (let k = 0; k < B; k++) { let b = 0; for (let a = 1; a < nA; a++) if (qo[k * nA + a] > qo[k * nA + b]) b = a; boot[k] = nxt[k * nA + b]; }
      } else for (let k = 0; k < B; k++) { let v = q2[k * nA]; for (let a = 1; a < nA; a++) v = Math.max(v, q2[k * nA + a]); boot[k] = v; }
      const q = this.q.forward(this.xb, B), td = new Float64Array(B), dq = this.dq.subarray(0, B * nA).fill(0);
      let loss = 0, tdSum = 0, qSum = 0;
      for (let k = 0; k < B; k++) {
        const j = idx[k], y = m.r[j] + cfg.gamma * (1 - m.done[j]) * boot[k], e = q[k * nA + m.a[j]] - y;
        td[k] = e;
        // Huber: squared for small errors, linear for large ones, so one surprise cannot blow up the step
        dq[k * nA + m.a[j]] = ((cfg.huber ? Math.max(-1, Math.min(1, e)) : e) * w[k]) / B;
        const ae = Math.abs(e);
        loss += w[k] * (ae < 1 ? 0.5 * e * e : ae - 0.5); tdSum += ae;
        let mx = q[k * nA]; for (let a = 1; a < nA; a++) mx = Math.max(mx, q[k * nA + a]); qSum += mx;
      }
      this.q.backward(dq, B);
      this.opt.step(this.q.net.grads);
      m.update(idx, td);
      this.log.loss.push(loss / B); this.log.td.push(tdSum / B); this.log.q.push(qSum / B);
    }
  }
  deep.DQN = DQN;
})(globalThis.RL = globalThis.RL || {});
