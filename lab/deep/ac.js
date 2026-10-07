/* Off-policy actor-critics for continuous actions: DDPG, TD3 and SAC. The JavaScript port of recorder/ac.py, same
   settings and same defaults.

   All three keep a replay memory and critics that learn Q(s, a) from targets computed by slowly following copies of
   the networks. The actor climbs the critic: DDPG and TD3 have a deterministic actor, a = high · tanh(net(s)), and
   follow the critic's gradient with respect to the action; SAC has a squashed Gaussian, a = high · tanh(μ + σ ε), and
   climbs the critic plus an entropy bonus through the same reparameterization. TD3 adds twin critics (the smaller
   target wins), delayed actor updates and noise on the target action; SAC uses twin critics and tunes the bonus's
   weight α so that the policy's entropy stays near a target. */
(function (RL) {
  "use strict";
  const D = RL.dmath, deep = (RL.deep = RL.deep || {});
  const LOG_STD = [-5, 2], LOG_2PI = 1.8378770664093453; // SAC: the log spread is kept in this range

  deep.AC_DEFAULTS = {
    algo: "td3", lr_actor: 1e-3, lr_critic: 1e-3, gamma: 0.99, tau: 0.005, batch: 128, buffer: 100000, start: 1000,
    noise: 0.1, policy_noise: 0.2, noise_clip: 0.5, delay: 2, alpha: 0.2, auto_alpha: true, lr_alpha: 3e-4,
    depth: 2, width: 64, act: "relu", reward_scale: 1,
  };
  const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  class AC {
    constructor(dim, world, config, rng, total) {
      const cfg = (this.cfg = { ...deep.AC_DEFAULTS, ...config }), hidden = deep.hiddenOf(cfg), sac = cfg.algo === "sac";
      this.rng = rng; this.total = total; this.t = 0; this.updates = 0;
      this.dim = dim; this.ad = world.act; this.high = world.hi;
      const X = dim + this.ad;
      this.actor = new deep.MLP([dim, ...hidden, (sac ? 2 : 1) * this.ad], rng, cfg.act, 0.1);
      this.critics = Array.from({ length: cfg.algo === "ddpg" ? 1 : 2 }, () => new deep.MLP([X, ...hidden, 1], rng, cfg.act));
      this.criticTargets = this.critics.map(() => new deep.MLP([X, ...hidden, 1], rng, cfg.act));
      this.criticTargets.forEach((ct, i) => ct.copyFrom(this.critics[i]));
      if (!sac) { this.actorTarget = new deep.MLP([dim, ...hidden, this.ad], rng, cfg.act); this.actorTarget.copyFrom(this.actor); }
      this.optActor = new deep.Adam(this.actor.params, cfg.lr_actor);
      this.optCritics = this.critics.map((c) => new deep.Adam(c.params, cfg.lr_critic));
      this.logAlpha = D.log(cfg.alpha); this.am = 0; this.av = 0; this.ab1 = 1; this.ab2 = 1;
      const C = cfg.buffer;
      this.mem = { S: new Float64Array(C * dim), S2: new Float64Array(C * dim), A: new Float64Array(C * this.ad), R: new Float64Array(C), D: new Float64Array(C), size: 0, at: 0, C };
      this.log = { q: [], loss: [], alpha: [] };
    }
    // ---- the actor ----
    // SAC's parts for states x (B of them): mean, clipped log spread, raw log spread, spread, and u = μ + σ ε.
    sacParts(x, B, eps) {
      const out = this.actor.forward(x, B), d = this.ad, mu = new Float64Array(B * d), ls = new Float64Array(B * d), raw = new Float64Array(B * d), sd = new Float64Array(B * d), u = new Float64Array(B * d);
      for (let r = 0; r < B; r++) for (let j = 0; j < d; j++) {
        const k = r * d + j;
        mu[k] = out[r * 2 * d + j]; raw[k] = out[r * 2 * d + d + j]; ls[k] = clip(raw[k], LOG_STD[0], LOG_STD[1]); sd[k] = D.exp(ls[k]);
        u[k] = mu[k] + sd[k] * (eps ? eps[k] : 0);
      }
      return { mu, ls, raw, sd, u };
    }
    // log π(a | s) of a = high · tanh(u), u = μ + σ ε: the Gaussian's, minus the log of the squashing's slope.
    sacLogp(eps, ls, u, B) {
      const d = this.ad, out = new Float64Array(B);
      for (let r = 0; r < B; r++) {
        let s = 0;
        for (let j = 0; j < d; j++) { const k = r * d + j, t = D.tanh(u[k]); s += -0.5 * eps[k] * eps[k] - ls[k] - 0.5 * LOG_2PI - D.log(this.high * (1 - t * t) + 1e-6); }
        out[r] = s;
      }
      return out;
    }
    normals(n, rng = this.rng) { return Float64Array.from({ length: n }, () => rng.normal()); }
    // Actions for states x (B × ad): the deterministic actor (plus noise when exploring), or SAC's sample (its mean
    // when greedy).
    policy(x, B, greedy = false, rng = this.rng) {
      const d = this.ad, h = this.high;
      if (this.cfg.algo === "sac") {
        const { u } = this.sacParts(x, B, greedy ? null : this.normals(B * d, rng));
        return Float64Array.from(u, (v) => h * D.tanh(v));
      }
      const a = Float64Array.from(this.actor.forward(x, B), (z) => h * D.tanh(z));
      if (!greedy) for (let k = 0; k < a.length; k++) a[k] += this.cfg.noise * h * rng.normal();
      return a.map((v) => clip(v, -h, h));
    }
    // How widely the policy tries actions around its choice: SAC's σ carried through the squashing's slope at the
    // mean (action units), or the exploration noise.
    spread(x, B) {
      if (this.cfg.algo !== "sac") return new Float64Array(B).fill(this.cfg.noise * this.high);
      const { mu, sd } = this.sacParts(x, B, null), d = this.ad;
      return Float64Array.from({ length: B }, (_, r) => { const t = D.tanh(mu[r * d]); return this.high * sd[r * d] * (1 - t * t); });
    }
    join(x, a, B) { const d = this.dim, ad = this.ad, X = new Float64Array(B * (d + ad)); for (let r = 0; r < B; r++) { X.set(x.subarray(r * d, r * d + d), r * (d + ad)); X.set(a.subarray(r * ad, r * ad + ad), r * (d + ad) + d); } return X; }
    // Q(s, the greedy action) by the first critic: what the agent thinks a state is worth.
    value(x, B) { const a = this.policy(x, B, true); return this.critics[0].forward(this.join(x, a, B), B).slice(); }
    // ---- acting and learning ----
    act(s) {
      if (this.t < this.cfg.start) return Array.from({ length: this.ad }, () => -this.high + 2 * this.high * this.rng.next());
      return Array.from(this.policy(Float64Array.from(s), 1));
    }
    observe(s, a, r, s2, done) {
      const m = this.mem, i = m.at;
      m.S.set(s, i * this.dim); m.S2.set(s2, i * this.dim); m.A.set(a, i * this.ad); m.R[i] = r * this.cfg.reward_scale; m.D[i] = done ? 1 : 0;
      m.at = (i + 1) % m.C; m.size = Math.min(m.size + 1, m.C);
      this.t++;
      if (this.t >= this.cfg.start) this.learn();
    }
    learn() {
      const cfg = this.cfg, B = cfg.batch, m = this.mem, dim = this.dim, ad = this.ad, h = this.high;
      const S = new Float64Array(B * dim), S2 = new Float64Array(B * dim), A = new Float64Array(B * ad), R = new Float64Array(B), Dn = new Float64Array(B);
      for (let k = 0; k < B; k++) {
        const i = this.rng.int(m.size);
        S.set(m.S.subarray(i * dim, i * dim + dim), k * dim); S2.set(m.S2.subarray(i * dim, i * dim + dim), k * dim);
        A.set(m.A.subarray(i * ad, i * ad + ad), k * ad); R[k] = m.R[i]; Dn[k] = m.D[i];
      }
      this.updates++;
      const alpha = D.exp(this.logAlpha);
      // the targets, from the slowly following copies
      let A2, logp2;
      if (cfg.algo === "sac") {
        const eps2 = this.normals(B * ad), p = this.sacParts(S2, B, eps2);
        A2 = Float64Array.from(p.u, (v) => h * D.tanh(v)); logp2 = this.sacLogp(eps2, p.ls, p.u, B);
      } else {
        A2 = Float64Array.from(this.actorTarget.forward(S2, B), (z) => h * D.tanh(z));
        if (cfg.algo === "td3") for (let k = 0; k < A2.length; k++) { // smooth the target: nearby actions should be worth about the same
          const noise = clip(cfg.policy_noise * h * this.rng.normal(), -cfg.noise_clip * h, cfg.noise_clip * h);
          A2[k] = clip(A2[k] + noise, -h, h);
        }
      }
      const X2 = this.join(S2, A2, B), q2 = this.criticTargets.map((ct) => ct.forward(X2, B).slice()), y = new Float64Array(B);
      for (let k = 0; k < B; k++) {
        let q = q2[0][k];
        for (let c = 1; c < q2.length; c++) q = Math.min(q, q2[c][k]);
        if (cfg.algo === "sac") q -= alpha * logp2[k];
        y[k] = R[k] + cfg.gamma * (1 - Dn[k]) * q;
      }
      // the critics: a regression step toward the targets
      const X = this.join(S, A, B);
      let loss = 0;
      this.critics.forEach((c, ci) => {
        const out = c.forward(X, B), err = new Float64Array(B);
        let sq = 0;
        for (let k = 0; k < B; k++) { const e = out[k] - y[k]; sq += e * e; err[k] = e / B; }
        loss += sq / B;
        c.backward(err);
        this.optCritics[ci].step(c.grads);
      });
      this.log.loss.push(loss / this.critics.length);
      let ys = 0;
      for (const v of y) ys += v;
      this.log.q.push(ys / B);
      // the actor, and the targets with it (TD3: every `delay` critic updates)
      if (cfg.algo === "td3" && this.updates % cfg.delay) return;
      if (cfg.algo === "sac") this.actorStepSac(S, B, alpha);
      else { this.actorStepDet(S, B); this.actorTarget.softUpdate(this.actor, cfg.tau); }
      this.criticTargets.forEach((ct, i) => ct.softUpdate(this.critics[i], cfg.tau));
    }
    // dQ/da of critic c at (S, a) for B rows
    dqda(c, X, B) {
      c.forward(X, B);
      const gx = c.backward(new Float64Array(B).fill(1)), w = this.dim + this.ad, out = new Float64Array(B * this.ad);
      for (let r = 0; r < B; r++) for (let j = 0; j < this.ad; j++) out[r * this.ad + j] = gx[r * w + this.dim + j];
      return out;
    }
    // Climb Q(s, μ(s)): the critic's gradient with respect to the action, carried back through tanh and the actor.
    actorStepDet(S, B) {
      const h = this.high, z = this.actor.forward(S, B).slice(), a = Float64Array.from(z, (v) => h * D.tanh(v));
      const g = this.dqda(this.critics[0], this.join(S, a, B), B), dz = new Float64Array(z.length);
      for (let k = 0; k < z.length; k++) { const t = D.tanh(z[k]); dz[k] = (-g[k] * h * (1 - t * t)) / B; } // loss = −mean Q
      this.actor.forward(S, B); // the actor's own inputs again, for its backward pass
      this.actor.backward(dz);
      this.optActor.step(this.actor.grads);
    }
    // Climb min(Q1, Q2)(s, a) − α log π(a | s), a = high · tanh(μ + σ ε), through the sample (reparameterization).
    actorStepSac(S, B, alpha) {
      const ad = this.ad, h = this.high, eps = this.normals(B * ad), p = this.sacParts(S, B, eps);
      const t = Float64Array.from(p.u, (v) => D.tanh(v)), a = Float64Array.from(t, (v) => h * v), X = this.join(S, a, B);
      const qs = this.critics.map((c) => c.forward(X, B).slice()), grads = this.critics.map((c) => this.dqda(c, X, B));
      const dl = new Float64Array(B * 2 * ad);
      for (let r = 0; r < B; r++) {
        let pick = 0; // the gradient of the min is the gradient of whichever critic is smaller
        for (let c = 1; c < qs.length; c++) if (qs[c][r] < qs[pick][r]) pick = c;
        for (let j = 0; j < ad; j++) {
          const k = r * ad + j, du = (alpha * 2 * t[k] - grads[pick][k] * h * (1 - t[k] * t[k])) / B;
          // loss = mean(α log π − min Q); log π depends on u through the squashing (d/du of −log(1 − tanh²) is 2 tanh)
          dl[r * 2 * ad + j] = du;
          dl[r * 2 * ad + ad + j] = p.raw[k] > LOG_STD[0] && p.raw[k] < LOG_STD[1] ? du * p.sd[k] * eps[k] - alpha / B : 0;
        }
      }
      this.actor.forward(S, B);
      this.actor.backward(dl);
      this.optActor.step(this.actor.grads);
      if (this.cfg.auto_alpha) { // α rises when the policy is less random than the target entropy, and falls when more
        const logp = this.sacLogp(eps, p.ls, p.u, B);
        let g = 0;
        for (const v of logp) g += v - ad;
        g = -g / B; // d/d(log α) of −log α · (log π + target), target = −(action dimensions)
        this.am = 0.9 * this.am + 0.1 * g; this.av = 0.999 * this.av + 0.001 * g * g;
        this.ab1 *= 0.9; this.ab2 *= 0.999;
        this.logAlpha -= (this.cfg.lr_alpha * (this.am / (1 - this.ab1))) / (Math.sqrt(this.av / (1 - this.ab2)) + 1e-8);
      }
      this.log.alpha.push(D.exp(this.logAlpha));
    }
  }
  deep.AC = AC;
})(globalThis.RL = globalThis.RL || {});
