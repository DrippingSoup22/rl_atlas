/* Policy-gradient methods: the agent learns its policy π(a | s, θ) directly (lab/policies.js), nudging θ so that
   actions that turned out better than expected become more likely.

   REINFORCE                plays an episode, then moves θ along γᵗ G_t ∇ ln π(A_t | S_t) for every step
   REINFORCE with baseline  the same with G_t − v̂(S_t, w), where v̂ is learned alongside (Monte Carlo)
   actor–critic             learns at every step, judging each action by the TD error δ of a learned v̂ (the critic);
                            with λ > 0 both learners keep eligibility traces
   A2C                      several workers play at once; every n steps, one update from all their n-step advantages
   TRPO                     a batch of episodes, then the largest natural-gradient step that keeps the average KL
                            divergence between the old and the new policy below δ
   PPO                      a batch of episodes, then several passes over it, pushing each action's probability ratio
                            only while it stays within 1 ± ε

   Same conventions as the other agents: generators that pause at every line of the pseudocode (the line ids match
   the {#ids} of the entries' pseudocode), with what the views need in each event. The batch methods play a round:
   every worker plays one episode, all of them stepping together (events carry the worker as `w`), and the round's
   numbers are the averages over the workers. An episode cut short by the Lab's step limit counts as ended. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const workersOf = (p) => Math.max(1, Math.round(p.workers ?? 4));
  const rolloutOf = (p) => Math.max(1, Math.round(p.n ?? 5));
  const sumOf = (xs) => xs.reduce((a, b) => a + b, 0);

  // What a policy looks like in one state, before and after an update: the probability of the action taken, or for a
  // Gaussian its mean and spread (in the world's units).
  function look(P, theta, s, a) {
    if (P.kind === "gaussian") return { mu: P.unit * P.mean(theta, s), sd: P.unit * P.sd(theta, s) };
    return { pa: P.prob(theta, s, a), pr: P.prob(theta, s, 1) };
  }
  const moved = (before, after) => (before.mu !== undefined
    ? { mu: before.mu, mu2: after.mu, sd: before.sd, sd2: after.sd }
    : { pa: before.pa, pa2: after.pa, pr: before.pr, pr2: after.pr });

  // ---- REINFORCE, with or without a learned baseline ----
  function reinforce(learned) {
    return function* ({ env, m, rng, p }) {
      const P = lab.policy(env, p), F = P.F, { theta, w } = m;
      const S = [env.reset(rng)], A = [], R = [0];
      yield { line: "start", type: "start", s: S[0] };
      for (let t = 0; !env.terminal(S[t]) && t < p.maxSteps; t++) {
        const s = S[t], a = P.sample(theta, s, rng);
        yield { line: "generate", type: "choose", s, a, ...look(P, theta, s, a) };
        const o = env.step(s, a, rng);
        yield { line: "generate", type: "move", s, a, ...o };
        A.push(a); S.push(o.s2); R.push(o.r);
      }
      const T = A.length, G = new Float64Array(T + 1);
      for (let t = T - 1; t >= 0; t--) G[t] = R[t + 1] + p.gamma * G[t + 1];
      let disc = 1;
      for (let t = 0; t < T; t++) {
        const s = S[t], a = A[t], x = F.of(s);
        yield { line: "return", type: "return", t, s, a, G: G[t], T };
        const b = learned ? lab.dot(w, x) : 0, delta = G[t] - b;
        if (learned) {
          lab.addTo(w, x, p.alphaW * delta);
          yield { line: "critic", type: "critic", t, s, a, G: G[t], old: b, value: lab.dot(w, x), delta };
        }
        const before = look(P, theta, s, a);
        P.grad(theta, s, a, p.alpha * disc * delta, theta);
        yield { line: "update", type: "update", t, s, a, G: G[t], baseline: learned ? b : undefined, delta, disc, T, ...moved(before, look(P, theta, s, a)) };
        disc *= p.gamma;
      }
    };
  }

  // ---- a fixed policy, followed and never changed: for the stories that show what a policy does before any learning ----
  function* follow({ env, m, rng, p }) {
    const P = lab.policy(env, p), { theta } = m;
    let s = env.reset(rng);
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = P.sample(theta, s, rng);
      yield { line: "choose", type: "choose", s, a };
      const o = env.step(s, a, rng);
      yield { line: "act", type: "move", s, a, ...o };
      s = o.s2;
    }
  }

  // ---- actor–critic: the critic's TD error judges every action as soon as it is taken ----
  // I = γᵗ discounts the actor's steps along the episode, as the gradient of the start state's value requires.
  function* actorCritic({ env, m, rng, p }) {
    const P = lab.policy(env, p), F = P.F, { theta, w, zt, zw } = m, lam = p.lambda ?? 0, fade = p.gamma * lam;
    zt.fill(0);
    zw.fill(0);
    let s = env.reset(rng), I = 1;
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = P.sample(theta, s, rng);
      yield { line: "choose", type: "choose", s, a, ...look(P, theta, s, a) };
      const o = env.step(s, a, rng);
      yield { line: "act", type: "move", s, a, ...o };
      const x = F.of(s), v = lab.dot(w, x), end = env.terminal(o.s2), next = end ? 0 : lab.dot(w, F.of(o.s2));
      const delta = o.r + p.gamma * next - v;
      yield { line: "error", type: "error", s, a, r: o.r, s2: o.s2, old: v, next, delta, end };
      const before = look(P, theta, s, a);
      if (!lam) {
        lab.addTo(w, x, p.alphaW * delta);
        P.grad(theta, s, a, p.alpha * I * delta, theta);
      } else {
        for (let i = 0; i < zw.length; i++) zw[i] *= fade;
        for (let i = 0; i < zt.length; i++) zt[i] *= fade;
        lab.addTo(zw, x, 1);
        P.grad(theta, s, a, I, zt);
        for (let i = 0; i < zw.length; i++) if (zw[i]) w[i] += p.alphaW * delta * zw[i];
        for (let i = 0; i < zt.length; i++) if (zt[i]) theta[i] += p.alpha * delta * zt[i];
      }
      yield {
        line: "update", type: "update", s, a, r: o.r, s2: o.s2, old: v, next, target: o.r + p.gamma * next, delta, value: lab.dot(w, x), I,
        traced: !!lam, ...moved(before, look(P, theta, s, a)),
      };
      I *= p.gamma;
      s = o.s2;
      yield { line: "next", type: "next", s };
    }
  }

  // ---- the batch methods: a round of episodes, one per worker, all stepping together ----
  // onTick(team, tick) may update during the round (A2C); the team is returned for the update after it.
  function* round({ env, m, rng, p }, P, onTick) {
    const N = workersOf(p), team = [];
    for (let k = 0; k < N; k++) {
      const s = env.reset(rng);
      team.push({ S: [s], A: [], R: [0], done: false, cut: false, from: 0, falls: 0 });
      yield { line: "start", type: "start", s, w: k };
    }
    for (let tick = 0; team.some((tr) => !tr.done); tick++) {
      for (let k = 0; k < N; k++) {
        const tr = team[k];
        if (tr.done) continue;
        const s = tr.S[tr.S.length - 1], a = P.sample(m.theta, s, rng);
        yield { line: "act", type: "choose", s, a, w: k };
        const o = env.step(s, a, rng);
        yield { line: "act", type: "move", s, a, ...o, w: k };
        tr.A.push(a); tr.S.push(o.s2); tr.R.push(o.r);
        if (o.fell !== undefined) tr.falls++;
        if (env.terminal(o.s2) || tr.A.length >= p.maxSteps) { tr.done = true; tr.cut = !env.terminal(o.s2); }
      }
      yield { line: "act", type: "tick", tick, active: team.filter((tr) => !tr.done).length };
      if (onTick) yield* onTick(team, tick);
    }
    return team;
  }

  // Advantages of steps from … to−1 of one worker's episode: GAE(λ), a discounted sum of TD errors with weights (γλ)ᵏ,
  // stopping at `to`, which is the end of the episode (worth 0) or a point the critic's estimate stands in for.
  // λ-returns (advantage + estimate) are what the critic learns toward.
  function advantages(tr, from, to, V, gamma, lambda) {
    const n = to - from, adv = new Float64Array(n), ret = new Float64Array(n), deltas = new Float64Array(n);
    const last = tr.A.length, ended = tr.done && to === last;
    let A = 0;
    for (let k = n - 1; k >= 0; k--) {
      const t = from + k, next = t + 1 === last && ended ? 0 : V(tr.S[t + 1]);
      deltas[k] = tr.R[t + 1] + gamma * next - V(tr.S[t]);
      A = deltas[k] + gamma * lambda * A;
      adv[k] = A;
      ret[k] = A + V(tr.S[t]);
    }
    return { adv, ret, deltas };
  }

  // What a round's episodes came to, averaged over the workers.
  const totals = (team, extra) => ({
    return: lab.mean(team.map((tr) => sumOf(tr.R))), steps: lab.mean(team.map((tr) => tr.A.length)), falls: lab.mean(team.map((tr) => tr.falls)), ...extra,
  });

  // ---- A2C: every n steps, one synchronous update from all the workers' last n steps ----
  function* a2c(ctx) {
    const { env, m, p } = ctx, P = lab.policy(env, p), F = P.F, { theta, w } = m, N = workersOf(p), n = rolloutOf(p), beta = p.beta || 0;
    const V = (s) => lab.dot(w, F.of(s));
    let updates = 0;
    const team = yield* round(ctx, P, function* (team, tick) {
      if ((tick + 1) % n !== 0 && team.some((tr) => !tr.done)) return;
      const gt = new Float64Array(theta.length), gw = new Float64Array(w.length), list = [];
      for (const tr of team) {
        const to = tr.A.length;
        if (to === tr.from) continue;
        const { adv, ret } = advantages(tr, tr.from, to, V, p.gamma, 1); // n-step returns: λ = 1 inside the stretch
        for (let k = 0; k < adv.length; k++) {
          const t = tr.from + k, s = tr.S[t];
          P.grad(theta, s, tr.A[t], adv[k], gt);
          if (beta) P.gradEntropy(theta, s, beta, gt);
          lab.addTo(gw, F.of(s), ret[k] - V(s));
          list.push({ s, a: tr.A[t], adv: adv[k] });
        }
        tr.from = to;
      }
      if (!list.length) return;
      for (let i = 0; i < theta.length; i++) if (gt[i]) theta[i] += (p.alpha / N) * gt[i];
      for (let i = 0; i < w.length; i++) if (gw[i]) w[i] += (p.alphaW / N) * gw[i];
      updates++;
      yield { line: "update", type: "update", list, samples: list.length, mean: lab.mean(list.map((u) => u.adv)), n: updates };
    });
    return totals(team, { updates });
  }

  // The round's experience as a batch of samples, with GAE advantages and λ-returns, and the policy's
  // probabilities as they were when the actions were taken.
  function batchOf(team, P, theta, V, p) {
    const batch = [];
    for (const tr of team) {
      const { adv, ret } = advantages(tr, 0, tr.A.length, V, p.gamma, p.lambda ?? 0.95);
      for (let t = 0; t < tr.A.length; t++) {
        const s = tr.S[t], a = tr.A[t];
        batch.push({ s, a, adv: adv[t], ret: ret[t], pi: Float64Array.from(P.probs(theta, s)) });
      }
    }
    return batch;
  }
  // The critic learns toward the λ-returns: a few passes over the batch, one small step per sample.
  function fitCritic(batch, F, w, p, passes) {
    let err = 0;
    for (let k = 0; k < passes; k++) {
      err = 0;
      for (const b of batch) {
        const x = F.of(b.s), e = b.ret - lab.dot(w, x);
        lab.addTo(w, x, p.alphaW * e);
        err += e * e;
      }
    }
    return Math.sqrt(err / Math.max(1, batch.length));
  }
  // The average KL divergence from the old policy to the current one, over the batch's states.
  function klOf(batch, P, theta, env) {
    let D = 0;
    for (const b of batch) {
      const q = P.probs(theta, b.s);
      for (const a of env.acts(b.s)) if (b.pi[a] > 0) D += b.pi[a] * Math.log(b.pi[a] / q[a]);
    }
    return D / Math.max(1, batch.length);
  }
  // The surrogate objective: the average of ratio × advantage (clipped for PPO), which equals the average advantage, 0
  // for normalized advantages, at the old policy.
  function surrogate(batch, P, theta, eps) {
    let L = 0;
    for (const b of batch) {
      const r = P.prob(theta, b.s, b.a) / b.pi[b.a];
      L += eps ? Math.min(r * b.adv, Math.max(1 - eps, Math.min(1 + eps, r)) * b.adv) : r * b.adv;
    }
    return L / Math.max(1, batch.length);
  }

  // ---- PPO: several passes over each batch, each sample pushing its action only while its ratio is within 1 ± ε ----
  function* ppo(ctx) {
    const { env, m, rng, p } = ctx, P = lab.policy(env, p), F = P.F, { theta, w } = m;
    const V = (s) => lab.dot(w, F.of(s)), eps = p.clip ?? 0.2, K = Math.max(1, Math.round(p.epochs ?? 4)), beta = p.beta || 0;
    const team = yield* round(ctx, P, null);
    const batch = batchOf(team, P, theta, V, p);
    // Advantages are standardized (mean 0, spread 1) in each batch, so that the step size means the same whatever the
    // rewards' scale.
    const mean = lab.mean(batch.map((b) => b.adv)), sd = Math.sqrt(lab.mean(batch.map((b) => (b.adv - mean) ** 2))) || 1;
    for (const b of batch) b.adv = (b.adv - mean) / sd;
    yield { line: "advantage", type: "advantage", list: batch.map((b) => ({ s: b.s, a: b.a, adv: b.adv })), samples: batch.length };
    const order = batch.map((_, i) => i);
    let clipped = 0, kl = 0;
    for (let k = 0; k < K; k++) {
      for (let i = order.length - 1; i > 0; i--) { const j = rng.int(i + 1); const x = order[i]; order[i] = order[j]; order[j] = x; }
      const L0 = surrogate(batch, P, theta, eps);
      clipped = 0;
      for (const i of order) {
        const b = batch[i], r = P.prob(theta, b.s, b.a) / b.pi[b.a];
        if (eps && ((b.adv > 0 && r > 1 + eps) || (b.adv < 0 && r < 1 - eps))) clipped++;
        else P.grad(theta, b.s, b.a, p.alpha * b.adv * r, theta);
        if (beta) P.gradEntropy(theta, b.s, p.alpha * beta, theta);
      }
      kl = klOf(batch, P, theta, env);
      yield { line: "update", type: "update", epoch: k + 1, epochs: K, clipped: clipped / batch.length, kl, L0, L1: surrogate(batch, P, theta, eps), samples: batch.length, clip: eps };
    }
    const err = fitCritic(batch, F, w, p, K);
    yield { line: "critic", type: "critic", err, samples: batch.length };
    return totals(team, { kl, clipped: clipped / batch.length });
  }

  // ---- TRPO: the natural-gradient step as large as the trust region allows, checked by a line search ----
  // For a softmax with one feature per state (a table, or the corridor's single feature), the natural gradient has a
  // closed form: in each state, the estimated advantage of each action, Σ 1[A = a] Â / π(a | s), averaged over the
  // state's visits. Its step size makes the quadratic estimate of the KL divergence equal δ; the line search halves
  // it until the true average KL is within δ and the surrogate objective has improved.
  function* trpo(ctx) {
    const { env, m, p } = ctx, P = lab.policy(env, p), F = P.F, { theta, w } = m, nA = env.nA, n = F.n, delta = p.delta ?? 0.01;
    if (P.kind !== "softmax" || F.k !== 1) throw new Error("TRPO in the Lab needs a softmax with one feature per state");
    const V = (s) => lab.dot(w, F.of(s));
    const team = yield* round(ctx, P, null);
    const batch = batchOf(team, P, theta, V, p);
    yield { line: "advantage", type: "advantage", list: batch.map((b) => ({ s: b.s, a: b.a, adv: b.adv })), samples: batch.length };
    const cells = new Map();
    for (const b of batch) {
      const c = F.of(b.s).idx[0];
      if (!cells.has(c)) cells.set(c, { s: b.s, n: 0, x: new Float64Array(nA), pi: b.pi });
      const e = cells.get(c);
      e.n++;
      e.x[b.a] += b.adv / b.pi[b.a];
    }
    let quad = 0; // xᵀ F x: the Fisher information along x, the curvature of the KL divergence
    for (const e of cells.values()) {
      let m1 = 0, m2 = 0;
      for (const a of env.acts(e.s)) { e.x[a] /= e.n; m1 += e.pi[a] * e.x[a]; m2 += e.pi[a] * e.x[a] ** 2; }
      quad += (e.n / batch.length) * (m2 - m1 * m1);
    }
    const full = quad > 1e-12 ? Math.sqrt((2 * delta) / quad) : 0, old = Float64Array.from(theta), L0 = surrogate(batch, P, theta, 0);
    let frac = 1, tries = 0, accepted = false, kl = 0, L1 = L0;
    for (; tries < 10 && full > 0; tries++, frac /= 2) {
      theta.set(old);
      for (const [c, e] of cells) for (let a = 0; a < nA; a++) theta[a * n + c] += frac * full * e.x[a];
      kl = klOf(batch, P, theta, env);
      L1 = surrogate(batch, P, theta, 0);
      if (kl <= delta && L1 > L0) { accepted = true; break; }
    }
    if (!accepted) { theta.set(old); kl = 0; L1 = L0; }
    yield { line: "update", type: "update", kl, delta, frac: accepted ? frac : 0, tries: tries + (accepted ? 1 : 0), accepted, L0, L1, samples: batch.length, step: full };
    const err = fitCritic(batch, F, w, p, 4);
    yield { line: "critic", type: "critic", err, samples: batch.length };
    return totals(team, { kl });
  }

  // ---- what they show: the policy, as probabilities (or a mean and a spread), and the critic's values ----
  function show(critic) {
    return (m, env, p) => {
      const P = lab.policy(env, p), d = { theta: m.theta, w: m.w, F: P.F };
      if (P.kind === "gaussian") {
        d.mu = P.unit * P.mean(m.theta, 0);
        d.sd = P.unit * P.sd(m.theta, 0);
        if (critic) d.base = lab.dot(m.w, P.F.of(0));
        return d;
      }
      if (!env.continuous) {
        d.P = P.table(m.theta);
        if (critic) d.V = lab.valuesOf(env, P.F, m.w);
        if (m.zw && (p.lambda ?? 0) > 0) d.Z = m.zw; // the critic's trace, one per state
      }
      return d;
    };
  }
  const memory = (env, p) => ({ theta: lab.policy(env, p).n, w: lab.features(env, p).n });
  const init = (m, env, p) => { lab.policy(env, p).init(m.theta, p); m.w.fill(p.w0 || 0); };

  // ---- the live formula: the rule, the update in numbers, and a sentence on what just happened ----
  const tex = lab.texNum;
  const fmt = (v, d = 3) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d));
  // Enough digits to see a change: REINFORCE's steps on the corridor move a probability by a hundred-thousandth.
  const digits = (a, b, base) => { const d = Math.abs(b - a); return d === 0 || d >= 0.5 * 10 ** -base ? base : Math.min(base + 3, Math.ceil(-Math.log10(d)) + 1); };
  const pair = (a, b, base, k = 1) => { const n = digits(k * a, k * b, base); return [(k * a).toFixed(n), (k * b).toFixed(n)]; };
  const prob = (ev) => {
    if (ev.mu !== undefined) {
      const [m1, m2] = pair(ev.mu, ev.mu2, 2), [s1, s2] = pair(ev.sd, ev.sd2, 2);
      return `\\pol{\\mu}: ${m1}^\\circ \\to ${m2}^\\circ \\qquad \\pol{\\sigma}: ${s1}^\\circ \\to ${s2}^\\circ`;
    }
    const [p1, p2] = pair(ev.pa, ev.pa2, 3);
    return `\\pol{\\pi(A \\mid S)}: ${p1} \\to ${p2}`;
  };
  const policyMove = (ev, where) => {
    if (ev.mu !== undefined) {
      const [m1, m2] = pair(ev.mu, ev.mu2, 1), [s1, s2] = pair(ev.sd, ev.sd2, 2);
      return `the aim moved from <b>${m1}°</b> to <b>${m2}°</b>, the spread from <b>${s1}°</b> to <b>${s2}°</b>`;
    }
    const [p1, p2] = pair(ev.pa, ev.pa2, 1, 100);
    return `${where}: its chance went from <b>${p1}%</b> to <b>${p2}%</b>`;
  };
  const batchNote = (ev) => (ev.epoch
    ? `Pass <b>${ev.epoch} of ${ev.epochs}</b> over ${ev.samples.toLocaleString("en")} samples · <b>${Math.round(100 * ev.clipped)}%</b> outside 1 ± ${ev.clip} and left alone · the policy has moved KL = <b>${ev.kl.toFixed(4)}</b> from the one that collected them`
    : ev.list
      ? `One update from <b>${ev.samples}</b> steps of the workers · average advantage <b class="q-err">${ev.mean >= 0 ? "+" : "−"}${Math.abs(ev.mean).toFixed(3)}</b>`
      : ev.accepted
        ? `Natural-gradient step${ev.frac < 1 ? `, cut to ${ev.frac === 0.5 ? "half" : `1/${Math.round(1 / ev.frac)}`} by the line search` : ""} · KL = <b>${ev.kl.toFixed(4)}</b> ≤ δ = ${ev.delta} · surrogate gain <b>${(ev.L1 - ev.L0).toFixed(4)}</b>`
        : "No step: the line search found none within the trust region that improves the surrogate");

  // How the Lab names their knobs: α moves the policy, A2C's n is how many steps pass between updates, λ is GAE's.
  const ACTOR = { alpha: { sym: "αθ", name: "policy step size" } };
  const GAE = { ...ACTOR, lambda: { name: "GAE λ: how far advantages look ahead" } };

  lab.algorithms = Object.assign(lab.algorithms || {}, {
    // not a station: a policy that never learns (the stories' fixed policies)
    "fixed-policy": { id: "fixed-policy", title: "A fixed policy", unit: "episode", run: follow, memory, init, show: show(false) },
    reinforce: {
      id: "reinforce", title: "REINFORCE", unit: "episode", run: reinforce(false), memory, init, show: show(false), actor: true, knobs: ACTOR,
      rule: "\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\alp\\,\\gam^t\\,\\rew{G_t}\\,\\nabla \\ln \\pol{\\pi(A_t \\mid S_t, \\boldsymbol\\theta)}",
      numbers: (ev) => `\\rew{G_t} = ${tex(ev.G, ev.G % 1 ? 2 : 0)} \\qquad ${prob(ev)}`,
      note: (ev, where, signed) => `Return <b>${signed(ev.G)}</b> from step ${ev.t + 1} of ${ev.T} · ${policyMove(ev, where)}`,
    },
    baseline: {
      id: "baseline", title: "REINFORCE with baseline", unit: "episode", run: reinforce(true), memory, init, show: show(true), actor: true, knobs: ACTOR,
      rule: "\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\alp\\,\\gam^t\\,\\big(\\rew{G_t} - \\val{\\hat v(S_t, \\mathbf w)}\\big)\\,\\nabla \\ln \\pol{\\pi(A_t \\mid S_t, \\boldsymbol\\theta)}",
      numbers: (ev) => `\\rew{${tex(ev.G, ev.G % 1 ? 2 : 0)}} - \\val{${tex(ev.baseline, 2)}} = \\err{${fmt(ev.delta, 2)}} \\qquad ${prob(ev)}`,
      note: (ev, where, signed) => `Return <b>${signed(ev.G)}</b>, baseline <b>${signed(ev.baseline)}</b>: ${ev.delta >= 0 ? "better" : "worse"} than expected by <b class="q-err">${signed(Math.abs(ev.delta))}</b> · ${policyMove(ev, where)}`,
    },
    "actor-critic": {
      id: "actor-critic", title: "Actor–critic", unit: "episode", run: actorCritic, init, show: show(true), actor: true, knobs: ACTOR,
      memory: (env, p) => ({ ...memory(env, p), zt: lab.policy(env, p).n, zw: lab.features(env, p).n }),
      rule: (p) => (p.lambda > 0
        ? "\\mathbf z^{\\boldsymbol\\theta} \\leftarrow \\gam\\lam\\,\\mathbf z^{\\boldsymbol\\theta} + I\\,\\nabla \\ln \\pol{\\pi(A \\mid S)} \\qquad \\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\alp\\,\\del\\,\\mathbf z^{\\boldsymbol\\theta}"
        : "\\del = \\rew{R} + \\gam\\,\\val{\\hat v(S')} - \\val{\\hat v(S)} \\qquad \\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\alp\\,I\\,\\del\\,\\nabla \\ln \\pol{\\pi(A \\mid S)}"),
      numbers: (ev, p) => `\\del = \\rew{${+ev.r.toFixed(2)}} + ${p.gamma}\\cdot\\val{${tex(ev.next)}} - \\val{${tex(ev.old)}} = \\err{${ev.delta.toFixed(2)}} \\qquad ${prob(ev)}`,
      note: (ev, where, signed) => `TD error <b class="q-err">δ = ${signed(ev.delta)}</b>: the step went ${ev.delta >= 0 ? "better" : "worse"} than the critic expected · its estimate here moved to <b>${signed(ev.value)}</b> · ${policyMove(ev, where)}`,
    },
    a2c: {
      id: "a2c", title: "A2C", unit: "round", run: a2c, memory, init, show: show(true), actor: true, batch: true,
      knobs: { ...ACTOR, n: { name: "steps between updates", choices: [1, 2, 5, 10, 20, 50] } },
      rule: "\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\alp\\,\\tfrac{1}{N}\\sum \\big[\\err{\\hat A_t}\\,\\nabla \\ln \\pol{\\pi(A_t \\mid S_t)} + \\beta\\,\\nabla \\pol{H}\\big]",
      numbers: (ev) => `${ev.samples}\\ \\text{steps},\\ \\text{average}\\ \\err{\\hat A} = ${tex(ev.mean, 3)}`,
      note: (ev) => batchNote(ev),
    },
    trpo: {
      id: "trpo", title: "TRPO", unit: "round", run: trpo, memory, init, show: show(true), actor: true, batch: true, knobs: GAE,
      rule: "\\max_{\\boldsymbol\\theta}\\ \\hat{\\mathbb E}\\Big[\\tfrac{\\pol{\\pi_{\\boldsymbol\\theta}(A \\mid S)}}{\\pol{\\pi_{\\text{old}}(A \\mid S)}}\\,\\err{\\hat A}\\Big] \\ \\text{ with }\\ \\overline{D}_{\\mathrm{KL}}(\\pol{\\pi_{\\text{old}}} \\,\\|\\, \\pol{\\pi_{\\boldsymbol\\theta}}) \\le \\delta",
      numbers: (ev) => (ev.accepted ? `\\overline{D}_{\\mathrm{KL}} = ${ev.kl.toFixed(4)} \\le ${ev.delta} \\qquad L: ${tex(ev.L0, 3)} \\to ${tex(ev.L1, 3)}` : "\\text{no step}"),
      note: (ev) => batchNote(ev),
    },
    ppo: {
      id: "ppo", title: "PPO", unit: "round", run: ppo, memory, init, show: show(true), actor: true, batch: true, knobs: GAE,
      rule: (p) => `\\hat{\\mathbb E}\\Big[\\min\\big(r\\,\\err{\\hat A},\\ \\operatorname{clip}(r, ${p.clip ? `1 - ${p.clip}, 1 + ${p.clip}` : "1 - \\epsilon, 1 + \\epsilon"})\\,\\err{\\hat A}\\big)\\Big],\\quad r = \\tfrac{\\pol{\\pi_{\\boldsymbol\\theta}(A \\mid S)}}{\\pol{\\pi_{\\text{old}}(A \\mid S)}}`,
      numbers: (ev) => `\\text{pass } ${ev.epoch}/${ev.epochs}: \\quad L^{\\text{CLIP}}: ${tex(ev.L0, 3)} \\to ${tex(ev.L1, 3)} \\qquad \\overline{D}_{\\mathrm{KL}} = ${ev.kl.toFixed(4)}`,
      note: (ev) => batchNote(ev),
    },
  });
})(globalThis.RL = globalThis.RL || {});
