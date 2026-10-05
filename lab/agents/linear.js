/* Linear function approximation: the estimate is w · x(s), and an update moves the weights, not one entry of a table.
   Gradient Monte Carlo moves them toward the return; semi-gradient TD and SARSA toward a target that uses the estimate
   itself (n steps ahead, 1 unless a preset says otherwise), and ignore that the target moves with the weights.
   Off-policy semi-gradient TD (p.behavior) weights each update by the importance-sampling ratio.
   Same conventions as td.js: generators that pause at each line of the pseudocode, with the feature vector of the
   updated state in the event, so the views can show which part of the space the update moved. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const steps = (p) => Math.max(1, Math.round(p.n ?? 1));
  const norm = (w) => { let s = 0; for (let i = 0; i < w.length; i++) s += w[i] * w[i]; return Math.sqrt(s); };

  // ---- gradient Monte Carlo: play the episode, then move each visited state's estimate toward its return ----
  function* gradientMC({ env, m, rng, p }) {
    const F = lab.features(env, p), w = m.w, P = lab.policyOf(env, p);
    const S = [env.reset(rng)], R = [0];
    yield { line: "start", type: "start", s: S[0] };
    for (let t = 0; !env.terminal(S[t]) && t < p.maxSteps; t++) {
      const a = lab.pickRow(P, S[t], env, rng);
      yield { line: "choose", type: "choose", s: S[t], a };
      const o = env.step(S[t], a, rng);
      yield { line: "act", type: "move", s: S[t], a, ...o };
      S.push(o.s2); R.push(o.r);
    }
    const T = S.length - 1, G = new Float64Array(T + 1);
    for (let t = T - 1; t >= 0; t--) G[t] = R[t + 1] + p.gamma * G[t + 1];
    for (let t = 0; t < T; t++) {
      const x = F.of(S[t]), old = lab.dot(w, x);
      lab.addTo(w, x, p.alpha * (G[t] - old));
      yield { line: "update", type: "update", s: S[t], a: -1, old, target: G[t], delta: G[t] - old, value: lab.dot(w, x), x, t, T };
    }
    return { weights: norm(w) };
  }

  // ---- semi-gradient n-step TD, on- or off-policy ----
  // The update of time τ happens at time τ + n. A world cut into stretches (env.stretch, for tasks that never end)
  // bootstraps from wherever the stretch stopped; an episode that ended does not.
  function* semiTD({ env, m, rng, p }) {
    const F = lab.features(env, p), w = m.w, n = steps(p), P = lab.policyOf(env, p);
    const B = p.behavior ? (p.behaviorTable ||= lab.policyTable(env, p.behavior)) : P, limit = env.stretch || p.maxSteps;
    const S = [env.reset(rng)], R = [0], rho = [];
    yield { line: "start", type: "start", s: S[0] };
    let T = Infinity;
    for (let t = 0; ; t++) {
      if (t < T) {
        const a = lab.pickRow(B, S[t], env, rng);
        yield { line: "choose", type: "choose", s: S[t], a };
        const o = env.step(S[t], a, rng);
        yield { line: "act", type: "move", s: S[t], a, ...o };
        S.push(o.s2); R.push(o.r);
        rho.push(B === P ? 1 : P[S[t] * env.nA + a] / B[S[t] * env.nA + a]);
        if (env.terminal(o.s2) || t + 1 >= limit) T = t + 1;
      }
      const tau = t - n + 1;
      if (tau < 0) continue;
      const end = Math.min(tau + n, T), boot = !env.terminal(S[end]);
      let G = 0, pow = 1, ratio = 1;
      for (let i = tau + 1; i <= end; i++) { G += pow * R[i]; pow *= p.gamma; ratio *= rho[i - 1]; }
      const rewards = G, next = boot ? lab.dot(w, F.of(S[end])) : 0;
      G += boot ? pow * next : 0;
      const x = F.of(S[tau]), old = lab.dot(w, x);
      lab.addTo(w, x, p.alpha * ratio * (G - old));
      yield {
        line: "update", type: "update", s: S[tau], a: -1, r: R[tau + 1], s2: S[end], old, next, target: G, delta: G - old, value: lab.dot(w, x),
        x, k: end - tau, boot, rewards, discount: pow, rho: ratio, offPolicy: B !== P,
      };
      if (tau >= T - 1) return { weights: norm(w) };
    }
  }

  // ---- semi-gradient n-step SARSA: ε-greedy on the action values w_a · x(s) ----
  function* semiSarsa({ env, m, rng, p }) {
    const F = lab.features(env, p), w = m.w, n = steps(p), N = F.n, nA = env.nA, q = new Float64Array(nA);
    const choose = (s, x) => {
      const acts = env.acts(s);
      if (rng.next() < p.epsilon) return acts[rng.int(acts.length)];
      for (const a of acts) q[a] = lab.dot(w, x, a * N);
      let best = -Infinity, ties = [];
      for (const a of acts) { if (q[a] > best) { best = q[a]; ties = [a]; } else if (q[a] === best) ties.push(a); }
      if (!ties.length) return acts[rng.int(acts.length)]; // weights that diverged to NaN rank nothing: act at random
      return ties.length > 1 ? ties[rng.int(ties.length)] : ties[0];
    };
    const S = [env.reset(rng)], X = [F.of(S[0])], A = [], R = [0];
    yield { line: "start", type: "start", s: S[0] };
    A[0] = choose(S[0], X[0]);
    yield { line: "choose", type: "choose", s: S[0], a: A[0] };
    let T = Infinity;
    for (let t = 0; ; t++) {
      if (t < T) {
        const o = env.step(S[t], A[t], rng);
        yield { line: "act", type: "move", s: S[t], a: A[t], ...o };
        S.push(o.s2); R.push(o.r);
        if (env.terminal(o.s2) || t + 1 >= p.maxSteps) T = t + 1;
        else {
          X[t + 1] = F.of(o.s2);
          A[t + 1] = choose(o.s2, X[t + 1]);
          yield { line: "choose-next", type: "choose", s: o.s2, a: A[t + 1] };
        }
      }
      const tau = t - n + 1;
      if (tau < 0) continue;
      const end = Math.min(tau + n, T), boot = tau + n < T;
      let G = 0, pow = 1;
      for (let i = tau + 1; i <= end; i++) { G += pow * R[i]; pow *= p.gamma; }
      const rewards = G, next = boot ? lab.dot(w, X[end], A[end] * N) : 0;
      G += boot ? pow * next : 0;
      const x = X[tau], off = A[tau] * N, old = lab.dot(w, x, off);
      lab.addTo(w, x, p.alpha * (G - old), off);
      yield {
        line: "update", type: "update", s: S[tau], a: A[tau], r: R[tau + 1], old, next, target: G, delta: G - old, value: lab.dot(w, x, off),
        x, k: end - tau, boot, rewards, discount: pow,
      };
      if (tau >= T - 1) return { weights: norm(w) };
      X[tau] = null; // done with it: long episodes keep only the window
    }
  }

  // What a linear learner shows: its weights and features, and for a world small enough, the estimates they give.
  const linear = (control) => ({
    unit: "episode",
    memory: (env, p) => ({ w: lab.features(env, p).n * (control ? env.nA : 1) }),
    // q0: where every estimate starts (each of the k features on at once carries its share), or w0 for each weight
    init: (m, env, p) => { const F = lab.features(env, p); m.w.fill(p.q0 !== undefined ? p.q0 / F.k : p.w0 || 0); F.init?.(m.w); },
    show(m, env, p) {
      const F = lab.features(env, p), d = { w: m.w, F };
      if (env.continuous) return d;
      if (!control) d.V = lab.valuesOf(env, F, m.w);
      else {
        d.Q = new Float64Array(env.nS * env.nA);
        for (let s = 0; s < env.nS; s++) for (const a of env.acts(s)) d.Q[s * env.nA + a] = lab.dot(m.w, F.of(s), a * F.n);
      }
      return d;
    },
  });

  const tex = lab.texNum;
  // The estimate of the updated state in numbers: with linear features it moves by α δ ‖x‖² (times ρ off-policy),
  // so with 50 tilings each update moves it 50 times as far as α alone says.
  const sq = (x) => { let v = 0; for (let i = 0; i < x.k; i++) v += x.val[i] * x.val[i]; return v; };
  const target = (ev, p) => {
    const g = ev.k === 1 ? p.gamma : +ev.discount.toFixed(3);
    return `\\rew{${tex(ev.rewards ?? ev.target)}}${ev.boot ? ` + ${g === 1 ? "" : `${g}\\,`}\\val{${tex(ev.next)}}` : ""}`;
  };
  const moved = (sym) => (ev, p) => {
    const n2 = sq(ev.x), ratio = ev.offPolicy ? `${+ev.rho.toFixed(2)}\\cdot ` : "";
    return `\\val{${sym}} \\leftarrow \\val{${ev.old.toFixed(2)}} + ${ratio}${p.alpha}\\,\\big[\\,${target(ev, p)} - \\val{${tex(ev.old)}}\\,\\big]${n2 === 1 ? "" : `\\cdot ${+n2.toFixed(2)}`} = \\val{${ev.value.toFixed(2)}}`;
  };
  const vNumbers = moved("\\hat v"), qNumbers = moved("\\hat q");
  // How many weights an update touched, and so how much of the space it moved.
  const shared = (ev) => {
    const k = ev.x.k;
    return k === 1 ? "one weight moved: every state in the same group moved with it" : `${k} weights moved: every state sharing any of them moved too`;
  };
  const note = (ev, where, signed, p) => {
    const own = ev.boot === undefined ? `Return <b>${signed(ev.target)}</b>` : `Target <b>${signed(ev.target)}</b>${ev.boot ? "" : " (no estimate at the end)"}`;
    const off = ev.offPolicy ? ` · importance ratio <b>${+ev.rho.toFixed(2)}</b>${ev.rho === 0 ? ": the target policy never takes this action, nothing is learned" : ""}` : "";
    const n2 = sq(ev.x), far = n2 === 1 ? "" : ` · the estimate here moves by α·δ·‖x‖², and ‖x‖² = ${+n2.toFixed(2)}`;
    return `${own} · surprise <b class="q-err">${signed(ev.delta)}</b>${off}${far} · ${where}: ${shared(ev)}`;
  };

  lab.algorithms = Object.assign(lab.algorithms || {}, {
    "gradient-mc": {
      id: "gradient-mc", title: "Gradient Monte Carlo", ...linear(false), run: gradientMC,
      rule: "\\mathbf w \\leftarrow \\mathbf w + \\alp\\,\\big[\\rew{G_t} - \\val{\\hat v(S_t)}\\big]\\,\\mathbf x(S_t)",
      numbers: vNumbers, note,
    },
    "semi-gradient-td": {
      id: "semi-gradient-td", title: "Semi-gradient TD", ...linear(false), run: semiTD,
      rule: (p) => steps(p) === 1
        ? `\\mathbf w \\leftarrow \\mathbf w + \\alp${p.behavior ? "\\,\\rho" : ""}\\,\\big[\\rew{R} + \\gam\\,\\val{\\hat v(S')} - \\val{\\hat v(S)}\\big]\\,\\mathbf x(S)`
        : `\\mathbf w \\leftarrow \\mathbf w + \\alp\\,\\big[\\rew{G_{\\tau:\\tau+${steps(p)}}} - \\val{\\hat v(S_\\tau)}\\big]\\,\\mathbf x(S_\\tau)`,
      numbers: vNumbers, note,
    },
    "semi-gradient-sarsa": {
      id: "semi-gradient-sarsa", title: "Semi-gradient SARSA", ...linear(true), run: semiSarsa,
      rule: (p) => steps(p) === 1
        ? "\\mathbf w_A \\leftarrow \\mathbf w_A + \\alp\\,\\big[\\rew{R} + \\gam\\,\\val{\\hat q(S', A')} - \\val{\\hat q(S, A)}\\big]\\,\\mathbf x(S)"
        : `\\mathbf w_A \\leftarrow \\mathbf w_A + \\alp\\,\\big[\\rew{G_{\\tau:\\tau+${steps(p)}}} - \\val{\\hat q(S_\\tau, A_\\tau)}\\big]\\,\\mathbf x(S_\\tau)`,
      numbers: qNumbers, note,
    },
  });
})(globalThis.RL = globalThis.RL || {});
