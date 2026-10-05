/* Methods that look further than one step: n-step TD and SARSA, which wait n steps and use the n rewards seen
   before trusting an estimate, and TD(λ) and SARSA(λ), which keep an eligibility trace, a fading memory of the
   states (or state–action pairs) just visited, and pass every TD error back along it at once.
   Same conventions as td.js: generators that pause at each line of the pseudocode. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const steps = (p) => Math.max(1, Math.round(p.n ?? 4)); // 4 unless a preset says otherwise (the sandbox does not)

  // ---- n-step methods, prediction (V) or control (Q) ----
  // The update of time τ happens at time τ + n, once the rewards R_{τ+1} … R_{τ+n} are known; after the episode ends,
  // the last n − 1 states are updated with the rewards that remain, and no estimate at the end.
  function nStep(control) {
    return function* ({ env, m, rng, p }) {
      const n = steps(p), nA = env.nA, table = control ? m.Q : m.V, P = control ? null : lab.policyOf(env, p);
      const choose = (s) => (control ? lab.epsGreedy(m.Q, s, env, p.epsilon, rng) : lab.pickRow(P, s, env, rng));
      const S = [env.reset(rng)], A = [], R = [0];
      const key = (t) => (control ? S[t] * nA + A[t] : S[t]);
      yield { line: "start", type: "start", s: S[0] };
      let T = Infinity;
      if (control) { A[0] = choose(S[0]); yield { line: "choose", type: "choose", s: S[0], a: A[0] }; }
      for (let t = 0; ; t++) {
        if (t < T) {
          if (!control) { A[t] = choose(S[t]); yield { line: "choose", type: "choose", s: S[t], a: A[t] }; }
          const o = env.step(S[t], A[t], rng);
          yield { line: "act", type: "move", s: S[t], a: A[t], ...o };
          S.push(o.s2); R.push(o.r);
          if (env.terminal(o.s2) || t + 1 >= p.maxSteps) T = t + 1;
          else if (control) { A[t + 1] = choose(o.s2); yield { line: "choose-next", type: "choose", s: o.s2, a: A[t + 1] }; }
        }
        const tau = t - n + 1;
        if (tau < 0) continue;
        const end = Math.min(tau + n, T), boot = tau + n < T;
        let G = 0, pow = 1;
        for (let i = tau + 1; i <= end; i++) { G += pow * R[i]; pow *= p.gamma; }
        const rewards = G, next = boot ? table[key(tau + n)] : 0;
        G += boot ? pow * next : 0;
        const window = S.slice(tau, end + 1);
        yield { line: "return", type: "window", s: S[tau], a: control ? A[tau] : -1, states: window, k: end - tau, boot, tau, G };
        const i = key(tau), old = table[i];
        table[i] = old + p.alpha * (G - old);
        yield {
          line: "update", type: "update", s: S[tau], a: control ? A[tau] : -1, r: R[tau + 1], old, target: G, delta: G - old, value: table[i],
          k: end - tau, boot, rewards, next, discount: pow, states: window, tau, late: t >= T,
        };
        if (tau >= T - 1) return;
      }
    };
  }

  // ---- TD(λ) and SARSA(λ): one TD error per step, passed back along the eligibility trace ----
  // Every step the trace fades by γλ, the pair just visited gains 1 (accumulating) or is set to 1 (replacing),
  // and every entry of the table moves by α δ z.
  function lambda(control) {
    return function* ({ env, m, rng, p }) {
      const { Z } = m, nA = env.nA, table = control ? m.Q : m.V, P = control ? null : lab.policyOf(env, p);
      const fade = p.gamma * (p.lambda ?? 0.9), replacing = p.trace === "replacing";
      const choose = (s) => (control ? lab.epsGreedy(m.Q, s, env, p.epsilon, rng) : lab.pickRow(P, s, env, rng));
      Z.fill(0);
      let s = env.reset(rng);
      yield { line: "start", type: "start", s };
      let a = choose(s);
      if (control) yield { line: "choose", type: "choose", s, a };
      for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
        if (!control && t > 0) { a = choose(s); }
        if (!control) yield { line: "choose", type: "choose", s, a };
        const o = env.step(s, a, rng);
        yield { line: "act", type: "move", s, a, ...o };
        const end = env.terminal(o.s2);
        const a2 = control && !end ? choose(o.s2) : -1;
        if (control) yield { line: "choose-next", type: "choose", s: o.s2, a: a2 };
        const i = control ? s * nA + a : s, old = table[i];
        const next = end ? 0 : control ? table[o.s2 * nA + a2] : table[o.s2], delta = o.r + p.gamma * next - old;
        yield { line: "error", type: "error", s, a: control ? a : -1, r: o.r, s2: o.s2, old, next, delta };
        for (let j = 0; j < Z.length; j++) Z[j] *= fade;
        Z[i] = replacing ? 1 : Z[i] + 1;
        yield { line: "trace", type: "trace", s, a: control ? a : -1, z: Z[i] };
        let touched = 0;
        for (let j = 0; j < Z.length; j++) {
          if (Z[j] < 1e-12) { Z[j] = 0; continue; }
          table[j] += p.alpha * delta * Z[j];
          touched++;
        }
        yield {
          line: "update", type: "update", s, a: control ? a : -1, r: o.r, s2: o.s2, old, next, target: o.r + p.gamma * next, delta, value: table[i],
          touched, traced: true,
        };
        s = o.s2;
        a = a2;
        yield { line: "next", type: "next", s, a };
      }
    };
  }

  // ---- the offline λ-return algorithm: after the episode, every state moves toward its λ-return ----
  // Not a station of its own (the λ-return is a concept): the Textbook's figures compare it with TD(λ).
  function* offlineLambda({ env, m, rng, p }) {
    const V = m.V, P = lab.policyOf(env, p), lam = p.lambda ?? 0.9, g = p.gamma;
    const S = [env.reset(rng)], R = [0];
    yield { line: "start", type: "start", s: S[0] };
    for (let t = 0; !env.terminal(S[t]) && t < p.maxSteps; t++) {
      const a = lab.pickRow(P, S[t], env, rng);
      const o = env.step(S[t], a, rng);
      yield { line: "act", type: "move", s: S[t], a, ...o };
      S.push(o.s2); R.push(o.r);
    }
    const T = S.length - 1, est = S.map((s, t) => (t < T ? V[s] : 0)); // the estimates as they were during the episode
    // G^λ_t = R_{t+1} + γ[(1 − λ) V(S_{t+1}) + λ G^λ_{t+1}], with G^λ_{T-1} = R_T: computed backward.
    let G = 0;
    const targets = new Float64Array(T);
    for (let t = T - 1; t >= 0; t--) { G = R[t + 1] + g * (t + 1 < T ? (1 - lam) * est[t + 1] + lam * G : 0); targets[t] = G; }
    for (let t = 0; t < T; t++) {
      const s = S[t], old = V[s];
      V[s] = old + p.alpha * (targets[t] - old);
      yield { line: "update", type: "update", s, a: -1, old, target: targets[t], delta: targets[t] - old, value: V[s] };
    }
  }

  const tex = lab.texNum;
  // The n-step return in numbers: the rewards seen, and γⁿ times the estimate n steps ahead when the episode goes on.
  const nStepNumbers = (sym) => (ev, p) => {
    const tail = ev.boot ? ` + ${+ev.discount.toFixed(3)}\\cdot\\val{${tex(ev.next)}}` : "";
    return `\\val{${sym}} \\leftarrow \\val{${ev.old.toFixed(2)}} + ${p.alpha}\\,\\big[\\,\\rew{${tex(ev.rewards)}}${tail} - \\val{${tex(ev.old)}}\\,\\big] = \\val{${ev.value.toFixed(2)}}`;
  };
  const nStepNote = (ev, where, signed) =>
    `${ev.k === 1 ? "One reward" : `${ev.k} rewards`}${ev.boot ? ` and the estimate ${ev.k} steps ahead` : ev.late ? ", the episode is over: no estimate at the end" : ", up to the end of the episode"} · return <b>${signed(ev.target)}</b> · update of ${where}`;
  const lambdaNumbers = (ev, p) =>
    `\\del = \\rew{${+ev.r.toFixed(2)}} + ${p.gamma}\\cdot\\val{${tex(ev.next)}} - \\val{${tex(ev.old)}} = \\err{${ev.delta.toFixed(3)}}`;
  const lambdaNote = (ev, where, signed, p) =>
    `Surprise <b class="q-err">δ = ${signed(ev.delta, 3)}</b> · ${ev.touched === 1 ? "one entry has" : `${ev.touched} entries have`} a trace, and each moves by α·δ·z: the brighter the glow, the bigger the share · ${where} moved by <b>${signed(ev.value - ev.old, 3)}</b>`;

  const vTable = {
    memory: (env) => ({ V: env.nS }),
    init: (m, env, p) => { for (let s = 0; s < env.nS; s++) m.V[s] = env.terminal(s) ? 0 : p.v0 || 0; },
    show: (m) => ({ V: m.V }),
  };
  lab.algorithms = Object.assign(lab.algorithms || {}, {
    "n-step-td": {
      id: "n-step-td", title: "n-step TD", unit: "episode", run: nStep(false), ...vTable,
      rule: (p) => `\\val{V(S_\\tau)} \\leftarrow \\val{V(S_\\tau)} + \\alp\\,\\big[\\,\\rew{G_{\\tau:\\tau+${steps(p)}}} - \\val{V(S_\\tau)}\\,\\big]`,
      numbers: nStepNumbers("V(S_\\tau)"), note: nStepNote,
    },
    "n-step-sarsa": {
      id: "n-step-sarsa", title: "n-step SARSA", unit: "episode", run: nStep(true), ...lab.qTable, show: (m) => ({ Q: m.Q }),
      rule: (p) => `\\val{Q(S_\\tau,A_\\tau)} \\leftarrow \\val{Q(S_\\tau,A_\\tau)} + \\alp\\,\\big[\\,\\rew{G_{\\tau:\\tau+${steps(p)}}} - \\val{Q(S_\\tau,A_\\tau)}\\,\\big]`,
      numbers: nStepNumbers("Q(S_\\tau,A_\\tau)"), note: nStepNote,
    },
    "td-lambda": {
      id: "td-lambda", title: "TD(λ)", unit: "episode", run: lambda(false),
      memory: (env) => ({ V: env.nS, Z: env.nS }),
      init: vTable.init,
      show: (m) => ({ V: m.V, Z: m.Z }),
      rule: "\\val{V(s)} \\leftarrow \\val{V(s)} + \\alp\\,\\del\\,\\htmlClass{q-trc}{z(s)} \\quad \\text{for every } s",
      numbers: lambdaNumbers, note: lambdaNote,
    },
    "sarsa-lambda": {
      id: "sarsa-lambda", title: "SARSA(λ)", unit: "episode", run: lambda(true),
      memory: (env) => ({ Q: env.nS * env.nA, Z: env.nS * env.nA }),
      init: (m, env, p) => m.Q.fill(p.q0 || 0),
      show: (m) => ({ Q: m.Q, Z: m.Z }),
      rule: "\\val{Q(s,a)} \\leftarrow \\val{Q(s,a)} + \\alp\\,\\del\\,\\htmlClass{q-trc}{z(s,a)} \\quad \\text{for every } s, a",
      numbers: lambdaNumbers, note: lambdaNote,
    },
    "offline-lambda": {
      id: "offline-lambda", title: "Offline λ-return", unit: "episode", run: offlineLambda, ...vTable,
      rule: "\\val{V(S_t)} \\leftarrow \\val{V(S_t)} + \\alp\\,\\big[\\,\\rew{G^\\lambda_t} - \\val{V(S_t)}\\,\\big]",
    },
  });
})(globalThis.RL = globalThis.RL || {});
