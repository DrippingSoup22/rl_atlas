/* Monte Carlo methods as generators. Each plays a whole episode first, then walks back through it from the end,
   adding up the return G and updating the states (or state–action pairs) it visited. Nothing is learned during
   the episode itself: that is the difference from temporal-difference methods. */
(function (RL) {
  "use strict";
  const lab = RL.lab;

  // The step size of an update: 1/N (an exact average of the returns seen so far) unless a constant α is set.
  const stepOf = (p, n) => (p.alpha > 0 ? p.alpha : 1 / n);
  const avg = (p) => (p.alpha > 0 ? "\\alp" : "\\tfrac{1}{N}");

  // Play an episode with choose(s) picking the actions, and keep its states, actions and rewards.
  // first[i] is the time of the first visit to i (a state, or a state–action pair when pairs = true).
  function* generate({ env, rng, p }, s, choose, pairs, ep) {
    const first = new Int32Array(pairs ? env.nS * env.nA : env.nS).fill(-1);
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = choose(s, t);
      yield { line: "generate", type: "choose", s, a };
      const o = env.step(s, a, rng);
      yield { line: "generate", type: "move", s, a, ...o };
      const i = pairs ? s * env.nA + a : s;
      if (first[i] < 0) first[i] = t;
      ep.S.push(s); ep.A.push(a); ep.R.push(o.r);
      s = o.s2;
    }
    ep.first = first;
  }

  // ---- Monte Carlo prediction: the value of a fixed policy, as the average of the returns that followed each state ----
  function* mcPrediction(ctx) {
    const { env, m, rng, p } = ctx, { V, N } = m, P = lab.policyOf(env, p), ep = { S: [], A: [], R: [] };
    const s0 = env.reset(rng);
    yield { line: "start", type: "start", s: s0 };
    yield* generate(ctx, s0, (s) => lab.pickRow(P, s, env, rng), false, ep);
    let G = 0;
    yield { line: "g0", type: "info" };
    for (let t = ep.S.length - 1; t >= 0; t--) {
      const s = ep.S[t];
      G = p.gamma * G + ep.R[t];
      yield { line: "return", type: "return", t, s, a: -1, G, r: ep.R[t] };
      if (p.visits !== "every" && ep.first[s] < t) { yield { line: "first", type: "skip", t, s, a: -1 }; continue; }
      N[s] += 1;
      const old = V[s], step = stepOf(p, N[s]);
      V[s] = old + step * (G - old);
      yield { line: "update", type: "update", t, s, a: -1, old, target: G, delta: G - old, value: V[s], n: N[s], step };
    }
  }

  // ---- Monte Carlo control: episodes, then every visited (state, action) pair moves toward its return ----
  // start(): where the episode begins; choose(Q, s, t): the action the behavior takes there.
  function control(start, choose) {
    return function* (ctx) {
      const { env, m, rng, p } = ctx, { Q, N } = m, nA = env.nA, ep = { S: [], A: [], R: [] };
      const [s0, a0] = start(ctx);
      yield { line: "start", type: "start", s: s0 };
      yield* generate(ctx, s0, (s, t) => (t === 0 && a0 >= 0 ? a0 : choose(ctx, s)), true, ep);
      let G = 0;
      yield { line: "g0", type: "info" };
      for (let t = ep.S.length - 1; t >= 0; t--) {
        const s = ep.S[t], a = ep.A[t], i = s * nA + a;
        G = p.gamma * G + ep.R[t];
        yield { line: "return", type: "return", t, s, a, G, r: ep.R[t] };
        if (ep.first[i] < t) { yield { line: "first", type: "skip", t, s, a }; continue; }
        N[i] += 1;
        const old = Q[i], step = stepOf(p, N[i]);
        Q[i] = old + step * (G - old);
        yield { line: "update", type: "update", t, s, a, old, target: G, delta: G - old, value: Q[i], n: N[i], step };
      }
    };
  }

  // Exploring starts: a random state and a random first action, then greedy (ties broken at random).
  const randomStart = ({ env, rng }) => {
    const s = env.starts[rng.int(env.starts.length)], acts = env.acts(s);
    return [env.reset(rng, s), acts[rng.int(acts.length)]];
  };
  const exploringStarts = control(randomStart, ({ env, m, rng }, s) => lab.greedy(m.Q, s, env, rng));
  // On-policy control: the usual start, and ε-greedy (an ε-soft policy) all the way.
  const onPolicy = control(({ env, rng }) => [env.reset(rng), -1], ({ env, m, rng, p }, s) => lab.epsGreedy(m.Q, s, env, p.epsilon, rng));

  // ---- Off-policy Monte Carlo control: learn the greedy policy from episodes of an ε-greedy one ----
  // Weighted importance sampling: walking back, each return counts with weight W, the product of 1 / b(A|S) since
  // the end. The first action the greedy policy would not have taken ends the walk: what came before it is not
  // what the greedy policy would have done.
  function* offPolicy({ env, m, rng, p }) {
    const { Q, C } = m, nA = env.nA, probs = new Array(nA);
    const S = [], A = [], R = [], B = [];
    let s = env.reset(rng);
    yield { line: "start", type: "start", s };
    for (let t = 0; !env.terminal(s) && t < p.maxSteps; t++) {
      const a = lab.epsGreedy(Q, s, env, p.epsilon, rng);
      lab.epsGreedyProbs(Q, s, env, p.epsilon, probs);
      yield { line: "generate", type: "choose", s, a };
      const o = env.step(s, a, rng);
      yield { line: "generate", type: "move", s, a, ...o };
      S.push(s); A.push(a); R.push(o.r); B.push(probs[a]);
      s = o.s2;
    }
    let G = 0, W = 1;
    yield { line: "g0", type: "info" };
    for (let t = S.length - 1; t >= 0; t--) {
      const s = S[t], a = A[t], i = s * nA + a;
      G = p.gamma * G + R[t];
      yield { line: "return", type: "return", t, s, a, G, r: R[t], W };
      C[i] += W;
      const old = Q[i], step = W / C[i];
      Q[i] = old + step * (G - old);
      yield { line: "update", type: "update", t, s, a, old, target: G, delta: G - old, value: Q[i], W, step };
      if (a !== lab.greedy(Q, s, env)) { yield { line: "cut", type: "cut", t, s, a }; break; }
      W /= B[t];
      yield { line: "weight", type: "info", t, s, a, W };
    }
  }

  const pairs = { unit: "episode", memory: (env) => ({ Q: env.nS * env.nA, N: env.nS * env.nA }), init: (m, env, p) => m.Q.fill(p.q0 || 0) };
  const numbers = (ev, step) => `\\val{${ev.old.toFixed(2)}} + ${step}\\,\\big[\\rew{${+ev.target.toFixed(2)}} - \\val{${lab.texNum(ev.old)}}\\big] = \\val{${ev.value.toFixed(2)}}`;
  const stepTex = (ev, p) => (p.alpha > 0 ? p.alpha : `\\tfrac{1}{${ev.n}}`);

  Object.assign(lab.algorithms, {
    "mc-prediction": {
      id: "mc-prediction", title: "Monte Carlo", unit: "episode", run: mcPrediction,
      memory: (env) => ({ V: env.nS, N: env.nS }),
      init: (m, env, p) => { for (let s = 0; s < env.nS; s++) m.V[s] = env.terminal(s) ? 0 : p.v0 || 0; },
      show: (m) => ({ V: m.V, N: m.N }),
      rule: (p) => `\\val{V(S_t)} \\leftarrow \\val{V(S_t)} + ${avg(p)}\\,\\big[\\rew{G_t} - \\val{V(S_t)}\\big]`,
      numbers: (ev, p) => numbers(ev, stepTex(ev, p)),
    },
    "exploring-starts": {
      id: "exploring-starts", title: "Monte Carlo ES", ...pairs, run: exploringStarts,
      show: (m) => ({ Q: m.Q, N: m.N, greedy: true }),
      rule: (p) => `\\val{Q(S_t,A_t)} \\leftarrow \\val{Q(S_t,A_t)} + ${avg(p)}\\,\\big[\\rew{G_t} - \\val{Q(S_t,A_t)}\\big]`,
      numbers: (ev, p) => numbers(ev, stepTex(ev, p)),
    },
    "mc-control": {
      id: "mc-control", title: "On-policy MC", ...pairs, run: onPolicy,
      show: (m) => ({ Q: m.Q, N: m.N }),
      rule: (p) => `\\val{Q(S_t,A_t)} \\leftarrow \\val{Q(S_t,A_t)} + ${avg(p)}\\,\\big[\\rew{G_t} - \\val{Q(S_t,A_t)}\\big]`,
      numbers: (ev, p) => numbers(ev, stepTex(ev, p)),
    },
    "off-policy-mc": {
      id: "off-policy-mc", title: "Off-policy MC", unit: "episode", run: offPolicy,
      memory: (env) => ({ Q: env.nS * env.nA, C: env.nS * env.nA }),
      init: (m, env, p) => m.Q.fill(p.q0 || 0),
      show: (m) => ({ Q: m.Q, greedy: true }),
      rule: "\\val{Q(S_t,A_t)} \\leftarrow \\val{Q(S_t,A_t)} + \\tfrac{W}{C(S_t,A_t)}\\,\\big[\\rew{G_t} - \\val{Q(S_t,A_t)}\\big]",
      numbers: (ev) => numbers(ev, ev.step === 1 ? "1" : `\\tfrac{${+ev.W.toFixed(2)}}{${+(ev.W / ev.step).toFixed(2)}}`),
    },
  });
})(globalThis.RL = globalThis.RL || {});
