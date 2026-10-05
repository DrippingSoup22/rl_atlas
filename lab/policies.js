/* Policies with weights: the policy itself is learned, as a function of features, instead of being read off values.

   softmax   discrete actions. Each action has a preference h(s, a) = θ_a · x(s), a weighted sum of the state's
             features with one copy of the weights per action (as q̂ does in semi-gradient SARSA), and the policy is
             their softmax over the actions available in s. With a table for features this is one preference per
             state and action, the tabular case; with the short corridor's single feature, every state looks alike.
   gaussian  a continuous action: a normal distribution with mean μ(s) = θ_μ · x(s) and spread σ(s) = exp(θ_σ · x(s)).
             The world says in what unit the policy measures its action (env.continuousActions.unit), so that one step
             size suits the mean and the spread alike.

   Both give what a policy-gradient method needs: an action drawn from π, the gradient of ln π(a | s), and the
   entropy of π(· | s) with its gradient. The features are the Lab's own (lab.features), set by p.features. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const cache = new Map();
  const HALF_LOG_2PIE = 0.5 * Math.log(2 * Math.PI * Math.E);

  lab.policy = function (env, p) {
    const F = lab.features(env, p);
    const key = `${env.key || env.name}|${F.kind}|${p.cells ?? ""}|${p.tilings ?? ""}|${p.order ?? ""}`;
    if (!cache.has(key)) cache.set(key, env.continuousActions ? gaussian(env, F) : softmax(env, F));
    return cache.get(key);
  };

  function softmax(env, F) {
    const n = F.n, nA = env.nA, h = new Float64Array(nA), own = new Float64Array(nA);
    const P = {
      kind: "softmax", n: n * nA, F,
      init(theta, p = {}) { theta.fill(0); env.policyInit?.(theta, p); },
      // π(· | s) into out (0 for actions not available in s). Without out, a shared buffer: copy it to keep it.
      probs(theta, s, out = own) {
        const x = F.of(s), acts = env.acts(s);
        out.fill(0);
        let top = -Infinity, sum = 0;
        for (const a of acts) { h[a] = lab.dot(theta, x, a * n); top = Math.max(top, h[a]); }
        for (const a of acts) sum += out[a] = Math.exp(h[a] - top);
        for (const a of acts) out[a] /= sum;
        return out;
      },
      prob: (theta, s, a) => P.probs(theta, s)[a],
      sample(theta, s, rng) {
        const q = P.probs(theta, s), acts = env.acts(s);
        let u = rng.next();
        for (const a of acts) { u -= q[a]; if (u < 0) return a; }
        for (let i = acts.length - 1; i > 0; i--) if (q[acts[i]] > 0) return acts[i];
        return acts[0];
      },
      // out += c ∇ ln π(a | s): the chosen action's weights move by c (1 − π(a | s)) x(s), every other action's by −c π(b | s) x(s)
      grad(theta, s, a, c, out) {
        const x = F.of(s), q = P.probs(theta, s);
        for (const b of env.acts(s)) { const k = (b === a ? 1 : 0) - q[b]; if (k) lab.addTo(out, x, c * k, b * n); }
      },
      entropy(theta, s) {
        const q = P.probs(theta, s);
        let H = 0;
        for (const b of env.acts(s)) if (q[b] > 0) H -= q[b] * Math.log(q[b]);
        return H;
      },
      // out += c ∇H(π(· | s)): with H = −Σ π ln π, ∂H/∂h(s, b) = −π(b | s) (ln π(b | s) + H)
      gradEntropy(theta, s, c, out) {
        const x = F.of(s), q = P.probs(theta, s), acts = env.acts(s);
        let H = 0;
        for (const b of acts) if (q[b] > 0) H -= q[b] * Math.log(q[b]);
        for (const b of acts) if (q[b] > 0) lab.addTo(out, x, -c * q[b] * (Math.log(q[b]) + H), b * n);
      },
      // Every state's probabilities at once (nS × nA), for the views and for evaluating the policy exactly.
      table(theta, out = new Float64Array(env.nS * nA)) {
        out.fill(0);
        for (let s = 0; s < env.nS; s++) {
          if (env.terminal(s) || env.blocked?.(s)) continue;
          const q = P.probs(theta, s);
          for (let a = 0; a < nA; a++) out[s * nA + a] = q[a];
        }
        return out;
      },
    };
    return P;
  }

  function gaussian(env, F) {
    const n = F.n, unit = env.continuousActions.unit;
    const P = {
      kind: "gaussian", n: 2 * n, F, unit,
      init(theta, p = {}) { theta.fill(0); env.policyInit?.(theta, p); },
      mean: (theta, s) => lab.dot(theta, F.of(s)),
      sd: (theta, s) => Math.exp(lab.dot(theta, F.of(s), n)),
      // an action in the world's own units (degrees, say): unit × a draw from N(μ, σ²)
      sample(theta, s, rng) { return unit * (P.mean(theta, s) + P.sd(theta, s) * rng.normal()); },
      density(theta, s, a) { const sd = P.sd(theta, s), z = (a / unit - P.mean(theta, s)) / sd; return Math.exp(-0.5 * z * z) / (sd * unit * Math.sqrt(2 * Math.PI)); },
      // ∂ ln π / ∂θ_μ = (u − μ) / σ² · x(s) and ∂ ln π / ∂θ_σ = ((u − μ)² / σ² − 1) · x(s), with u the action in the policy's unit
      grad(theta, s, a, c, out) {
        const x = F.of(s), sd = P.sd(theta, s), z = (a / unit - P.mean(theta, s)) / sd;
        lab.addTo(out, x, (c * z) / sd);
        lab.addTo(out, x, c * (z * z - 1), n);
      },
      entropy: (theta, s) => lab.dot(theta, F.of(s), n) + HALF_LOG_2PIE, // ln σ + ½ ln(2πe), in the policy's unit
      gradEntropy(theta, s, c, out) { lab.addTo(out, F.of(s), c, n); }, // ∂H/∂θ_σ = x(s)
    };
    return P;
  }
})(globalThis.RL = globalThis.RL || {});
