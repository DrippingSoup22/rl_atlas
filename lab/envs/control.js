/* Classic control, as Gymnasium plays it, with the physics ported line for line so that runs can be trained in the
   page. Each world's state is a few real numbers; the Lab scales them to the unit interval (coords) for tile coding.
   - CartPole-v1 (Barto, Sutton & Anderson, 1983): push a cart left or right to keep a pole balanced on it.
   - Acrobot-v1 (Sutton, 1996): two links hanging from a pivot, a motor only at the joint between them; swing the tip
     above a line one link above the pivot. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});
  const clamp01 = (v) => Math.max(0, Math.min(1, v));

  // ---- CartPole: a state is (x, ẋ, θ, θ̇) ----
  // The cart's position and speed, the pole's angle from upright and its angular speed. Each push is 10 N, for 0.02 s
  // (Euler steps). An episode ends when the pole leans more than 12° or the cart leaves the track (|x| > 2.4), or after
  // 500 steps; every step pays +1, the last one included. Episodes start with all four numbers within ±0.05.
  const CP = { g: 9.8, mCart: 1.0, mPole: 0.1, half: 0.5, force: 10, tau: 0.02, xMax: 2.4, thetaMax: (12 * 2 * Math.PI) / 360 };
  CP.total = CP.mCart + CP.mPole;
  CP.pml = CP.mPole * CP.half;
  // The ranges the features cover (the speeds have no limit; these hold nearly every state an episode visits).
  const CP_RANGE = [[-CP.xMax, CP.xMax], [-3, 3], [-CP.thetaMax, CP.thetaMax], [-3.5, 3.5]];
  // The map the views draw: angle × spin, the cart at rest in the middle (31 × 31, the spin from −2 to +2 rad/s).
  const phaseGrid = (N, at) => { const out = []; for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) out.push(at(i / (N - 1), j / (N - 1))); return out; };
  const CP_PHASE = { N: 31, states: phaseGrid(31, (u, v) => [0, 0, (2 * u - 1) * CP.thetaMax, (2 * v - 1) * 2]) };

  lab.cartpole = function () {
    const all = [0, 1];
    const env = {
      name: "cartpole", key: "cartpole", kind: "cartpole", title: "CartPole", nA: 2, actionNames: ["push left", "push right"],
      xMax: CP.xMax, thetaMax: CP.thetaMax, poleLength: 2 * CP.half, maxSteps: 500, valueRange: 100, dims: 4, continuous: true,
      pace: { line: 200, step: 22 }, phase: CP_PHASE,
      acts: () => all,
      terminal: (s) => s[0] < -CP.xMax || s[0] > CP.xMax || s[2] < -CP.thetaMax || s[2] > CP.thetaMax,
      reset: (rng, s0) => s0 || [0, 0, 0, 0].map(() => -0.05 + 0.1 * rng.next()),
      step(s, a) {
        const [x, v, th, w] = s, F = a === 1 ? CP.force : -CP.force, cos = Math.cos(th), sin = Math.sin(th);
        const temp = (F + CP.pml * w * w * sin) / CP.total;
        const thAcc = (CP.g * sin - cos * temp) / (CP.half * (4 / 3 - (CP.mPole * cos * cos) / CP.total));
        const xAcc = temp - (CP.pml * thAcc * cos) / CP.total;
        return { s2: [x + CP.tau * v, v + CP.tau * xAcc, th + CP.tau * w, w + CP.tau * thAcc], r: 1 };
      },
      coords: (s) => s.map((v, j) => clamp01((v - CP_RANGE[j][0]) / (CP_RANGE[j][1] - CP_RANGE[j][0]))),
      describe: (s, a) => (s && s.length === 4
        ? `cart at ${lab.fmtSigned(s[0], 2)} m, pole at ${lab.fmtSigned((s[2] * 180) / Math.PI, 1)}°${a >= 0 ? `; ${env.actionNames[a]}` : ""}`
        : a >= 0 ? (a ? "pushing right" : "pushing left") : "the cart and pole"),
    };
    return env;
  };
  lab.worlds.cartpole = () => lab.cartpole();

  // ---- Acrobot: a state is (θ₁, θ₂, θ̇₁, θ̇₂) ----
  // θ₁ is the first link's angle from hanging straight down, θ₂ the second's relative to the first. The motor at the
  // joint gives a torque of −1, 0 or +1, too weak to lift the links directly: they must be swung. Each step lasts 0.2 s
  // (one Runge–Kutta step, as in Sutton & Barto's book version of the dynamics). Every step costs −1 until the tip
  // rises more than one link length above the pivot; episodes are cut at 500 steps.
  const AC = { l1: 1, m1: 1, m2: 1, lc1: 0.5, lc2: 0.5, I1: 1, I2: 1, g: 9.8, dt: 0.2, v1: 4 * Math.PI, v2: 9 * Math.PI };
  const AC_RANGE = [[-Math.PI, Math.PI], [-Math.PI, Math.PI], [-AC.v1, AC.v1], [-AC.v2, AC.v2]];
  function dsdt(y, a) {
    const [t1, t2, d1t, d2t] = y, { m1, m2, l1, lc1, lc2, I1, I2, g } = AC;
    const d1 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * Math.cos(t2)) + I1 + I2;
    const d2 = m2 * (lc2 * lc2 + l1 * lc2 * Math.cos(t2)) + I2;
    const phi2 = m2 * lc2 * g * Math.cos(t1 + t2 - Math.PI / 2);
    const phi1 = -m2 * l1 * lc2 * d2t * d2t * Math.sin(t2) - 2 * m2 * l1 * lc2 * d2t * d1t * Math.sin(t2) + (m1 * lc1 + m2 * l1) * g * Math.cos(t1 - Math.PI / 2) + phi2;
    const dd2 = (a + (d2 / d1) * phi1 - m2 * l1 * lc2 * d1t * d1t * Math.sin(t2) - phi2) / (m2 * lc2 * lc2 + I2 - (d2 * d2) / d1);
    const dd1 = -(d2 * dd2 + phi1) / d1;
    return [d1t, d2t, dd1, dd2];
  }
  const wrap = (x) => { while (x > Math.PI) x -= 2 * Math.PI; while (x < -Math.PI) x += 2 * Math.PI; return x; };
  const bound = (x, m) => Math.max(-m, Math.min(m, x));
  const add = (y, k, h) => y.map((v, i) => v + h * k[i]);
  const tipHeight = (s) => -Math.cos(s[0]) - Math.cos(s[0] + s[1]);

  lab.acrobot = function () {
    const all = [0, 1, 2];
    const env = {
      name: "acrobot", key: "acrobot", kind: "acrobot", title: "Acrobot", nA: 3, actionNames: ["torque −1", "no torque", "torque +1"],
      maxSteps: 500, valueRange: 100, dims: 4, continuous: true, pace: { line: 200, step: 22 }, tipHeight,
      // The map the views draw: the two angles, both links at rest.
      phase: { N: 31, states: phaseGrid(31, (u, v) => [(2 * u - 1) * Math.PI, (2 * v - 1) * Math.PI, 0, 0]) },
      acts: () => all,
      terminal: (s) => tipHeight(s) > 1,
      reset: (rng, s0) => s0 || [0, 0, 0, 0].map(() => -0.1 + 0.2 * rng.next()),
      step(s, a) {
        const u = a - 1, h = AC.dt, f = (y) => dsdt(y, u);
        const k1 = f(s), k2 = f(add(s, k1, h / 2)), k3 = f(add(s, k2, h / 2)), k4 = f(add(s, k3, h));
        const y = s.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
        const s2 = [wrap(y[0]), wrap(y[1]), bound(y[2], AC.v1), bound(y[3], AC.v2)];
        return { s2, r: tipHeight(s2) > 1 ? 0 : -1 };
      },
      coords: (s) => s.map((v, j) => clamp01((v - AC_RANGE[j][0]) / (AC_RANGE[j][1] - AC_RANGE[j][0]))),
      describe: (s, a) => `links at ${lab.fmtSigned((s[0] * 180) / Math.PI, 0)}° and ${lab.fmtSigned((s[1] * 180) / Math.PI, 0)}°${a >= 0 ? `; ${env.actionNames[a]}` : ""}`,
    };
    return env;
  };
  lab.worlds.acrobot = () => lab.acrobot();

  // ---- what a live learner thinks, for the views: over the world's map (env.phase), and in one state ----
  // From a snapshot of a linear learner (w, F: action values) or of a policy method (theta, F, and w for its critic).
  // The map: v, the value of each state (the larger action value; the critic's estimate); act, the action it prefers;
  // pr, for two actions, the probability of the second (a policy) or 0/1 (action values). Kept per snapshot.
  const maps = new WeakMap();
  lab.controlMap = function (env, d) {
    if (maps.has(d)) return maps.get(d);
    const { N, states } = env.phase, nA = env.nA, F = d.F, n = F.n, policy = !!d.theta;
    const P = d.P, critic = policy && d.w && d.w.some((x) => x !== 0);
    const v = new Float64Array(states.length), act = new Uint8Array(states.length), pr = new Float64Array(states.length);
    states.forEach((s, k) => {
      const x = F.of(s);
      if (policy) {
        const q = P.probs(d.theta, s);
        let best = 0;
        for (let a = 1; a < nA; a++) if (q[a] > q[best]) best = a;
        act[k] = best; pr[k] = nA === 2 ? q[1] : q[best];
        v[k] = critic ? lab.dot(d.w, x) : 0;
      } else {
        let best = 0, top = -Infinity;
        for (let a = 0; a < nA; a++) { const q = lab.dot(d.w, x, a * n); if (q > top) { top = q; best = a; } }
        act[k] = best; pr[k] = best; v[k] = top;
      }
    });
    const out = { N, v, act, pr, kind: policy ? "pg" : "q", noValue: policy && !critic };
    maps.set(d, out);
    return out;
  };
  // The numbers a learner gives one state: its action values, or its probabilities.
  lab.controlNumbers = function (env, d, s) {
    const x = d.F.of(s);
    if (d.theta) return { kind: "pg", vals: Array.from(d.P.probs(d.theta, s)) };
    return { kind: "q", vals: Array.from({ length: env.nA }, (_, a) => lab.dot(d.w, x, a * d.F.n)) };
  };
})(globalThis.RL = globalThis.RL || {});
