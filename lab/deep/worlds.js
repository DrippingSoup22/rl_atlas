/* The deep trainer's worlds: Gymnasium's CartPole-v1 and Pendulum-v1, their equations ported line for line onto the
   deterministic math (dmath.js). Each world also says what a snapshot maps (a 31 × 31 grid over two of its numbers)
   and how its states are stored (the ranges an 8-bit recording packs them into), as recorder/record.py did.
   Episodes start from the trainer's own generator, not Gymnasium's: the runs are not Gymnasium's runs, only its
   physics. */
(function (RL) {
  "use strict";
  const D = RL.dmath;
  const deep = (RL.deep = RL.deep || {});
  const GRID = 31;
  const axis = (lo, hi) => Array.from({ length: GRID }, (_, i) => lo + ((hi - lo) * i) / (GRID - 1));
  const grid = (xs, ys, at) => { const out = []; for (const y of ys) for (const x of xs) out.push(at(x, y)); return out; };

  // CartPole: (x, ẋ, θ, θ̇); push left or right with 10 N; ends past 12° or off the track; +1 a step; at most 500.
  const CP = { g: 9.8, mc: 1, mp: 0.1, l: 0.5, f: 10, tau: 0.02, xMax: 2.4, thMax: (12 * 2 * Math.PI) / 360 };
  const cartpole = {
    id: "cartpole", obs: 4, nA: 2, maxSteps: 500,
    x: ["angle", -0.21, 0.21], y: ["angular velocity", -2, 2],
    ranges: [[-2.4, 2.4], [-3, 3], [-0.21, 0.21], [-3.5, 3.5]],
    grid: () => grid(axis(-0.21, 0.21), axis(-2, 2), (th, om) => [0, 0, th, om]), // the cart at rest in the middle
    reset: (r) => [0, 0, 0, 0].map(() => -0.05 + 0.1 * r.next()),
    observe: (s) => s,
    state: (s) => s,
    step(s, a) {
      const [x, v, th, w] = s, F = a === 1 ? CP.f : -CP.f, c = D.cos(th), sn = D.sin(th), tot = CP.mc + CP.mp, pml = CP.mp * CP.l;
      const temp = (F + pml * w * w * sn) / tot;
      const thAcc = (CP.g * sn - c * temp) / (CP.l * (4 / 3 - (CP.mp * c * c) / tot));
      const xAcc = temp - (pml * thAcc * c) / tot;
      const s2 = [x + CP.tau * v, v + CP.tau * xAcc, th + CP.tau * w, w + CP.tau * thAcc];
      return { s2, r: 1, term: s2[0] < -CP.xMax || s2[0] > CP.xMax || s2[2] < -CP.thMax || s2[2] > CP.thMax };
    },
  };

  // Pendulum: (θ, θ̇), θ = 0 upright; a torque in [−2, 2]; pays −(θ² + 0.1 θ̇² + 0.001 u²); 200 steps. The network sees
  // (cos θ, sin θ, θ̇).
  const PD = { g: 10, m: 1, l: 1, dt: 0.05, maxSpeed: 8, maxTorque: 2 };
  const wrap = (th) => ((((th + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
  const pendulum = {
    id: "pendulum", obs: 3, act: 1, lo: -2, hi: 2, maxSteps: 200,
    x: ["angle", -Math.PI, Math.PI], y: ["angular velocity", -8, 8],
    ranges: [[-Math.PI, Math.PI], [-8, 8]],
    grid: () => grid(axis(-Math.PI, Math.PI), axis(-8, 8), (th, om) => [D.cos(th), D.sin(th), om]),
    reset: (r) => [-Math.PI + 2 * Math.PI * r.next(), -1 + 2 * r.next()],
    observe: (s) => [D.cos(s[0]), D.sin(s[0]), s[1]],
    state: (s) => [wrap(s[0]), s[1]],
    step(s, a) {
      const [th, w] = s, u = Math.max(-PD.maxTorque, Math.min(PD.maxTorque, a)), n = wrap(th);
      const cost = n * n + 0.1 * w * w + 0.001 * u * u;
      let w2 = w + ((3 * PD.g) / (2 * PD.l) * D.sin(th) + (3 / (PD.m * PD.l * PD.l)) * u) * PD.dt;
      w2 = Math.max(-PD.maxSpeed, Math.min(PD.maxSpeed, w2));
      return { s2: [th + w2 * PD.dt, w2], r: -cost, term: false };
    },
  };

  deep.worlds = { cartpole, pendulum };
  deep.GRID = GRID;
})(globalThis.RL = globalThis.RL || {});
