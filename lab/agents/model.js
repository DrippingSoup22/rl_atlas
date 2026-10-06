/* Planning with a learned model that generalizes, and exploiting its blind spots (model-based deep RL, in a grid).

   The agent's model knows how moves work: a move goes one tile in its direction, the edge of the world stops it, and
   it knows where the goal is. What each tile does when stepped on (an ordinary tile, a cliff that sends it back to
   the start, the goal) it learns only by stepping on it; on a tile it has never seen, the model predicts an ordinary
   one. That is a model's blind spot: confident where it has no data.

   model-planner  every step, value iteration in the model (cheap: a few dozen tiles), then the first move of the
                  best plan, in the real world; whatever happens is written into the model, and it plans again. The
                  first episode, p.guided, is a walk shown along a world's own route (p.behavior, "careful" on the
                  cliff), which teaches it those tiles. p.doubt is added to the cost of every planned step onto a tile
                  it has never seen: 0 trusts the model's guess, larger values keep it where it has data (an ensemble's
                  disagreement does that for a network). */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];

  // KT: what the model knows of each tile (0: never seen, 1: ordinary, 2: sends you elsewhere, 3: the goal, 4: a wall)
  // · KR: what stepping on it paid · KL: where it left you · V, Q: the plan's values, from the last planning
  const memory = (env) => ({ KT: env.nS, KR: env.nS, KL: env.nS, V: env.nS, Q: env.nS * env.nA, PL: 1 });
  function init(m, env) {
    for (let s = 0; s < env.nS; s++) if (env.terminal(s)) { m.KT[s] = 3; m.KL[s] = s; m.KR[s] = env.rewards.goal ?? env.rewards.step; }
    m.KT[env.start] = 1; m.KL[env.start] = env.start; m.KR[env.start] = env.rewards.step;
  }

  // What the model predicts for a move: { s2, r, guess } (guess: onto a tile it has never seen).
  function predict(m, env, s, a, p) {
    const [r, c] = env.rc(s), r2 = r + MOVES[a][0], c2 = c + MOVES[a][1];
    if (r2 < 0 || c2 < 0 || r2 >= env.rows || c2 >= env.cols) return { s2: s, r: env.rewards.step, guess: false };
    const t = r2 * env.cols + c2;
    if (!m.KT[t]) return { s2: t, r: env.rewards.step - (p.doubt || 0), guess: true };
    if (m.KT[t] === 4) return { s2: s, r: m.KR[t], guess: false };
    return { s2: m.KL[t], r: m.KR[t], guess: false };
  }

  // Value iteration in the model, undiscounted by default: the best predicted return from every tile.
  function plan(m, env, p) {
    const { V, Q } = m, nA = env.nA, gamma = p.gamma ?? 1;
    V.fill(0);
    for (let sweep = 0; sweep < 4 * env.nS; sweep++) {
      let change = 0;
      for (let s = 0; s < env.nS; s++) {
        if (m.KT[s] >= 2) continue; // nobody stands on a cliff, the goal or a wall
        let best = -Infinity;
        for (let a = 0; a < nA; a++) {
          const o = predict(m, env, s, a, p), q = o.r + gamma * (m.KT[o.s2] === 3 ? 0 : V[o.s2]);
          Q[s * nA + a] = q;
          if (q > best) best = q;
        }
        change = Math.max(change, Math.abs(best - V[s]));
        V[s] = best;
      }
      if (change < 1e-9) break;
    }
    m.PL[0] += 1;
  }

  // The plan from s, as the model imagines it: up to 40 moves.
  function imagined(m, env, s, p) {
    const list = [], seen = new Set([s]);
    for (let k = 0; k < 40 && m.KT[s] !== 3; k++) {
      const a = lab.argmax(m.Q.subarray(s * env.nA, (s + 1) * env.nA)), o = predict(m, env, s, a, p);
      list.push({ s, a });
      if (seen.has(o.s2)) break;
      seen.add(o.s2);
      s = o.s2;
    }
    return list;
  }

  function* planner({ env, m, rng, p, t }) {
    const guided = t < (p.guided ?? 1), B = guided ? env.policy(p.behavior || "careful") : null, nA = env.nA;
    let s = env.reset(rng), falls = 0;
    yield { line: guided ? "guided" : "start", type: "start", s };
    for (let k = 0; !env.terminal(s) && k < p.maxSteps; k++) {
      let a;
      if (guided) a = lab.pickRow(B, s, env, rng);
      else {
        plan(m, env, p);
        const list = imagined(m, env, s, p);
        yield { line: "plan", type: "plan", s, list, ordered: true, cost: -m.V[s] };
        a = rng.next() < (p.epsilon || 0) ? rng.int(nA) : lab.argmax(m.Q.subarray(s * nA, (s + 1) * nA));
      }
      yield { line: guided ? "guided" : "act", type: "choose", s, a };
      const o = env.step(s, a, rng);
      yield { line: guided ? "guided" : "act", type: "move", s, a, ...o };
      // what the move taught: the tile it stepped on, or the wall that stopped it (the edge teaches nothing new)
      const [r0, c0] = env.rc(s), r1 = r0 + MOVES[a][0], c1 = c0 + MOVES[a][1];
      const inside = r1 >= 0 && c1 >= 0 && r1 < env.rows && c1 < env.cols, aimed = inside ? r1 * env.cols + c1 : s;
      const tile = o.fell !== undefined ? o.fell : o.s2 === s ? aimed : o.s2;
      if (tile !== s && m.KT[tile] !== 3) {
        const kind = o.fell !== undefined ? 2 : o.s2 === s ? 4 : 1, fresh = m.KT[tile] !== kind;
        m.KT[tile] = kind; m.KR[tile] = o.r; m.KL[tile] = o.s2;
        if (fresh) yield { line: "model", type: "model", s, a, r: o.r, s2: o.s2, tile, kind };
      }
      if (o.fell !== undefined) falls++;
      s = o.s2;
    }
    if (!guided) plan(m, env, p);
    return { falls };
  }

  // The model's map: tiles it has seen show clear, the rest under fog.
  function known(m, env) {
    const M = new Float64Array(env.nS * env.nA).fill(-1);
    for (let s = 0; s < env.nS; s++) if (m.KT[s]) M.fill(1, s * env.nA, (s + 1) * env.nA);
    return M;
  }

  lab.algorithms = Object.assign(lab.algorithms || {}, {
    "model-planner": {
      id: "model-planner", station: "model-based-deep", title: "Planning in a learned model", unit: "episode", memory, init, run: planner,
      show: (m, env, p) => { if (!m.PL[0]) plan(m, env, p); return { Q: m.Q, V: m.V, model: known(m, env), greedy: true }; },
      knobs: { doubt: { sym: "d", name: "doubt: extra cost of an unseen tile", min: 0, max: 3, step: 0.1, sweep: [0, 0.2, 0.4, 0.6, 0.8, 1, 2] } },
      rule: "\\pi(s) = \\operatorname{arg\\,max}_a \\big[\\hat r(s, a) - d\\,[\\text{unseen}] + \\hat v(\\hat s'(s, a))\\big]",
    },
  });
})(globalThis.RL = globalThis.RL || {});
