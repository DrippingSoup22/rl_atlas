/* Features: how a world too big for a table is described to a learner with a few weights. A feature vector x(s)
   turns a state into numbers, and the estimate is their weighted sum, v̂(s) = w · x(s). Every construction here
   works on the state's coordinates, which the world scales to the unit interval (env.coords(s) → [u₁, u₂, …]):

   groups    state aggregation: the space is cut into `cells` equal pieces per coordinate, one weight per piece
   tiles     tile coding: `tilings` grids of `cells` tiles per coordinate, each shifted a little; one tile of each is on
   poly      polynomials: every product u₁^i u₂^j … with powers up to `order`
   fourier   cosines: cos(π (c₁u₁ + c₂u₂ + …)) for every whole c with entries up to `order`
   table     one weight per state: the tabular case, written as features
   own       the world's own features (Baird's counterexample)

   A feature vector is kept sparse, as the indices of the features that are not zero and their values, so a state of
   Mountain Car with 8 tilings costs 8 multiplications and not 648. Action values use one copy of the weights per
   action: q̂(s, a) = w[a·n …] · x(s). */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const cache = new Map();

  // The features a run uses, from its settings: p.features, p.cells, p.tilings, p.order.
  lab.features = function (env, p) {
    const kind = p.features || "table";
    const key = `${env.key || env.name}|${kind}|${p.cells ?? ""}|${p.tilings ?? ""}|${p.order ?? ""}`;
    if (!cache.has(key)) cache.set(key, build(env, kind, p));
    return cache.get(key);
  };

  function build(env, kind, p) {
    const d = env.dims || 1;
    let F;
    if (kind === "table") {
      F = { n: env.nS, k: 1, at: (s, out) => { out.idx[0] = s; out.val[0] = 1; out.k = 1; } };
      F.init = (w) => { if (env.initV) w.set(env.initV); };
    } else if (kind === "own") {
      F = { n: env.features.n, k: env.features.k, at: env.features.at, init: env.features.init };
    } else if (kind === "groups") {
      const c = p.cells ?? 10;
      F = { n: c ** d, k: 1, cells: c, at(s, out) {
        const u = env.coords(s);
        let i = 0;
        for (let j = 0; j < d; j++) i = i * c + Math.min(c - 1, Math.max(0, Math.floor(u[j] * c)));
        out.idx[0] = i; out.val[0] = 1; out.k = 1;
      } };
    } else if (kind === "tiles") {
      // Each tiling is a grid of tiles `1/cells` wide, with one tile more per coordinate so that a shifted grid still
      // covers the space. Tiling i is shifted by i/tilings of a tile, times 1, 3, 5, … along the coordinates: the
      // uneven shifts Sutton & Barto recommend, so the tilings do not all line up along a diagonal.
      const c = p.cells ?? 8, T = p.tilings ?? 8, per = (c + 1) ** d;
      F = { n: T * per, k: T, cells: c, tilings: T, at(s, out) {
        const u = env.coords(s);
        for (let t = 0; t < T; t++) {
          let i = 0;
          for (let j = 0; j < d; j++) {
            const shift = ((t * (2 * j + 1)) % T) / T;
            i = i * (c + 1) + Math.min(c, Math.max(0, Math.floor(u[j] * c + shift)));
          }
          out.idx[t] = t * per + i; out.val[t] = 1;
        }
        out.k = T;
      } };
    } else if (kind === "poly" || kind === "fourier") {
      const n = p.order ?? 5, combos = [];
      for (let i = 0; i < (n + 1) ** d; i++) {
        const c = [];
        for (let j = 0, r = i; j < d; j++, r = Math.floor(r / (n + 1))) c.unshift(r % (n + 1));
        combos.push(c);
      }
      const f = kind === "poly"
        ? (u, c) => { let v = 1; for (let j = 0; j < d; j++) v *= u[j] ** c[j]; return v; }
        : (u, c) => { let z = 0; for (let j = 0; j < d; j++) z += c[j] * u[j]; return Math.cos(Math.PI * z); };
      F = { n: combos.length, k: combos.length, order: n, combos, at(s, out) {
        const u = env.coords(s);
        for (let i = 0; i < combos.length; i++) { out.idx[i] = i; out.val[i] = f(u, combos[i]); }
        out.k = combos.length;
      } };
    } else throw new Error(`unknown features '${kind}'`);
    F.kind = kind;
    F.vector = () => ({ idx: new Int32Array(F.k), val: new Float64Array(F.k), k: 0 });
    // A world with a few hundred states keeps every state's vector, worked out once.
    if (env.nS && env.nS <= 5000 && !env.continuous) {
      const all = Array.from({ length: env.nS }, (_, s) => { const x = F.vector(); F.at(s, x); return x; });
      F.of = (s) => all[s];
    } else F.of = (s) => { const x = F.vector(); F.at(s, x); return x; };
    return F;
  }

  // w · x, for the copy of the weights that starts at `off` (an action's copy, for action values).
  lab.dot = function (w, x, off = 0) {
    let v = 0;
    for (let i = 0; i < x.k; i++) v += w[off + x.idx[i]] * x.val[i];
    return v;
  };
  // w ← w + c x: the step of a linear method, whose gradient is x itself.
  lab.addTo = function (w, x, c, off = 0) {
    for (let i = 0; i < x.k; i++) w[off + x.idx[i]] += c * x.val[i];
  };
  // The estimates of every state of a small world, from the weights.
  lab.valuesOf = function (env, F, w, out = new Float64Array(env.nS)) {
    for (let s = 0; s < env.nS; s++) out[s] = env.terminal(s) ? 0 : lab.dot(w, F.of(s));
    return out;
  };
})(globalThis.RL = globalThis.RL || {});
