/* Story scene chain: the random walk, on the Lab's chain view.
   Step keys: run, at, play, pace (see sceneKit.runScene); truth, numbers, and ghosts: earlier moments of the run
   ([0, 1, 10] units) kept in view as faint lines, as in Sutton & Barto's Figure 6.2. seeds = n: the same run from
   seeds 1 to n instead, each a faint line at the same moment, and their average as the bold one. */
(function (RL) {
  "use strict";
  const others = new WeakMap(); // run → its copies from other seeds, made once
  function seedsOf(r, n) {
    let list = others.get(r);
    if (!list || list.length !== n) {
      list = Array.from({ length: n }, (_, k) => RL.lab.simulate({ world: r.world, algorithm: r.algorithm, params: r.params, units: r.units, seed: k + 1 }));
      others.set(r, list);
    }
    return list;
  }
  RL.scenes.chain = RL.sceneKit.runScene(RL.ChainView, {
    options: (st) => ({ truth: st.truth !== false, numbers: !!st.numbers }),
    more(st, view, r, t) {
      if (!st.seeds) return view.ghosts((st.ghosts || []).map((k) => ({ V: r.at(k).V, label: k === 0 ? "start" : `${k}` })));
      const Vs = seedsOf(r, st.seeds).map((x) => x.at(t).V), mean = new Float64Array(Vs[0].length);
      for (const V of Vs) V.forEach((v, s) => (mean[s] += v / Vs.length));
      view.ghosts(Vs.map((V) => ({ V })));
      view.show({ V: mean });
    },
  });
})(globalThis.RL = globalThis.RL || {});
