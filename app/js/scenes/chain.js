/* Story scene chain: the random walk, on the Lab's chain view.
   Step keys: run, at, play, pace (see sceneKit.runScene); truth, numbers, and ghosts: earlier moments of the run
   ([0, 1, 10] units) kept in view as faint lines, as in Sutton & Barto's Figure 6.2. */
(function (RL) {
  "use strict";
  RL.scenes.chain = RL.sceneKit.runScene(RL.ChainView, {
    options: (st) => ({ truth: st.truth !== false, numbers: !!st.numbers }),
    more: (st, view, r) => view.ghosts((st.ghosts || []).map((k) => ({ V: r.at(k).V, label: k === 0 ? "start" : `${k}` }))),
  });
})(globalThis.RL = globalThis.RL || {});
