/* Story scene line: the 1000-state walk, on the Lab's line view.
   Step keys: run, at, play, pace (see sceneKit.runScene); truth, visits, footprint, and ghosts: earlier moments of the
   run ([0, 100] units) kept in view as faint lines. */
(function (RL) {
  "use strict";
  RL.scenes.line = RL.sceneKit.runScene(RL.LineView, {
    options: (st) => ({ truth: st.truth !== false, visits: st.visits !== false, footprint: st.footprint !== false }),
    more: (st, view, r) => view.ghosts((st.ghosts || []).map((k) => ({ V: r.algorithm.show(r.at(k), r.env, r.params).V, label: k === 0 ? "start" : `after ${k}` }))),
  });
})(globalThis.RL = globalThis.RL || {});
