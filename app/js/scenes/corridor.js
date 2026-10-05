/* Story scene corridor: the short corridor, on the Lab's corridor view.
   Step keys: run, at, play, pace (see sceneKit.runScene), numbers (the chances in the cells), and p: show a fixed
   policy that steps right with chance p, instead of the run's own. */
(function (RL) {
  "use strict";
  const fixed = (p) => Float64Array.from([1 - p, p, 1 - p, p, 1 - p, p, 0, 0]);
  RL.scenes.corridor = RL.sceneKit.runScene(RL.CorridorView, {
    options: (st) => ({ numbers: st.numbers !== false, agent: st.agent !== false }),
    more: (st, view) => { if (st.p !== undefined) { view.show({ P: fixed(st.p) }); view.rest([]); } },
  });
})(globalThis.RL = globalThis.RL || {});
