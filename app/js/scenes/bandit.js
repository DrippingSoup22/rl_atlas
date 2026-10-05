/* Story scene bandit: the slot machines of a k-armed bandit, on the Lab's arms view.
   Step keys: run, at, play, pace (see sceneKit.runScene); truth (draw the true payouts), learner (draw what the
   learner knows), numbers, focus (outline one arm, counted from 1), curves and metric (averages under the picture). */
(function (RL) {
  "use strict";
  RL.scenes.bandit = RL.sceneKit.runScene(RL.ArmsView, {
    options: (st) => ({ truth: st.truth !== false, learner: st.learner !== false, numbers: !!st.numbers }),
    more: (st, view) => view.mark(st.focus ? st.focus - 1 : -1),
  });
})(globalThis.RL = globalThis.RL || {});
