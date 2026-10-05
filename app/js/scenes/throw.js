/* Story scene throw: one throw per episode, on the Lab's throw view.
   Step keys: run, at, play, pace (see sceneKit.runScene) and truth (draw how far each angle flies). */
(function (RL) {
  "use strict";
  RL.scenes.throw = RL.sceneKit.runScene(RL.ThrowView, {
    options: (st) => ({ truth: st.truth !== false }),
  });
})(globalThis.RL = globalThis.RL || {});
