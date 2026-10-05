/* Story scenes on runs trained offline (recorder/record.py): CartPole and Pendulum, on the Lab's views.
   Step keys: run, at, play, pace (see sceneKit.runScene), map ("value" or "action") and path (false hides it). */
(function (RL) {
  "use strict";
  const options = (st) => ({ map: st.map || "value", path: st.path !== false });
  RL.scenes.cartpole = RL.sceneKit.runScene(RL.labViews.cartpole, { options });
  RL.scenes.pendulum = RL.sceneKit.runScene(RL.labViews.pendulum, { options });
})(globalThis.RL = globalThis.RL || {});
