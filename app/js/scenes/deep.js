/* Story scenes on runs trained offline (recorder/record.py): CartPole and Pendulum, on the Lab's views.
   Step keys: run, at, play, pace (see sceneKit.runScene), map ("value" or "action") and path (false hides it). A
   step after some training plays its test episode in a loop. */
(function (RL) {
  "use strict";
  const options = (st) => ({ map: st.map || "value", path: st.path !== false });
  RL.scenes.cartpole = RL.sceneKit.runScene(RL.labViews.cartpole, { options, test: true });
  RL.scenes.pendulum = RL.sceneKit.runScene(RL.labViews.pendulum, { options, test: true });
})(globalThis.RL = globalThis.RL || {});
