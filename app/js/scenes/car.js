/* Story scene car: Mountain Car, on the Lab's car view. Step keys: run, at, play, pace (see sceneKit.runScene);
   look ("3d" or "map") and path (draw the states of the episode). */
(function (RL) {
  "use strict";
  RL.scenes.car = RL.sceneKit.runScene(RL.CarView, {
    options: (st) => ({ look: st.look || "3d", path: st.path !== false }),
  });
})(globalThis.RL = globalThis.RL || {});
