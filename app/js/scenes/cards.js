/* Story scene cards: Blackjack, on the Lab's cards view.
   Step keys: run, at, play, pace (see sceneKit.runScene); tiles ("policy" or "v") and numbers. */
(function (RL) {
  "use strict";
  RL.scenes.cards = RL.sceneKit.runScene(RL.CardsView, {
    options: (st) => ({ tiles: st.tiles || "policy", numbers: !!st.numbers }),
  });
})(globalThis.RL = globalThis.RL || {});
