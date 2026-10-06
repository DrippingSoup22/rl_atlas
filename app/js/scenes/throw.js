/* Story scene throw: one throw per episode, on the Lab's throw view.
   Step keys: run, at, play, pace (see sceneKit.runScene), truth (draw how far each angle flies) and recent = n: the last
   n throws before the moment, drawn at once (a critic's data). */
(function (RL) {
  "use strict";
  function lastThrows(r, t, n) {
    const list = [];
    for (let u = Math.max(0, t - n); u < t; u++) {
      for (const ev of r.replay(u).events) if (ev.type === "move") list.push({ angle: ev.a, land: ev.r });
    }
    return list;
  }
  RL.scenes.throw = RL.sceneKit.runScene(RL.ThrowView, {
    options: (st) => ({ truth: st.truth !== false }),
    more: (st, view, r, t) => { if (st.recent) view.recent(lastThrows(r, t, st.recent)); },
  });
})(globalThis.RL = globalThis.RL || {});
