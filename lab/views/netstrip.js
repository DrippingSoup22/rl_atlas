/* The network, as a strip under a deep run's drawing: one row per hidden layer, one cell per unit, shaded by how often
   the unit fires over the map's states (lab/deep/ counts it for every snapshot). With ReLU, an empty cell is a dead
   unit: it never fires, so it never learns again. With tanh, the shading shows saturation instead (|h| > 0.97, where
   the unit's slope, and so its gradient, is nearly zero). The depth and width knobs reshape the strip. It is HTML, in
   the grid of the drawing's bars (lab/views/readouts.js), so that its words keep their size in a narrow pane. */
(function (RL) {
  "use strict";
  RL.netStrip = {
    // Draw net (sizes, act, of) and units (per hidden layer, a share per unit) at the end of grid, replacing the last.
    draw(grid, net, units) {
      grid.querySelectorAll(".net").forEach((e) => e.remove());
      if (!net || !units) return;
      const relu = net.act === "relu";
      grid.appendChild(RL.h('<div class="net net-title"></div>')).textContent = `${net.of}: ${net.sizes.join(" → ")} · ${relu ? "ReLU" : "tanh"}`;
      units.forEach((f, i) => {
        let marked = 0, cells = "";
        for (const share of f) {
          const off = relu && share < 0.005;
          if (relu ? off : share > 0.5) marked++;
          cells += off ? '<i class="dead"></i>' : `<i style="--s: ${Math.round(100 * share)}%"></i>`;
        }
        const n = f.length, row = grid.appendChild(RL.h(`<div class="net ctl-row net-row${relu ? "" : " tanh"}"><span class="bar-name">layer ${i + 1}</span><span class="cells">${cells}</span><span class="bar-num">${marked} ${relu ? "dead" : "saturated"}</span></div>`));
        row.title = relu
          ? `Layer ${i + 1}: ${n} units, each shaded by how often it fires over the map's states. ${marked} never fire${marked === 1 ? "s" : ""} (dead: no gradient reaches ${marked === 1 ? "it" : "them"}, so ${marked === 1 ? "it stays" : "they stay"} dead).`
          : `Layer ${i + 1}: ${n} units, each shaded by how often it is saturated (|h| > 0.97) over the map's states. ${marked} ${marked === 1 ? "is" : "are"} saturated on most states, where ${marked === 1 ? "its" : "their"} gradient nearly vanishes.`;
      });
    },
  };
})(globalThis.RL = globalThis.RL || {});
