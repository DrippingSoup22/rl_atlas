/* The network, as a strip under a deep run's drawing: one row per hidden layer, one cell per unit, shaded by how often
   the unit fires over the map's states (lab/deep/ counts it for every snapshot). With ReLU, an empty cell is a dead
   unit: it never fires, so it never learns again. With tanh, the shading shows saturation instead (|h| > 0.97, where
   the unit's slope, and so its gradient, is nearly zero). The depth and width knobs reshape the strip. It is HTML, in
   the grid of the drawing's bars (lab/views/readouts.js), so that its words keep their size in a narrow pane. */
(function (RL) {
  "use strict";
  const off = (relu, share) => relu && share < 0.005;
  const marks = (relu, f) => f.reduce((k, share) => k + (relu ? off(relu, share) : share > 0.5), 0);
  const tip = (relu, i, n, marked) => (relu
    ? `Layer ${i + 1}: ${n} units, each shaded by how often it fires over the map's states. ${marked} never fire${marked === 1 ? "s" : ""} (dead: no gradient reaches ${marked === 1 ? "it" : "them"}, so ${marked === 1 ? "it stays" : "they stay"} dead).`
    : `Layer ${i + 1}: ${n} units, each shaded by how often it is saturated (|h| > 0.97) over the map's states. ${marked} ${marked === 1 ? "is" : "are"} saturated on most states, where ${marked === 1 ? "its" : "their"} gradient nearly vanishes.`);
  RL.netStrip = {
    // Draw net (sizes, act, of) and units (per hidden layer, a share per unit) at the end of grid, replacing the last.
    // The same network at another snapshot only reshades its cells and counts: a run played fast does it every frame.
    draw(grid, net, units) {
      const key = net && units ? `${net.of} ${net.sizes} ${net.act} ${units.map((f) => f.length)}` : "";
      const rows = grid.querySelectorAll(".net-row");
      if (key && key === grid.dataset.net && rows.length === units.length) return this.reshade(rows, net, units);
      grid.dataset.net = key;
      grid.querySelectorAll(".net").forEach((e) => e.remove());
      if (!net || !units) return;
      const relu = net.act === "relu";
      grid.appendChild(RL.h('<div class="net net-title"></div>')).textContent = `${net.of}: ${net.sizes.join(" → ")} · ${relu ? "ReLU" : "tanh"}`;
      units.forEach((f, i) => {
        const cells = Array.from(f, (share) => (off(relu, share) ? '<i class="dead"></i>' : `<i style="--s: ${Math.round(100 * share)}%"></i>`)).join(""), marked = marks(relu, f);
        const row = grid.appendChild(RL.h(`<div class="net ctl-row net-row${relu ? "" : " tanh"}"><span class="bar-name">layer ${i + 1}</span><span class="cells">${cells}</span><span class="bar-num">${marked} ${relu ? "dead" : "saturated"}</span></div>`));
        row.title = tip(relu, i, f.length, marked);
      });
    },
    reshade(rows, net, units) {
      const relu = net.act === "relu";
      units.forEach((f, i) => {
        const cells = rows[i].querySelector(".cells").children, marked = marks(relu, f), num = rows[i].querySelector(".bar-num"), text = `${marked} ${relu ? "dead" : "saturated"}`;
        f.forEach((share, j) => {
          const dead = off(relu, share), s = dead ? "" : `${Math.round(100 * share)}%`, c = cells[j];
          c.classList.toggle("dead", dead);
          if (c.style.getPropertyValue("--s") !== s) c.style.setProperty("--s", s);
        });
        if (num.textContent !== text) { num.textContent = text; rows[i].title = tip(relu, i, f.length, marked); }
      });
    },
  };
})(globalThis.RL = globalThis.RL || {});
