/* The network, as a strip under a deep run's drawing: one row per hidden layer, one cell per unit, shaded by how often
   the unit fires over the map's states (lab/deep/ counts it for every snapshot). With ReLU, an empty cell is a dead
   unit: it never fires, so it never learns again. With tanh, the shading shows saturation instead (|h| > 0.97, where
   the unit's slope, and so its gradient, is nearly zero). The depth and width knobs reshape the strip. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg", ROW = 12;
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  RL.netStrip = {
    // The height the strip needs for this network (0 without one).
    height: (net) => (net ? 16 + (net.sizes.length - 2) * ROW + 2 : 0),
    // Draw net (sizes, act, of) and units (per hidden layer, a share per unit) at y in svg, between x0 and x1.
    draw(svg, net, units, { y, x0 = 88, x1 = 278 }) {
      svg.querySelector(".netstrip")?.remove();
      if (!net || !units) return;
      const g = el("g", { class: "netstrip" }, svg), relu = net.act === "relu", w = x1 - x0;
      el("text", { class: "net-title", x: 8, y: y + 9 }, g).textContent = `${net.of}: ${net.sizes.join(" → ")} · ${relu ? "ReLU" : "tanh"}`;
      units.forEach((f, i) => {
        const yy = y + 14 + i * ROW, n = f.length, c = w / n, row = el("g", { class: `net-row${relu ? "" : " tanh"}` }, g);
        el("text", { class: "bar-name", x: 8, y: yy + 8 }, row).textContent = `layer ${i + 1}`;
        let marked = 0;
        for (let j = 0; j < n; j++) {
          const share = f[j], off = relu ? share < 0.005 : false;
          if (relu ? off : share > 0.5) marked++;
          el("rect", { class: off ? "unit dead" : "unit", x: (x0 + j * c).toFixed(2), y: yy, width: Math.max(0.6, c - (c > 3 ? 0.6 : 0.15)).toFixed(2), height: 9,
            style: off ? "" : `fill: color-mix(in oklab, var(${relu ? "--v-pos" : "--v-neg"}) ${Math.round(100 * share)}%, var(--v-mid))` }, row);
        }
        el("text", { class: "bar-num", x: x1 + 6, y: yy + 8 }, row).textContent = relu ? `${marked} dead` : `${marked} saturated`;
        el("title", {}, row).textContent = relu
          ? `Layer ${i + 1}: ${n} units, each shaded by how often it fires over the map's states. ${marked} never fire${marked === 1 ? "s" : ""} (dead: no gradient reaches ${marked === 1 ? "it" : "them"}, so ${marked === 1 ? "it stays" : "they stay"} dead).`
          : `Layer ${i + 1}: ${n} units, each shaded by how often it is saturated (|h| > 0.97) over the map's states. ${marked} ${marked === 1 ? "is" : "are"} saturated on most states, where ${marked === 1 ? "its" : "their"} gradient nearly vanishes.`;
      });
    },
  };
})(globalThis.RL = globalThis.RL || {});
