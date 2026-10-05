/* Backup diagrams, in the style of Sutton & Barto: open circles are states, dots are actions.
   A diagram shows which values an update looks at. Each one draws itself when it scrolls into view. */
(function (RL) {
  "use strict";

  // Levels from top to bottom. Every node of a level has `fan` children on the next level (default 1).
  // label: the level's name; edge: the label of the edges into it; max: an arc across each fan, meaning "take the best".
  const DIAGRAMS = {
    sarsa: [
      { kind: "action", label: "S, A" },
      { kind: "state", label: "S′", edge: "R" },
      { kind: "action", label: "A′" },
    ],
    "q-learning": [
      { kind: "action", label: "S, A" },
      { kind: "state", label: "S′", edge: "R" },
      { kind: "action", label: "every a", fan: 3, max: true },
    ],
    "v-pi": [
      { kind: "state", label: "s" },
      { kind: "action", label: "a", fan: 3, edge: "π" },
      { kind: "state", label: "s′", fan: 2, edge: "r" },
    ],
    "q-pi": [
      { kind: "action", label: "s, a" },
      { kind: "state", label: "s′", fan: 2, edge: "r" },
      { kind: "action", label: "a′", fan: 3, edge: "π" },
    ],
    "v-star": [
      { kind: "state", label: "s" },
      { kind: "action", label: "a", fan: 3, max: true },
      { kind: "state", label: "s′", fan: 2, edge: "r" },
    ],
    "q-star": [
      { kind: "action", label: "s, a" },
      { kind: "state", label: "s′", fan: 2, edge: "r" },
      { kind: "action", label: "a′", fan: 3, max: true },
    ],
  };
  const EDGE_CLASS = { R: "q-rew", r: "q-rew", "π": "q-pol" };

  RL.demos.backup = function (host, id) {
    const levels = DIAGRAMS[id];
    if (!levels) { RL.warn(`no backup diagram for '${id}'`); return; }
    // Place the bottom row evenly, then every parent above the middle of its children.
    const counts = [];
    levels.forEach((lv, k) => counts.push((counts[k - 1] || 1) * (k ? lv.fan || 1 : 1)));
    const last = counts[counts.length - 1], spread = last > 3 ? 34 : 56, gap = 80, top = 22;
    const W = Math.max(240, (last - 1) * spread + 150), cx = (W - 70) / 2;
    const xs = levels.map(() => []);
    xs[levels.length - 1] = Array.from({ length: last }, (_, i) => cx + (i - (last - 1) / 2) * spread);
    for (let k = levels.length - 2; k >= 0; k--) {
      const fan = levels[k + 1].fan || 1;
      xs[k] = Array.from({ length: counts[k] }, (_, i) => xs[k + 1].slice(i * fan, (i + 1) * fan).reduce((a, b) => a + b, 0) / fan);
    }
    const H = top + gap * (levels.length - 1) + 24;
    let svg = `<svg class="backup" viewBox="0 0 ${W} ${H}" role="img" aria-label="Backup diagram">`, d = 0;
    levels.forEach((lv, k) => {
      const y = top + k * gap;
      if (k > 0) {
        const py = y - gap, fan = lv.fan || 1;
        xs[k].forEach((x, i) => {
          svg += `<line class="edge" pathLength="1" style="--d:${d}" x1="${xs[k - 1][Math.floor(i / fan)]}" y1="${py}" x2="${x}" y2="${y}"/>`;
        });
        if (lv.edge) {
          const px = xs[k - 1][xs[k - 1].length - 1], x = xs[k][xs[k].length - 1];
          svg += `<text class="edge-label ${EDGE_CLASS[lv.edge] || ""}" x="${(px + x) / 2 + 9}" y="${py + gap / 2 + 4}">${lv.edge}</text>`;
        }
        if (lv.max) {
          xs[k - 1].forEach((px, i) => {
            const kids = xs[k].slice(i * fan, (i + 1) * fan), t = 28 / gap;
            const xa = px + (kids[0] - px) * t, xb = px + (kids[kids.length - 1] - px) * t;
            svg += `<path class="arc" pathLength="1" style="--d:${d + 1}" d="M${xa - 3} ${py + 28} Q${px} ${py + 40} ${xb + 3} ${py + 28}"/>`;
          });
          const px = xs[k - 1][xs[k - 1].length - 1], kids = xs[k].slice(-fan);
          svg += `<text class="arc-label" x="${px + (kids[kids.length - 1] - px) * (28 / gap) + 10}" y="${py + 36}">max</text>`;
        }
        d += 1;
      }
      for (const x of xs[k]) {
        svg += lv.kind === "state"
          ? `<circle class="node state" style="--d:${d}" cx="${x}" cy="${y}" r="12"/>`
          : `<circle class="node action" style="--d:${d}" cx="${x}" cy="${y}" r="6.5"/>`;
      }
      svg += `<text class="node-label" x="${xs[k][xs[k].length - 1] + (lv.kind === "state" ? 20 : 14)}" y="${y + 5}">${lv.label}</text>`;
      d += 1;
    });
    host.innerHTML = `${svg}</svg>`;
    const drawing = host.firstElementChild;
    const io = new IntersectionObserver(([it]) => {
      if (it.isIntersecting) { drawing.classList.add("on"); io.disconnect(); }
    }, { threshold: 0.4 });
    io.observe(drawing);
  };
})(globalThis.RL = globalThis.RL || {});
