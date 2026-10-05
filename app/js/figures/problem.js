/* Figures of Part 1, the RL problem: the loop, the recycling robot and the gridworld's values. */
(function (RL) {
  "use strict";
  const { lab } = RL;

  // ---- drawings shared with the stories: the agent–environment loop and the recycling robot ----
  RL.demos.loop = (host) => { host.innerHTML = RL.scenes.loop.svg(true); };
  RL.demos["mdp-graph"] = (host, name = "robot") => { host.innerHTML = RL.scenes.mdp.svg(lab.mdp(name), true); };

  // ---- the gridworld of Sutton & Barto (Example 3.5): "random" draws Figure 3.2, "optimal" Figure 3.5 ----
  const ARROW = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  RL.demos.gridworld = function (host, arg = "random") {
    const env = lab.make("gridworld"), T = 34, GAP = 34, gamma = 0.9, optimal = arg === "optimal";
    const V = optimal ? lab.valueIteration(env, gamma) : lab.evaluate(env, lab.randomPolicy(env), gamma);
    const P = optimal ? lab.greedyPolicy(env, V, gamma) : null;
    const panels = optimal ? ["world", "values", "policy"] : ["world", "values"];
    const W = panels.length * 5 * T + (panels.length - 1) * GAP + 24, H = 5 * T + 52, id = RL.fig.uid();
    const ox = (i) => 12 + i * (5 * T + GAP), cell = (i, s) => [ox(i) + (s % 5) * T, 8 + Math.floor(s / 5) * T];
    const NAMES = { A: "A", B: "B", a: "A′", b: "B′" };
    let g = `<defs><marker id="gw-tip-${id}" class="tip-mark" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>`;
    panels.forEach((kind, i) => {
      for (let s = 0; s < 25; s++) {
        const [x, y] = cell(i, s), cx = x + T / 2, cy = y + T / 2;
        g += `<rect class="cell" x="${x}" y="${y}" width="${T}" height="${T}" fill="none"/>`;
        if (kind === "world" && NAMES[env.tile(s)]) g += `<text class="big" x="${cx}" y="${cy + 6}" text-anchor="middle">${NAMES[env.tile(s)]}</text>`;
        if (kind === "values") g += `<text class="num" x="${cx}" y="${cy + 5}" text-anchor="middle">${(V[s] < 0 ? "−" : "") + Math.abs(V[s]).toFixed(1)}</text>`;
        if (kind === "policy") {
          for (let a = 0; a < 4; a++) {
            if (!P[s * 4 + a]) continue;
            const [dx, dy] = ARROW[a];
            g += `<path class="pol-arrow" marker-end="url(#gw-tip-${id})" d="M${cx + dx * 3} ${cy + dy * 3} L${cx + dx * 12} ${cy + dy * 12}"/>`;
          }
        }
      }
      g += `<rect class="frame" x="${ox(i)}" y="8" width="${5 * T}" height="${5 * T}"/>`;
      if (kind === "world") {
        for (const j of env.jumps) {
          const [x1, y1] = cell(i, j.from), [x2, y2] = cell(i, j.to), mx = x1 + T + 18, my = (y1 + y2) / 2 + T / 2;
          g += `<path class="back" marker-end="url(#gw-tip-${id})" d="M${x1 + T - 6} ${y1 + T / 2} Q${mx + 14} ${my} ${x2 + T - 6} ${y2 + T / 2}"/>`;
          g += `<text class="note" x="${x1 + 41}" y="${my + 4}" text-anchor="end">+${j.reward}</text>`; // just left of the arc's bulge
        }
      }
      const caption = { world: "the gridworld", values: optimal ? "v*" : "vπ, random policy", policy: "π*" }[kind];
      g += `<text class="note" x="${ox(i) + 2.5 * T}" y="${5 * T + 34}" text-anchor="middle">${caption}</text>`;
    });
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="The gridworld${optimal ? ", its optimal values and an optimal policy" : " and the values of the random policy"}">${g}</svg>`;
  };
})(globalThis.RL = globalThis.RL || {});
