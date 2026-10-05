/* Figures of Part 5, temporal-difference learning: the cliff, the random walk, Expected SARSA and maximization bias. */
(function (RL) {
  "use strict";
  const { lab } = RL;

  // ---- the cliff world (Sutton & Barto, Example 6.6) ----
  const CLIFF = { alpha: 0.5, epsilon: 0.1, gamma: 1 };
  const cliffRun = (id, seed) => lab.simulate({ world: "cliff", algorithm: lab.algorithms[id], params: CLIFF, units: 500, seed });
  const ROUTES = [
    { id: "sarsa", label: "SARSA: the safe path", dashed: true, dx: -3 },
    { id: "q-learning", label: "Q-learning: the optimal path", dashed: false, dx: 3 },
  ];

  // The grid, the cliff, and the path each algorithm's greedy policy takes after 500 episodes.
  RL.demos["cliff-paths"] = function (host) {
    const env = lab.make("cliff"), T = 34, X = 12, Y = 10, W = env.cols * T + 2 * X, H = env.rows * T + Y + 50;
    const id = RL.fig.uid();
    host.innerHTML = `<svg class="fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="The cliff world and the paths SARSA and Q-learning learn"></svg>`;
    const svg = host.firstElementChild;
    RL.fig.whenVisible(host, () => {
      const cx = (s) => X + (s % env.cols) * T + T / 2, cy = (s) => Y + Math.floor(s / env.cols) * T + T / 2;
      const bottom = Y + env.rows * T, S = env.start, G = env.nS - 1;
      let g = `<defs>
          <pattern id="hatch-${id}" class="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7"/></pattern>
          <marker id="tip-${id}" class="tip-mark" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker>
        </defs>
        <rect class="cliff" x="${X + T}" y="${Y + 3 * T}" width="${10 * T}" height="${T}" fill="url(#hatch-${id})"/>`;
      for (let r = 0; r <= env.rows; r++) g += `<line class="cell" x1="${X}" x2="${X + env.cols * T}" y1="${Y + r * T}" y2="${Y + r * T}"/>`;
      for (let c = 0; c <= env.cols; c++) g += `<line class="cell" x1="${X + c * T}" x2="${X + c * T}" y1="${Y}" y2="${bottom}"/>`;
      g += `<rect class="frame" x="${X}" y="${Y}" width="${env.cols * T}" height="${env.rows * T}"/>
        <text class="halo cliff-name" x="${X + 6 * T}" y="${cy(S) + 5}" text-anchor="middle">The Cliff</text>
        <text class="big" x="${cx(S)}" y="${cy(S) + 6}" text-anchor="middle">S</text>
        <text class="big" x="${cx(G)}" y="${cy(G) + 6}" text-anchor="middle">G</text>
        <path class="back" marker-end="url(#tip-${id})" d="M${X + 6 * T} ${bottom + 3} C${X + 6 * T} ${bottom + 34} ${cx(S)} ${bottom + 34} ${cx(S)} ${bottom + 5}"/>
        <text class="note" x="${X + 3.5 * T}" y="${bottom + 44}" text-anchor="middle">R = −100, and back to the start</text>
        <clipPath id="wipe-${id}"><rect class="wipe" x="0" y="0" width="${W}" height="${H}"/></clipPath>`;

      let routes = "";
      for (const route of ROUTES) {
        const { path } = lab.greedyPath(env, cliffRun(route.id, 7).at(500).Q);
        const pts = path.map((s) => [cx(s) + route.dx, cy(s)]);
        pts[0][1] = Y + 3 * T; // leave S and G readable: start on the edge of S, stop just above G
        pts[pts.length - 1][1] = Y + 3 * T - 4;
        routes += `<path class="route${route.dashed ? " dashed" : ""}" marker-end="url(#tip-${id})" d="M${pts.map((p) => p.join(" ")).join(" L")}"/>`;
        // The label goes under the longest straight stretch of the path.
        let best = [0, 0];
        for (let i = 0, j = 0; i < pts.length; i = j) {
          for (j = i + 1; j < pts.length && pts[j][1] === pts[i][1]; j++);
          if (j - i > best[1] - best[0]) best = [i, j];
        }
        const [a, b] = best, lx = (pts[a][0] + pts[b - 1][0]) / 2;
        routes += `<text class="halo route-name" x="${lx}" y="${pts[a][1] + 15}" text-anchor="middle">${route.label} (${path.length - 1} steps)</text>`;
      }
      svg.innerHTML = `${g}<g clip-path="url(#wipe-${id})">${routes}</g>`;
      requestAnimationFrame(() => svg.classList.add("on"));
    });
  };

  // Reward per episode while learning, averaged over many runs: the curves smooth out as runs come in.
  RL.demos["cliff-curves"] = function (host) {
    RL.fig.average(host, {
      label: "Reward per episode while learning, SARSA and Q-learning",
      x: { max: 500, ticks: [0, 100, 200, 300, 400, 500], label: "Episodes" },
      y: { min: -100, max: 0, ticks: [-100, -75, -50, -25, 0], label: "Sum of rewards during episode", digits: 1 },
      curves: ROUTES.map((r) => ({ id: r.id, name: r.id === "sarsa" ? "SARSA" : "Q-learning", dash: r.dashed })),
      at: (e) => `Episode ${e}`,
    }, {
      runs: 100, smooth: 10,
      sample: (seed) => Object.fromEntries(ROUTES.map((r) => [r.id, cliffRun(r.id, seed).metrics.return])),
    });
  };
})(globalThis.RL = globalThis.RL || {});
