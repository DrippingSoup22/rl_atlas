/* Figures of Part 13's search: how often Monte Carlo tree search, and flat Monte Carlo with the same number of random
   games, choose the one winning move of the tic-tac-toe position in the MCTS story, against the number of simulations.
   Computed live from lab/mcts.js when the figure scrolls into view. */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const BOARD = "O....X...", WIN = 2, SIMS = [10, 30, 100, 300, 1000, 3000], RUNS = 100;

  RL.demos["mcts-odds"] = function (host, arg) {
    const small = arg === "small";
    const p = RL.fig.plot(host, {
      label: "How often the search chooses the winning corner, against the number of simulations, for UCT and for flat Monte Carlo",
      h: small ? 230 : 280, right: 132,
      x: { min: 1, max: 3000, log: true, ticks: [10, 30, 100, 300, 1000, 3000], label: "Simulations (random games) per decision, log scale" },
      y: { min: 0, max: 1, percent: true, ticks: [0, 0.25, 0.5, 0.75, 1], label: "Chooses the winning corner" },
      curves: [{ id: "uct", name: "tree search (UCT)" }, { id: "flat", name: "flat Monte Carlo", dash: true }],
      at: (n) => `${n.toLocaleString("en")} simulations`,
    });
    const wins = { uct: SIMS.map(() => 0), flat: SIMS.map(() => 0) };
    RL.fig.whenVisible(host, () => {
      let done = 0;
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (done < RUNS && performance.now() - t0 < 30) {
          const seed = 1 + done, search = lab.mcts(BOARD, { c: 1.4, seed });
          let ran = 0;
          SIMS.forEach((n, i) => {
            search.run(n - ran); // one search per seed, read at each budget on its way up
            ran = n;
            if (search.best() === WIN) wins.uct[i]++;
            if (lab.flatMC(BOARD, Math.max(1, Math.round(n / 7)), { seed }).best === WIN) wins.flat[i]++;
          });
          done++;
        }
        p.draw({ uct: { xs: SIMS, ys: wins.uct.map((w) => w / done) }, flat: { xs: SIMS, ys: wins.flat.map((w) => w / done) } });
        p.status(done < RUNS ? `Searching, seed ${done} of ${RUNS}…` : `Each point: ${RUNS} searches from the story's position, each with its own seed. Flat Monte Carlo spreads the same number of random games evenly over the seven moves. UCT uses c = 1.4 and plays its most visited move.`);
        if (done < RUNS) setTimeout(more, 16);
      };
      more();
    });
  };
})(globalThis.RL = globalThis.RL || {});
