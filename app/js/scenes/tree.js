/* Story scene tree: Monte Carlo tree search on a tic-tac-toe position (lab/mcts.js). On the left, the position: each
   empty cell shows the share of the simulations that started with a move there (or, for flat Monte Carlo, that move's
   average result), and during a simulation the moves it played, those inside the tree solid and the random rollout
   faint. On the right, the tree three moves deep: the mover's moves as small boards, the replies as circles, the moves
   after them as ticks, each as large as its visit count and colored by its value for X (blue good for X, orange good
   for O). Config: board (nine cells, row by row), c, seed. Step keys: sims (simulations done), play (run this many more,
   one at a time, pace ms each), flat (n: flat Monte Carlo with n random games after each move, no tree), chart (the odds
   of choosing the winning move against the number of simulations), mark (a cell to outline). */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const { timers, formula, FORMULA } = RL.sceneKit;
  const W = 690, H = 372, BX = 18, BY = 52, CELL = 48, TX = 206, TW = W - TX - 64, L1 = 112, L2 = 228, L3 = 304;
  const pct = (v) => `${Math.round(100 * v)}%`;

  RL.scenes.tree = {
    create(card, cfg, states = []) {
      const board = cfg.board || "O....X...", c = cfg.c ?? 1.4, seed = cfg.seed ?? 1, mover = lab.ttt.toMove(board);
      const first = lab.ttt.moves(board), xs = first.map((_, i) => TX + TW * ((i + 0.5) / first.length)), span = TW / first.length;
      card.innerHTML = `<svg class="tree-scene" viewBox="0 0 ${W} ${H}" role="img" aria-label="A search tree growing over a tic-tac-toe position"></svg>
        ${FORMULA}<div class="scene-chart" hidden></div><div class="scene-foot"><span class="scene-note"></span></div>`;
      const svg = card.querySelector("svg"), note = card.querySelector(".scene-note"), chart = card.querySelector(".scene-chart");
      const showFormula = formula(card, cfg), { later, stop } = timers();
      let search = null, oddsDrawn = false;

      // A fresh search run to k simulations (searches are cheap: rebuilding one is simpler than rewinding it).
      const searchTo = (k) => { search = lab.mcts(board, { c, seed }); search.run(k); return search; };
      // Flat Monte Carlo's averages, worked out once for each number of games (thousands of games take a moment): those
      // the steps will show are played ahead, while the reader is not scrolling
      const flats = {}, flatOf = (n) => (flats[n] ||= lab.flatMC(board, n, { seed }));
      const ahead = [...new Set(states.map((st) => st.flat).filter(Boolean))];
      const warm = () => {
        if (!card.isConnected || !ahead.length) return;
        if (!RL.scrolling()) flatOf(ahead.shift());
        setTimeout(warm, RL.scrolling() ? 120 : 50);
      };
      setTimeout(warm, 300);
      const colorOf = (n) => (n && n.N ? `color-mix(in oklab, var(${n.X / n.N >= 0.5 ? "--v-pos" : "--v-neg"}) ${Math.round(Math.min(1, Math.abs(n.X / n.N - 0.5) * 2.4) * 100)}%, var(--v-mid))` : "var(--surface-2)");

      function boardSvg(st, sim) {
        const share = {}, flat = st.flat ? flatOf(st.flat) : null;
        if (!flat && search.sims) for (const k of search.root.kids) share[k.move] = k.N / search.sims;
        let g = `<text class="t-title" x="${BX}" y="22">${mover} to move</text>`;
        // the moves of the current simulation, in order: inside the tree, then the rollout
        const played = new Map();
        if (sim) {
          sim.path.slice(1).forEach((n, i) => played.set(n.move, { p: n.by, k: i + 1, tree: true }));
          let b = sim.path[sim.path.length - 1].b;
          sim.rollout.forEach((cell, i) => { const p = lab.ttt.toMove(b); b = lab.ttt.play(b, cell, p); played.set(cell, { p, k: sim.path.length + i, tree: false }); });
        }
        for (let i = 0; i < 9; i++) {
          const x = BX + (i % 3) * CELL, y = BY + Math.floor(i / 3) * CELL, ch = board[i], pl = played.get(i);
          let fill = "var(--surface)";
          if (ch === "." && !sim) {
            const v = flat ? flat.value[i] : share[i];
            if (v !== undefined) fill = flat ? `color-mix(in oklab, var(${v >= 0.5 ? "--v-pos" : "--v-neg"}) ${Math.round(Math.min(1, Math.abs(v - 0.5) * 3) * 100)}%, var(--v-mid))` : `color-mix(in oklab, var(--pol) ${Math.round(Math.min(1, v * 1.6) * 70)}%, var(--surface))`;
          }
          g += `<rect class="cell${st.mark === i ? " marked" : ""}" x="${x}" y="${y}" width="${CELL}" height="${CELL}" style="fill:${fill}"/>`;
          if (ch !== ".") g += `<text class="piece" x="${x + CELL / 2}" y="${y + CELL / 2 + 9}">${ch}</text>`;
          else if (pl) g += `<text class="piece${pl.tree ? "" : " faint"}" x="${x + CELL / 2}" y="${y + CELL / 2 + 9}">${pl.p}</text><text class="order" x="${x + 5}" y="${y + 13}">${pl.k}</text>`;
          else if (!sim && (flat || share[i] !== undefined)) g += `<text class="share" x="${x + CELL / 2}" y="${y + CELL / 2 + 5}">${pct(flat ? flat.value[i] : share[i])}</text>`;
        }
        const yb = BY + 3 * CELL;
        if (sim) {
          const w = sim.winner;
          g += `<text class="t-note" x="${BX}" y="${yb + 24}"><tspan class="strong">${w === "draw" ? "A draw" : `${w} wins`}</tspan> (½ counts as a draw)</text>
            <text class="t-note" x="${BX}" y="${yb + 42}">solid: moves inside the tree</text><text class="t-note" x="${BX}" y="${yb + 58}">faint: the random rollout</text>`;
        } else if (flat) {
          g += `<text class="t-note" x="${BX}" y="${yb + 24}">each cell: ${mover}'s average result</text><text class="t-note" x="${BX}" y="${yb + 40}">over ${flat ? st.flat.toLocaleString("en") : ""} random games</text><text class="t-note" x="${BX}" y="${yb + 56}">after a move there</text>`;
        } else if (search.sims) {
          g += `<text class="t-note" x="${BX}" y="${yb + 24}">each cell: the share of</text><text class="t-note" x="${BX}" y="${yb + 40}">simulations that began there</text>`;
        }
        return g;
      }

      // The tree, three levels deep. Every child of a node has a fixed slot (its cell), tried or not, so the picture
      // keeps its place as the tree grows.
      function treeSvg(st, sim) {
        if (st.flat) return flatSvg(st);
        const root = search.root, on = new Set(sim ? sim.path : []), top = Math.max(1, ...root.kids.map((k) => k.N));
        const rx = TX + TW / 2;
        let g = `<text class="t-title" x="${TX}" y="22">the tree after ${search.sims.toLocaleString("en")} simulation${search.sims === 1 ? "" : "s"}</text>`;
        g += `<circle class="t-root" cx="${rx}" cy="44" r="7"/><text class="t-note" x="${rx + 12}" y="48">now</text>`;
        first.forEach((m, i) => {
          const k = root.kids.find((n) => n.move === m), x = xs[i];
          const w = k ? 1 + 7 * Math.sqrt(k.N / top) : 0.8;
          g += `<path class="t-edge${on.has(k) ? " on" : ""}${k ? "" : " untried"}" d="M${rx} 51 C${rx} 80 ${x} 70 ${x} ${L1 - 26}" style="stroke-width:${w.toFixed(1)}"/>`;
          // a small board with this move in it
          const s = 11, bx = x - 1.5 * s, by = L1 - 24;
          g += `<g class="mini${on.has(k) ? " on" : ""}${st.mark === m ? " marked" : ""}">`;
          for (let j = 0; j < 9; j++) {
            const cx = bx + (j % 3) * s, cy = by + Math.floor(j / 3) * s, ch = j === m ? mover : board[j];
            g += `<rect x="${cx}" y="${cy}" width="${s}" height="${s}" style="fill:${j === m ? colorOf(k) : "var(--surface)"}"/>${ch !== "." ? `<text x="${cx + s / 2}" y="${cy + s - 2}" class="${j === m ? "new" : ""}">${ch}</text>` : ""}`;
          }
          g += `</g><text class="t-n" x="${x}" y="${L1 + 25}">${k ? k.N.toLocaleString("en") : "–"}</text><text class="t-v" x="${x}" y="${L1 + 38}">${k ? pct(k.X / k.N) : ""}</text>`;
          // replies: circles in slots under the board
          const replies = lab.ttt.moves(lab.ttt.play(board, m, mover)), top2 = Math.max(1, ...(k ? k.kids.map((n) => n.N) : [1]));
          replies.forEach((r, j) => {
            const x2 = x - span * 0.42 + span * 0.84 * ((j + 0.5) / replies.length), k2 = k?.kids.find((n) => n.move === r);
            if (k2) g += `<line class="t-edge thin${on.has(k2) ? " on" : ""}" x1="${x}" y1="${L1 + 43}" x2="${x2}" y2="${L2 - 6}"/>`;
            const r2 = k2 ? 1.6 + 4.2 * Math.sqrt(k2.N / top2) : 1.2;
            g += `<circle class="t-node${on.has(k2) ? " on" : ""}${k2 ? "" : " untried"}" cx="${x2}" cy="${L2}" r="${r2.toFixed(1)}" style="fill:${colorOf(k2)}"><title>${k2 ? `${k2.N} visits, ${pct(k2.X / k2.N)} for X` : "not tried yet"}</title></circle>`;
            // the moves after the reply: ticks
            if (k2?.kids.length) {
              const top3 = Math.max(1, ...k2.kids.map((n) => n.N)), n3 = lab.ttt.moves(k2.b).length;
              for (const k3 of k2.kids) {
                const slot = lab.ttt.moves(k2.b).indexOf(k3.move), x3 = x2 - 4 + 8 * ((slot + 0.5) / n3), h = 3 + 22 * Math.sqrt(k3.N / top3);
                g += `<line class="t-tick${on.has(k3) ? " on" : ""}" x1="${x3.toFixed(1)}" x2="${x3.toFixed(1)}" y1="${L3 - 10}" y2="${(L3 - 10 + h).toFixed(1)}" style="stroke:${colorOf(k3)}"/>`;
              }
            }
          });
        });
        g += `<text class="t-level" x="${W - 6}" y="${L1 - 6}">${mover}'s move</text><text class="t-level" x="${W - 6}" y="${L2 + 4}">${mover === "X" ? "O" : "X"}'s reply</text><text class="t-level" x="${W - 6}" y="${L3 + 2}">${mover}'s next</text>`;
        let deepest = 0;
        (function walk(n) { deepest = Math.max(deepest, n.depth); n.kids.forEach(walk); })(root);
        if (deepest > 3) g += `<text class="t-note" x="${TX + TW / 2}" y="${H - 10}" text-anchor="middle">deeper levels not drawn: the tree reaches ${deepest} moves ahead</text>`;
        return g;
      }

      // Flat Monte Carlo keeps no tree: instead, each first move's average over the random games (a bar), against what
      // the move is worth when both sides play their best (a line: a win, a draw or a loss).
      function flatSvg(st) {
        const flat = flatOf(st.flat), Y0 = 300, HB = 150, y = (v) => Y0 - v * HB;
        let g = `<text class="t-title" x="${TX}" y="22">each move: random games against best play</text>`;
        for (const v of [0, 0.5, 1]) g += `<line class="t-grid" x1="${TX}" x2="${TX + TW}" y1="${y(v)}" y2="${y(v)}"/><text class="t-level" x="${W - 6}" y="${y(v) + 4}">${v === 1 ? "X wins" : v === 0.5 ? "draw" : "X loses"}</text>`;
        first.forEach((m, i) => {
          const x = xs[i], s = 11, bx = x - 1.5 * s, by = 70, v = flat.value[m], best = lab.ttt.perfect(lab.ttt.play(board, m, mover));
          g += `<g class="mini${st.mark === m ? " marked" : ""}">`;
          for (let j = 0; j < 9; j++) {
            const cx = bx + (j % 3) * s, cy = by + Math.floor(j / 3) * s, ch = j === m ? mover : board[j];
            g += `<rect x="${cx}" y="${cy}" width="${s}" height="${s}" style="fill:${j === m ? "var(--surface-3)" : "var(--surface)"}"/>${ch !== "." ? `<text x="${cx + s / 2}" y="${cy + s - 2}" class="${j === m ? "new" : ""}">${ch}</text>` : ""}`;
          }
          g += `</g><rect class="t-bar${m === flat.best ? " top" : ""}" x="${x - 12}" y="${y(v)}" width="24" height="${Y0 - y(v)}"/><text class="t-n" x="${x}" y="${y(v) - 6}">${pct(v)}</text>
            <line class="t-best" x1="${x - 17}" x2="${x + 17}" y1="${y(best)}" y2="${y(best)}"/>`;
        });
        g += `<rect class="t-bar" x="${TX}" y="${H - 40}" width="12" height="10"/><text class="t-note" x="${TX + 18}" y="${H - 31}">average over ${st.flat.toLocaleString("en")} random games after the move</text>
          <line class="t-best" x1="${TX}" x2="${TX + 12}" y1="${H - 17}" y2="${H - 17}"/><text class="t-note" x="${TX + 18}" y="${H - 13}">what the move is worth if both sides play their best</text>`;
        return g;
      }

      function draw(st, sim = null) { svg.innerHTML = boardSvg(st, sim) + treeSvg(st, sim); }

      // The odds figure, drawn once, the first time a step asks for it.
      function odds() {
        if (oddsDrawn) return;
        oddsDrawn = true;
        RL.demos["mcts-odds"](chart, "small");
      }

      return {
        apply(st) {
          stop();
          const k = st.sims ?? 0;
          searchTo(k);
          draw(st);
          chart.hidden = !st.chart;
          if (st.chart) odds();
          note.textContent = st.note || (st.flat ? `Flat Monte Carlo would play the cell with the best average` : k ? `${k.toLocaleString("en")} simulations · the search would now play the most visited move` : "");
          if (st.play) {
            let done = 0;
            const tick = () => {
              if (done >= st.play) { draw(st); note.textContent = `${search.sims.toLocaleString("en")} simulations played`; return; }
              const sim = search.step();
              done++;
              draw(st, sim);
              note.textContent = `Simulation ${search.sims}: ${sim.added ? `adds one node, ${sim.path.length - 1} move${sim.path.length > 2 ? "s" : ""} deep` : "reaches a finished game inside the tree"}, then ${sim.rollout.length ? `${sim.rollout.length} random move${sim.rollout.length > 1 ? "s" : ""}` : "no rollout"}: ${sim.winner === "draw" ? "a draw" : `${sim.winner} wins`}`;
              later(tick, st.pace ?? 1100);
            };
            later(tick, 500);
          }
          showFormula(st);
        },
        destroy() { stop(); },
      };
    },
  };
})(globalThis.RL = globalThis.RL || {});
