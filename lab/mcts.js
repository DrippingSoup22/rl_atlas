/* Monte Carlo tree search on tic-tac-toe, for the MCTS story. A board is a string of nine cells, "X", "O" or ".",
   read row by row; X moves first. UCT grows a tree from the current position one simulation at a time: select by UCB
   down to a node with an untried move, expand it, finish the game at random (the rollout), and pass the result back up.
   Flat Monte Carlo, for comparison, plays random games after each move and keeps no tree. No DOM: the tests use it. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

  const ttt = (lab.ttt = {
    // "X" or "O" for a win, "draw" for a full board, null while the game goes on.
    winner(b) {
      for (const [a, c, d] of LINES) if (b[a] !== "." && b[a] === b[c] && b[a] === b[d]) return b[a];
      return b.includes(".") ? null : "draw";
    },
    toMove: (b) => ([...b].filter((x) => x === "X").length > [...b].filter((x) => x === "O").length ? "O" : "X"),
    moves: (b) => [...b].flatMap((x, i) => (x === "." ? [i] : [])),
    play: (b, i, p) => b.slice(0, i) + p + b.slice(i + 1),
    // The result for X: 1 a win, ½ a draw, 0 a loss.
    forX: (w) => (w === "X" ? 1 : w === "draw" ? 0.5 : 0),
    // The value of a position for X with perfect play on both sides (minimax), for the tests and the story's notes.
    perfect(b) {
      const memo = new Map();
      const go = (s) => {
        if (memo.has(s)) return memo.get(s);
        const w = ttt.winner(s);
        let v;
        if (w) v = ttt.forX(w);
        else {
          const p = ttt.toMove(s), vals = ttt.moves(s).map((i) => go(ttt.play(s, i, p)));
          v = p === "X" ? Math.max(...vals) : Math.min(...vals);
        }
        memo.set(s, v);
        return v;
      };
      return go(b);
    },
  });

  // A random game from b to its end: the cells played, in order, and the winner.
  function rollout(b, rng) {
    const cells = [];
    let w = ttt.winner(b);
    while (!w) {
      const ms = ttt.moves(b), i = ms[rng.int(ms.length)];
      b = ttt.play(b, i, ttt.toMove(b));
      cells.push(i);
      w = ttt.winner(b);
    }
    return { cells, winner: w };
  }

  // UCT from board b. Each node keeps its visits N and the sum X of the results for X, so X / N is its value for X;
  // a node's value for the player who moved into it is X / N for X and 1 − X / N for O.
  lab.mcts = function (board, { c = 1.4, seed = 1 } = {}) {
    const rng = lab.rng(seed);
    const make = (b, parent, move) => ({ b, parent, move, by: parent ? ttt.toMove(parent.b) : null, N: 0, X: 0, kids: [], untried: ttt.moves(b).filter(() => !ttt.winner(b)), end: ttt.winner(b), depth: parent ? parent.depth + 1 : 0 });
    const root = make(board, null, -1);
    const mine = (n) => (n.by === "X" ? n.X / n.N : 1 - n.X / n.N); // the mover's own view of a child
    const search = {
      root, sims: 0,
      // One simulation. Returns what it did, for the view: the path through the tree, the node it added (or null if the
      // path ended in a finished game), the random moves after it, and the winner.
      step() {
        let n = root;
        const path = [n];
        while (!n.end && !n.untried.length && n.kids.length) {
          const logN = Math.log(n.N);
          let best = null, top = -Infinity;
          for (const k of n.kids) {
            const u = mine(k) + c * Math.sqrt(logN / k.N);
            if (u > top) { top = u; best = k; }
          }
          n = best;
          path.push(n);
        }
        let added = null;
        if (!n.end && n.untried.length) {
          const i = n.untried.splice(rng.int(n.untried.length), 1)[0];
          added = make(ttt.play(n.b, i, ttt.toMove(n.b)), n, i);
          n.kids.push(added);
          n = added;
          path.push(n);
        }
        const r = n.end ? { cells: [], winner: n.end } : rollout(n.b, rng), x = ttt.forX(r.winner);
        for (const m of path) { m.N += 1; m.X += x; }
        search.sims += 1;
        return { path, added, rollout: r.cells, winner: r.winner };
      },
      run(k) { let last = null; for (let i = 0; i < k; i++) last = search.step(); return last; },
      // The move the search would play: the most visited child (the usual choice: visits are what UCB spent effort on).
      best: () => (root.kids.length ? root.kids.reduce((a, b) => (b.N > a.N ? b : a)).move : -1),
    };
    return search;
  };

  // Flat Monte Carlo from board b: n random games after each move in turn, no tree. Returns { move: value for the mover }
  // and the move with the best average.
  lab.flatMC = function (board, n, { seed = 1 } = {}) {
    const rng = lab.rng(seed), p = ttt.toMove(board), value = {};
    for (const i of ttt.moves(board)) {
      const b = ttt.play(board, i, p);
      let sum = 0;
      for (let k = 0; k < n; k++) { const x = ttt.forX(ttt.winner(b) || rollout(b, rng).winner); sum += p === "X" ? x : 1 - x; }
      value[i] = sum / n;
    }
    const best = +Object.keys(value).reduce((a, b) => (value[b] > value[a] ? b : a));
    return { value, best };
  };
})(globalThis.RL = globalThis.RL || {});
