/* Grid worlds. A world is a few lines of text: S start, G goal (a gem), g a small gem (it pays less, and also ends
   the episode), T exit (an ending without a prize), . free, # wall, C cliff (a fall: back to the start), H hole (the
   episode ends), k a checkpoint (a bonus, reward.checkpoint, every time it is entered), and letters for tiles that jump
   elsewhere (A jumps to a, B to b). On slippery ice a move can slide to either side. A world with `shaping` adds
   γΦ(s′) − Φ(s) to every move, where Φ is shaping.potential on the tiles named there and 0 elsewhere. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  lab.ACTIONS = ["up", "right", "down", "left"];
  const MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];

  const WORLDS = {
    cliff: {
      title: "Cliff walking",
      map: ["............", "............", "............", "SCCCCCCCCCCG"],
      reward: { step: -1, cliff: -100 },
      valueRange: 20, // values beyond ±20 (like stepping into the cliff) show at full color
      // A careful walker, for logs (offline RL): up the first column, along the top row, down the last column.
      policies: { careful: (r, c, rows, cols) => (c === cols - 1 ? 2 : r > 0 ? 0 : 1) },
    },
    // Sutton & Barto, Example 3.5: every move from A lands on A′ with +10, every move from B on B′ with +5,
    // a move into the edge costs 1 and leaves the agent in place, every other move is free. It never ends.
    gridworld: {
      title: "Gridworld",
      map: [".A.B.", ".....", "...b.", ".....", ".a..."],
      reward: { step: 0, wall: -1 },
      jumps: { A: ["a", 10], B: ["b", 5] },
      valueRange: 10,
    },
    // Sutton & Barto, Example 4.1: every move costs 1 until one of the two shaded corners is reached.
    "small-gridworld": {
      title: "4 × 4 gridworld",
      map: ["T...", "....", "....", "...T"],
      reward: { step: -1 },
      valueRange: 22,
    },
    // The classic Frozen Lake: reach the gem without falling through the ice. Each move goes the intended way
    // only one time in three; otherwise it slides to the left or the right of it. Only the gem pays.
    "frozen-lake": {
      title: "Frozen Lake",
      map: ["S...", ".H.H", "...H", "H..G"],
      reward: { step: 0, goal: 1 },
      slip: 2 / 3,
      ice: true,
      valueRange: 1,
    },
    // Gymnasium's larger Frozen Lake, the same slippery ice on an 8 × 8 lake with ten holes. Gems are rare: a random
    // walker reaches this one about once in five hundred episodes.
    "frozen-lake-8": {
      title: "Frozen Lake 8 × 8",
      map: ["S.......", "........", "...H....", ".....H..", "...H....", ".HH...H.", ".H..H.H.", "...H...G"],
      reward: { step: 0, goal: 1 },
      slip: 2 / 3,
      ice: true,
      valueRange: 1,
    },
    // Sutton & Barto, Example 6.5: a wind blows up through the middle columns, pushing every move up by the strength of
    // the column it starts from (0 to 2 tiles). Each step costs 1 until the goal; the walls of the grid stop the agent,
    // and the wind cannot push it past the top row.
    windy: {
      title: "Windy gridworld",
      map: ["..........", "..........", "..........", "S......G..", "..........", "..........", ".........."],
      wind: [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
      reward: { step: -1 },
      valueRange: 20,
    },
    // Sutton & Barto, Example 8.1: a small maze where only the gem pays (+1), discounted by γ = 0.95.
    "dyna-maze": {
      title: "Dyna maze",
      map: [".......#G", "..#....#.", "S.#....#.", "..#......", ".....#...", "........."],
      reward: { step: 0, goal: 1 },
      valueRange: 1,
    },
    // Example 8.2: after 1000 steps the gap in the wall moves from the right end to the left end.
    "blocking-maze": {
      title: "Blocking maze",
      map: ["........G", ".........", ".........", "########.", ".........", "...S....."],
      change: { at: 1000, map: ["........G", ".........", ".........", ".########", ".........", "...S....."] },
      reward: { step: 0, goal: 1 },
      valueRange: 1,
    },
    // Example 8.3: after 3000 steps a shortcut opens at the right end of the wall; the long way round stays open.
    "shortcut-maze": {
      title: "Shortcut maze",
      map: ["........G", ".........", ".........", ".########", ".........", "...S....."],
      change: { at: 3000, map: ["........G", ".........", ".........", ".#######.", ".........", "...S....."] },
      reward: { step: 0, goal: 1 },
      valueRange: 1,
    },
    // A small gem two steps from the start pays 0.3; the big one, seven steps away, pays 1. With γ = 0.95 the big one is
    // worth more than twice as much from the start, but a learner that stops exploring early never finds out.
    "two-gems": {
      title: "Two gems",
      map: ["........", ".g......", "..S.....", "........", ".......G"],
      reward: { step: 0, goal: 1, small: 0.3 },
      valueRange: 1,
    },
    // Cliffs scattered between the start and the goal, for planning in a learned model: the way round by the top row
    // is safe and takes 14 steps; the best way through takes 12, past four cliffs a model cannot know without trying.
    "hidden-cliffs": {
      title: "Hidden cliffs",
      map: ["...........", "..C....C...", "S...C.C...G", "..C.....C..", "....C.C....", "..........."],
      reward: { step: -1, cliff: -100 },
      valueRange: 20,
      // the safe way, shown to a learner: up to the top row, along it, and down the last column to the goal
      policies: { careful: (r, c, rows, cols) => (c === cols - 1 ? (r < 2 ? 2 : 0) : r > 0 ? 0 : 1) },
    },
    // A designer's bonus, for reward design: the gem pays 1, and a checkpoint on the way pays 0.2 every time it is
    // entered. Stepping off and back on pays it every second step: with γ = 0.95 that loop is worth about 2.05, more
    // than the gem.
    checkpoint: {
      title: "Checkpoint",
      map: ["S....#....", ".....#....", "..k..#....", ".....#....", "..........", ".........G"],
      reward: { step: 0, goal: 1, checkpoint: 0.2 },
      valueRange: 2,
    },
    // The same bonus as a potential (Ng, Harada & Russell, 1999): entering the checkpoint pays γ·0.2, leaving it costs
    // 0.2, so a loop through it loses a little and the gem stays the goal.
    "checkpoint-shaped": {
      title: "Checkpoint, shaped",
      map: ["S....#....", ".....#....", "..k..#....", ".....#....", "..........", ".........G"],
      reward: { step: 0, goal: 1 },
      shaping: { gamma: 0.95, potential: { k: 0.2 } },
      valueRange: 2,
    },
    // A bridge of ice between holes, for imitation: the way across is three tiles wide, and one move in ten slides
    // to a side. An expert recovers from a slide; a learner that never saw one has to guess.
    "ice-bridge": {
      title: "Ice bridge",
      map: ["HHHHHHHHH", "H.......H", "S.......G", "H.......H", "HHHHHHHHH"],
      reward: { step: 0, goal: 1 },
      slip: 0.1,
      ice: true,
      valueRange: 1,
    },
  };

  // name: one of the worlds above, or a world spec of your own ({ title, map, reward, slip, … }).
  lab.grid = function (name) {
    const w = typeof name === "string" ? WORLDS[name] : name;
    if (!w) throw new Error(`unknown world '${name}'`);
    const rows = w.map.length, cols = w.map[0].length, first = w.map.join(""), later = w.change ? w.change.map.join("") : first;
    let cells = first;
    const reward = w.reward, slip = w.slip || 0;
    const wall = reward.wall ?? reward.step;
    const start = Math.max(0, cells.indexOf("S"));
    const jumps = Object.entries(w.jumps || {}).map(([from, [to, r]]) => ({ from: cells.indexOf(from), to: cells.indexOf(to), reward: r }));
    const isTerminal = (s) => cells[s] === "G" || cells[s] === "g" || cells[s] === "T" || cells[s] === "H";
    const all = [0, 1, 2, 3];

    // Where a move in direction d from s ends, and what it pays. In the wind, the move is shifted up by the wind of the
    // column it starts from, and the edges of the grid clamp it.
    const wind = w.wind || null;
    function land(s, d) {
      let r = Math.floor(s / cols) + MOVES[d][0], c = (s % cols) + MOVES[d][1];
      if (wind) { r = Math.min(rows - 1, Math.max(0, r - wind[s % cols])); c = Math.min(cols - 1, Math.max(0, c)); }
      if (r < 0 || c < 0 || r >= rows || c >= cols) return { s2: s, r: wall };
      const s2 = r * cols + c, k = cells[s2];
      if (k === "#") return { s2: s, r: wall };
      if (k === "C") return { s2: start, r: reward.cliff, fell: s2 };
      if (k === "G" || k === "T") return { s2, r: reward.goal ?? reward.step };
      if (k === "g") return { s2, r: reward.small ?? reward.goal ?? reward.step };
      if (k === "H") return { s2, r: reward.hole ?? reward.step };
      if (k === "k") return { s2, r: reward.step + (reward.checkpoint ?? 0) };
      return { s2, r: reward.step };
    }
    // Potential-based shaping: γΦ(s′) − Φ(s) on every move, with Φ read off the tiles (0 at the end of an episode).
    const phi = (s) => (isTerminal(s) ? 0 : w.shaping?.potential?.[cells[s]] ?? 0);
    const shaped = (s, o) => (w.shaping ? { ...o, r: o.r + w.shaping.gamma * phi(o.s2) - phi(s) } : o);
    // On ice the intended direction holds with probability 1 − slip; the rest is split between its two sides.
    const sideways = (d) => [(d + 3) % 4, (d + 1) % 4];

    const env = {
      name: typeof name === "string" ? name : "custom", key: typeof name === "string" ? name : JSON.stringify(w),
      kind: "grid", title: w.title, rows, cols, nS: rows * cols, nA: 4,
      start, valueRange: w.valueRange ?? 10, jumps, slip, ice: !!w.ice, map: w.map, rewards: reward, wind,
      // Tiles an episode may start from (for exploring starts): anything that is not a wall, a cliff or an ending.
      starts: Array.from({ length: rows * cols }, (_, s) => s).filter((s) => !"#CGgTH".includes(cells[s])),
      tile: (s) => cells[s],
      // A named policy of the world's own (w.policies: (row, column, rows, columns) → action), as a table.
      policy(name) {
        const f = w.policies?.[name];
        if (!f) throw new Error(`world '${env.name}' has no policy '${name}'`);
        const P = new Float64Array(rows * cols * 4);
        for (let s = 0; s < rows * cols; s++) P[s * 4 + f(Math.floor(s / cols), s % cols, rows, cols)] = 1;
        return P;
      },
      // A world that changes (a wall that moves) is told the time, in steps; it returns true when its layout changed.
      // `version` counts the changes, so a view knows when to draw the walls again.
      changes: w.change ? w.change.at : null, version: 0,
      setTime(t) {
        const now = t >= (w.change?.at ?? Infinity) ? later : first;
        if (now === cells) return false;
        cells = now;
        env.version++;
        env.map = (now === first ? w.map : w.change.map);
        return true;
      },
      terminal: isTerminal,
      blocked: (s) => cells[s] === "#" || cells[s] === "C",
      acts: () => all,
      rc: (s) => [Math.floor(s / cols), s % cols],
      describe: (s, a) => `row ${Math.floor(s / cols) + 1}, column ${(s % cols) + 1}${a >= 0 ? `, moving ${lab.ACTIONS[a]}` : ""}`,
      reset: (rng, s0) => (s0 === undefined ? start : s0),
      // Moving off the grid or into a wall leaves the agent in place. Stepping into the cliff costs dearly and
      // sends it back to the start; `fell` says which cliff tile it stepped into.
      step(s, a, rng) {
        const jump = jumps.find((j) => j.from === s);
        if (jump) return { s2: jump.to, r: jump.reward };
        if (!slip) return shaped(s, land(s, a));
        const u = rng.next(), [left, right] = sideways(a);
        const d = u < 1 - slip ? a : u < 1 - slip / 2 ? left : right;
        return { ...shaped(s, land(s, d)), slid: d !== a ? d : undefined };
      },
      // The same moves as a model, for dynamic programming: every outcome with its probability.
      model(s, a) {
        const jump = jumps.find((j) => j.from === s);
        if (jump) return [{ p: 1, s2: jump.to, r: jump.reward }];
        if (!slip) { const { s2, r } = shaped(s, land(s, a)); return [{ p: 1, s2, r }]; }
        const out = [];
        const [left, right] = sideways(a);
        for (const [d, p] of [[a, 1 - slip], [left, slip / 2], [right, slip / 2]]) {
          const { s2, r } = shaped(s, land(s, d)), same = out.find((o) => o.s2 === s2 && o.r === r);
          if (same) same.p += p;
          else out.push({ p, s2, r });
        }
        return out;
      },
    };
    return env;
  };

  for (const name of Object.keys(WORLDS)) lab.worlds[name] = () => lab.grid(name);

  // The sandbox: a world you draw in the Lab. lab.sandbox holds the current drawing.
  lab.SANDBOX = {
    title: "Your world",
    map: ["S...#....", ".##.#.##.", ".#..#..#.", ".#.###.#.", ".#.....#G", "...CCC..."],
    reward: { step: -1, cliff: -100, hole: -100, goal: 0 },
    slip: 0,
  };
  lab.sandbox = lab.SANDBOX;
  lab.worlds.sandbox = () => lab.grid(lab.sandbox);
})(globalThis.RL = globalThis.RL || {});
