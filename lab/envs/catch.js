/* Catch, from DeepMind's bsuite (Osband et al., 2020): a ball falls down a board 10 rows high and 5 columns wide, one
   row per step, from a random column of the top row; a paddle on the bottom row moves left, stays or moves right.
   When the ball reaches the bottom row the episode ends: +1 if the paddle is under it, −1 if not. A state is the ball's
   row and column and the paddle's column: 10 × 5 × 5 = 250, few enough for a table (the deep agents of bsuite see the
   same board as 50 pixels). */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});
  const ROWS = 10, COLS = 5;
  const encode = (by, bx, px) => (by * COLS + bx) * COLS + px;
  const decode = (s) => ({ px: s % COLS, bx: Math.floor(s / COLS) % COLS, by: Math.floor(s / (COLS * COLS)) });

  function act(s, a) {
    const { by, bx, px } = decode(s), p2 = Math.max(0, Math.min(COLS - 1, px + a - 1)), b2 = by + 1;
    return { s2: encode(b2, bx, p2), r: b2 === ROWS - 1 ? (p2 === bx ? 1 : -1) : 0 };
  }

  lab.catchGame = function () {
    const nS = ROWS * COLS * COLS, all = [0, 1, 2];
    const starts = Array.from({ length: COLS }, (_, bx) => encode(0, bx, COLS >> 1));
    const env = {
      name: "catch", key: "catch", kind: "catch", title: "Catch", nS, nA: 3, actionNames: ["left", "stay", "right"],
      rows: ROWS, cols: COLS, encode, decode, starts, valueRange: 1, maxSteps: ROWS,
      terminal: (s) => decode(s).by === ROWS - 1,
      acts: () => all,
      reset: (rng, s0) => (s0 === undefined ? starts[Math.floor(rng.next() * starts.length)] : s0),
      step: (s, a) => act(s, a),
      model: (s, a) => { const { s2, r } = act(s, a); return [{ p: 1, s2, r }]; },
      caught: (s) => { const d = decode(s); return d.by === ROWS - 1 && d.px === d.bx; },
      describe(s, a) {
        const d = decode(s);
        return `ball at row ${d.by + 1}, column ${d.bx + 1}; paddle at column ${d.px + 1}${a >= 0 ? `; ${env.actionNames[a]}` : ""}`;
      },
    };
    return env;
  };
  lab.worlds.catch = () => lab.catchGame();
})(globalThis.RL = globalThis.RL || {});
