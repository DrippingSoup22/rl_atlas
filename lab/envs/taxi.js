/* Taxi (Dietterich, 2000), as Gymnasium's Taxi-v3 plays it. A 5 × 5 city with a few walls and four stands, R, G, Y and B.
   A passenger waits at one stand and wants to go to another. The taxi moves south, north, east or west, picks up and
   drops off. Every action costs 1; dropping the passenger at the right stand pays +20 and ends the episode; picking up
   or dropping off where it makes no sense costs 10. A state is the taxi's tile, where the passenger is (a stand, or in
   the taxi) and the destination: 25 × 5 × 4 = 500 states, few enough for a table. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  // Gymnasium's map: ":" is open between two tiles, "|" a wall.
  const MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
  ];
  const STANDS = [[0, 0], [0, 4], [4, 0], [4, 3]]; // R, G, Y, B
  const STAND_NAMES = ["R", "G", "Y", "B"];
  const ACTIONS = ["south", "north", "east", "west", "pick up", "drop off"];
  const N = 5, IN_TAXI = 4;

  const encode = (row, col, pass, dest) => ((row * N + col) * 5 + pass) * 4 + dest;
  const decode = (s) => ({ dest: s % 4, pass: Math.floor(s / 4) % 5, col: Math.floor(s / 20) % N, row: Math.floor(s / 100) });
  const standAt = (row, col) => STANDS.findIndex(([r, c]) => r === row && c === col);
  // Can the taxi cross from (row, col) east, or west?
  const openEast = (row, col) => MAP[1 + row][2 * col + 2] === ":";
  const openWest = (row, col) => MAP[1 + row][2 * col] === ":";

  // One action from a state: the next state, the reward, and whether the passenger arrived.
  function act(s, a) {
    let { row, col, pass, dest } = decode(s), r = -1, done = false;
    if (a === 0) row = Math.min(row + 1, N - 1);
    else if (a === 1) row = Math.max(row - 1, 0);
    else if (a === 2 && openEast(row, col)) col = Math.min(col + 1, N - 1);
    else if (a === 3 && openWest(row, col)) col = Math.max(col - 1, 0);
    else if (a === 4) {
      if (pass < IN_TAXI && standAt(row, col) === pass) pass = IN_TAXI;
      else r = -10;
    } else if (a === 5) {
      const here = standAt(row, col);
      if (pass === IN_TAXI && here === dest) { pass = dest; done = true; r = 20; }
      else if (pass === IN_TAXI && here >= 0) pass = here; // set down at another stand: allowed, and pointless
      else r = -10;
    }
    return { s2: encode(row, col, pass, dest), r, done };
  }

  lab.taxi = function () {
    const nS = 500, nA = 6, all = [0, 1, 2, 3, 4, 5];
    // Episodes start anywhere, with the passenger at a stand and a different stand as the destination.
    const starts = [];
    for (let row = 0; row < N; row++) for (let col = 0; col < N; col++) for (let pass = 0; pass < 4; pass++) for (let dest = 0; dest < 4; dest++) if (pass !== dest) starts.push(encode(row, col, pass, dest));
    const done = new Uint8Array(nS); // states reached by a successful drop-off: the episode is over there
    for (let s = 0; s < nS; s++) { const d = decode(s); if (d.pass === d.dest && d.pass < 4 && STANDS[d.dest][0] === d.row && STANDS[d.dest][1] === d.col) done[s] = 1; }
    return {
      name: "taxi", key: "taxi", kind: "taxi", title: "Taxi", nS, nA, actionNames: ACTIONS, rows: N, cols: N,
      map: MAP, stands: STANDS, standNames: STAND_NAMES, IN_TAXI, encode, decode, starts, valueRange: 20, maxSteps: 200,
      terminal: (s) => done[s] === 1,
      acts: () => all,
      reset: (rng, s0) => (s0 === undefined ? starts[Math.floor(rng.next() * starts.length)] : s0),
      step: (s, a) => act(s, a),
      model: (s, a) => { const { s2, r } = act(s, a); return [{ p: 1, s2, r }]; },
      openEast, openWest,
      describe(s, a) {
        const d = decode(s), where = d.pass === IN_TAXI ? "in the taxi" : `at ${STAND_NAMES[d.pass]}`;
        return `taxi at row ${d.row + 1}, column ${d.col + 1}; passenger ${where}, going to ${STAND_NAMES[d.dest]}${a >= 0 ? `; ${ACTIONS[a]}` : ""}`;
      },
    };
  };
  lab.worlds.taxi = () => lab.taxi();
})(globalThis.RL = globalThis.RL || {});
