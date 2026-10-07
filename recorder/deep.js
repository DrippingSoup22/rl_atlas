/* Record training runs for RL Atlas with the JavaScript trainer (lab/deep/), the one the page also trains with.

     node recorder/deep.js                  # every recording in the catalog
     node recorder/deep.js dqn-cartpole     # some of them
     node recorder/deep.js --sweep [names]  # the sweeps: each recording's knobs over several values, curves only

   Writes content/recordings/<name>.json and content/recordings/sweeps/<name>.json, as recorder/record.py did, or to
   $RECORDINGS_OUT to compare before replacing. Seeds run in parallel, one thread per core. Each finished run is kept
   in recorder/.cache, so an interrupted job resumes where it stopped (delete the folder after changing the trainer). */
"use strict";
const { Worker, isMainThread, parentPort, workerData } = require("worker_threads");
const fs = require("fs"), path = require("path"), os = require("os"), crypto = require("crypto");

if (!isMainThread) { // a worker: train the runs it is sent
  const deep = require("../lab/deep/node.js");
  parentPort.on("message", ({ id, spec, seed, snapshots }) => parentPort.postMessage({ id, result: deep.run(spec, seed, { snapshots }) }));
  return;
}

const ROOT = path.resolve(__dirname, ".."), OUT = process.env.RECORDINGS_OUT || path.join(ROOT, "content", "recordings");
const CACHE = path.join(__dirname, ".cache");
const SEEDS = Array.from({ length: 20 }, (_, i) => i + 1);

// DQN on CartPole: tuned until most seeds balance for the full 500 steps. A squared loss, not Huber's: CartPole's values
// run up to 1/(1 − γ) = 100, and clipped errors learn them too slowly. Each comparison of Part 9 changes one thing.
const DQN_CARTPOLE = { lr: 5e-4, buffer: 10000, batch: 128, train_every: 4, target_every: 500, eps: [1, 0.05, 20000], huber: false };
const CARTPOLE = { world: "cartpole", learner: "dqn", steps: 200000, block: 5000, seeds: SEEDS };
const CATALOG = {
  "dqn-cartpole": { ...CARTPOLE, cfg: DQN_CARTPOLE, station: "dqn", title: "DQN" },
  "dqn-cartpole-no-replay": { ...CARTPOLE, cfg: { ...DQN_CARTPOLE, buffer: 128 }, station: "dqn", title: "DQN without replay" },
  "dqn-cartpole-no-target": { ...CARTPOLE, cfg: { ...DQN_CARTPOLE, target_every: 0 }, station: "dqn", title: "DQN without a target network" },
  "double-dqn-cartpole": { ...CARTPOLE, cfg: { ...DQN_CARTPOLE, double: true }, station: "dqn-extensions", title: "Double DQN" },
  "dueling-dqn-cartpole": { ...CARTPOLE, cfg: { ...DQN_CARTPOLE, dueling: true }, station: "dqn-extensions", title: "Dueling DQN" },
  "prioritized-dqn-cartpole": { ...CARTPOLE, cfg: { ...DQN_CARTPOLE, prioritized: true }, station: "dqn-extensions", title: "Prioritized replay" },
};
// The knobs each recording's sweep tries, its own value among them. target_every = 0: no target network; buffer = 128
// (the batch size): no replay, each batch is the latest experience.
const SWEEPS = {
  "dqn-cartpole": { lr: [1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3], target_every: [0, 100, 500, 2000], buffer: [128, 1000, 10000, 100000], huber: [0, 1] },
};
const DEFAULTS = { dqn: require("../lab/deep/node.js").DQN_DEFAULTS };

// ---- a pool of threads, with every finished run cached on disk ----
function pool(jobs, label) {
  fs.mkdirSync(CACHE, { recursive: true });
  const keyOf = (j) => crypto.createHash("sha1").update(JSON.stringify([j.spec.world, j.spec.learner, j.spec.steps, j.spec.block, j.spec.cfg, j.seed, j.snapshots])).digest("hex");
  const results = new Array(jobs.length), todo = [];
  jobs.forEach((j, i) => { const f = path.join(CACHE, `${keyOf(j)}.json`); if (fs.existsSync(f)) results[i] = JSON.parse(fs.readFileSync(f, "utf8")); else todo.push(i); });
  if (!todo.length) return Promise.resolve(results);
  const t0 = Date.now(), n = Math.min(os.cpus().length, todo.length);
  let done = 0;
  return new Promise((resolve, reject) => {
    for (let w = 0; w < n; w++) {
      const worker = new Worker(__filename);
      const next = () => { const i = todo.shift(); if (i === undefined) return worker.terminate(); worker.postMessage({ id: i, ...jobs[i] }); };
      worker.on("message", ({ id, result }) => {
        fs.writeFileSync(path.join(CACHE, `${keyOf(jobs[id])}.json`), JSON.stringify(result));
        results[id] = result; done++;
        const left = jobs.length - results.filter(Boolean).length;
        process.stdout.write(`\r${label}: ${done} run${done > 1 ? "s" : ""} trained, ${left} to go, ${Math.round((Date.now() - t0) / 1000)} s   `);
        if (!left) { process.stdout.write("\n"); resolve(results); }
        next();
      });
      worker.on("error", reject);
      next();
    }
  });
}

const write = (file, data) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, JSON.stringify(data)); return Math.round(fs.statSync(file).size / 1024); };

async function record(name) {
  const deep = require("../lab/deep/node.js"), spec = CATALOG[name], shown = spec.shown ?? spec.seeds[0];
  const results = await pool(spec.seeds.map((seed) => ({ spec, seed, snapshots: seed === shown })), name);
  const kb = write(path.join(OUT, `${name}.json`), deep.recording(name, spec, results));
  const ends = results.map((r) => r.test.at(-1));
  console.log(`${name}: ${kb} KB; final test returns ${ends.join(" ")}`);
}

// The recording's own value of a knob: its setting, or the learner's default (a switch is swept as 0 and 1).
function current(spec, knob) {
  const v = { ...DEFAULTS[spec.learner], ...spec.cfg }[knob];
  return typeof v === "boolean" ? +v : v;
}

async function sweep(name) {
  const spec = CATALOG[name], knobs = SWEEPS[name], seeds = spec.sweepSeeds || spec.seeds;
  const jobs = Object.entries(knobs).flatMap(([knob, values]) => values.flatMap((v) => seeds.map((seed) => ({ knob, v, seed }))));
  const results = await pool(jobs.map(({ knob, v, seed }) => ({ spec: { ...spec, cfg: { ...spec.cfg, [knob]: typeof DEFAULTS[spec.learner][knob] === "boolean" ? !!v : v } }, seed, snapshots: false })), `${name} sweep`);
  const at = (knob, v, seed) => results[jobs.findIndex((j) => j.knob === knob && j.v === v && j.seed === seed)];
  const out = {
    name, world: spec.world, steps: spec.steps, block: spec.block, seeds, config: spec.cfg, trainer: "lab/deep",
    knobs: Object.fromEntries(Object.entries(knobs).map(([knob, values]) => [knob, {
      values, current: current(spec, knob),
      train: values.map((v) => seeds.map((s) => at(knob, v, s).train)),
      test: values.map((v) => seeds.map((s) => at(knob, v, s).test)),
    }])),
  };
  console.log(`${name} sweep: ${write(path.join(OUT, "sweeps", `${name}.json`), out)} KB`);
}

(async () => {
  const args = process.argv.slice(2), isSweep = args.includes("--sweep"), names = args.filter((a) => !a.startsWith("--"));
  const known = isSweep ? SWEEPS : CATALOG, unknown = names.filter((n) => !(n in known));
  if (unknown.length) { console.error(`unknown ${isSweep ? "sweeps" : "recordings"}: ${unknown.join(", ")} (known: ${Object.keys(known).join(", ")})`); process.exit(1); }
  for (const name of names.length ? names : Object.keys(known)) await (isSweep ? sweep(name) : record(name));
})();
