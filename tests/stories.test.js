// The runs the stories play are pinned: a seed and settings chosen offline (tools/story-runs.js) because they show the
// idea well. Code changes that alter a pinned run (one random draw more or less is enough) must not slip through
// unnoticed, so each run's fingerprint is kept here. When this test fails: check with tools/story-runs.js that the new
// run still shows what its story says (re-pin it if not, and update the story's numbers), then refresh the
// fingerprints with  UPDATE=1 node --test tests/stories.test.js
// Run from rl_atlas/ after python build.py.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

require("../lab/node.js");
require("../app/content.js");
const { lab, content } = globalThis.RL;
const STORE = path.join(__dirname, "story-runs.json");

// Every live run a story plays, as the story scenes make it (app/js/scenes/common.js: runs, and the grid scene's main run).
function storyRuns() {
  const out = {};
  for (const [id, entry] of Object.entries(content.entries)) {
    const cfg = entry.story?.config;
    if (!cfg) continue;
    for (const [name, spec] of Object.entries(cfg.runs || {})) {
      if (!spec.algorithm) continue;
      const { algorithm, units, seed, world, measures, name: _label, ...params } = spec;
      out[`${id}/${name}`] = { world: world || cfg.world || cfg.env, algorithm, params, units: units || cfg.units || 100, seed: seed ?? cfg.seed ?? 1, measures };
    }
    if (cfg.algorithm && cfg.episodes) out[`${id}/main`] = { world: cfg.env, algorithm: cfg.algorithm, params: cfg, units: cfg.episodes, seed: cfg.seed };
  }
  return out;
}

// A run's fingerprint: a hash of every measure it recorded, unit by unit (FNV-1a over values rounded to 9 digits).
function fingerprint(metrics) {
  let h = 0x811c9dc5;
  for (const key of Object.keys(metrics).sort()) {
    for (const ch of key) h = Math.imul(h ^ ch.charCodeAt(0), 0x01000193);
    for (const v of metrics[key]) for (const ch of Number.isFinite(v) ? v.toPrecision(9) : "x") h = Math.imul(h ^ ch.charCodeAt(0), 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

test("the runs the stories play are the ones pinned", () => {
  const runs = storyRuns(), now = {};
  for (const [key, r] of Object.entries(runs)) {
    const { metrics } = lab.simulate({ world: r.world, algorithm: lab.algorithms[r.algorithm], params: r.params, units: r.units, seed: r.seed, snapshots: false, measures: r.measures });
    now[key] = fingerprint(metrics);
  }
  if (process.env.UPDATE) { fs.writeFileSync(STORE, JSON.stringify(now, null, 1) + "\n"); return; }
  const pinned = JSON.parse(fs.readFileSync(STORE, "utf8"));
  const changed = Object.keys(now).filter((k) => pinned[k] !== now[k]), gone = Object.keys(pinned).filter((k) => !(k in now));
  assert.deepEqual([...changed, ...gone], [], "story runs that changed or are new (see the note at the top of this file)");
});

// The averages the story charts show are worked out offline (tools/story-curves.js) with the page's own runs and seeds.
// They must still be the engine's: each curve's first run is made again and compared with the fingerprint kept for it
// (the sum of its finite values, and how many are not finite). When this fails, or a chart has no averages kept (it
// would be averaged live, slowly), run  node tools/story-curves.js  and then python build.py.
test("the story charts' averages kept offline are the engine's, and no chart is left to average live", () => {
  const kept = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "content", "story-curves.json"), "utf8"));
  const seen = new Set(), missing = [], stale = [];
  for (const [id, entry] of Object.entries(content.entries)) {
    const cfg = entry.story?.config;
    if (!cfg?.runs) continue;
    for (const { state: st } of entry.story.steps) {
      if (!st.curves?.length || st.curves.every((n) => cfg.runs[n].recording)) continue;
      const key = lab.curveKey(cfg, st.curves, st.metric || "optimal");
      if (seen.has(key)) continue;
      seen.add(key);
      if (!kept[key]) { missing.push(`${id}: ${st.curves.join(", ")}`); continue; }
      for (const n of st.curves) {
        const v = lab.simulate(lab.curveRun(cfg, n, 0)).metrics[kept[key].metric];
        let sum = 0, odd = 0;
        for (const x of v) if (Number.isFinite(x)) sum += x; else odd++;
        const [s0, o0] = kept[key].check[n];
        if (odd !== o0 || Math.abs(sum - s0) > 1e-9 * Math.max(1, Math.abs(s0))) stale.push(`${id}/${n}`);
      }
    }
  }
  const unused = Object.keys(kept).filter((k) => !seen.has(k)).map((k) => kept[k].story);
  assert.deepEqual({ missing, stale, unused }, { missing: [], stale: [], unused: [] }, "run node tools/story-curves.js, then python build.py");
});
