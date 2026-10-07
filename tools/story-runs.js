// The offline check of the runs the stories play. For every live run: its pinned seed's score against seeds 1..N, and
// whether the order between a story's runs on the pinned seed is the order of their medians. A run is shown well when
// it is good but no freak: far from either end of its seeds (flagged <<) and in the usual order (flagged FLIP). The score
// is the first measure that tells seeds apart: the last tenth of the run, or for episode lengths the whole run.
// Some runs are rare on purpose (a failure the story names as rare); the flags only say where to look.
// Usage, from rl_atlas/ after python build.py:  N=40 node tools/story-runs.js [story ids...]
// To see the typical seed of one story (its runs' ranks closest to the middle):  TYPICAL=1 node tools/story-runs.js id
require("../lab/node.js");
require("../app/content.js");
const { lab, content } = globalThis.RL;
const N = Number(process.env.N || 40), only = process.argv.slice(2);
const KEYS = ["optimal", "greedy", "match", "left", "error", "ve", "optimal-error", "delta", "return", "steps"];
const tail = (v) => { const k = Math.max(1, Math.floor(v.length / 10)); let s = 0, c = 0; for (let i = v.length - k; i < v.length; i++) if (Number.isFinite(v[i])) { s += v[i]; c++; } return c ? s / c : NaN; };
const score = (m, k) => (k === "steps" ? m[k].reduce((a, b) => a + b, 0) / m[k].length : tail(m[k]));
const median = (a) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };
const fmt = (x) => (Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(3));

for (const [id, entry] of Object.entries(content.entries)) {
  const cfg = entry.story?.config;
  if (!cfg || (only.length && !only.includes(id))) continue;
  const specs = Object.entries(cfg.runs || {}).filter(([, r]) => r.algorithm).map(([name, spec]) => {
    const { algorithm, units, seed, world, measures, name: _label, ...params } = spec;
    return { name, world: world || cfg.world || cfg.env, algorithm, params, units: units || cfg.units || 100, seed: seed ?? cfg.seed ?? 1, measures };
  });
  if (cfg.algorithm && cfg.episodes) specs.push({ name: "main", world: cfg.env, algorithm: cfg.algorithm, params: cfg, units: cfg.episodes, seed: cfg.seed });
  if (!specs.length) continue;
  const rows = [];
  for (const r of specs) {
    const one = (seed) => lab.simulate({ world: r.world, algorithm: lab.algorithms[r.algorithm], params: r.params, units: r.units, seed, snapshots: false, measures: r.measures }).metrics;
    const pinned = one(r.seed), seeds = Array.from({ length: N }, (_, i) => one(i + 1));
    const key = KEYS.find((k) => pinned[k] && new Set(seeds.map((m) => score(m, k).toFixed(6))).size > 1);
    if (!key) { rows.push({ ...r, key: null }); continue; }
    const lower = lab.lowerIsBetter({ metric: key }), scores = seeds.map((m) => score(m, key)), p = score(pinned, key);
    const pct = Math.round(100 * (1 - scores.filter((x) => (lower ? x < p : x > p)).length / N));
    rows.push({ ...r, key, lower, p, median: median(scores), pct, scores });
  }
  const flips = [];
  const scored = rows.filter((r) => r.key);
  for (let i = 0; i < scored.length; i++) for (let j = i + 1; j < scored.length; j++) {
    const a = scored[i], b = scored[j];
    if (a.key !== b.key) continue;
    const order = (x, y) => Math.sign(a.lower ? y - x : x - y);
    if (order(a.median, b.median) && order(a.p, b.p) !== order(a.median, b.median)) flips.push(`${a.name}/${b.name}`);
  }
  console.log(`${id}${flips.length ? "  FLIP " + flips.join(" ") : ""}`);
  for (const r of rows) {
    console.log(!r.key ? `   ${r.name}: no measure tells its seeds apart` :
      `   ${r.name.padEnd(10)} seed ${String(r.seed).padEnd(5)} ${r.key.padEnd(8)} ${fmt(r.p).padStart(8)}  median ${fmt(r.median).padStart(8)}  better than ${r.pct}%${r.pct < 10 || r.pct > 90 ? "  <<" : ""}`);
  }
  if (process.env.TYPICAL && scored.length) {
    const ranks = scored.map((r) => { const rk = []; r.scores.map((v, i) => [r.lower ? v : -v, i]).sort((x, y) => x[0] - y[0]).forEach(([, i], k) => (rk[i] = k + 1)); return rk; });
    const off = (s) => ranks.reduce((sum, rk) => sum + Math.abs(rk[s] - (N + 1) / 2), 0);
    const best = Array.from({ length: N }, (_, s) => s).sort((x, y) => off(x) - off(y)).slice(0, 5);
    console.log(`   typical seeds: ${best.map((s) => `${s + 1} (ranks ${ranks.map((rk) => rk[s]).join("/")})`).join(", ")}`);
  }
}
