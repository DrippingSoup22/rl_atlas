// The averaged curves the story charts show, worked out once here rather than in every reader's browser, where a chart
// of 100 runs could keep the page busy for a minute or more. The runs, seeds and sums are the page's own
// (lab.curveRun, app/js/scenes/common.js: curves); the means are kept to four significant digits, and each curve's
// first run leaves a fingerprint, so that tests/stories.test.js can tell when the engine no longer makes these curves.
// A curve longer than 1000 units keeps the mean of each stretch of units instead (the chart draws a few hundred pixels:
// it spreads them over the run, lab/views/chart.js). A chart missing here (a story changed since) is averaged live.
// Usage, from rl_atlas/ after python build.py:  node tools/story-curves.js   (then python build.py again)
require("../lab/node.js");
require("../app/content.js");
const fs = require("node:fs");
const path = require("node:path");
const { lab, content } = globalThis.RL;

// at most 1000 points: the mean of each stretch of k units (k = 20 for 20,000 units), its finite values only
const POINTS = 1000;
function thin(v) {
  if (v.length <= POINTS) return Array.from(v);
  const k = Math.ceil(v.length / POINTS), out = [];
  for (let i = 0; i < v.length; i += k) {
    let sum = 0, n = 0;
    for (let j = i; j < Math.min(v.length, i + k); j++) if (Number.isFinite(v[j])) { sum += v[j]; n++; }
    out.push(n ? sum / n : NaN);
  }
  return out;
}
// a run's numbers in short: the sum of its finite values, and how many are not finite
const fingerprint = (v) => { let sum = 0, odd = 0; for (const x of v) if (Number.isFinite(x)) sum += x; else odd++; return [sum, odd]; };

const out = {}, t0 = Date.now();
for (const [id, entry] of Object.entries(content.entries)) {
  const cfg = entry.story?.config;
  if (!cfg?.runs) continue;
  const charts = new Map();
  for (const { state: st } of entry.story.steps) {
    if (!st.curves?.length || st.curves.every((n) => cfg.runs[n].recording)) continue;
    const metric = st.metric || "optimal";
    charts.set(lab.curveKey(cfg, st.curves, metric), { names: st.curves, metric });
  }
  for (const [key, { names, metric }] of charts) {
    const total = cfg.average || 200, len = cfg.runs[names[0]].units || cfg.units || 100;
    const sums = names.map(() => new Float64Array(len)), check = {};
    for (let k = 0; k < total; k++) {
      names.forEach((n, i) => {
        const v = lab.simulate(lab.curveRun(cfg, n, k)).metrics[metric];
        for (let t = 0; t < len; t++) sums[i][t] += v[t];
        if (k === 0) check[n] = fingerprint(v);
      });
    }
    out[key] = {
      story: id, metric, runs: total, check,
      curves: Object.fromEntries(names.map((n, i) => [n, thin(sums[i].map((s) => s / total)).map((v) => (Number.isFinite(v) ? +v.toPrecision(4) : null))])),
    };
    console.log(`${id}: ${names.join(", ")} (${metric}), ${total} runs`);
  }
}
const file = path.join(__dirname, "..", "content", "story-curves.json");
fs.writeFileSync(file, `${JSON.stringify(out, null, 0)}\n`);
console.log(`${Object.keys(out).length} charts in ${((Date.now() - t0) / 1000).toFixed(0)} s: ${(fs.statSync(file).size / 1024).toFixed(0)} KB`);
