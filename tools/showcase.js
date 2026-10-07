// The showcase study (recorder/WORLD-STUDY.md, part 2): every comparison lab of the classic algorithms, in its own
// world and in every shared world its racers may run, with 20 seeds each. Prints one JSON line per lab and world:
// how many seeds each racer ends well on (the world's odds rule), its median score, and how often each racer is the
// best of the race on the same seed.
// Usage, from rl_atlas/ after python build.py:  node tools/showcase.js [lab ids...] > showcase.jsonl
require("../lab/node.js");
require("../app/content.js");
const { lab, content } = globalThis.RL;
const only = process.argv.slice(2), SEEDS = Array.from({ length: 20 }, (_, i) => i + 1);
const median = (a) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[(s.length - 1) >> 1] : NaN; };

for (const [id, base] of Object.entries(content.presets)) {
  if (only.length ? !only.includes(id) : base.racers.length < 2 || base.racers.some((r) => r.recording) || base.env === "sandbox") continue;
  const algos = base.racers.map((r) => lab.algorithms[r.algorithm]);
  const others = base.home ? [] : lab.worldsFor(algos).flatMap((g) => g.worlds.map((w) => w.id)).filter((w) => w !== base.env);
  for (const w of [base.env, ...others]) {
    const t0 = Date.now(), home = w === base.env;
    let params, units, rule, measures;
    if (home) ({ params, units, measures } = base), (rule = base.success || { metric: base.charts[0] });
    else { // as the Lab's world panel sets it up (lab.js, switchWorld)
      const prof = lab.worldProfile(w, algos);
      params = { ...base.params, ...prof.params };
      for (const k of ["alpha", "alphaW"]) if (!(k in params) && base.racers.some((r) => k in r.params)) params[k] = 0.1;
      if (prof.gamma !== undefined && "gamma" in base.params) params.gamma = prof.gamma;
      if (prof.domain) params.domain = prof.domain; else delete params.domain;
      if (prof.maxSteps) params.maxSteps = prof.maxSteps; else delete params.maxSteps;
      ({ units, measures } = prof);
      rule = prof.success || { metric: prof.charts[0] };
    }
    const env = lab.make(w);
    const racers = base.racers.map((r) => ({ name: r.name, algorithm: lab.algorithms[r.algorithm],
      params: { ...params, ...(home ? r.params : lab.carryRacer(r.params, { env, base: base.params, racers: base.racers.map((x) => x.params), knobs: params })) } }));
    const bench = new lab.Bench({ world: () => lab.make(w), racers, units, measures, rule, seeds: SEEDS, giveUp: 20 });
    bench.step(Infinity);
    const lower = lab.lowerIsBetter(rule), better = (a, b) => (lower ? a < b : a > b);
    const best = racers.map(() => 0);
    SEEDS.forEach((_, si) => { // the best racer on this seed (ties: nobody)
      const sc = racers.map((_, ri) => bench.results[ri][si].score);
      const top = sc.reduce((b, s, ri) => (b < 0 || better(s, sc[b]) ? ri : b), -1);
      if (sc.filter((s) => s === sc[top]).length === 1) best[top]++;
    });
    console.log(JSON.stringify({ lab: id, world: w, home, rule: rule.metric + (rule.min !== undefined ? `>=${rule.min}` : rule.max !== undefined ? `<=${rule.max}` : ""), units,
      racers: racers.map((r, ri) => ({ name: r.name, wins: bench.results[ri].filter((d) => d.ok).length, stuck: bench.results[ri].filter((d) => d.stopped).length, median: +median(bench.results[ri].map((d) => d.score)).toPrecision(3), best: best[ri] })),
      secs: Math.round((Date.now() - t0) / 1000) }));
  }
}
