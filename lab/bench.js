/* The bench: the same settings run on a fixed set of seeds (1 to 20), so that what the settings do can be told apart
   from what luck does. From the bench come the odds (how many seeds end well), the spread over time (a band: the
   middle half of the seeds, and their median), the typical seed (the one whose ending is the median; for a race, the
   seed closest to every racer's median), and where any other seed falls among the bench.
   The work is done in slices, so a page can run it in the background; results are kept per settings, so going back
   to settings already benched costs nothing. */
(function (RL) {
  "use strict";
  const lab = RL.lab;
  const SEEDS = Array.from({ length: 20 }, (_, i) => i + 1);
  const POINTS = 400; // the band is kept at this many points at most: enough for any chart's width
  // Measures where less is better; for the others, more is better. A success rule with a max says "less" too.
  const LOWER = new Set(["steps", "error", "ve", "delta", "optimal-error", "weights", "left", "kl"]);
  lab.lowerIsBetter = (rule) => (rule.max !== undefined ? true : rule.min !== undefined ? false : LOWER.has(rule.metric));

  // A long series, smoothed (a trailing average over w units, as the charts draw it) and cut into at most POINTS
  // buckets, each the average of the units in it.
  function shrink(values, w) {
    const n = values.length, k = Math.min(n, POINTS), out = new Float32Array(k);
    let sum = 0;
    const sm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      sum += values[i];
      if (w > 1 && i >= w) sum -= values[i - w];
      sm[i] = w > 1 ? sum / Math.min(i + 1, w) : values[i];
    }
    for (let b = 0; b < k; b++) {
      const i0 = Math.floor((b * n) / k), i1 = Math.max(i0 + 1, Math.floor(((b + 1) * n) / k));
      let s = 0, c = 0;
      for (let i = i0; i < i1; i++) if (Number.isFinite(sm[i])) { s += sm[i]; c++; }
      out[b] = c ? s / c : NaN;
    }
    return out;
  }

  const quantile = (sorted, q) => {
    if (!sorted.length) return NaN;
    const at = (sorted.length - 1) * q, lo = Math.floor(at), hi = Math.ceil(at);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo);
  };

  // Ranks of a list of scores, 1 for the best; tied scores share the average of their ranks.
  // The typical seed of a bench: scores[racer][i] is how seed seeds[i] ended; the median seed, or for a race the one
  // whose ranks are closest, summed over the racers, to the middle.
  lab.typicalSeed = function (scores, seeds, lower) {
    const n = scores[0].length, per = scores.map((s) => ranks(s, lower)), middle = (n + 1) / 2;
    let best = 0, bestD = Infinity;
    for (let si = 0; si < n; si++) {
      const d = per.reduce((sum, rk) => sum + Math.abs(rk[si] - middle), 0);
      if (d < bestD - 1e-9) { bestD = d; best = si; }
    }
    return seeds[best];
  };

  function ranks(scores, lower) {
    const order = scores.map((s, i) => [Number.isNaN(s) ? (lower ? Infinity : -Infinity) : s, i]).sort((a, b) => (lower ? a[0] - b[0] : b[0] - a[0]));
    const out = new Array(scores.length);
    for (let i = 0; i < order.length;) {
      let j = i;
      while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
      for (let k = i; k <= j; k++) out[order[k][1]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return out;
  }

  const cache = new Map();

  // world: a world id or a function making one; racers: [{ algorithm, params }]; rule: a success rule, or
  // { metric } to judge by a measure's last tenth; keys: the measures whose spread is kept; smooth: { key: w }.
  class Bench {
    constructor({ world, racers, units, measures = null, rule, keys = [], smooth = {}, seeds = SEEDS, giveUp = 20, key = null }) {
      Object.assign(this, { world, racers, units, measures, rule, keys, smooth, seeds, giveUp });
      this.lower = lab.lowerIsBetter(rule);
      this.key = key && JSON.stringify([key, racers.map((r) => [r.algorithm.id, r.params]), units, seeds]);
      const kept = this.key && cache.get(this.key);
      this.results = kept || racers.map(() => []);
      this.next = kept ? seeds.length * racers.length : 0;
    }

    get total() { return this.seeds.length * this.racers.length; }
    get complete() { return this.next >= this.total; }
    get seedsDone() { return Math.floor(this.next / this.racers.length); }

    // Run until the budget (in ms) is spent or the bench is complete; true when complete. A long run is itself
    // spread over several steps (lab.simulateJob), so no step overruns its budget by more than one unit of a run.
    step(budget = 14) {
      const t0 = Date.now();
      while (!this.complete && (budget === Infinity || Date.now() - t0 < budget)) {
        const si = Math.floor(this.next / this.racers.length), ri = this.next % this.racers.length, r = this.racers[ri];
        const seed = this.seeds[si];
        this.job ||= lab.simulateJob({ world: this.world, algorithm: r.algorithm, params: r.params, units: this.units, seed, snapshots: false, measures: this.measures, giveUp: this.giveUp });
        if (!this.job.step(budget === Infinity ? Infinity : Math.max(1, budget - (Date.now() - t0)))) break;
        const { metrics, stopped } = this.job.result;
        this.job = null;
        const { score, ok } = lab.success(this.rule, metrics);
        const series = {};
        for (const k of this.keys) if (metrics[k]) series[k] = shrink(metrics[k], this.smooth[k] || 1);
        this.results[ri][si] = { seed, score, ok, stopped: !!stopped, series };
        this.next++;
      }
      if (this.complete && this.key) cache.set(this.key, this.results);
      return this.complete;
    }

    // Per racer, from the seeds done so far: scores, wins, stuck runs, and per kept measure its band (quartiles,
    // median) and its mean.
    // Kept until another seed is done (the charts and the odds both ask). Thousands of seeds make a band of hundreds
    // of points slow to sort, so each column is sorted as numbers, in a typed array.
    stats() {
      const n = this.seedsDone;
      if (this.lastStats?.n === n) return this.lastStats.stats;
      const col = new Float64Array(n);
      const stats = this.racers.map((_, ri) => {
        const done = this.results[ri].slice(0, n), scores = done.map((d) => d.score);
        const band = {};
        for (const k of this.keys) {
          const lines = done.map((d) => d.series[k]).filter(Boolean);
          if (!lines.length) continue;
          const len = lines[0].length, lo = new Float32Array(len), mid = new Float32Array(len), hi = new Float32Array(len), mean = new Float32Array(len);
          for (let b = 0; b < len; b++) {
            let m = 0;
            for (const l of lines) if (Number.isFinite(l[b])) col[m++] = l[b];
            const sorted = col.subarray(0, m).sort();
            lo[b] = quantile(sorted, 0.25); mid[b] = quantile(sorted, 0.5); hi[b] = quantile(sorted, 0.75);
            let sum = 0;
            for (let j = 0; j < m; j++) sum += sorted[j];
            mean[b] = m ? sum / m : NaN;
          }
          band[k] = { lo, mid, hi, mean, units: this.units };
        }
        const sorted = scores.filter(Number.isFinite).sort((x, y) => x - y);
        return { n, scores, wins: done.filter((d) => d.ok).length, stuck: done.filter((d) => d.stopped).length, median: quantile(sorted, 0.5), band };
      });
      this.lastStats = { n, stats };
      return stats;
    }

    // The typical seed: the median ending. For a race, the seed whose ranks are closest, summed over the racers, to
    // the middle rank; ties go to the smaller seed.
    typical() {
      const n = this.seedsDone;
      if (!n) return this.seeds[0];
      return lab.typicalSeed(this.racers.map((_, ri) => this.results[ri].slice(0, n).map((d) => d.score)), this.seeds, this.lower);
    }

    // Where a score falls among racer ri's bench: how many seeds it beats, ties, and loses to.
    place(ri, score) {
      const scores = this.results[ri].slice(0, this.seedsDone).map((d) => d.score).filter(Number.isFinite);
      const better = (a, b) => (this.lower ? a < b : a > b);
      return { beats: scores.filter((s) => better(score, s)).length, ties: scores.filter((s) => s === score).length, of: scores.length };
    }

    // Seeds on which racer a ends better than racer b (the same seed, the same luck).
    pairedWins(a, b) {
      const n = this.seedsDone, A = this.results[a], B = this.results[b];
      let wins = 0;
      for (let si = 0; si < n; si++) if (this.lower ? A[si].score < B[si].score : A[si].score > B[si].score) wins++;
      return { wins, of: n };
    }
  }

  lab.Bench = Bench;
  lab.BENCH_SEEDS = SEEDS;
  lab.benchQuantile = quantile;
})(globalThis.RL = globalThis.RL || {});
