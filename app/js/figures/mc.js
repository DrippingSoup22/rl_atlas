/* Figures of Part 4, Monte Carlo methods: Blackjack's values and strategy as Monte Carlo learns them, Monte Carlo
   control against Q-learning on Frozen Lake, and the two classic examples of importance sampling. Long runs are
   computed a slice at a time, so the page stays responsive while the figures fill in. */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const RANK = (d) => (d === 1 ? "A" : String(d));

  // Play an algorithm a few units at a time; every `every` units, call look(memory, units played).
  // Returns a function that cancels the run.
  function chunked({ world, algorithm, params, seed, units, every, look, done }) {
    const env = lab.make(world), rng = lab.rng(seed), p = { maxSteps: 5000, ...params };
    env.init?.(rng);
    const { m } = lab.memory(algorithm.memory(env, p));
    algorithm.init?.(m, env, p);
    let t = 0, stopped = false;
    const slice = () => {
      if (stopped) return;
      const t0 = performance.now();
      while (t < units && performance.now() - t0 < 24) {
        lab.play(algorithm, { env, m, rng, p, t });
        t += 1;
        if (t % every === 0 || t === units) look(m, t, env);
      }
      if (t < units) setTimeout(slice, 0);
      else done?.();
    };
    slice();
    return () => { stopped = true; };
  }

  // Blackjack's 200 states as two 10 × 10 maps (usable ace above, no usable ace below): fill(s) colors a cell,
  // mark(s) can outline it, text(s) can write in it. Returns SVG markup at (x, y).
  function bjMaps(env, { x = 0, y = 0, C = 15, fill, text, mark, title }) {
    let g = title ? `<text class="note" x="${x + 5 * C + 14}" y="${y + 14}" text-anchor="middle">${title}</text>` : "";
    [true, false].forEach((usable, k) => {
      const oy = y + 24 + k * (10 * C + 34), ox = x + 26;
      g += `<text class="tick small" x="${ox + 5 * C}" y="${oy + 10 * C + 26}" text-anchor="middle">${usable ? "usable ace" : "no usable ace"}</text>`;
      for (let sum = 21; sum >= 12; sum--) {
        const yy = oy + (21 - sum) * C;
        if (sum % 3 === 0 || sum === 21 || sum === 12) g += `<text class="tick small" x="${ox - 4}" y="${yy + C / 2 + 4}" text-anchor="end">${sum}</text>`;
        for (let d = 1; d <= 10; d++) {
          const s = env.encode(sum, d, usable), xx = ox + (d - 1) * C;
          g += `<rect class="bj-cell${mark?.(s) ? " off" : ""}" x="${xx}" y="${yy}" width="${C}" height="${C}" style="fill:${fill(s)}"/>`;
          const t = text?.(s);
          if (t) g += `<text class="bj-txt" x="${xx + C / 2}" y="${yy + C / 2 + 3.5}" text-anchor="middle">${t}</text>`;
        }
      }
      for (const d of [1, 10]) g += `<text class="tick small" x="${ox + (d - 0.5) * C}" y="${oy + 10 * C + 12}" text-anchor="middle">${RANK(d)}</text>`;
      g += `<rect class="frame" x="${ox}" y="${oy}" width="${10 * C}" height="${10 * C}"/>`;
    });
    return g;
  }
  // The policy's two actions in its own color, as in the Lab's cards view: hit dark, stick light.
  const HIT_FILL = "color-mix(in oklab, var(--pol) 72%, var(--surface))", STICK_FILL = "color-mix(in oklab, var(--pol) 10%, var(--surface))";

  // ---- Sutton & Barto, Figure 5.1: the values of "stick on 20 or 21" after 10,000 and 500,000 hands, and exactly ----
  RL.demos["blackjack-values"] = function (host) {
    const env = lab.make("blackjack"), C = 15, panelW = 10 * C + 40, W = 3 * panelW + 20, H = 2 * (10 * C + 34) + 36;
    const exact = lab.blackjackExact(env.policy("stick-20")).V;
    const shots = { 10000: null, 500000: null };
    const svg = () => {
      const panel = (V, i, title) => bjMaps(env, { x: i * (panelW + 10), C, title, fill: (s) => (V ? RL.GridView.valueColor(V[s], 1) : "var(--surface-2)") });
      return `<svg class="fig bj" viewBox="0 0 ${W} ${H}" role="img" aria-label="The values of the stick-on-20 policy at Blackjack, learned by Monte Carlo and exact">
        ${panel(shots[10000], 0, "after 10,000 hands")}${panel(shots[500000], 1, "after 500,000 hands")}${panel(exact, 2, "exact, from the rules")}</svg>
        <div class="scale fig-scale"><span>−1</span><i></i><span>0</span><i class="up"></i><span>+1</span></div><p class="fig-status"></p>`;
    };
    host.innerHTML = svg();
    RL.fig.whenVisible(host, () => {
      const status = () => host.querySelector(".fig-status");
      chunked({
        world: "blackjack", algorithm: lab.algorithms["mc-prediction"], params: { gamma: 1, policy: "stick-20" }, seed: 5, units: 500000, every: 10000,
        look: (m, t) => {
          if (t === 10000 || t === 500000) { shots[t] = Float64Array.from(m.V); host.innerHTML = svg(); }
          status().textContent = t < 500000 ? `Playing hand ${t.toLocaleString("en")} of 500,000…` : `RMS error after 500,000 hands: ${lab.rms(m.V, exact, env).toFixed(3)}.`;
        },
      });
    });
  };

  // ---- Sutton & Barto, Figure 5.2: the optimal strategy, and the one Monte Carlo ES has learned ----
  RL.demos["blackjack-policy"] = function (host) {
    const env = lab.make("blackjack"), C = 15, panelW = 10 * C + 40, W = 2 * panelW + 10, H = 2 * (10 * C + 34) + 36;
    const best = lab.blackjackExact().policy, units = 300000;
    const stickBest = (s) => best[s * 2] > 0.5;
    let learned = null, last = 0;
    const draw = () => {
      const stickLearned = (s) => learned[s * 2] >= learned[s * 2 + 1];
      const exactPanel = bjMaps(env, { x: 0, C, title: "optimal (exact)", fill: (s) => (stickBest(s) ? STICK_FILL : HIT_FILL) });
      const learnedPanel = bjMaps(env, {
        x: panelW + 10, C, title: learned ? `Monte Carlo ES, ${last.toLocaleString("en")} hands` : "Monte Carlo ES",
        fill: (s) => (!learned ? "var(--surface-2)" : stickLearned(s) ? STICK_FILL : HIT_FILL),
        mark: (s) => learned && stickLearned(s) !== stickBest(s),
      });
      host.innerHTML = `<svg class="fig bj" viewBox="0 0 ${W} ${H}" role="img" aria-label="The optimal Blackjack strategy and the one learned by Monte Carlo with exploring starts">${exactPanel}${learnedPanel}</svg>
        <p class="arms-legend fig-legend"><i class="lg-hit"></i>hit · <i class="lg-stick"></i>stick · outlined: differs from the optimal strategy</p><p class="fig-status"></p>`;
    };
    draw();
    RL.fig.whenVisible(host, () => {
      chunked({
        world: "blackjack", algorithm: lab.algorithms["exploring-starts"], params: { gamma: 1 }, seed: 2, units, every: 25000,
        look: (m, t) => {
          learned = Float64Array.from(m.Q);
          last = t;
          draw();
          let off = 0;
          for (let s = 0; s < 200; s++) if ((learned[s * 2] >= learned[s * 2 + 1]) !== stickBest(s)) off++;
          host.querySelector(".fig-status").textContent = t < units ? `Playing hand ${t.toLocaleString("en")} of ${units.toLocaleString("en")}…` : `${off} of 200 states differ from the optimal strategy, most of them near the boundary, where the two moves are worth almost the same.`;
        },
      });
    });
  };

  // ---- How fast each control method finds the optimal strategy: Monte Carlo ES against off-policy MC from random play ----
  RL.demos["blackjack-match"] = function (host) {
    const units = 100000, every = 500, runs = 3, n = units / every;
    const p = RL.fig.plot(host, {
      label: "States where the greedy action is optimal, while learning Blackjack",
      right: 150,
      x: { max: units, from: 1, ticks: [0, 25000, 50000, 75000, 100000], label: "Hands played", format: (u) => (u ? `${u / 1000}k` : "0") },
      y: { min: 0.4, max: 1, ticks: [0.4, 0.6, 0.8, 1], label: "Greedy action optimal", percent: true },
      curves: [{ id: "es", name: "Monte Carlo ES" }, { id: "off", name: "off-policy MC", dash: true }],
      at: (t) => `After ${t.toLocaleString("en")} hands`,
    });
    const env = lab.make("blackjack"), optimal = lab.optimalActions(env, { gamma: 1 });
    const states = optimal.map((set, s) => s).filter((s) => optimal[s].size && optimal[s].size < env.acts(s).length);
    const match = (Q) => states.filter((s) => env.acts(s).every((a) => Q[s * 2 + a] < Math.max(Q[s * 2], Q[s * 2 + 1]) || optimal[s].has(a))).length / states.length;
    const sums = { es: new Float64Array(n), off: new Float64Array(n) }, counts = { es: 0, off: 0 };
    const jobs = [];
    for (let k = 0; k < runs; k++) jobs.push(["es", "exploring-starts", { gamma: 1 }, 100 + k], ["off", "off-policy-mc", { gamma: 1, epsilon: 1 }, 100 + k]);
    // Stretch each curve over the points computed so far, so the plot fills in from the left.
    const lines = () => Object.fromEntries(Object.keys(sums).filter((id) => counts[id]).map((id) => [id, Float64Array.from({ length: units }, (_, t) => sums[id][Math.min(n - 1, Math.floor(t / every))] / counts[id])]));
    RL.fig.whenVisible(host, () => {
      const next = (j) => {
        if (j >= jobs.length || !host.isConnected) { p.status(`Each curve: the average of ${runs} runs, measured every ${every} hands against the exact optimal strategy.`); return; }
        const [id, algo, params, seed] = jobs[j];
        p.status(`Run ${Math.floor(j / 2) + 1} of ${runs}: ${id === "es" ? "Monte Carlo ES" : "off-policy MC"}…`);
        const row = new Float64Array(n);
        chunked({
          world: "blackjack", algorithm: lab.algorithms[algo], params, seed, units, every,
          look: (m, t) => { row[t / every - 1] = match(m.Q); },
          done: () => { for (let i = 0; i < n; i++) sums[id][i] += row[i]; counts[id] += 1; p.draw(lines()); next(j + 1); },
        });
      };
      next(0);
    });
  };

  // ---- Monte Carlo control against Q-learning on Frozen Lake: how good the greedy policy is, episode by episode ----
  RL.demos["frozen-mc"] = function (host) {
    const units = 3000, params = { gamma: 0.99, epsilon: 0.1, judge: 1 }, env = lab.make("frozen-lake");
    const best = lab.evaluate(env, lab.greedyPolicy(env, lab.optimalValues(env, params), params.gamma, 1e-6), 1, { theta: 1e-10 })[env.start];
    RL.fig.average(host, {
      label: "The chance that the greedy policy reaches the gem, for on-policy Monte Carlo control and Q-learning",
      right: 150,
      x: { max: units, ticks: [0, 1000, 2000, 3000], label: "Episodes" },
      y: { min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], label: "Greedy policy reaches the gem", percent: true },
      curves: [{ id: "mc", name: "MC control (averages)" }, { id: "q", name: "Q-learning (α = 0.1)", dash: true }],
      refs: [{ value: best, label: `the optimal policy: ${Math.round(best * 100)}%` }],
      at: (t) => `After ${t.toLocaleString("en")} episodes`,
    }, {
      runs: 20,
      sample: (seed) => ({
        mc: lab.simulate({ world: "frozen-lake", algorithm: lab.algorithms["mc-control"], params: { ...params, alpha: 0 }, units, seed, snapshots: false, measures: ["greedy"] }).metrics.greedy,
        q: lab.simulate({ world: "frozen-lake", algorithm: lab.algorithms["q-learning"], params: { ...params, alpha: 0.1 }, units, seed, snapshots: false, measures: ["greedy"] }).metrics.greedy,
      }),
    });
  };

  // ---- Sutton & Barto, Example 5.4 and Figure 5.3: ordinary and weighted importance sampling on one Blackjack state ----
  RL.demos["is-blackjack"] = function (host) {
    const env = lab.make("blackjack"), s0 = env.encode(13, 2, true), episodes = 10000, runs = 100;
    const truth = lab.blackjackExact(env.policy("stick-20")).V[s0];
    const p = RL.fig.plot(host, {
      label: "Mean squared error of ordinary and weighted importance sampling, estimating one Blackjack state",
      right: 150,
      x: { max: episodes, log: true, ticks: [1, 10, 100, 1000, 10000], label: "Episodes (log scale)" },
      y: { min: 0, max: 9, ticks: [0, 2, 4, 6, 8], label: "Mean squared error", digits: 2, tickDigits: 0 },
      curves: [{ id: "ord", name: "ordinary" }, { id: "wei", name: "weighted", dash: true }],
      at: (k) => `After ${k.toLocaleString("en")} episodes`,
    });
    const se = { ord: new Float64Array(episodes), wei: new Float64Array(episodes) };
    let done = 0;
    // One episode from s0 under the random behavior policy: the return G and the ratio ρ for "stick on 20 or 21".
    const episode = (rng) => {
      let s = env.reset(rng, s0), rho = 1, G = 0;
      while (!env.terminal(s)) {
        const a = rng.next() < 0.5 ? env.STICK : env.HIT, target = env.decode(s).sum >= 20 ? env.STICK : env.HIT;
        rho = a === target ? rho * 2 : 0; // π(a|s) / b(a|s) = 1 / 0.5 when the target would act the same, else 0
        const o = env.step(s, a, rng);
        G += o.r;
        s = o.s2;
      }
      return [rho, G];
    };
    RL.fig.whenVisible(host, () => {
      const more = () => {
        if (!host.isConnected) return;
        const t0 = performance.now();
        while (done < runs && performance.now() - t0 < 30) {
          const rng = lab.rng(500 + done);
          let num = 0, den = 0;
          for (let k = 0; k < episodes; k++) {
            const [rho, G] = episode(rng);
            num += rho * G;
            den += rho;
            se.ord[k] += (num / (k + 1) - truth) ** 2;
            se.wei[k] += ((den ? num / den : 0) - truth) ** 2;
          }
          done++;
        }
        p.draw({ ord: se.ord.map((v) => v / done), wei: se.wei.map((v) => v / done) });
        p.status(done < runs ? `Averaging run ${done} of ${runs}…` : `Average of ${runs} runs. The true value of the state, computed from the rules, is ${RL.fig.num(truth, 5)}.`);
        if (done < runs) setTimeout(more, 16);
      };
      more();
    });
  };

  // ---- Sutton & Barto, Example 5.5 and Figure 5.4: ordinary importance sampling with infinite variance ----
  RL.demos["is-infinite"] = function (host) {
    const episodes = 1000000, runs = 10, points = 240;
    const at = Array.from({ length: points }, (_, i) => Math.max(1, Math.round(episodes ** (i / (points - 1)))));
    const p = RL.fig.plot(host, {
      label: "Ten runs of ordinary importance sampling whose estimates have infinite variance",
      right: 40,
      x: { max: episodes, log: true, ticks: [1, 10, 100, 1000, 10000, 100000, 1000000], label: "Episodes (log scale)", format: (u) => (u >= 1000 ? `10${"⁰¹²³⁴⁵⁶"[Math.round(Math.log10(u))]}` : String(u)) },
      y: { min: 0, max: 3, ticks: [0, 1, 2, 3], label: "Estimate of v(s)", digits: 2, tickDigits: 0 },
      curves: Array.from({ length: runs }, (_, i) => ({ id: `r${i}`, name: "", light: i > 0, marks: false })),
      refs: [{ value: 1, label: "true value 1" }],
      at: (k) => `After ${k.toLocaleString("en")} episodes`,
    });
    const lines = {};
    let run = 0;
    RL.fig.whenVisible(host, () => {
      const one = () => {
        if (!host.isConnected || run >= runs) { p.status("Each line is one run. The target policy always goes left; the behavior goes left or right with equal probability."); return; }
        const rng = lab.rng(900 + run), xs = [], ys = [];
        let sum = 0, k = 0, next = 0;
        const slice = () => {
          if (!host.isConnected) return;
          const t0 = performance.now();
          while (k < episodes && performance.now() - t0 < 24) {
            // From s: right ends at once with reward 0 (ρ = 0); left returns to s with probability 0.9, else ends with +1.
            let rho = 1, G = 0;
            for (;;) {
              if (rng.next() < 0.5) { rho = 0; break; }
              rho *= 2;
              if (rng.next() < 0.1) { G = 1; break; }
            }
            sum += rho * G;
            k += 1;
            while (next < points && at[next] === k) { xs.push(k); ys.push(sum / k); next++; }
          }
          if (k < episodes) { setTimeout(slice, 0); return; }
          lines[`r${run}`] = { xs, ys };
          p.draw(lines);
          p.status(`Run ${run + 1} of ${runs}…`);
          run += 1;
          one();
        };
        slice();
      };
      one();
    });
  };
})(globalThis.RL = globalThis.RL || {});
