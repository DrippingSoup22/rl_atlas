/* Story scene grid: a grid world with values, a policy and the agent (it uses the Lab's grid view). */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const { timers, formula, FORMULA, ACTION, signed, runs, player, showRun, curves } = RL.sceneKit;

  // ---- grid: a grid world with values, a policy and the agent ----
  RL.scenes.grid = {
    create(card, cfg) {
      const env = lab.make(cfg.env), nA = env.nA, gamma = cfg.gamma ?? 1;
      const algorithm = cfg.algorithm ? lab.algorithms[cfg.algorithm] : null;
      card.innerHTML = `<div class="scene-grid"></div>${FORMULA}
        <div class="scene-foot"><span class="scene-note"></span>
          <span class="scale" title="Colors of the values"><span>worse</span><i></i><span>0</span><i class="up"></i><span>better</span></span></div>
        <div class="scene-chart" hidden></div>`;
      // Without an ε of its own, a story draws the greedy arrows of Q.
      const view = new RL.GridView(card.querySelector(".scene-grid"), env, { tiles: "q", epsilon: cfg.epsilon ?? 0, digits: cfg.digits ?? 2 });
      const note = card.querySelector(".scene-note"), showFormula = formula(card, cfg);
      const { later, stop } = timers(), runOf = runs(cfg), play = player(later);
      const chart = cfg.runs ? curves(card.querySelector(".scene-chart"), cfg, runOf) : () => {};
      const tile = (v) => (v === "start" ? env.start : v[0] * env.cols + v[1]);

      // What a step can show, each computed once: a learned Q, and exact values from dynamic programming.
      const memo = new Map();
      const once = (key, make) => (memo.has(key) ? memo.get(key) : memo.set(key, make()).get(key));
      const zeros = new Float64Array(env.nS * nA);
      const V = {
        random: () => once("v-random", () => lab.evaluate(env, lab.randomPolicy(env), gamma)),
        optimal: () => once("v-optimal", () => lab.valueIteration(env, gamma)),
      };
      const Q = {
        zero: () => zeros,
        trained: () => once("q-trained", () => lab.simulate({ world: cfg.env, algorithm, params: cfg, units: cfg.episodes, seed: cfg.seed }).at(cfg.episodes).Q),
        random: () => once("q-random", () => lab.qFromV(env, V.random(), gamma)),
        optimal: () => once("q-optimal", () => lab.qFromV(env, V.optimal(), gamma)),
        advantage: () => once("advantage", () => Q.random().map((q, i) => q - V.random()[Math.floor(i / nA)])),
      };
      const POLICY = {
        random: () => lab.randomPolicy(env),
        improved: () => once("p-improved", () => lab.greedyPolicy(env, V.random(), gamma)),
        optimal: () => once("p-optimal", () => lab.greedyPolicy(env, V.optimal(), gamma)),
      };
      const successors = (s) => [...new Set([0, 1, 2, 3].map((a) => env.step(s, a).s2))].filter((s2) => s2 !== s);

      function toward(from, to) {
        const [r1, c1] = env.rc(from), [r2, c2] = env.rc(to);
        return r2 < r1 ? 0 : c2 > c1 ? 1 : r2 > r1 ? 2 : 3;
      }

      function glowing(glow, q) {
        if (!glow) return [];
        const [r, c, a] = glow, s = r * env.cols + c;
        if (a !== "best") return [[s, ACTION[a]]];
        const best = lab.maxQ(q, s, env);
        return [0, 1, 2, 3].filter((k) => q[s * nA + k] === best).map((k) => [s, k]);
      }

      // The trained agent walks again and again with ε-greedy, so you can see how often it slips.
      function explore(q) {
        const rng = lab.rng(cfg.seed + 1);
        let s = env.start, t = 0, walks = 0, falls = 0;
        const tally = () => { note.textContent = `Walking with ε = ${cfg.epsilon}: ${walks} ${walks === 1 ? "walk" : "walks"}, ${falls} ${falls === 1 ? "fall" : "falls"}`; };
        const step = () => {
          if (env.terminal(s) || t > 60) {
            walks += 1;
            tally();
            s = env.start;
            t = 0;
            later(() => { view.place(s, 1, true); later(step, 350); }, 500);
            return;
          }
          const a = lab.epsGreedy(q, s, env, cfg.epsilon, rng);
          const { s2, fell } = env.step(s, a);
          if (fell !== undefined) {
            falls += 1;
            tally();
            view.fall(fell, s2);
            view.pop(fell, "−100");
            later(step, 1000);
          } else {
            view.move(s2, a);
            later(step, 150);
          }
          s = s2;
          t += 1;
        };
        tally();
        later(step, 600);
      }

      // The agent walks from one tile with a fixed policy, again and again. Each walk's discounted return
      // is one sample of the tile's value; their average creeps toward it.
      function wander(from, name) {
        const rng = lab.rng(cfg.seed || 1), P = POLICY[name](), value = (name === "random" ? V.random() : V.optimal())[from];
        const LENGTH = 40; // later rewards count less than γ⁴⁰ ≈ 1.5% of their size
        let s = from, t = 0, G = 0, walks = 0, total = 0;
        const tally = () => {
          note.textContent = `Walk ${walks + 1}: return so far ${G.toFixed(1)}` +
            (walks ? ` · average of ${walks} ${walks === 1 ? "walk" : "walks"}: ${(total / walks).toFixed(1)}, true value ${value.toFixed(1)}` : "");
        };
        const pick = () => {
          let u = rng.next();
          for (let a = 0; a < nA; a++) { u -= P[s * nA + a]; if (u <= 0) return a; }
          return nA - 1;
        };
        const step = () => {
          if (t === LENGTH) {
            walks += 1;
            total += G;
            s = from;
            t = 0;
            G = 0;
            tally();
            later(() => { view.place(s, -1, true); view.spark(s, -1); later(step, 450); }, 500);
            return;
          }
          const a = pick(), { s2, r } = env.step(s, a), jumped = env.jumps.some((j) => j.from === s);
          G += gamma ** t * r;
          if (jumped) { view.spark(s, -1); view.place(s2, a, true); view.spark(s2, -1); }
          else if (s2 === s) view.look(a);
          else view.move(s2, a);
          if (r) view.pop(jumped ? s2 : s, signed(r));
          s = s2;
          t += 1;
          tally();
          later(step, jumped ? 420 : 130);
        };
        view.place(from, -1, true);
        tally();
        later(step, 500);
      }

      // A step that names a run shows it at a moment (st.at) and can replay some of it (st.play), as in the Lab:
      // sweeps of dynamic programming, or Monte Carlo episodes walked and then added up backward.
      function fromRun(st) {
        const r = runOf(st.run), t = Math.min(st.at ?? 0, r.units);
        view.setOptions({ arrows: st.arrows !== false, tiles: st.tiles || (r.algorithm.show(r.at(0), env, r.params).Q ? "q" : "v"), numbers: !!st.values, range: st.range || cfg.range, trail: st.trail !== false, agent: r.algorithm.unit !== "sweep", traces: st.traces !== false, fog: !!st.fog });
        view.glow([]);
        view.path(null);
        showRun(view, r, t);
        const focus = st.focus ? tile(st.focus) : -1;
        view.mark(focus < 0 ? [] : [focus], "focus");
        view.mark(focus < 0 || !st.next ? [] : successors(focus), "next");
        showFormula(st);
        note.textContent = st.note || "";
        if (st.play) play(view, r, t, st.play, { pace: st.pace || 200, fine: !!st.fine, after: (u) => { if (!st.note) note.textContent = `${u} ${r.algorithm.unit}${u === 1 ? "" : "s"} done`; } });
      }

      return {
        apply(st, forward) {
          stop();
          note.textContent = "";
          chart(st);
          if (st.run) return fromRun(st);
          const q = Float64Array.from(st.q ? Q[st.q]() : zeros);
          for (const [r, c, a, v] of st.set || []) q[(r * env.cols + c) * nA + ACTION[a]] = v;
          view.setOptions({ arrows: !!(st.arrows || st.policy), tiles: st.v ? "v" : st.q ? "q" : "none", numbers: !!st.values, range: st.range || cfg.range });
          view.setV(st.v ? V[st.v]() : null);
          view.setPolicy(st.policy ? POLICY[st.policy]() : null);
          view.setQ(q, { ripple: !!st.ripple && forward });
          view.glow(glowing(st.glow, q));
          view.trail(null);
          view.path(st.path ? lab.greedyPath(env, q).path : null, forward);
          const focus = st.focus ? tile(st.focus) : -1;
          view.mark(focus < 0 ? [] : [focus], "focus");
          view.mark(focus < 0 || !st.next ? [] : successors(focus), "next");

          const hidden = st.agent === "none";
          view.showAgent(!hidden);
          const at = st.agent === undefined || hidden ? env.start : tile(st.agent);
          if (st.from !== undefined && forward && !hidden) {
            const from = tile(st.from);
            view.place(from, -1, true);
            later(() => {
              view.move(at, toward(from, at));
              if (st.reward !== undefined) view.pop(at, signed(st.reward));
            }, 450);
          } else view.place(at, -1, true);

          showFormula(st);
          if (st.explore) explore(q);
          else if (st.wander) wander(focus >= 0 ? focus : at, st.wander);
          else if (st.q === "trained" && algorithm) note.textContent = `After ${cfg.episodes} episodes of ${algorithm.title}`;
          else if (st.note) note.textContent = st.note;
        },
        destroy() { stop(); view.destroy(); },
      };
    },
  };
})(globalThis.RL = globalThis.RL || {});
