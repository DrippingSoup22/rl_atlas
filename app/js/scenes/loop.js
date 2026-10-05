/* Story scene loop: the agent acts, the environment answers with a reward and the next state. */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const { timers, along, svgEl, sub, signed } = RL.sceneKit;

  // ---- loop: the agent acts, the environment answers with a reward and the next state ----
  // The environment is a small corridor: the agent mostly walks right; the gem at the end pays +10, every other step −1.
  const WIRES = { action: "M430 57 H520 V262 H437", state: "M230 248 H150 V44 H223", reward: "M230 280 H110 V72 H223" };

  function loopSvg(dry) {
    const tip = dry ? "loop-tip-dry" : "loop-tip";
    const mini = dry ? "" : `<g class="mini">${Array.from({ length: 6 }, (_, i) => `<rect class="cell" x="${262 + i * 26}" y="262" width="22" height="22" rx="5"/>`).join("")}
        <polygon class="gem" points="403,265 410,273 403,282 396,273"/><circle class="mini-dot" cx="299" cy="273" r="7"/></g>`;
    const brain = dry ? "" : `<g class="face" transform="translate(276 66)"><circle r="15"/><circle class="eye" cx="-5" cy="-2" r="2.5"/><circle class="eye" cx="5" cy="-2" r="2.5"/></g>
        <text class="box-note" x="300" y="71">policy π</text><text class="tally" x="416" y="42" text-anchor="end"></text>`;
    return `<svg class="${dry ? "fig " : ""}loop" viewBox="0 0 600 320" role="img" aria-label="The agent–environment loop">
      <defs><marker id="${tip}" class="wire-tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>
      ${dry ? "" : '<rect class="boundary" x="218" y="6" width="224" height="102" rx="24"/><text class="boundary-name" x="442" y="124" text-anchor="end">the agent: only what it controls</text>'}
      <g class="part-agent"><rect class="box" x="230" y="18" width="200" height="78" rx="16"/><text class="box-name" x="246" y="42">Agent</text>${brain}</g>
      <g class="part-env"><rect class="box" x="230" y="222" width="200" height="84" rx="16"/><text class="box-name" x="246" y="246">Environment</text>${mini}</g>
      <g class="wires">
        <path class="wire w-action" d="${WIRES.action}" marker-end="url(#${tip})"/>
        <path class="wire w-state" d="${WIRES.state}" marker-end="url(#${tip})"/>
        <path class="wire w-reward" d="${WIRES.reward}" marker-end="url(#${tip})"/>
        <text class="wire-name" x="528" y="152">action</text><text class="wire-sym" x="528" y="174">Aₜ</text>
        <text class="wire-name" x="158" y="140">state</text><text class="wire-sym" x="158" y="162">Sₜ₊₁</text>
        <text class="wire-name" x="102" y="168" text-anchor="end">reward</text><text class="wire-sym rew" x="102" y="190" text-anchor="end">Rₜ₊₁</text>
      </g>
      <g class="tokens"></g>
    </svg>`;
  }

  RL.scenes.loop = {
    svg: loopSvg,
    create(card) {
      card.innerHTML = `${loopSvg(false)}<div class="traj" aria-label="The trajectory so far"></div>`;
      const svg = card.querySelector("svg"), tokens = svg.querySelector(".tokens"), dot = svg.querySelector(".mini-dot");
      const traj = card.querySelector(".traj"), tally = svg.querySelector(".tally");
      const { later, stop } = timers();
      let anims = [], w = null;

      function reset() {
        w = { p: 1, t: 0, total: 0, rng: lab.rng(3) };
        traj.replaceChildren();
        dot.style.cx = "299px";
        tally.textContent = "";
      }
      function chip(kind, text) {
        traj.insertAdjacentHTML("beforeend", `<span class="tchip k-${kind}">${text}</span>`);
        while (traj.children.length > 16) traj.firstElementChild.remove();
      }
      function fly(kind, text, done) {
        const g = svgEl("g", { class: `token k-${kind}` }, tokens);
        g.innerHTML = `<rect x="-31" y="-11" width="62" height="22" rx="11"/><text y="4" text-anchor="middle">${text}</text>`;
        const a = along(g, svg.querySelector(`.w-${kind}`), 750);
        anims.push(a);
        a.onfinish = () => { g.remove(); done?.(); };
      }
      const stateText = () => `S${sub(w.t)} = ${w.p}`;
      function observe(animate, done) {
        const text = stateText();
        if (animate) fly("state", text, () => { chip("state", text); done?.(); });
        else { chip("state", text); done?.(); }
      }
      // The agent's policy: right three times in four. The world: −1 a step, +10 at the gem.
      function act(animate, done) {
        const a = w.rng.next() < 0.75 ? 1 : -1, text = `A${sub(w.t)} = ${a > 0 ? "→" : "←"}`;
        const go = () => {
          chip("action", text);
          w.p = Math.max(0, Math.min(5, w.p + a));
          dot.style.cx = `${273 + w.p * 26}px`;
          const r = w.p === 5 ? 10 : -1;
          w.t += 1;
          w.total += r;
          done?.(r);
        };
        if (animate) fly("action", text, go);
        else go();
      }
      function respond(r, animate, done) {
        const rText = `R${sub(w.t)} = ${signed(r)}`, sText = stateText();
        const go = () => { chip("reward", rText); chip("state", sText); tally.textContent = `total reward ${signed(w.total)}`; done?.(); };
        if (!animate) return go();
        let n = 0;
        const both = () => { if (++n === 2) go(); };
        fly("reward", rText, both);
        fly("state", sText, both);
      }
      function run() {
        act(true, (r) => respond(r, true, () => {
          if (w.p < 5) return later(run, 450);
          later(() => {
            chip("sep", "new episode");
            Object.assign(w, { p: 1, t: 0 });
            dot.style.cx = "299px";
            observe(false);
            later(run, 500);
          }, 600);
        }));
      }

      return {
        apply(st) {
          stop();
          anims.forEach((a) => a.cancel());
          anims = [];
          tokens.replaceChildren();
          svg.classList.toggle("no-env", st.env === false);
          svg.classList.toggle("no-wires", st.wires === false);
          svg.classList.toggle("with-boundary", !!st.boundary);
          svg.classList.toggle("with-tally", !!st.tally);
          traj.classList.toggle("on", !!st.trajectory);
          reset();
          if (st.phase === "observe") observe(true);
          else if (st.phase === "act") { observe(false); act(true); }
          else if (st.phase === "respond") { observe(false); act(false, (r) => respond(r, true)); }
          else if (st.phase === "loop") { observe(false); later(run, 300); }
        },
        destroy() { stop(); anims.forEach((a) => a.cancel()); },
      };
    },
  };
})(globalThis.RL = globalThis.RL || {});
