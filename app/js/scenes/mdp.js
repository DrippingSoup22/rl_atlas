/* Story scene mdp: a small MDP drawn as a graph of states, actions and outcomes (the recycling robot). */
(function (RL) {
  "use strict";
  const { lab } = RL;
  const { timers, formula, along, svgEl, FORMULA, signed } = RL.sceneKit;

  // ---- mdp: the recycling robot (after Sutton & Barto, Example 3.3) as a graph of states, actions and outcomes ----
  const ROBOT = {
    states: { high: [150, 170], low: [450, 170] },
    actions: { // where each action's dot sits, and where its name goes
      "high-search": [300, 72, 0, -14], "high-wait": [54, 98, 0, -14], "low-search": [300, 270, 0, 26],
      "low-wait": [546, 98, 0, -14], "low-recharge": [300, 170, 0, 24],
    },
    edges: { // from a state to one of its actions
      "high-search": "M180 148 L295 76", "high-wait": "M119 150 L60 103", "low-search": "M420 192 L306 266",
      "low-wait": "M481 150 L540 103", "low-recharge": "M411 170 L307 170",
    },
    outcomes: { // from an action to where it may lead: path, then where its probability and reward are written
      "high-search>high": ["M293 67 Q205 28 162 134", 205, 44],
      "high-search>low": ["M307 67 Q395 28 438 134", 395, 44],
      "high-wait>high": ["M47 105 Q30 192 112 183", 34, 204],
      "low-search>low": ["M307 276 Q395 314 438 206", 395, 316],
      "low-search>high": ["M293 276 Q205 314 162 206", 205, 316],
      "low-wait>low": ["M553 105 Q570 192 488 183", 566, 204],
      "low-recharge>high": ["M293 170 L191 170", 242, 161],
    },
  };

  function mdpSvg(env, dry) {
    const tip = dry ? "mdp-tip-dry" : "mdp-tip";
    let g = `<svg class="${dry ? "fig " : ""}mdp" viewBox="0 0 600 340" role="img" aria-label="${env.title}: its states, actions and outcomes">
      <defs><marker id="${tip}" class="out-tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>`;
    for (const [key, d] of Object.entries(ROBOT.edges)) g += `<path class="act-edge" data-act="${key}" d="${d}"/>`;
    for (const [key, [d, lx, ly]] of Object.entries(ROBOT.outcomes)) {
      const [act, to] = key.split(">"), [s, a] = act.split("-");
      const [, , , p, r] = env.dynamics.find(([s0, a0, s2]) => s0 === s && a0 === a && s2 === to);
      g += `<path class="out-edge" data-act="${act}" data-out="${key}" d="${d}" marker-end="url(#${tip})"/>
        <text class="out" data-act="${act}" x="${lx}" y="${ly}" text-anchor="middle"><tspan class="p">${p}</tspan><tspan class="r"> · ${signed(r)}</tspan></text>`;
    }
    for (const [key, [x, y, dx, dy]] of Object.entries(ROBOT.actions)) {
      g += `<circle class="act" data-act="${key}" cx="${x}" cy="${y}" r="6.5"/><text class="act-name" data-act="${key}" x="${x + dx}" y="${y + dy}" text-anchor="middle">${key.split("-")[1]}</text>`;
    }
    for (const [name, [x, y]] of Object.entries(ROBOT.states)) {
      g += `<g class="state" data-state="${name}"><circle cx="${x}" cy="${y}" r="38"/><text x="${x}" y="${y + 6}" text-anchor="middle">${name}</text></g>`;
    }
    return `${g}<g class="tokens"></g></svg>`;
  }

  RL.scenes.mdp = {
    svg: mdpSvg,
    create(card, cfg) {
      const env = lab.mdp(cfg.mdp || "robot");
      card.innerHTML = `${mdpSvg(env, false)}<div class="mdp-table"><table>
          <thead><tr><th>s</th><th>a</th><th>s′</th><th>r</th><th>p(s′, r | s, a)</th></tr></thead>
          <tbody>${env.dynamics.map(([s, a, s2, p, r]) => `<tr data-act="${s}-${a}"><td>${s}</td><td>${a}</td><td>${s2}</td><td class="rew">${signed(r)}</td><td>${p}</td></tr>`).join("")}</tbody>
        </table></div>${FORMULA}<div class="scene-foot"><span class="scene-note"></span></div>`;
      const svg = card.querySelector("svg"), table = card.querySelector(".mdp-table"), note = card.querySelector(".scene-note");
      const tokens = svg.querySelector(".tokens"), showFormula = formula(card, cfg);
      const { later, stop } = timers();
      let anims = [];

      // The robot acts at random; the note keeps the last few steps, which its future does not depend on.
      function walk() {
        const rng = lab.rng(11), robot = svgEl("g", { class: "robot" }, tokens);
        robot.innerHTML = '<circle r="11"/><circle class="eye" cx="-3.5" cy="-1.5" r="2"/><circle class="eye" cx="3.5" cy="-1.5" r="2"/>';
        let s = 0;
        const story = [env.states[0]];
        const [x0, y0] = ROBOT.states.high;
        robot.style.transform = `translate(${x0}px, ${y0 - 38}px)`;
        const turn = () => {
          const acts = env.available(s), a = acts[Math.floor(rng.next() * acts.length)], outs = env.model(s, a);
          let u = rng.next(), o = outs[outs.length - 1];
          for (const x of outs) { u -= x.p; if (u <= 0) { o = x; break; } }
          const act = `${env.states[s]}-${env.actions[a]}`;
          const first = along(robot, svg.querySelector(`.act-edge[data-act="${act}"]`), 650);
          anims.push(first);
          first.onfinish = () => {
            const second = along(robot, svg.querySelector(`.out-edge[data-out="${act}>${env.states[o.s2]}"]`), 850);
            anims.push(second);
            second.onfinish = () => {
              s = o.s2;
              story.push(`${env.actions[a]} (${signed(o.r)})`, env.states[s]);
              note.textContent = `${story.length > 9 ? "… " : ""}${story.slice(-9).join(" → ")}`;
              later(turn, 450);
            };
          };
        };
        later(turn, 400);
      }

      return {
        apply(st) {
          stop();
          anims.forEach((a) => a.cancel());
          anims = [];
          tokens.replaceChildren();
          svg.setAttribute("class", `mdp lv-${st.level ?? 4}${st.highlight ? " focus" : ""}`);
          svg.querySelectorAll(".hot").forEach((e) => e.classList.remove("hot"));
          if (st.highlight) svg.querySelectorAll(`[data-act="${st.highlight}"]`).forEach((e) => e.classList.add("hot"));
          table.classList.toggle("on", !!st.table);
          table.querySelectorAll("tr[data-act]").forEach((tr) => tr.classList.toggle("hot", tr.dataset.act === st.highlight));
          showFormula(st);
          note.textContent = st.note || "";
          if (st.walk) walk();
        },
        destroy() { stop(); anims.forEach((a) => a.cancel()); },
      };
    },
  };
})(globalThis.RL = globalThis.RL || {});
