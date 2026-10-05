/* Demo "be the agent": you play an unknown world with two buttons before any algorithm does. */
(function (RL) {
  "use strict";

  // ---- be the agent: five rooms in a row, two buttons, and no explanation ----
  // One button moves you on to the next room and pays nothing, except in the last room, which pays +10 and keeps you there.
  // The other sends you back to the first room and pays +2. One press in ten does the opposite.
  // (A small version of the "chain" problems used to test exploration.)
  const ROOMS = ["●", "▲", "■", "◆", "★"];
  const PRESSES = 10, EPISODES = 3, SLIP = 0.1;
  const outcome = (room, forward) => (forward ? (room === 4 ? [4, 10] : [room + 1, 0]) : [0, 2]);

  // Expected return over h presses from the first room: playing as well as possible (choose = null), or with a fixed rule.
  function expected(h, choose = null) {
    let V = [0, 0, 0, 0, 0];
    for (let k = 0; k < h; k++) {
      V = V.map((_, s) => {
        const value = (f) => {
          const [s1, r1] = outcome(s, f), [s2, r2] = outcome(s, !f);
          return (1 - SLIP) * (r1 + V[s1]) + SLIP * (r2 + V[s2]);
        };
        return choose ? value(choose(s)) : Math.max(value(true), value(false));
      });
    }
    return V[0];
  }

  function chainSvg() {
    const x = (i) => 46 + i * 112, y = 66;
    let g = `<defs><marker id="bta-tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>`;
    for (let i = 0; i < 4; i++) {
      g += `<path class="fwd" marker-end="url(#bta-tip)" d="M${x(i) + 25} ${y} H${x(i + 1) - 27}"/><text class="lbl" x="${(x(i) + x(i + 1)) / 2}" y="${y - 9}" text-anchor="middle">0</text>`;
    }
    g += `<path class="fwd" marker-end="url(#bta-tip)" d="M${x(4) + 12} ${y - 21} C${x(4) + 30} ${y - 60} ${x(4) - 30} ${y - 60} ${x(4) - 12} ${y - 22}"/><text class="lbl good" x="${x(4) + 28}" y="${y - 42}">+10</text>`;
    for (let i = 4; i >= 1; i--) {
      g += `<path class="back" marker-end="url(#bta-tip)" d="M${x(i) - 6} ${y + 24} Q${(x(i) + x(0)) / 2} ${y + 34 + i * 13} ${x(0) + 8} ${y + 24}"/>`;
    }
    g += `<path class="back" marker-end="url(#bta-tip)" d="M${x(0) - 14} ${y + 18} C${x(0) - 40} ${y + 52} ${x(0) + 8} ${y + 60} ${x(0) - 2} ${y + 25}"/>`;
    g += `<text class="lbl" x="${(x(0) + x(4)) / 2}" y="${y + 104}" text-anchor="middle">every “back” press: +2, and back to ●</text>`;
    ROOMS.forEach((r, i) => { g += `<circle class="room" cx="${x(i)}" cy="${y}" r="23"/><text class="sym" x="${x(i)}" y="${y + 7}" text-anchor="middle">${r}</text>`; });
    return `<svg class="bta-chain" viewBox="0 0 560 182" role="img" aria-label="The hidden world: five rooms in a row">${g}</svg>`;
  }

  RL.demos["be-the-agent"] = function (host) {
    host.innerHTML = `<div class="bta">
      <div class="bta-head"><span class="bta-where"></span><span class="bta-return"></span></div>
      <div class="bta-stage"><div class="bta-room" aria-live="polite"></div><div class="bta-pop"></div></div>
      <div class="bta-buttons"><button class="btn" type="button" data-b="0">Button 1</button><button class="btn" type="button" data-b="1">Button 2</button></div>
      <ol class="bta-log" aria-label="This episode so far"></ol>
      <div class="bta-end"></div>
    </div>`;
    const box = host.querySelector(".bta"), q = (s) => box.querySelector(s);
    let forwardButton, room, episode, press, G, found, returns, stats, reached;

    function newGame() {
      forwardButton = Math.random() < 0.5 ? 0 : 1;
      returns = [];
      stats = [{ n: 0, sum: 0 }, { n: 0, sum: 0 }];
      reached = 0;
      episode = 0;
      newEpisode();
    }
    function newEpisode() {
      episode += 1;
      room = 0;
      press = 0;
      G = 0;
      found = false;
      q(".bta-log").replaceChildren();
      q(".bta-end").replaceChildren();
      box.classList.remove("over", "revealed");
      show(false);
    }
    function show(moved) {
      q(".bta-where").textContent = `Episode ${episode} of ${EPISODES} · press ${Math.min(press + 1, PRESSES)} of ${PRESSES}`;
      q(".bta-return").innerHTML = `Return so far <b>${G}</b>`;
      q(".bta-room").innerHTML = `<span>${ROOMS[room]}</span>`;
      if (moved && !RL.reducedMotion()) {
        q(".bta-room span").animate([{ transform: "scale(0.4) rotate(-25deg)", opacity: 0 }, { transform: "none", opacity: 1 }], { duration: 380, easing: "cubic-bezier(.2,.9,.25,1.25)" });
      }
    }
    function pop(r) {
      const p = q(".bta-pop");
      p.textContent = r ? `+${r}` : "0";
      p.classList.toggle("good", r > 0);
      if (!RL.reducedMotion()) p.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(-26px)" }], { duration: 1000, easing: "ease-out" });
    }
    function act(b) {
      if (press >= PRESSES) return;
      const forward = (b === forwardButton) !== (Math.random() < SLIP);
      const [next, r] = outcome(room, forward);
      q(".bta-log").insertAdjacentHTML("beforeend", `<li><span class="r">${ROOMS[room]}</span><span class="b">${b + 1}</span><span class="w${r ? " good" : ""}">${r ? `+${r}` : "0"}</span></li>`);
      stats[b].n += 1;
      stats[b].sum += r;
      found ||= next === 4;
      room = next;
      press += 1;
      G += r;
      pop(r);
      show(true);
      if (press === PRESSES) end();
    }
    function end() {
      returns.push(G);
      if (found) reached += 1;
      box.classList.add("over");
      const last = episode === EPISODES;
      q(".bta-end").innerHTML = `<p>Episode ${episode} is over. Its <b>return</b>, the sum of its rewards, is <b>${G}</b>.</p>
        <button class="btn" type="button" data-next>${last ? "Reveal the world ▸" : "Next episode ▸"}</button>`;
    }
    function reveal() {
      box.classList.add("revealed");
      const fb = forwardButton + 1, bb = 2 - forwardButton;
      const avg = (s) => (s.n ? (s.sum / s.n).toFixed(1) : "–");
      const best = expected(PRESSES), back = expected(PRESSES, () => false);
      q(".bta-end").innerHTML = `<div class="bta-reveal">
        ${chainSvg()}
        <p>Button ${fb} moved you <b>forward</b>, one room on, for nothing, until the last room, which pays +10 at every press.
          Button ${bb} sent you <b>back</b> to ● with +2. One press in ten did the opposite.</p>
        <p>Your returns: <b>${returns.join(" · ")}</b>. You reached ★ in ${reached} of ${EPISODES} episodes.
          Pressing back every time earns about ${back.toFixed(0)} per episode; the best way to play, forward all the way and then stay, earns about ${best.toFixed(0)}.</p>
        <p>Per press, button 1 paid you ${avg(stats[0])} on average and button 2 paid ${avg(stats[1])}. Judged by the next reward alone,
          going back looks better. Judged by where it leads, forward is far better. Telling the two apart is the whole job of reinforcement learning.</p>
        <button class="btn ghost" type="button" data-again>Play again in a new world ▸</button>
      </div>`;
    }
    box.addEventListener("click", (e) => {
      const b = e.target.closest("[data-b]");
      if (b) return act(+b.dataset.b);
      if (e.target.closest("[data-next]")) return episode === EPISODES ? reveal() : newEpisode();
      if (e.target.closest("[data-again]")) newGame();
    });
    newGame();
  };
})(globalThis.RL = globalThis.RL || {});
