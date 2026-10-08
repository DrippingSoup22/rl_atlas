/* Stories: the text scrolls on the left while the picture on the right follows it.
   Each "::: step" of an entry carries a small state; a scene (scenes.js) turns that state into a picture.
   States are complete descriptions, so scrolling back up shows the same picture again. */
(function (RL) {
  "use strict";

  // One faint line under the last step: the seeds of the runs the story plays, so anyone can replay them in the Lab.
  // Runs trained offline are left out (their seeds are in the recordings).
  function seedLine(config) {
    const chance = (id) => RL.lab?.algorithms[id] && RL.lab.algorithms[id].unit !== "sweep"; // sweeps draw nothing at random
    const live = Object.values(config.runs || {}).filter((r) => chance(r.algorithm)).map((r) => r.seed ?? config.seed ?? 1);
    if (chance(config.algorithm)) live.push(config.seed ?? 1);
    if (!live.length) return "";
    const list = [...new Set(live)].sort((a, b) => a - b).map((s) => s.toLocaleString("en-US"));
    const named = list.length === 1 ? `seed ${list[0]}` : `seeds ${list.slice(0, -1).join(", ")} and ${list.at(-1)}`;
    return `<p class="seed-line">Played with ${named}. The same ${list.length > 1 ? "seeds" : "seed"} and settings in the Lab replay ${live.length > 1 ? "these runs" : "this run"}.</p>`;
  }

  RL.story = function (host, entry) {
    const { config, steps } = entry.story;
    host.innerHTML = `
      <div class="story">
        <div class="story-steps">
          ${steps.map((s, i) => `<section class="story-step" data-i="${i}"><span class="step-n">${i + 1}</span><div class="step-body">${s.html}</div></section>`).join("")}
          <div class="story-end">${seedLine(config)}</div>
        </div>
        <div class="story-stage"><div class="stage-card card scene-${config.scene}"></div></div>
      </div>`;
    RL.math.render(host);
    const scene = RL.scenes[config.scene];
    if (!scene) { RL.warn(`unknown story scene '${config.scene}'`); return {}; }
    const els = Array.from(host.querySelectorAll(".story-step"));
    const stage = scene.create(host.querySelector(".stage-card"), config, steps.map((s) => s.state)); // the states: a scene may prepare ahead
    let current = -1;

    function activate(i) {
      if (i === current) return;
      const forward = i > current;
      current = i;
      els.forEach((el, k) => { el.classList.toggle("active", k === i); el.classList.toggle("seen", k < i); });
      stage.apply(steps[i].state, forward);
    }

    // The active step is the one whose top has passed a line across the screen, a little above the middle (lower on
    // phones, below the picture). It is worked out again as the page scrolls, once a frame, so a fast scroll that
    // jumps over a step lands on the right one, the last included.
    const phone = matchMedia("(max-width: 900px)").matches;
    let frame = 0;
    function follow() {
      frame = 0;
      const line = innerHeight * (phone ? 0.74 : 0.475);
      let k = 0;
      for (let i = 0; i < els.length && els[i].getBoundingClientRect().top <= line; i++) k = i;
      activate(k);
    }
    const onScroll = () => { frame ||= requestAnimationFrame(follow); };
    addEventListener("scroll", onScroll, { passive: true });
    els.forEach((el, i) => el.addEventListener("click", () => activate(i)));
    activate(0);
    frame = requestAnimationFrame(follow); // a page that opens scrolled down (going back to it) starts at its step
    return { destroy() { removeEventListener("scroll", onScroll); cancelAnimationFrame(frame); stage.destroy(); } };
  };
})(globalThis.RL = globalThis.RL || {});
