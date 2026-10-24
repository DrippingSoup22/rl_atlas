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
    const card = host.querySelector(".stage-card");
    const stage = scene.create(card, config, steps.map((s) => s.state)); // the states: a scene may prepare ahead
    let current = -1;

    // The stage stays inside the window, under the bar: when a step makes it taller than that, its picture shrinks to
    // make room, and keeps that size for the rest of the story, so that it does not change size at every step. The
    // formula counts whole even while it is closed or shows only its first rows, so the picture has its final size
    // before the formula opens. On a phone, where the stage sits above the text, it takes at most 60% of the window's
    // height, so that the text keeps the rest.
    let fitting = 0;
    // The drawings on the stage, other than the formula and the chart, largest first, with their twins (the two maps
    // of a blackjack table shrink together). The largest shrinks first; if that does not make the stage lower (a
    // drawing beside a taller one), it keeps its size and the next one shrinks.
    const drawings = () => {
      const all = [...card.querySelectorAll("svg, canvas")].filter((e) => !e.closest(".scene-formula, .scene-chart, .katex"))
        .map((e) => [e, e.getBoundingClientRect()]).filter(([, b]) => b.width && b.height).sort((a, b) => b[1].width * b[1].height - a[1].width * a[1].height);
      const groups = [];
      for (const d of all) {
        const twin = groups.find((g) => Math.abs(g[0][1].width - d[1].width) <= 2 && Math.abs(g[0][1].height - d[1].height) <= 2);
        if (twin) twin.push(d); else groups.push([d]);
      }
      return groups;
    };
    // The first step starts lower when the stage is tall, so that by the time it is read, a third of the way down the
    // window, the stage has reached its place under the bar and shows whole.
    const column = host.querySelector(".story-steps");
    function keepInView() {
      fitting = 0;
      if (!card.isConnected) return;
      const phone = innerWidth <= 900, top = parseFloat(getComputedStyle(phone ? card.parentElement : card).top) || 0;
      const closed = [...card.querySelectorAll(".f-sym, .f-num")].reduce((h, e) => h + Math.max(0, (+e.dataset.full || e.scrollHeight) - e.clientHeight), 0);
      const room = innerHeight * (phone ? 0.6 : 1) - top - 12;
      let over = card.offsetHeight + closed - room;
      for (const group of over > 1 ? drawings() : []) {
        const [pic, box] = group[0], natural = +(pic.dataset.natural || box.width);
        const width = Math.max(natural * 0.55, (box.width * (box.height - over)) / box.height); // at most to 55% of its size
        if (width >= box.width - 1) continue;
        const before = card.offsetHeight, was = group.map(([e]) => e.style.maxWidth);
        const set = (w) => group.forEach(([e], i) => { e.style.maxWidth = w ?? was[i]; e.style.marginInline = "auto"; });
        set(`${width.toFixed(0)}px`); // narrower, so lower: it keeps its shape
        if (card.offsetHeight >= before - 1) { set(null); continue; }
        group.forEach(([e]) => { e.dataset.natural ||= natural; });
        over = card.offsetHeight + closed - room;
        if (over <= 1) break;
      }
      const lower = phone ? 0 : Math.round(card.offsetHeight + closed + 12 - innerHeight * 0.65);
      column.style.paddingTop = lower > innerHeight * 0.04 ? `${lower}px` : ""; // more than the usual 4vh
    }
    const later = () => { cancelAnimationFrame(fitting); fitting = requestAnimationFrame(() => { fitting = requestAnimationFrame(keepInView); }); };
    // fitted again from its full size when the window changes, but not when a phone's address bar comes and goes
    let resized = 0, size = [innerWidth, innerHeight];
    const onResize = () => {
      clearTimeout(resized);
      resized = setTimeout(() => {
        if (innerWidth === size[0] && innerWidth <= 900 && Math.abs(innerHeight - size[1]) < 120) return;
        size = [innerWidth, innerHeight];
        for (const [e] of drawings().flat()) if (e.dataset.natural) { e.style.maxWidth = ""; delete e.dataset.natural; }
        later();
      }, 200);
    };
    addEventListener("resize", onResize);
    // a step can add to the stage after it starts (a chart, a line of numbers): fit again whenever the stage grows
    const grown = typeof ResizeObserver === "function" ? new ResizeObserver(later) : null;
    grown?.observe(card);

    function activate(i) {
      if (i === current) return;
      const forward = i > current;
      current = i;
      els.forEach((el, k) => { el.classList.toggle("active", k === i); el.classList.toggle("seen", k < i); });
      stage.apply(steps[i].state, forward);
      later();
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
    return { destroy() { removeEventListener("scroll", onScroll); removeEventListener("resize", onResize); cancelAnimationFrame(frame); cancelAnimationFrame(fitting); clearTimeout(resized); grown?.disconnect(); stage.destroy(); } };
  };
})(globalThis.RL = globalThis.RL || {});
