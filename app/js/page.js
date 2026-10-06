/* Pages: an entry (its Story, Textbook and Card), a station that is still planned, and the symbols page. */
(function (RL) {
  "use strict";
  const { esc, station, entry, store, lineColor, order } = RL;
  // The three ways to read an entry, in the order they are meant to be read.
  const MODES = [
    { id: "story", name: "Story", sub: "the idea", next: "Get the idea from the Story" },
    { id: "textbook", name: "Textbook", sub: "the theory", next: "Study the theory in the Textbook" },
    { id: "card", name: "Card", sub: "the summary", next: "Sum it up with the Card" },
  ];

  // Bring rendered content to life: math, planned links, demos and flip-card questions.
  function wire(root) {
    RL.math.render(root);
    for (const a of root.querySelectorAll("a.term")) if (!entry(a.dataset.term)) a.classList.add("planned");
    for (const d of root.querySelectorAll(".demo[data-demo]")) {
      const demo = RL.demos[d.dataset.demo];
      if (demo) demo(d, d.dataset.arg);
      else RL.warn(`unknown demo '${d.dataset.demo}'`);
    }
    for (const q of root.querySelectorAll(".question")) {
      const flip = () => q.classList.toggle("flipped");
      q.addEventListener("click", (e) => { if (!e.target.closest("a")) flip(); });
      q.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); } });
    }
  }

  function neighbours(st) {
    const link = (s, dir) => s
      ? `<a class="nb ${dir}${entry(s.id) ? "" : " planned"}" href="#/e/${s.id}" data-term="${s.id}"><span class="faint">${dir === "prev" ? "← Previous" : "Next →"}</span><b>${esc(s.title)}</b></a>`
      : "<span></span>";
    return `<nav class="entry-nav">${link(order[st.index - 1], "prev")}${link(order[st.index + 1], "next")}</nav>`;
  }

  const badge = (st) => `<a class="part-badge" href="#/"><i>${st.part}</i>${esc(st.partTitle)}</a>`;

  // Sections that take the whole width of a card; the others pair up two by two.
  const FULL = new Set(["idea", "knobs", "check"]);

  function card(e) {
    const sources = e.sources.map((s) => `<li>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.text)}</a>` : esc(s.text)}</li>`).join("");
    // A half-width section left without a partner (the next one is full-width, or there is none) takes the whole row.
    const wide = new Set();
    let open = -1;
    e.card.forEach((s, i) => {
      if (FULL.has(s.id)) { if (open >= 0) wide.add(open); open = -1; }
      else open = open >= 0 ? -1 : i;
    });
    if (open >= 0) wide.add(open);
    return `<div class="card-grid">
      ${e.card.map((s, i) => `<section class="sec sec-${s.id}${wide.has(i) ? " wide" : ""}" style="--i:${i}"><h2>${s.title}</h2>${s.html}</section>`).join("")}
      <section class="sec sec-sources" style="--i:${e.card.length}"><h2>Sources</h2><ul>${sources}</ul></section>
    </div>`;
  }

  RL.views.entry = function (host, id, mode) {
    const st = station(id);
    if (!st) { location.replace("#/"); return {}; }
    const e = entry(id);
    if (!e) return planned(host, st);
    const firstVisit = !store.visited(id);
    store.visit(id);
    const labels = (st.labels || []).map((l) => RL.content.labels[l]).filter(Boolean)
      .map((l) => `<a class="chip" href="#/e/${l.station}" data-term="${l.station}" data-tip="${esc(l.tip)}">${esc(l.text)}</a>`).join("");
    const parent = st.parent && e.change && station(st.parent);
    const modes = MODES.filter((m) => m.id === "card" || e[m.id]);
    host.innerHTML = `
      <article class="entry" style="--c:${lineColor(st.line)}">
        <header class="hero">
          ${badge(st)}
          <h1>${esc(st.title)}</h1>
          <p class="summary">${esc(e.summary)}</p>
          <div class="labels">${labels}</div>
          ${parent ? `<p class="change"><span class="eyebrow">One change from <a class="term as-is" data-term="${parent.id}" href="#/e/${parent.id}">${esc(parent.title)}</a></span>${esc(e.change)}</p>` : ""}
          <nav class="modes">
            ${modes.length > 1 ? `<div class="seg big" role="tablist" aria-label="Ways to read this entry">
              ${modes.map((m) => `<button type="button" role="tab" data-mode="${m.id}"><b>${m.name}</b><small>${m.sub}</small></button>`).join("")}
            </div>` : ""}
            ${e.lab ? `<a class="btn lab-btn" href="#/lab/${e.lab}">Open the Lab ▸</a>` : ""}
          </nav>
          ${!e.story && e.storyIn?.length ? `<p class="story-in">See this idea at work in the ${e.storyIn.length > 1 ? "stories" : "story"} of ${e.storyIn.map((s) => `<a class="term" data-term="${s}" href="#/e/${s}">${esc(station(s).title)}</a>`).join(e.storyIn.length > 2 ? ", " : " and ").replace(/, ([^,]*)$/, " and $1")}.</p>` : ""}
        </header>
        <div class="mode-body"></div>
        ${neighbours(st)}
      </article>`;
    wire(host.querySelector(".hero"));
    wire(host.querySelector(".entry-nav"));

    // Coming back from the Lab: the Lab shrinks back into its button.
    const labBtn = host.querySelector(".lab-btn");
    if (RL.app?.from?.name === "lab" && labBtn) {
      const r = labBtn.getBoundingClientRect();
      document.documentElement.style.setProperty("--vx", `${r.left + r.width / 2}px`);
      document.documentElement.style.setProperty("--vy", `${r.top + r.height / 2}px`);
    }

    const body = host.querySelector(".mode-body");
    let current = null;
    // Each way of reading ends by pointing to the next one, and the last one to the Lab.
    function onward(m) {
      const next = modes[modes.findIndex((k) => k.id === m) + 1];
      if (next) return `<div class="onward"><span class="faint">Next</span><button class="btn ghost" type="button" data-mode="${next.id}">${next.next} ▸</button></div>`;
      return e.lab ? `<div class="onward"><span class="faint">Next</span><a class="btn" href="#/lab/${e.lab}">Watch it learn in the Lab ▸</a></div>` : "";
    }
    function show(m) {
      if (!modes.some((k) => k.id === m)) m = modes[0].id;
      current?.destroy?.();
      current = null;
      host.querySelectorAll(".seg [data-mode]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
      body.className = `mode-body mode-${m}`;
      if (m === "story") current = RL.story(body, e);
      else if (m === "textbook") { current = RL.textbook(body, e); wire(body); }
      else { body.innerHTML = card(e); wire(body); }
      body.insertAdjacentHTML("beforeend", onward(m));
      if (location.hash.startsWith(`#/e/${id}`)) history.replaceState(null, "", `#/e/${id}/${m}`);
      store.set("mode", m);
    }
    // The host outlives this page (the next view mounts into it), so the listener goes when the page does.
    function onClick(ev) {
      const b = ev.target.closest("button[data-mode]");
      if (!b || b.getAttribute("aria-selected") === "true") return;
      show(b.dataset.mode);
      if (body.getBoundingClientRect().top < 0) body.scrollIntoView({ behavior: RL.reducedMotion() ? "auto" : "smooth" });
    }
    host.addEventListener("click", onClick);
    show(mode || (firstVisit ? "story" : store.get("mode", "story"))); // a first visit starts with the first way of reading
    return { destroy() { host.removeEventListener("click", onClick); current?.destroy?.(); } };
  };

  function planned(host, st) {
    host.innerHTML = `
      <article class="entry planned-entry" style="--c:${lineColor(st.line)}">
        <header class="hero">
          ${badge(st)}
          <h1>${esc(st.title)}</h1>
          <p class="summary">This station is on the map but not written yet. It will be filled in as the atlas grows.</p>
          <a class="btn ghost" href="#/">Back to the map</a>
        </header>
        ${neighbours(st)}
      </article>`;
    wire(host);
    return {};
  }

  RL.views.symbols = function (host) {
    const N = RL.content.notation;
    host.innerHTML = `
      <article class="entry symbols">
        <header class="hero">
          <div class="eyebrow">Symbols and colors</div>
          <h1>Reading the formulas</h1>
          <p class="summary">Every formula in the atlas uses the same colors. Hover any colored symbol, here or on any page, and every symbol of the same kind lights up.</p>
        </header>
        <section class="quantities">
          ${N.quantity.map((q) => `<div class="qty card" data-q="${q.key}"><div class="sym"><span class="tex tex-display">${esc(q.tex)}</span></div><h3>${esc(q.name)}</h3><p>${esc(q.text)}</p></div>`).join("")}
        </section>
        <section class="notation card">
          <h2>Notation</h2>
          <table>${N.symbol.map((s) => `<tr><td class="sym"><span class="tex">${esc(s.tex)}</span></td><td>${esc(s.text)}</td>
            <td>${s.station ? `<a class="term" data-term="${s.station}" href="#/e/${s.station}">${esc(station(s.station).title)}</a>` : ""}</td></tr>`).join("")}</table>
        </section>
      </article>`;
    wire(host);
    return {};
  };
})(globalThis.RL = globalThis.RL || {});
