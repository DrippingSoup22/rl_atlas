/* Pages: an entry (its Story, Textbook and Card), a station that is still planned, the symbols page, and the appendix
   (an entry of it, and the list of them all). */
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
    for (const a of root.querySelectorAll("a.term:not(.appx)")) if (!entry(a.dataset.term)) a.classList.add("planned");
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

  // Sections that take the whole width of a card: the idea, the formulas (they need the room), the knobs and the
  // questions.
  const FULL = new Set(["idea", "formula", "knobs", "check"]);

  function card(e) {
    const sources = e.sources.map((s) => `<li>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.text)}</a>` : esc(s.text)}</li>`).join("");
    const section = (s, i) => `<section class="sec sec-${s.id}" style="--i:${i}"><h2>${s.title}</h2>${s.html}</section>`;
    // The other sections, between two full-width ones, flow into two balanced columns: each keeps its own height,
    // instead of being stretched to its neighbour's and left half empty. One alone takes the whole row.
    let html = "", run = [];
    const flush = () => {
      if (run.length) html += run.length === 1 ? run[0] : `<div class="card-flow">${run.join("")}</div>`;
      run = [];
    };
    e.card.forEach((s, i) => {
      if (FULL.has(s.id)) { flush(); html += section(s, i); }
      else run.push(section(s, i));
    });
    flush();
    return `<div class="card-grid">${html}
      <section class="sec sec-sources" style="--i:${e.card.length}"><h2>Sources</h2><ul>${sources}</ul></section>
    </div>`;
  }

  // A card's formulas written as pairs, a line of text and then its formula, become a sheet: the words on the left,
  // each formula on the right of its own words.
  function sheets(root) {
    for (const sec of root.querySelectorAll(".sec-formula")) {
      const kids = [...sec.children].slice(1);
      if (kids.length >= 2 && kids.length % 2 === 0 && kids.every((k, i) => k.matches(i % 2 ? ".tex-display" : "p"))) sec.classList.add("sheet");
    }
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
      else { body.innerHTML = card(e); wire(body); sheets(body); }
      body.insertAdjacentHTML("beforeend", onward(m));
      if (location.hash.startsWith(`#/e/${id}`)) history.replaceState(history.state, "", `#/e/${id}/${m}`); // the state keeps the scroll
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
        ${appendixInvite()}
      </article>`;
    wire(host);
    return {};
  };

  // ---- the appendix: the mathematics the atlas uses, explained apart from reinforcement learning ----
  const APPX = () => RL.content.appendix || {};
  const appxLink = (id) => `<a class="term appx" data-appendix="${id}" href="#/a/${id}">${esc(APPX()[id].title)}</a>`;
  const appendixInvite = () => Object.keys(APPX()).length ? `<a class="appx-invite card" href="#/appendix">
      <span class="eyebrow">Appendix</span><b>The mathematics behind the formulas</b>
      <span>Sums, expectations, norms, fixed points and the other tools the atlas uses, each explained on its own page.</span></a>` : "";
  // Where "back" goes: the page this one was opened from, if it was opened from inside the atlas.
  function backLink() {
    const from = RL.app?.from;
    if (!from || from.name === "map") return "";
    const name = from.name === "entry" ? station(from.id)?.title : from.name === "appendix" && from.id ? APPX()[from.id]?.title
      : from.name === "appendix" ? "the appendix" : from.name === "symbols" ? "Symbols and colors" : from.name === "lab" ? "the Lab" : "";
    return name ? `<button class="btn ghost back" type="button">← Back to ${esc(name)}</button>` : "";
  }

  RL.views.appendix = function (host, id) {
    if (id) return appendixEntry(host, id);
    const all = APPX();
    const groups = RL.content.appendixGroups.map((g) => [g, Object.keys(all).filter((k) => all[k].group === g)
      .sort((a, b) => all[a].title.localeCompare(all[b].title))]).filter(([, ids]) => ids.length);
    host.innerHTML = `
      <article class="entry appendix-index" style="--c:var(--ink-3)">
        <header class="hero">
          <div class="eyebrow">Appendix</div>
          <h1>The mathematics behind the formulas</h1>
          <p class="summary">Short explanations of the mathematics the atlas leans on, none of it about reinforcement learning. Each one is linked from the pages that use it, and you can read them in any order.</p>
          ${backLink()}
        </header>
        <div class="appx-groups">
          ${groups.map(([g, ids], i) => `<section class="appx-group card" style="--i:${i}"><h2>${esc(g)}</h2><ul>
            ${ids.map((k) => `<li><a href="#/a/${k}" data-appendix="${k}"><b>${esc(all[k].title)}</b><span>${esc(all[k].summary)}</span></a></li>`).join("")}
          </ul></section>`).join("")}
        </div>
      </article>`;
    host.querySelector(".back")?.addEventListener("click", () => history.back());
    return {};
  };

  function appendixEntry(host, id) {
    const a = APPX()[id];
    if (!a) { location.replace("#/appendix"); return {}; }
    const sources = a.sources.map((s) => `<li>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.text)}</a>` : esc(s.text)}</li>`).join("");
    const used = a.usedIn.filter(station).map((s) => `<li><a class="term" data-term="${s}" href="#/e/${s}">${esc(station(s).title)}</a></li>`).join("");
    const related = a.related.filter((r) => APPX()[r]).map((r) => `<li>${appxLink(r)}</li>`).join("");
    host.innerHTML = `
      <article class="entry appendix-page" style="--c:var(--ink-3)">
        <header class="hero">
          <div class="eyebrow"><a href="#/appendix">Appendix</a> · ${esc(a.group)}</div>
          <h1>${esc(a.title)}</h1>
          <p class="summary">${esc(a.summary)}</p>
          ${backLink()}
        </header>
        <div class="appx-layout">
          <article class="tb-paper appx-paper">
            ${a.sections.map((s) => `<section class="tb-sec" data-anchor="${s.id}">${s.title ? `<h2>${s.title}</h2>` : ""}${s.html}</section>`).join("")}
            ${sources ? `<section class="tb-sec tb-refs"><h2>Further reading</h2><ol>${sources}</ol></section>` : ""}
          </article>
          <aside class="appx-side">
            ${used ? `<div><div class="eyebrow">Used in</div><ul>${used}</ul></div>` : ""}
            ${related ? `<div><div class="eyebrow">See also</div><ul>${related}</ul></div>` : ""}
            <div><a class="all" href="#/appendix">All the appendix ›</a></div>
          </aside>
        </div>
      </article>`;
    wire(host);
    host.querySelector(".back")?.addEventListener("click", () => history.back());
    const refs = RL.refs(host.querySelector(".appx-paper"));
    return { destroy: refs.destroy };
  }
})(globalThis.RL = globalThis.RL || {});
