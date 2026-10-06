/* The shell: header, routing, zoom transitions between views, search and theme.
   Routes: #/ map · #/e/<id>[/story|/textbook|/card] entry · #/lab/<preset> lab · #/symbols symbols. */
(function (RL) {
  "use strict";
  const { $, h, esc, station, entry, store, lineColor } = RL;
  const root = document.documentElement, top = $("#top"), view = $("#view");
  const app = (RL.app = { origin: null, from: null });

  const LOGO = '<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="11" fill="none" stroke="var(--line-tabular)" stroke-width="5"/><rect x="2" y="13" width="28" height="6" rx="3" fill="var(--ink)"/></svg>';
  const SEARCH = '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="2.2"/><path d="M15.5 15.5 20 20" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>';
  const MOON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4 8.5 8.5 0 1 0 20 14.5z" fill="currentColor"/></svg>';
  const SUN = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="currentColor"/><g stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></g></svg>';

  top.innerHTML = `
    <a class="brand" href="#/" aria-label="RL Atlas: the map">${LOGO}<span>RL Atlas</span></a>
    <nav class="crumbs" aria-label="You are here"></nav>
    <span class="spacer"></span>
    <button class="pill search" type="button" aria-label="Search the atlas (Ctrl K)">${SEARCH}<span>Search</span><kbd>Ctrl K</kbd></button>
    <a class="pill" href="#/symbols">Symbols</a>
    <button class="icon-btn theme" type="button" aria-label="Switch between light and dark"></button>`;

  // ---- routes ----
  function parse() {
    const path = location.hash.replace(/^#/, "") || "/";
    let m;
    if ((m = path.match(/^\/e\/([\w-]+)(?:\/(story|textbook|card))?$/))) return { name: "entry", id: m[1], mode: m[2], depth: 1 };
    if ((m = path.match(/^\/lab\/([\w-]+)$/))) return { name: "lab", id: m[1], depth: 2 };
    if (path === "/symbols") return { name: "symbols", depth: 1 };
    return { name: "map", depth: 0 };
  }

  function crumbs(route) {
    const sep = '<span class="sep">›</span>', home = '<a href="#/">Map</a>';
    let html = "";
    if (route.name === "entry" && station(route.id)) {
      const st = station(route.id);
      html = `${home}${sep}<span>${st.part} · ${esc(st.partTitle)}</span>${sep}<span class="here">${esc(st.title)}</span>`;
    } else if (route.name === "lab") {
      const preset = RL.content.presets[route.id];
      const last = preset?.racers.at(-1).algorithm, via = app.from?.name === "entry" ? app.from.id : RL.lab?.algorithms?.[last]?.station || last;
      html = `${home}${sep}${via && station(via) ? `<a href="#/e/${via}">${esc(station(via).title)}</a>${sep}` : ""}<span class="here">Lab${preset ? ` · ${esc(preset.title)}` : ""}</span>`;
    } else if (route.name === "symbols") html = `${home}${sep}<span class="here">Symbols and colors</span>`;
    $(".crumbs", top).innerHTML = html;
  }

  let current = null, route = null, started = false;
  const same = (a, b) => a && b && a.name === b.name && a.id === b.id;
  // Mount whatever the address says when this runs: a transition can be overtaken by a newer navigation.
  function mount() {
    const next = parse();
    if (same(next, route)) return;
    RL.hideCard();
    RL.tip.hide();
    current?.destroy?.();
    app.from = route;
    route = next;
    view.replaceChildren();
    scrollTo(0, 0);
    crumbs(next);
    current = RL.views[next.name](view, next.id, next.mode) || {};
    const st = next.id && station(next.id);
    document.title = next.name === "entry" && st ? `${st.title} · RL Atlas` : next.name === "lab" ? "Lab · RL Atlas" : "RL Atlas";
  }

  // Remember where the pointer was, so the next view grows out of what was clicked.
  let pointer = null;
  addEventListener("pointerdown", (e) => { pointer = { x: e.clientX, y: e.clientY }; }, true);

  function navigate() {
    const next = parse();
    if (same(next, route)) return; // only the story/card mode changed
    const dir = !route ? "none" : next.depth > route.depth ? "in" : next.depth < route.depth ? "out" : "side";
    const o = app.origin || pointer || { x: innerWidth / 2, y: innerHeight / 2 };
    app.origin = null;
    root.style.setProperty("--vx", `${o.x}px`);
    root.style.setProperty("--vy", `${o.y}px`);
    if (!started || dir === "none" || !document.startViewTransition || RL.reducedMotion()) { mount(); return; }
    root.classList.add(`vt-${dir}`);
    const t = document.startViewTransition(mount);
    t.ready.catch(() => {}); // skipped (e.g. the page is hidden): the view still changes, just without the zoom
    t.finished.finally(() => root.classList.remove("vt-in", "vt-out", "vt-side"));
  }
  addEventListener("hashchange", navigate);

  // ---- search ----
  function openSearch() {
    if ($(".palette")) return;
    const box = h(`<div class="palette" role="dialog" aria-label="Search the atlas">
      <div class="box"><input type="search" placeholder="Search: Q-learning, discount, PPO…" aria-label="Search"><ul role="listbox"></ul></div></div>`);
    document.body.appendChild(box);
    const input = $("input", box), list = $("ul", box);
    let hits = [], on = 0;
    const close = () => box.remove();
    const go = (s) => { close(); location.hash = `#/e/${s.id}`; };
    function render() {
      const words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      hits = RL.order.filter((s) => {
        const text = `${s.title} ${s.short || ""} ${s.id} ${s.partTitle}`.toLowerCase();
        return words.every((w) => text.includes(w));
      }).slice(0, 40);
      on = Math.min(on, Math.max(0, hits.length - 1));
      list.innerHTML = hits.map((s, i) => `<li class="${entry(s.id) ? "written" : "planned"}${i === on ? " on" : ""}" data-i="${i}" role="option" style="--c:${lineColor(s.line)}">
        <span class="dot"></span><span class="title">${esc(s.title)}</span><span class="part">${entry(s.id) ? "" : "planned · "}Part ${s.part}</span></li>`).join("")
        || '<li class="planned">Nothing matches</li>';
      list.querySelector(".on")?.scrollIntoView({ block: "nearest" });
    }
    input.addEventListener("input", () => { on = 0; render(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); on = (on + (e.key === "ArrowDown" ? 1 : -1) + hits.length) % Math.max(1, hits.length); render(); }
      else if (e.key === "Enter" && hits[on]) go(hits[on]);
      else if (e.key === "Escape") close();
    });
    list.addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) go(hits[+li.dataset.i]); });
    box.addEventListener("pointerdown", (e) => { if (e.target === box) close(); });
    render();
    input.focus();
  }
  $(".search", top).addEventListener("click", openSearch);
  addEventListener("keydown", (e) => {
    const typing = e.target.closest?.("input, textarea, select");
    if ((e.key === "k" && (e.ctrlKey || e.metaKey)) || (e.key === "/" && !typing)) { e.preventDefault(); openSearch(); }
  });

  // ---- theme: follows the system until you pick one ----
  const themeBtn = $(".theme", top);
  const isDark = () => (root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
  function setTheme(t) {
    if (t) root.dataset.theme = t;
    else delete root.dataset.theme;
    themeBtn.innerHTML = isDark() ? SUN : MOON;
  }
  themeBtn.addEventListener("click", () => {
    const t = isDark() ? "light" : "dark";
    store.set("theme", t);
    setTheme(t);
  });
  setTheme(store.get("theme", null));

  navigate();
  started = true;
})(globalThis.RL = globalThis.RL || {});
