/* RL Atlas core: the RL namespace, DOM helpers, content lookups, saved progress,
   hover cards for terms, quantity highlighting and the shared tooltip. */
(function (RL) {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  // Escaped text for an uppercase title: words with Greek letters or math signs keep their case (θ must not become Θ).
  const asIs = (s) => esc(s).replace(/\S*[\u0370-\u03ff‖√∇|]\S*/g, (w) => `<span class="as-is">${w}</span>`);
  const warn = (...args) => console.warn("[RL Atlas]", ...args);
  const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  function h(html) {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  }

  // ---- content lookups: every station in journey order, with its part and line ----
  const stations = new Map();
  const order = [];
  for (const part of RL.content.parts) {
    for (const s of part.stations) {
      const st = { ...s, part: part.n, partTitle: part.title, line: part.line, index: order.length };
      stations.set(s.id, st);
      order.push(st);
    }
  }
  const station = (id) => stations.get(id);
  const entry = (id) => RL.content.entries[id];
  const lineColor = (line) => `var(--line-${line})`;

  // ---- saved progress: per browser, and the atlas works the same without it ----
  const KEY = "rl-atlas:v1";
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY)) || {}; } catch { saved = {}; }
  const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch { /* storage unavailable */ } };
  const store = {
    get: (k, fallback) => (k in saved ? saved[k] : fallback),
    set: (k, v) => { saved[k] = v; persist(); },
    visited: (id) => !!(saved.visited && saved.visited[id]),
    visit: (id) => { (saved.visited ||= {})[id] = 1; persist(); },
  };

  // ---- hover cards: anything with data-term="<station>" explains itself on hover ----
  let card = null, cardTimer = 0, cardFor = null;
  function hideCard() {
    clearTimeout(cardTimer);
    card?.remove();
    card = null;
    cardFor = null;
  }
  function showCard(target) {
    const st = station(target.dataset.term);
    if (!st) return;
    const e = entry(st.id);
    const tip = target.dataset.tip;
    const parent = !tip && e?.change && st.parent && station(st.parent);
    card = h(`<div class="hovercard" style="--c:${lineColor(st.line)}">
      <div class="meta">${tip ? "Label" : st.kind === "algorithm" ? "Algorithm" : "Concept"} · Part ${st.part} · ${esc(st.partTitle)}</div>
      <h4>${esc(tip ? target.textContent : st.title)}</h4>
      <p>${esc(tip || (e ? e.summary : ""))}</p>
      ${parent ? `<p class="chg"><b>One change from ${esc(parent.title)}:</b> ${esc(e.change)}</p>` : ""}
      ${tip ? `<span class="soon">Explained in: ${esc(st.title)}${e ? "" : " (planned)"}</span>` : e ? "" : `<span class="soon">Planned: this station is not written yet.</span>`}
    </div>`);
    document.body.appendChild(card);
    const r = target.getBoundingClientRect(), w = card.offsetWidth, hgt = card.offsetHeight;
    const x = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), innerWidth - w - 8);
    const below = r.bottom + 10 + hgt < innerHeight;
    card.style.left = `${x}px`;
    card.style.top = `${below ? r.bottom + 10 : r.top - hgt - 10}px`;
  }
  document.addEventListener("pointerover", (ev) => {
    const t = ev.target.closest?.("[data-term]") || null;
    if (t === cardFor) return;
    hideCard();
    cardFor = t;
    if (t) cardTimer = setTimeout(() => showCard(t), 260);
  });
  document.addEventListener("pointerdown", hideCard);

  // ---- quantity highlighting: hover a colored term and every term of that kind lights up ----
  const KINDS = ["val", "rew", "pol", "err", "knob"];
  document.addEventListener("pointerover", (ev) => {
    const el = ev.target.closest?.(".q-val, .q-rew, .q-pol, .q-err, .q-knob, [data-q]");
    const kind = el && (el.dataset.q || KINDS.find((k) => el.classList.contains(`q-${k}`)));
    if (kind) document.body.dataset.hot = kind;
    else delete document.body.dataset.hot;
  });

  // ---- tooltip shared by the lab views and charts (content is built by our own code) ----
  let tipEl = null;
  const tip = {
    show(x, y, html) {
      if (!tipEl) { tipEl = h('<div class="tip"></div>'); document.body.appendChild(tipEl); }
      tipEl.innerHTML = html;
      const w = tipEl.offsetWidth, hgt = tipEl.offsetHeight;
      tipEl.style.left = `${x + 16 + w > innerWidth ? x - w - 16 : x + 16}px`;
      tipEl.style.top = `${Math.min(y + 16, innerHeight - hgt - 8)}px`;
    },
    hide() { tipEl?.remove(); tipEl = null; },
  };

  Object.assign(RL, { $, $$, h, esc, asIs, warn, reducedMotion, station, entry, stations, order, lineColor, store, hideCard, tip });
  RL.views = RL.views || {};
  RL.demos = RL.demos || {};
})(globalThis.RL = globalThis.RL || {});
