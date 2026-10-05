/* The home screen: the whole curriculum, seen three ways.
   Metro map: each part is a branch hanging from its family's line. Family tree: every algorithm under the one it changes.
   Unified view: the tabular methods, placed by how deep and how wide their updates are (after Sutton & Barto, Figure 8.11).
   Stations glide from one view to the next; the filters dim what does not match.
   Drag to pan, scroll to move, Ctrl + scroll to zoom, click a station to dive in. */
(function (RL) {
  "use strict";
  const { esc, station, entry, store, lineColor } = RL;
  const COL = 178, STEP = 27, FIRST = 50, LEFT = 40, TOP = 96, ROW_GAP = 132;
  const LENSES = {
    map: { name: "Metro map", caption: "Every station is one idea. Follow the lines from the top left: first the problem, then methods that keep a table, then methods that scale." },
    tree: { name: "Family tree", caption: "Every algorithm is the one it hangs from, plus one change. Hover a written station to read the change." },
    unified: { name: "Unified view", caption: "The tabular methods, by how far an update looks ahead (downward) and whether it samples one outcome or averages over all of them (across)." },
  };
  const cameras = {}; // one camera per view, kept between visits
  let drawn = false; // the lines draw themselves only the first time
  const filters = new Set();

  // ---- layouts: where every station sits in each view ----
  function metroLayout() {
    const C = RL.content, pos = new Map(), rows = [];
    let y = TOP;
    C.rows.forEach((nums, ri) => {
      const parts = nums.map((n) => C.parts.find((p) => p.n === n));
      parts.forEach((p, ci) => {
        const x = LEFT + 30 + ci * COL;
        pos.set(`part-${p.n}`, { x, y });
        p.stations.forEach((s, k) => pos.set(s.id, { x, y: y + FIRST + k * STEP }));
      });
      const bottom = y + FIRST + (Math.max(...parts.map((p) => p.stations.length)) - 1) * STEP;
      rows.push({ title: C.rowTitles[ri], y, parts, bottom });
      y = bottom + ROW_GAP;
    });
    const width = LEFT + 30 + Math.max(...C.rows.map((r) => r.length)) * COL;
    return { pos, rows, width, height: rows[rows.length - 1].bottom + 30 };
  }

  // A tidy tree, growing to the right: leaves one row apart, each parent level with the middle of its children.
  function treeLayout() {
    const algos = RL.order.filter((s) => s.kind === "algorithm"), kids = new Map(algos.map((s) => [s.id, []])), roots = [];
    for (const s of algos) (s.parent ? kids.get(s.parent) : roots).push(s.id);
    const pos = new Map(), DX = 150, DY = 30, X = 70, Y = 60;
    let rowsUsed = 0, depthMax = 0;
    const place = (id, depth) => {
      depthMax = Math.max(depthMax, depth);
      const c = kids.get(id);
      const y = c.length ? (() => { const ys = c.map((k) => place(k, depth + 1)); return (ys[0] + ys[ys.length - 1]) / 2; })() : Y + rowsUsed++ * DY;
      pos.set(id, { x: X + depth * DX, y });
      return y;
    };
    roots.forEach((r, i) => { rowsUsed += i ? 1 : 0; place(r, 0); });
    return { pos, kids, width: X + depthMax * DX + 150, height: Y + rowsUsed * DY };
  }

  function unifiedLayout() {
    const X = 110, Y = 110, PW = 900, PH = 560;
    const pos = new Map(Object.entries(RL.content.unified).map(([id, [w, d]]) => [id, { x: X + w * PW, y: Y + d * PH }]));
    return { pos, plane: { X, Y, PW, PH }, width: X + PW + 150, height: Y + PH + 60 };
  }

  // ---- drawings that belong to one view ----
  function metroDrawing(L) {
    const C = RL.content;
    let svg = "";
    for (const row of L.rows) svg += `<text class="row-title" x="${LEFT}" y="${row.y - 56}">${esc(row.title)}</text>`;
    // The trunk of each row: same family = its color, a change of family = a thin transfer.
    for (const row of L.rows) {
      row.parts.slice(0, -1).forEach((p, i) => {
        const a = L.pos.get(`part-${p.n}`), q = row.parts[i + 1], b = L.pos.get(`part-${q.n}`);
        const same = q.line === p.line;
        svg += `<line class="trunk${same ? "" : " transfer"}" pathLength="1" style="--c:${lineColor(same ? p.line : "next")};--d:${q.n}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`;
      });
    }
    // From the end of the first row to the start of the second: the journey continues.
    if (L.rows.length > 1) {
      const [r1, r2] = L.rows;
      const a = L.pos.get(`part-${r1.parts[r1.parts.length - 1].n}`), b = L.pos.get(`part-${r2.parts[0].n}`);
      const right = L.width - 8, mid = r1.bottom + 58;
      svg += `<path class="ribbon" d="M${a.x + 18} ${a.y} H${right - 24} Q${right} ${a.y} ${right} ${a.y + 24} V${mid - 24} Q${right} ${mid} ${right - 24} ${mid} H${b.x + 24} Q${b.x} ${mid} ${b.x} ${mid + 24} V${b.y - 14}"/>`;
      svg += `<text class="ribbon-label" x="${right - 30}" y="${mid - 9}" text-anchor="end">When tables are not enough, the journey continues</text>`;
    }
    for (const p of C.parts) {
      const hub = L.pos.get(`part-${p.n}`), last = L.pos.get(p.stations[p.stations.length - 1].id);
      svg += `<line class="branch" pathLength="1" style="--c:${lineColor(p.line)};--d:${p.n}" x1="${hub.x}" y1="${hub.y}" x2="${last.x}" y2="${last.y}"/>`;
      svg += `<g class="hub" style="--c:${lineColor(p.line)};--d:${p.n}" transform="translate(${hub.x} ${hub.y})"><rect x="-15" y="-11" width="30" height="22" rx="11"/><text y="4.5">${p.n}</text></g>`;
      svg += `<text class="part-title" style="--d:${p.n}" x="${hub.x - 15}" y="${hub.y - 21}">${esc(p.title)}</text>`;
    }
    return svg;
  }

  function treeDrawing(T) {
    let svg = "";
    for (const [id, kids] of T.kids) {
      const a = T.pos.get(id);
      for (const k of kids) {
        const b = T.pos.get(k), mx = (a.x + b.x) / 2;
        svg += `<path class="edge" pathLength="1" data-from="${id}" data-to="${k}" style="--c:${lineColor(station(k).line)}" d="M${a.x + 9} ${a.y} C${mx} ${a.y} ${mx} ${b.y} ${b.x - 9} ${b.y}"/>`;
      }
    }
    return svg;
  }

  function unifiedDrawing(U) {
    const { X, Y, PW, PH } = U.plane;
    return `<rect class="plane" x="${X - 24}" y="${Y - 10}" width="${PW + 170}" height="${PH + 24}" rx="18"/>
      <text class="axis-title" x="${X + PW / 2}" y="${Y - 70}" text-anchor="middle">Width of update</text>
      <text class="axis-end" x="${X - 24}" y="${Y - 48}">← sample updates: one outcome at a time</text>
      <text class="axis-end" x="${X + PW + 146}" y="${Y - 48}" text-anchor="end">expected updates: every outcome, weighted →</text>
      <text class="axis-title" transform="translate(${X - 58} ${Y + PH / 2}) rotate(-90)" text-anchor="middle">Depth of update</text>
      <text class="axis-end" transform="translate(${X - 40} ${Y}) rotate(-90)" text-anchor="end">one step: bootstrapping</text>
      <text class="axis-end" transform="translate(${X - 40} ${Y + PH}) rotate(-90)">the full return</text>
      <text class="corner" x="${X - 6}" y="${Y - 20}">Temporal-difference learning</text>
      <text class="corner" x="${X + PW + 140}" y="${Y - 20}" text-anchor="end">Dynamic programming</text>
      <text class="corner" x="${X - 6}" y="${Y + PH + 40}">Monte Carlo</text>
      <text class="corner" x="${X + PW + 140}" y="${Y + PH + 40}" text-anchor="end">Exhaustive search</text>
      <text class="region" x="${X + 0.22 * PW}" y="${Y + 0.41 * PH}">n-step methods: n rewards, then a guess</text>
      <text class="region" x="${X + 0.22 * PW}" y="${Y + 0.63 * PH}">eligibility traces: a blend of every depth</text>
      <text class="credit" x="${X + PW + 140}" y="${Y + PH + 58}" text-anchor="end">after Sutton &amp; Barto, Figure 8.11</text>`;
  }

  function stations(next) {
    let svg = "";
    for (const p of RL.content.parts) {
      for (const s of p.stations) {
        const st = station(s.id);
        const state = !entry(s.id) ? "planned" : store.visited(s.id) ? "visited" : "written";
        svg += `<g class="st ${st.kind} ${state}${next && next.id === s.id ? " next" : ""}" data-term="${s.id}" data-id="${s.id}" tabindex="0" role="link" aria-label="${esc(st.title)}"
            style="--c:${lineColor(p.line)};--d:${p.n}">
          <circle class="halo" r="15"/><circle class="dot" r="${st.kind === "algorithm" ? 7.5 : 6}"/>
          <text x="${st.kind === "algorithm" ? 15 : 13}" y="4.5">${esc(s.short || s.title)}</text></g>`;
      }
    }
    return svg;
  }

  RL.views.map = function (host) {
    const C = RL.content;
    const layouts = { map: metroLayout(), tree: treeLayout(), unified: unifiedLayout() };
    let lens = LENSES[store.get("lens")] ? store.get("lens") : "map";
    const next = RL.order.find((s) => entry(s.id) && !store.visited(s.id)) || RL.order.find((s) => entry(s.id));
    const algorithms = RL.order.filter((s) => s.kind === "algorithm");
    const count = (line) => {
      const ids = C.parts.filter((p) => !line || p.line === line).flatMap((p) => p.stations.map((s) => s.id));
      return `${ids.filter((id) => entry(id)).length}/${ids.length}`;
    };
    host.innerHTML = `
      <section class="map-view">
        <div class="map-banner">
          <div class="mb-row">
            <div class="seg lenses" role="tablist" aria-label="How to look at the atlas">
              ${Object.entries(LENSES).map(([id, l]) => `<button type="button" role="tab" data-lens="${id}">${l.name}</button>`).join("")}
            </div>
            <span class="muted lens-caption"></span>
            ${next ? `<a class="btn" href="#/e/${next.id}">${store.visited(next.id) ? "Continue with" : "Start with"} ${esc(next.title)} ▸</a>` : ""}
          </div>
          <div class="mb-row map-filters">
            <span class="eyebrow">Show only</span>
            <div class="mf-chips">${Object.entries(C.labels).map(([id, l]) => `<button class="chip" type="button" data-label="${id}" aria-pressed="false" data-tip="${esc(l.tip)}" data-term="${l.station}">${esc(l.text)}</button>`).join("")}</div>
            <span class="mf-count faint"></span><button class="mf-clear" type="button">clear</button>
          </div>
        </div>
        <svg class="metro${drawn ? "" : " intro"}" aria-label="Map of the atlas"><g class="cam">
          <g class="deco deco-map">${metroDrawing(layouts.map)}</g>
          <g class="deco deco-tree">${treeDrawing(layouts.tree)}</g>
          <g class="deco deco-unified">${unifiedDrawing(layouts.unified)}</g>
          <g class="links"></g>
          ${stations(next)}
        </g></svg>
        <div class="map-legend card">
          <div class="map-zoom">
            <button class="icon-btn" type="button" data-zoom="in" aria-label="Zoom in">+</button>
            <button class="icon-btn" type="button" data-zoom="out" aria-label="Zoom out">−</button>
            <button class="icon-btn" type="button" data-zoom="fit" aria-label="Fit the map to the screen">⤢</button>
            <span class="faint">Drag or scroll to move<br>Ctrl + scroll to zoom</span>
          </div>
          ${Object.entries(C.lines).map(([id, name]) => `<div class="lg" style="--c:${lineColor(id)}"><i></i><span>${esc(name)}</span><b>${count(id)}</b></div>`).join("")}
          <div class="lg-note"><span class="dot-written"></span>written <span class="dot-planned"></span>planned · ${count()} written</div>
        </div>
      </section>`;
    drawn = true;

    const svg = host.querySelector(".metro"), cam = host.querySelector(".cam"), links = host.querySelector(".links");
    const nodes = new Map(Array.from(svg.querySelectorAll(".st"), (g) => [g.dataset.id, g]));
    const L = () => layouts[lens];
    let view = null, raf = 0;
    const apply = () => cam.setAttribute("transform", `translate(${view.x.toFixed(1)} ${view.y.toFixed(1)}) scale(${view.k.toFixed(4)})`);
    const keep = () => { cameras[lens] = { ...view }; };

    // Put every station where this view wants it; stations the view leaves out fade away where they are.
    function arrange() {
      const pos = L().pos;
      for (const [id, g] of nodes) {
        const at = pos.get(id);
        g.classList.toggle("out", !at);
        if (at) g.style.transform = `translate(${at.x}px, ${at.y}px)`;
      }
      // In the family tree the names sit above the dots, so the branches can leave from the dots.
      for (const g of nodes.values()) {
        const text = g.querySelector("text");
        text.style.transform = lens === "tree" ? `translate(${-(+text.getAttribute("x")) - text.getComputedTextLength() / 2}px, -15px)` : "";
      }
      svg.dataset.lens = lens;
      host.querySelector(".map-view").dataset.lens = lens;
      host.querySelector(".lens-caption").textContent = LENSES[lens].caption;
      host.querySelectorAll("[data-lens]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.lens === lens)));
    }
    // The whole view when it stays readable; on small screens fit the width instead (the rest is a scroll away).
    function fitted() {
      const W = svg.clientWidth, H = svg.clientHeight, { width, height } = L();
      const top = host.querySelector(".map-banner").offsetHeight + 20; // the banner floats over the top of the map
      if (!W || H <= top) return null;
      const whole = Math.min((W - 24) / width, (H - top - 16) / height);
      const k = Math.max(0.3, Math.min(1.25, whole >= 0.8 ? whole : (W - 24) / width));
      return { k, x: (W - width * k) / 2, y: top + (whole >= 0.8 ? (H - top - height * k) / 2 : 6) };
    }
    function fit() {
      const v = fitted();
      if (!v) return;
      view = v;
      apply();
      keep();
    }
    const resized = new ResizeObserver(() => { if (!cameras[lens]?.moved) fit(); });
    resized.observe(svg);
    function zoomAt(px, py, f) {
      const k = Math.max(0.3, Math.min(3, view.k * f));
      view = { k, x: px - ((px - view.x) * k) / view.k, y: py - ((py - view.y) * k) / view.k, moved: true };
      apply();
      keep();
    }
    function tween(to, ms, done) {
      cancelAnimationFrame(raf);
      const from = { ...view }, t0 = performance.now();
      const ease = (t) => 1 - (1 - t) ** 3;
      const frame = (t) => {
        const u = Math.min(1, (t - t0) / ms), e = ease(u);
        view = { ...to, k: from.k + (to.k - from.k) * e, x: from.x + (to.x - from.x) * e, y: from.y + (to.y - from.y) * e };
        apply();
        if (u < 1) raf = requestAnimationFrame(frame);
        else done?.();
      };
      raf = requestAnimationFrame(frame);
    }
    function screenOf(id) {
      const at = L().pos.get(id), r = svg.getBoundingClientRect();
      return { x: r.left + view.x + at.x * view.k, y: r.top + view.y + at.y * view.k };
    }
    // Clicking a station flies the camera into it, and the page grows out of it.
    function dive(id) {
      const at = L().pos.get(id), W = svg.clientWidth, H = svg.clientHeight, k = Math.max(view.k * 1.8, 1.6);
      if (!at || RL.reducedMotion()) { location.hash = `#/e/${id}`; return; }
      let gone = false;
      const go = () => {
        if (gone) return;
        gone = true;
        const r = svg.getBoundingClientRect();
        RL.app.origin = { x: r.left + W / 2, y: r.top + H / 2 };
        location.hash = `#/e/${id}`;
      };
      tween({ k, x: W / 2 - at.x * k, y: H / 2 - at.y * k }, 300, go);
      setTimeout(go, 450); // in case animation frames are paused
    }
    function switchTo(next) {
      if (next === lens) return;
      lens = next;
      store.set("lens", lens);
      clear();
      arrange();
      svg.classList.remove("redraw");
      void svg.getBoundingClientRect();
      svg.classList.add("redraw"); // the family tree's branches draw themselves again
      const to = cameras[lens] || fitted();
      if (to) tween(to, 650, keep);
    }

    arrange();
    view = cameras[lens] ? { ...cameras[lens] } : null;
    if (view) apply();
    else fit();
    // Coming back from a page: that page shrinks back into its station, which pulses once.
    const back = RL.app?.from?.name === "entry" && L().pos.has(RL.app.from.id) ? RL.app.from.id : null;
    if (back && view) {
      const p = screenOf(back);
      document.documentElement.style.setProperty("--vx", `${p.x}px`);
      document.documentElement.style.setProperty("--vy", `${p.y}px`);
      nodes.get(back)?.classList.add("arrived");
    }

    // ---- filters: dim every station that has none of the chosen labels ----
    function filter() {
      const chosen = [...filters];
      let matches = 0;
      for (const [id, g] of nodes) {
        const st = station(id), ok = !chosen.length || (st.kind === "algorithm" && chosen.some((l) => st.labels?.includes(l)));
        g.classList.toggle("dim", !ok);
        if (ok && st.kind === "algorithm") matches += 1;
      }
      host.querySelectorAll("[data-label]").forEach((b) => b.setAttribute("aria-pressed", String(filters.has(b.dataset.label))));
      host.querySelector(".mf-count").textContent = chosen.length ? `${matches} of ${algorithms.length} algorithms` : `${algorithms.length} algorithms, ${RL.order.length - algorithms.length} concepts`;
      host.querySelector(".mf-clear").hidden = !chosen.length;
      svg.classList.toggle("filtered", chosen.length > 0);
    }
    host.querySelector(".mf-chips").addEventListener("click", (e) => {
      const b = e.target.closest("[data-label]");
      if (!b) return;
      e.preventDefault();
      filters.has(b.dataset.label) ? filters.delete(b.dataset.label) : filters.add(b.dataset.label);
      filter();
    });
    host.querySelector(".mf-clear").addEventListener("click", () => { filters.clear(); filter(); });
    filter();
    host.querySelector(".lenses").addEventListener("click", (e) => { const b = e.target.closest("[data-lens]"); if (b) switchTo(b.dataset.lens); });

    // ---- pan, zoom, click ----
    let drag = null;
    svg.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false, target: e.target.closest(".st:not(.out)") };
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 4) return;
      drag.moved = true;
      svg.classList.add("dragging");
      RL.hideCard();
      view = { ...view, x: drag.vx + dx, y: drag.vy + dy, moved: true };
      apply();
    });
    svg.addEventListener("pointerup", () => {
      const d = drag;
      drag = null;
      svg.classList.remove("dragging");
      if (d?.moved) keep();
      else if (d?.target) dive(d.target.dataset.id);
    });
    // Scrolling moves the map like a page; Ctrl + scroll (or a pinch) zooms.
    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      RL.hideCard();
      if (e.ctrlKey || e.metaKey) {
        const r = svg.getBoundingClientRect();
        zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.004));
        return;
      }
      view = { ...view, x: view.x - (e.shiftKey ? e.deltaY : e.deltaX), y: view.y - (e.shiftKey ? 0 : e.deltaY), moved: true };
      apply();
      keep();
    }, { passive: false });
    svg.addEventListener("keydown", (e) => {
      const st = e.target.closest?.(".st");
      if (st && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); dive(st.dataset.id); }
    });
    host.querySelectorAll("[data-zoom]").forEach((b) => b.addEventListener("click", () => {
      if (b.dataset.zoom === "fit") return tween(fitted() || view, 400, keep);
      zoomAt(svg.clientWidth / 2, svg.clientHeight / 2, b.dataset.zoom === "in" ? 1.3 : 1 / 1.3);
    }));

    // ---- hover: a station's family, its parent and its children ----
    svg.addEventListener("pointerover", (e) => {
      const g = e.target.closest?.(".st:not(.out)");
      if (!g || g.classList.contains("hot")) return;
      clear();
      g.classList.add("hot");
      const id = g.dataset.id, pos = L().pos, a = pos.get(id);
      const family = [station(id).parent, ...algorithms.filter((s) => s.parent === id).map((s) => s.id)].filter((r) => r && pos.has(r));
      for (const r of family) nodes.get(r)?.classList.add("rel");
      if (lens === "tree") {
        svg.querySelectorAll(`.edge[data-from="${id}"], .edge[data-to="${id}"]`).forEach((p) => p.classList.add("hot"));
        return;
      }
      links.innerHTML = family.map((r) => {
        const b = pos.get(r), bend = Math.min(a.x, b.x) - 46 - Math.abs(a.y - b.y) * 0.15;
        return `<path d="M${a.x - 9} ${a.y} Q${bend} ${(a.y + b.y) / 2} ${b.x - 9} ${b.y}"/>`;
      }).join("");
    });
    svg.addEventListener("pointerout", (e) => {
      const g = e.target.closest?.(".st");
      if (g && !g.contains(e.relatedTarget)) clear();
    });
    function clear() {
      links.innerHTML = "";
      svg.querySelectorAll(".st.hot, .st.rel, .edge.hot").forEach((s) => s.classList.remove("hot", "rel"));
    }

    return { destroy() { cancelAnimationFrame(raf); resized.disconnect(); } };
  };
})(globalThis.RL = globalThis.RL || {});
