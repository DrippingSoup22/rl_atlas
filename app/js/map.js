/* The home screen: the whole curriculum, seen three ways.
   Metro: each part is a branch hanging from its family's line. Tree: one tree per line, of what builds on what.
   Unified: every algorithm, placed by how deep and how wide its updates are (after Sutton & Barto, Figure 8.11).
   Stations glide from one view to the next, part by part; the filters and the legend dim what does not match.
   Drag to pan, scroll to move, Ctrl + scroll to zoom, click a station to open it. */
(function (RL) {
  "use strict";
  const { esc, station, entry, store, lineColor } = RL;
  const COL = 178, STEP = 27, FIRST = 50, LEFT = 40, TOP = 96, ROW_GAP = 132;
  const LENSES = {
    map: { name: "Metro", caption: "Every station is one idea, in reading order. Start at the top left: first the problem, then methods that keep a table, then methods that scale." },
    tree: { name: "Tree", caption: "One tree per line, read from the top. Each station grows from the one it builds on; an algorithm from the algorithm it changes." },
    unified: { name: "Unified", caption: "Every algorithm, by how far its update looks ahead (down) and whether it samples one outcome or averages over all of them (across)." },
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

  // One tree per line, stacked in reading order. Every station hangs from the station of its line it builds on
  // most: an algorithm from the algorithm it changes, any other station from its latest prerequisite. Stations
  // that build on nothing in their line hang from the line's badge. Leaves sit one row apart, each parent level
  // with the middle of its children, and the trees grow to the right.
  function treeLayout() {
    const DX = 156, DY = 30, X = 40, HUB = 188, GAP = 58;
    const pos = new Map(), kids = new Map(), up = new Map(), hubs = [];
    let y = 44, depthMax = 0;
    for (const [line, name] of Object.entries(RL.content.lines)) {
      const ids = RL.order.filter((s) => s.line === line), hub = `line-${line}`;
      const before = (a, s) => station(a)?.line === line && station(a).index < s.index;
      kids.set(hub, []);
      for (const s of ids) kids.set(s.id, []);
      for (const s of ids) {
        const pre = (entry(s.id)?.prereqs || []).filter((q) => before(q, s)).sort((a, b) => station(b).index - station(a).index);
        const p = s.parent && before(s.parent, s) ? s.parent : pre[0] || hub;
        up.set(s.id, p);
        kids.get(p).push(s.id);
      }
      const top = y;
      let leaves = 0;
      // A station with branches carries its name above its dot: unless it is the first child, it gets half a row of
      // headroom so the name clears the sibling above.
      const place = (id, d, first = true) => {
        depthMax = Math.max(depthMax, d);
        const c = kids.get(id);
        if (c.length && !first) leaves += 0.5;
        const at = c.length ? (() => { const ys = c.map((k, i) => place(k, d + 1, i === 0)); return (ys[0] + ys[ys.length - 1]) / 2; })() : top + leaves++ * DY;
        if (d) pos.set(id, { x: X + HUB + 80 + (d - 1) * DX, y: at });
        return at;
      };
      hubs.push({ id: hub, line, name, x: X, w: HUB, y: place(hub, 0) });
      y = top + leaves * DY + GAP;
    }
    return { pos, kids, up, hubs, width: X + HUB + 80 + (depthMax - 1) * DX + 90, height: y - GAP + 10 };
  }

  // Sutton & Barto's plane, with every algorithm on it. A bandit has no next state, so one step is already the
  // whole return: the bandits sit in their own strip above the plane, where the reading starts.
  function unifiedLayout() {
    const X = 110, Y = 250, PW = 1300, PH = 560, STRIP = Y - 168;
    const pos = new Map(Object.entries(RL.content.unified).map(([id, [w, d]]) => [id, { x: X + w * PW, y: d < 0 ? STRIP + 44 : Y + d * PH }]));
    return { pos, plane: { X, Y, PW, PH, STRIP }, width: X + PW + 150, height: Y + PH + 60 };
  }

  // ---- drawings that belong to one view ----
  function metroDrawing(L) {
    const C = RL.content;
    let svg = "";
    for (const row of L.rows) svg += `<text class="row-title" x="${LEFT}" y="${row.y - 70}">${esc(row.title)}</text>`;
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
      // It ends where the second row's title begins, so the dotted line runs into that title.
      const titleEnd = LEFT + r2.title.length * 9.6 + 14;
      svg += `<path class="ribbon" d="M${a.x + 18} ${a.y} H${right - 24} Q${right} ${a.y} ${right} ${a.y + 24} V${mid - 24} Q${right} ${mid} ${right - 24} ${mid} H${titleEnd}"/>`;
      svg += `<text class="ribbon-label" x="${right - 30}" y="${mid - 9}" text-anchor="end">When tables are not enough, the journey continues</text>`;
    }
    for (const p of C.parts) {
      const hub = L.pos.get(`part-${p.n}`), last = L.pos.get(p.stations[p.stations.length - 1].id);
      svg += `<line class="branch" pathLength="1" style="--c:${lineColor(p.line)};--d:${p.n}" x1="${hub.x}" y1="${hub.y}" x2="${last.x}" y2="${last.y}"/>`;
      svg += `<g class="hub" style="--c:${lineColor(p.line)};--d:${p.n}" transform="translate(${hub.x} ${hub.y})"><rect x="-15" y="-11" width="30" height="22" rx="11"/><text y="4.5">${p.n}</text></g>`;
      // A long title takes two lines, so it never runs into the next part's title.
      const words = p.title.split(" "), cut = p.title.length > 18 && words.length > 1 ? Math.ceil(words.length / 2) : words.length;
      const lines = [words.slice(0, cut).join(" "), words.slice(cut).join(" ")].filter(Boolean);
      svg += `<text class="part-title" style="--d:${p.n}" x="${hub.x - 15}" y="${hub.y - 21 - 16 * (lines.length - 1)}">${lines.map((t, i) => `<tspan x="${hub.x - 15}"${i ? ' dy="16"' : ""}>${esc(t)}</tspan>`).join("")}</text>`;
    }
    return svg;
  }

  function treeDrawing(T) {
    let svg = "";
    const hub = new Map(T.hubs.map((h) => [h.id, h]));
    // The trees follow one another down the page: a dotted spine joins their badges in reading order.
    T.hubs.slice(0, -1).forEach((h, i) => {
      const n = T.hubs[i + 1], cx = h.x + 14;
      svg += `<path class="spine" d="M${cx} ${h.y + 13} V${n.y - 13}"/>`;
    });
    for (const [id, kids] of T.kids) {
      const h = hub.get(id), a = h ? { x: h.x + h.w - 9, y: h.y } : T.pos.get(id);
      for (const k of kids) {
        const b = T.pos.get(k), mx = (a.x + b.x) / 2;
        svg += `<path class="edge" pathLength="1" data-from="${id}" data-to="${k}" style="--c:${lineColor(station(k).line)}" d="M${a.x + 9} ${a.y} C${mx} ${a.y} ${mx} ${b.y} ${b.x - 9} ${b.y}"/>`;
      }
    }
    for (const h of T.hubs) {
      svg += `<g class="tree-hub" data-line="${h.line}" style="--c:${lineColor(h.line)}" transform="translate(${h.x} ${h.y})"><rect y="-12" width="${h.w}" height="24" rx="12"/><text x="${h.w / 2}" y="4.5">${esc(h.name)}</text></g>`;
    }
    return svg;
  }

  function unifiedDrawing(U) {
    const { X, Y, PW, PH, STRIP } = U.plane, R = X + PW + 146;
    return `<rect class="plane" x="${X - 24}" y="${STRIP}" width="${0.52 * PW}" height="64" rx="16"/>
      <text class="corner" x="${X - 6}" y="${STRIP - 12}">Bandits</text>
      <text class="region" x="${X + 62}" y="${STRIP - 12}">one situation and no next state, so one step is the whole return</text>
      <rect class="plane" x="${X - 24}" y="${Y - 10}" width="${PW + 170}" height="${PH + 24}" rx="18"/>
      <text class="axis-title" x="${X + PW / 2}" y="${Y - 70}" text-anchor="middle">Width of update</text>
      <text class="axis-end" x="${X - 24}" y="${Y - 48}">← sample updates: one outcome at a time</text>
      <text class="axis-end" x="${R}" y="${Y - 48}" text-anchor="end">expected updates: every outcome, weighted →</text>
      <text class="axis-title" transform="translate(${X - 58} ${Y + PH / 2}) rotate(-90)" text-anchor="middle">Depth of update</text>
      <text class="axis-end" transform="translate(${X - 40} ${Y}) rotate(-90)" text-anchor="end">one step: bootstrapping</text>
      <text class="axis-end" transform="translate(${X - 40} ${Y + PH}) rotate(-90)">the full return</text>
      <text class="corner" x="${X - 6}" y="${Y - 20}">Temporal-difference learning</text>
      <text class="corner" x="${R - 6}" y="${Y - 20}" text-anchor="end">Dynamic programming</text>
      <text class="corner" x="${X - 6}" y="${Y + PH + 40}">Monte Carlo</text>
      <text class="corner" x="${R - 6}" y="${Y + PH + 40}" text-anchor="end">Exhaustive search</text>
      <text class="region" x="${X + 0.74 * PW}" y="${Y + 0.455 * PH}">n-step methods: n rewards, then a guess</text>
      <text class="region" x="${X + 0.74 * PW}" y="${Y + 0.655 * PH}">traces and GAE: a blend of every depth</text>
      <text class="credit" x="${R - 6}" y="${Y + PH + 58}" text-anchor="end">after Sutton &amp; Barto, Figure 8.11</text>`;
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
    const read = (line) => {
      const ids = RL.order.filter((s) => !line || s.line === line);
      return [ids.filter((s) => store.visited(s.id)).length, ids.length];
    };
    const panelOpen = store.get("mapPanel", false);
    host.innerHTML = `
      <section class="map-view" data-panel="${panelOpen ? "open" : "closed"}">
        <button class="panel-toggle" type="button" aria-controls="map-panel" aria-expanded="${panelOpen}" title="Views and filters (V)">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 5h14M3 10h9M3 15h5"/><circle cx="15" cy="10" r="2"/><circle cx="11" cy="15" r="2"/></svg>
          <span>Views &amp; filters</span><b class="pt-badge" hidden></b>
        </button>
        <aside class="map-panel" id="map-panel" aria-label="Views and filters">
          <div class="mp-head">
            <span class="eyebrow">Explore the atlas</span>
            <button class="icon-btn mp-close" type="button" aria-label="Close the panel" title="Close (Esc)"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15"/></svg></button>
          </div>
          <section class="mp-sec">
            <div class="lenses" role="tablist" aria-label="How to look at the atlas">
              ${Object.entries(LENSES).map(([id, l]) => `<button type="button" role="tab" data-lens="${id}">${l.name}</button>`).join("")}
            </div>
            <p class="lens-caption"></p>
          </section>
          <section class="mp-sec mp-lines">
            <h2 class="mp-title">Lines <span>read</span></h2>
            ${Object.entries(C.lines).map(([id, name]) => { const [n, of] = read(id); return `<button class="lg" type="button" data-line="${id}" style="--c:${lineColor(id)};--f:${n / of}"><i></i><span>${esc(name)}</span><b>${n}/${of}</b></button>`; }).join("")}
            <p class="lg-note"><span class="dot-read"></span>read <span class="dot-unread"></span>not yet · ${read()[0]} of ${read()[1]} read</p>
          </section>
          <section class="mp-sec map-filters">
            <h2 class="mp-title">Show only <button class="mf-clear" type="button">clear</button></h2>
            <div class="mf-chips">${Object.entries(C.labels).map(([id, l]) => `<button class="chip" type="button" data-label="${id}" aria-pressed="false" data-tip="${esc(l.tip)}" data-term="${l.station}">${esc(l.text)}</button>`).join("")}</div>
            <p class="mf-count faint"></p>
          </section>
          <p class="mp-hint faint">Drag to move the map; Ctrl + scroll or pinch to zoom.</p>
        </aside>
        ${next ? `<a class="btn map-cta" href="#/e/${next.id}">${store.visited(next.id) ? "Continue with" : "Start with"} ${esc(next.title)} ▸</a>` : ""}
        <svg class="metro${drawn ? "" : " intro"}" aria-label="Map of the atlas"><g class="cam">
          <g class="deco deco-map">${metroDrawing(layouts.map)}</g>
          <g class="deco deco-tree">${treeDrawing(layouts.tree)}</g>
          <g class="deco deco-unified">${unifiedDrawing(layouts.unified)}</g>
          ${stations(next)}
        </g></svg>
        <div class="map-zoom" role="group" aria-label="Zoom">
          <button class="icon-btn" type="button" data-zoom="in" aria-label="Zoom in" title="Zoom in">+</button>
          <button class="icon-btn" type="button" data-zoom="out" aria-label="Zoom out" title="Zoom out">−</button>
          <button class="icon-btn" type="button" data-zoom="fit" aria-label="Fit the map to the screen" title="Fit to the screen"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 8V3h5M17 8V3h-5M3 12v5h5M17 12v5h-5"/></svg></button>
        </div>
      </section>`;
    drawn = true;

    const svg = host.querySelector(".metro"), cam = host.querySelector(".cam");
    const nodes = new Map(Array.from(svg.querySelectorAll(".st"), (g) => [g.dataset.id, g]));
    const L = () => layouts[lens];
    let view = null, raf = 0, switching = 0;
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
      // In the tree a name sits above its dot, so the branches can leave from the dot; a leaf keeps its name on
      // the right, where no branch goes.
      for (const [id, g] of nodes) {
        const text = g.querySelector("text"), inner = lens === "tree" && layouts.tree.kids.get(id)?.length;
        text.style.transform = inner ? `translate(${-(+text.getAttribute("x")) - text.getComputedTextLength() / 2}px, -15px)` : "";
      }
      svg.dataset.lens = lens;
      host.querySelector(".map-view").dataset.lens = lens;
      host.querySelectorAll("[data-lens]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.lens === lens)));
      host.querySelector(".lens-caption").textContent = LENSES[lens].caption;
    }
    // The part of the screen the map may use: below the floating buttons, and right of the panel when it is open
    // beside the map (on a narrow screen it covers the map instead, so the map keeps the whole width).
    const view$ = host.querySelector(".map-view");
    const wide = () => svg.clientWidth > 1200; // narrower, the panel covers the map instead of shrinking it
    function insets() {
      const open = view$.dataset.panel === "open" && wide();
      return { left: open ? host.querySelector(".map-panel").offsetWidth + 24 : 12, top: 64, right: 60, bottom: 12 };
    }
    // The zoom that shows the whole view in that area.
    function wholeK() {
      const W = svg.clientWidth, H = svg.clientHeight, I = insets(), { width, height } = L();
      return Math.min((W - I.left - I.right) / width, (H - I.top - I.bottom) / height);
    }
    // Zoom stays between a little less than the whole view and close enough to read comfortably.
    const kRange = () => { const w = wholeK(); return [Math.min(0.85 * w, 0.6), Math.max(1.8, w)]; };
    // Keep the map in view: along an axis where it is smaller than the screen it stays centered; where it is
    // larger, it may move only until its edge reaches the edge of the screen (plus a small margin).
    function clamp(v) {
      const W = svg.clientWidth, H = svg.clientHeight;
      if (!W || !H) return v;
      const I = insets(), { width, height } = L(), [k0, k1] = kRange(), M = 48;
      const k = Math.max(k0, Math.min(k1, v.k));
      const axis = (pos, size, lo, hi) => {
        const room = hi - lo;
        if (size <= room) return lo + (room - size) / 2;
        return Math.max(hi - size - M, Math.min(lo + M, pos));
      };
      return { ...v, k, x: axis(v.x, width * k, I.left, W - I.right), y: axis(v.y, height * k, I.top, H - I.bottom) };
    }
    // The whole view when it stays readable; on small screens fit the width instead (the rest is a scroll away).
    // On a phone even the width is too much to read, so the view opens readable at the top left, where reading
    // starts, and the rest of the map is a swipe away.
    function fitted() {
      const W = svg.clientWidth, H = svg.clientHeight, I = insets(), { width, height } = L();
      if (!W || H <= I.top) return null;
      const whole = wholeK();
      if (whole < 0.8 && (W - I.left - I.right) / width < 0.6) return clamp({ k: 0.8, x: I.left, y: I.top + 6 });
      const k = Math.max(0.3, Math.min(1.25, whole >= 0.8 ? whole : (W - I.left - I.right) / width));
      return clamp({ k, x: I.left + (W - I.left - I.right - width * k) / 2, y: I.top + (whole >= 0.8 ? (H - I.top - I.bottom - height * k) / 2 : 6) });
    }
    function fit() {
      const v = fitted();
      if (!v) return;
      view = v;
      apply();
      keep();
    }
    const resized = new ResizeObserver(() => {
      if (!view) return;
      if (!cameras[lens]?.moved) fit();
      else { view = clamp(view); apply(); keep(); }
    });
    resized.observe(svg);
    function zoomAt(px, py, f) {
      const [k0, k1] = kRange(), k = Math.max(k0, Math.min(k1, view.k * f));
      view = clamp({ k, x: px - ((px - view.x) * k) / view.k, y: py - ((py - view.y) * k) / view.k, moved: true });
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
    // Clicking a station presses it, and its page opens out of it.
    function dive(id) {
      const g = nodes.get(id);
      if (!L().pos.get(id) || RL.reducedMotion()) { location.hash = `#/e/${id}`; return; }
      const r = g.querySelector(".dot").getBoundingClientRect();
      RL.app.origin = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
      g.classList.add("pressed");
      setTimeout(() => { location.hash = `#/e/${id}`; }, 160);
    }
    function switchTo(next) {
      if (next === lens) return;
      lens = next;
      store.set("lens", lens);
      arrange();
      svg.classList.remove("redraw");
      void svg.getBoundingClientRect();
      svg.classList.add("redraw"); // the family tree's branches draw themselves again
      // Stations set off part by part, so the eye sees the reading order arrive.
      svg.classList.add("switching");
      clearTimeout(switching);
      switching = setTimeout(() => svg.classList.remove("switching"), 1600);
      const to = cameras[lens] ? clamp(cameras[lens]) : fitted();
      if (to) tween(to, 650, keep);
    }

    arrange();
    view = cameras[lens] ? clamp({ ...cameras[lens] }) : null;
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
      const badge = host.querySelector(".pt-badge");
      badge.hidden = !chosen.length;
      badge.textContent = chosen.length;
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

    // ---- the panel: views, filters and the legend, opened and closed by the button at the top left ----
    const toggle = host.querySelector(".panel-toggle"), panel = host.querySelector(".map-panel");
    function setPanel(open, focus = false) {
      if ((view$.dataset.panel === "open") === open) return;
      const before = insets().left;
      view$.dataset.panel = open ? "open" : "closed";
      toggle.setAttribute("aria-expanded", String(open));
      store.set("mapPanel", open);
      // The map makes room for the panel: an untouched camera refits, a moved one shifts by half the room taken.
      const to = cameras[lens]?.moved ? clamp({ ...view, x: view.x + (insets().left - before) / 2 }) : fitted();
      if (to) tween(to, 350, keep);
      if (focus) (open ? panel.querySelector('[aria-selected="true"]') : toggle)?.focus({ preventScroll: true });
    }
    // Focus follows the panel only for the keyboard (a click has a detail count; Enter and Space do not).
    toggle.addEventListener("click", (e) => setPanel(true, e.detail === 0));
    host.querySelector(".mp-close").addEventListener("click", (e) => setPanel(false, e.detail === 0));
    function onKey(e) {
      if (e.target.closest?.("input, textarea, select, [contenteditable]") || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Escape" && view$.dataset.panel === "open") setPanel(false, true);
      else if (e.key === "v" || e.key === "V") setPanel(view$.dataset.panel !== "open", true);
    }
    document.addEventListener("keydown", onKey);

    // ---- pan, zoom, click ----
    let drag = null;
    svg.addEventListener("pointerdown", (e) => {
      if (!wide() && view$.dataset.panel === "open") setPanel(false); // on a small screen the panel covers the map
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
      view = clamp({ ...view, x: drag.vx + dx, y: drag.vy + dy, moved: true });
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
      view = clamp({ ...view, x: view.x - (e.shiftKey ? e.deltaY : e.deltaX), y: view.y - (e.shiftKey ? 0 : e.deltaY), moved: true });
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

    // ---- the legend ----
    // No highlighting on hover, of the stations or of the legend's lines: a cursor sweeping across them made the map
    // flicker. Clicking still finds things.
    const legend = host.querySelector(".mp-lines");
    // Clicking a line brings its stations to the middle of the screen.
    legend.addEventListener("click", (e) => {
      const line = e.target.closest("[data-line]")?.dataset.line;
      const pts = [...L().pos].filter(([id]) => station(id)?.line === line).map(([, p]) => p);
      if (!pts.length) return;
      const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y), I = insets(), k = view.k;
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2 + 60, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      const W = svg.clientWidth, H = svg.clientHeight;
      tween(clamp({ k, x: (I.left + W - I.right) / 2 - cx * k, y: (I.top + H - I.bottom) / 2 - cy * k, moved: true }), 500, keep);
    });

    return { destroy() { cancelAnimationFrame(raf); resized.disconnect(); document.removeEventListener("keydown", onKey); } };
  };
})(globalThis.RL = globalThis.RL || {});
