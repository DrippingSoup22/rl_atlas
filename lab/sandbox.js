/* The sandbox: draw a grid world of your own (walls, a cliff, holes in the ice, gems), choose one or two algorithms,
   and the Lab runs them on it. Every change runs them again. The drawing is kept in this browser. */
(function (RL) {
  "use strict";
  const { esc, lab, store } = RL;
  const TOOLS = [
    { tile: "S", name: "Start", tip: "where every episode begins (there is one)" },
    { tile: "G", name: "Gem", tip: "reaching it ends the episode" },
    { tile: ".", name: "Floor", tip: "each step costs 1" },
    { tile: "#", name: "Wall", tip: "bumping into it costs a step and goes nowhere" },
    { tile: "C", name: "Cliff", tip: "−100, and back to the start" },
    { tile: "H", name: "Hole", tip: "−100, and the episode ends" },
  ];
  // Grid algorithms you can pick; a race needs two of the same kind (episodes, or sweeps of the states).
  const CHOICES = ["q-learning", "sarsa", "expected-sarsa", "double-q", "mc-control", "off-policy-mc", "value-iteration", "policy-iteration"];
  const LIMITS = { rows: [3, 10], cols: [3, 14] };

  RL.sandbox = function (lab_) {
    const saved = store.get("sandbox", null);
    let map = (saved?.map || lab.SANDBOX.map).slice(), slip = saved?.slip ?? 0, tool = "#", painting = null;
    const spec = () => ({ ...lab.SANDBOX, map: map.slice(), slip });
    lab.sandbox = spec();
    const keep = () => { lab.sandbox = spec(); store.set("sandbox", { map, slip, algorithms: lab_.algorithms() }); };
    const set = (s, tile) => {
      const cols = map[0].length, r = Math.floor(s / cols), c = s % cols;
      if (map[r][c] === tile) return false;
      if (tile === "S") map = map.map((row) => row.replace("S", "."));
      if (map[r][c] === "S") return false; // the start moves by painting it somewhere else
      map[r] = map[r].slice(0, c) + tile + map[r].slice(c + 1);
      return true;
    };
    const has = (t) => map.some((row) => row.includes(t));

    function render() {
      const host = lab_.host(), chosen = lab_.algorithms(), units = (id) => lab.algorithms[id].unit;
      const option = (id, sel) => `<option value="${id}"${id === sel ? " selected" : ""}>${esc(lab.algorithms[id].title)}</option>`;
      host.innerHTML = `
        <div class="sb-row">
          <span class="eyebrow">Draw</span>
          <div class="sb-tools" role="radiogroup" aria-label="What to draw">${TOOLS.map((t) => `<button type="button" role="radio" data-tool="${esc(t.tile)}" aria-checked="${t.tile === tool}" title="${esc(t.tip)}"><i class="sw k-${{ ".": "free", "#": "wall", S: "start", G: "gem", C: "cliff", H: "hole" }[t.tile]}"></i>${t.name}</button>`).join("")}</div>
          <span class="sb-size">rows <button type="button" data-size="rows:-1" aria-label="One row less">−</button><b>${map.length}</b><button type="button" data-size="rows:1" aria-label="One row more">+</button></span>
          <span class="sb-size">columns <button type="button" data-size="cols:-1" aria-label="One column less">−</button><b>${map[0].length}</b><button type="button" data-size="cols:1" aria-label="One column more">+</button></span>
          <label class="knob"><span class="name">ice</span><input type="range" data-slip min="0" max="0.66" step="0.02" value="${slip}"><output>${slip ? `slides ${Math.round(slip * 100)}%` : "none"}</output></label>
          <button class="mf-clear sb-reset" type="button">start over</button>
        </div>
        <div class="sb-row">
          <span class="eyebrow">Run</span>
          <select data-racer="0" aria-label="Algorithm">${CHOICES.map((id) => option(id, chosen[0])).join("")}</select>
          <span class="faint">against</span>
          <select data-racer="1" aria-label="Second algorithm"><option value="">nobody</option>${CHOICES.filter((id) => units(id) === units(chosen[0]) && id !== chosen[0]).map((id) => option(id, chosen[1])).join("")}</select>
          <span class="faint sb-hint">${has("G") || has("H") ? "Click or drag on the world to draw. Each step costs 1; the gem ends the episode." : "Add a gem or a hole, or episodes never end."}</span>
        </div>`;
    }

    function changed(racers) {
      keep();
      lab_.changed(spec(), racers);
      render();
    }

    function onClick(e) {
      const t = e.target.closest("[data-tool]");
      if (t) { tool = t.dataset.tool; render(); return; }
      const z = e.target.closest("[data-size]");
      if (z) {
        const [what, d] = z.dataset.size.split(":"), n = +d, [lo, hi] = LIMITS[what];
        if (what === "rows") {
          if (map.length + n < lo || map.length + n > hi) return;
          map = n > 0 ? [...map, ".".repeat(map[0].length)] : map.slice(0, -1);
        } else {
          if (map[0].length + n < lo || map[0].length + n > hi) return;
          map = map.map((row) => (n > 0 ? `${row}.` : row.slice(0, -1)));
        }
        if (!has("S")) map[0] = `S${map[0].slice(1)}`;
        changed();
        return;
      }
      if (e.target.closest(".sb-reset")) { map = lab.SANDBOX.map.slice(); slip = 0; changed(); }
    }

    function onInput(e) {
      if (e.target.matches("[data-slip]")) {
        slip = +e.target.value;
        e.target.nextElementSibling.textContent = slip ? `slides ${Math.round(slip * 100)}%` : "none";
        clearTimeout(onInput.timer);
        onInput.timer = setTimeout(() => changed(), 120);
      }
    }

    function onChange(e) {
      const sel = e.target.closest("[data-racer]");
      if (!sel) return;
      const chosen = lab_.algorithms().slice();
      chosen[+sel.dataset.racer] = sel.value;
      const first = chosen[0], second = chosen[1] && lab.algorithms[chosen[1]].unit === lab.algorithms[first].unit && chosen[1] !== first ? chosen[1] : null;
      changed(second ? [first, second] : [first]);
    }

    // Painting on the world itself, in any of the stages.
    function stageOf(e) {
      const svg = e.target.closest?.(".gridview");
      if (!svg) return null;
      const i = +svg.closest(".stage").dataset.i;
      return lab_.views()[i];
    }
    function onDown(e) {
      const view = stageOf(e);
      if (!view) return;
      const s = view.tileAt(e);
      if (s < 0) return;
      e.preventDefault();
      e.stopPropagation();
      painting = { view, dirty: false };
      paint(view, s);
    }
    function paint(view, s) {
      if (set(s, tool)) { painting.dirty = true; lab_.views().forEach((v) => v.preview(s, tool)); }
    }
    function onMove(e) {
      if (!painting) return;
      const s = painting.view.tileAt(e);
      if (s >= 0) paint(painting.view, s);
    }
    function onUp() {
      if (!painting) return;
      const dirty = painting.dirty;
      painting = null;
      if (dirty) changed();
    }

    return {
      spec,
      mount() {
        const host = lab_.host(), stages = host.parentElement.querySelector(".stages");
        render();
        host.addEventListener("click", onClick);
        host.addEventListener("input", onInput);
        host.addEventListener("change", onChange);
        stages.addEventListener("pointerdown", onDown, true);
        addEventListener("pointermove", onMove);
        addEventListener("pointerup", onUp);
        stages.classList.add("drawable");
        const first = saved?.algorithms?.filter((id) => CHOICES.includes(id));
        if (first?.length) lab_.setRacers(first);
      },
      destroy() {
        removeEventListener("pointermove", onMove);
        removeEventListener("pointerup", onUp);
      },
    };
  };
})(globalThis.RL = globalThis.RL || {});
