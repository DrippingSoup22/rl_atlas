/* The filmstrip: what an algorithm knew at a few moments of its run, side by side, so you see learning spread
   at a glance. Each frame is a small drawing made by the world's view (View.thumb); a click jumps there. */
(function (RL) {
  "use strict";

  class Film {
    constructor(host, { onSeek } = {}) {
      this.host = host;
      this.onSeek = onSeek;
      this.frames = [];
      this.el = RL.h('<ol class="filmstrip"></ol>');
      host.appendChild(this.el);
      this.el.addEventListener("click", (e) => {
        const li = e.target.closest("li[data-t]");
        if (li) this.onSeek?.(+li.dataset.t);
      });
    }

    // frames: [{ t, label, html }]
    set(frames) {
      this.frames = frames;
      this.el.innerHTML = frames.map((f, i) => `<li data-t="${f.t}" style="--i:${i}" tabindex="0" role="button" aria-label="Jump to ${RL.esc(f.label)}">
        <div class="thumb">${f.html}</div><span>${RL.esc(f.label)}</span></li>`).join("");
      this.at(this.t ?? 0);
    }

    // Light up the last frame at or before t.
    at(t) {
      this.t = t;
      let on = -1;
      this.frames.forEach((f, i) => { if (f.t <= t) on = i; });
      this.el.querySelectorAll("li").forEach((li, i) => li.classList.toggle("on", i === on));
    }

    destroy() { this.el.remove(); }

    // A frame's map of N × N cells, c units wide, the cell at row j and column i filled with fill(j * N + i), row 0 at
    // the bottom. The cells of one fill make one path: 961 cells of their own, five frames over, would hold the page up.
    // Edge to edge and without smoothing, no seam shows between neighbours.
    static cells(N, c, fill) {
      const paths = new Map();
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const f = fill(j * N + i);
        paths.set(f, `${paths.get(f) ?? ""}M${i * c} ${(N - 1 - j) * c}h${c}v${c}h-${c}z`);
      }
      return `<g shape-rendering="crispEdges">${[...paths].map(([f, d]) => `<path d="${d}" fill="${f}"/>`).join("")}</g>`;
    }
  }

  RL.Film = Film;
})(globalThis.RL = globalThis.RL || {});
