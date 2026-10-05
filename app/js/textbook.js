/* Textbook: the theory of an entry, written like a book chapter. Numbered sections, equations and figures,
   a contents list that follows your reading, and references that jump to what they name (hover one to peek at it). */
(function (RL) {
  "use strict";
  const { esc, h, station } = RL;
  const PEEK = new Set(["equation", "definition", "theorem", "lemma", "example"]);

  RL.textbook = function (host, e) {
    const tb = e.textbook;
    const prereqs = e.prereqs.filter(station).map((id) => `<a class="term" data-term="${id}" href="#/e/${id}">${esc(station(id).title)}</a>`);
    const sources = e.sources.map((s) => `<li>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.text)}</a>` : esc(s.text)}</li>`).join("");
    const contents = [...tb.sections, { id: "references", n: "", title: "References" }];
    host.innerHTML = `
      <div class="tb">
        <nav class="tb-toc" aria-label="Contents">
          <div class="eyebrow">Contents</div>
          <ol>${contents.map((s) => `<li><a href="#" data-go="${s.id}"><span class="no">${s.n}</span><span>${s.title}</span></a></li>`).join("")}</ol>
          <div class="tb-progress" aria-hidden="true"><i></i></div>
        </nav>
        <article class="tb-paper">
          <header class="tb-top">
            <span class="eyebrow">Textbook · about ${tb.minutes} min</span>
            ${prereqs.length ? `<p>Before this chapter: ${prereqs.join(", ")}.</p>` : ""}
          </header>
          ${tb.sections.map((s) => `<section class="tb-sec" data-anchor="${s.id}"><h2><span class="no">${s.n}</span>${s.title}</h2>${s.html}</section>`).join("")}
          <section class="tb-sec tb-refs" data-anchor="references"><h2>References</h2><ol>${sources}</ol></section>
        </article>
      </div>`;
    const links = Array.from(host.querySelectorAll(".tb-toc a")), sections = Array.from(host.querySelectorAll(".tb-sec"));
    const bar = host.querySelector(".tb-progress"), paper = host.querySelector(".tb-paper");
    const target = (label) => host.querySelector(`[data-anchor="${label}"]`);

    // Jump to a section, equation, figure or statement, and light it up so the eye finds it.
    function jump(label) {
      const t = target(label);
      if (!t) return;
      const smooth = RL.reducedMotion() ? "auto" : "smooth";
      t.scrollIntoView({ behavior: smooth, block: t.classList.contains("tb-sec") ? "start" : "center" });
      t.classList.remove("flash");
      void t.offsetWidth; // restart the animation
      t.classList.add("flash");
    }
    host.addEventListener("click", (ev) => {
      const a = ev.target.closest("a[data-go], a.ref");
      if (!a) return;
      ev.preventDefault();
      hidePeek();
      jump(a.dataset.go || a.dataset.ref);
    });

    // Peek: hover a reference to an equation or a statement to read it without leaving your place.
    let peek = null;
    function hidePeek() { peek?.remove(); peek = null; }
    host.addEventListener("pointerover", (ev) => {
      const a = ev.target.closest("a.ref");
      if (!a || !PEEK.has(a.dataset.kind)) return;
      const t = target(a.dataset.ref);
      if (!t) return;
      hidePeek();
      peek = h(`<div class="hovercard peek" style="--c:var(--ink-3)"></div>`);
      peek.append(t.cloneNode(true));
      document.body.appendChild(peek);
      const r = a.getBoundingClientRect(), w = peek.offsetWidth, ht = peek.offsetHeight;
      peek.style.left = `${Math.min(Math.max(8, r.left + r.width / 2 - w / 2), innerWidth - w - 8)}px`;
      peek.style.top = `${r.bottom + 10 + ht < innerHeight ? r.bottom + 10 : r.top - ht - 10}px`;
    });
    host.addEventListener("pointerout", (ev) => { if (ev.target.closest("a.ref")) hidePeek(); });

    // The contents list follows the reading: the section under the top of the screen is lit, and a bar fills up.
    let raf = 0;
    function follow() {
      raf = 0;
      let on = 0;
      sections.forEach((s, i) => { if (s.getBoundingClientRect().top < 160) on = i; });
      links.forEach((a, i) => a.classList.toggle("on", i === on));
      const r = paper.getBoundingClientRect();
      bar.style.setProperty("--p", Math.min(1, Math.max(0, -r.top / Math.max(1, r.height - innerHeight))).toFixed(3));
    }
    const onScroll = () => { hidePeek(); if (!raf) raf = requestAnimationFrame(follow); };
    addEventListener("scroll", onScroll, { passive: true });
    follow();

    return { destroy() { removeEventListener("scroll", onScroll); cancelAnimationFrame(raf); hidePeek(); } };
  };
})(globalThis.RL = globalThis.RL || {});
