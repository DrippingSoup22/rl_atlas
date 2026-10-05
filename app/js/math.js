/* Math: KaTeX with the atlas's color macros.
   \val{..} value · \rew{..} reward · \pol{..} policy · \err{..} surprise · \knob{..} settings,
   \alp \gam \eps \lam \del: the common symbols, colored and linked to their stations,
   \step{n}{..}: a piece of a story formula that appears at step n. */
(function (RL) {
  "use strict";

  const MACROS = {
    "\\val": "\\htmlClass{q-val}{#1}",
    "\\rew": "\\htmlClass{q-rew}{#1}",
    "\\pol": "\\htmlClass{q-pol}{#1}",
    "\\err": "\\htmlClass{q-err}{#1}",
    "\\knob": "\\htmlClass{q-knob}{#1}",
    "\\alp": "\\htmlData{term=step-size}{\\knob{\\alpha}}",
    "\\gam": "\\htmlData{term=discount}{\\knob{\\gamma}}",
    "\\eps": "\\htmlData{term=epsilon-greedy}{\\knob{\\varepsilon}}",
    "\\lam": "\\htmlData{term=lambda-return}{\\knob{\\lambda}}",
    "\\del": "\\htmlData{term=td-error}{\\err{\\delta}}",
    "\\step": "\\htmlData{step=#1}{#2}",
  };
  const OPTIONS = {
    macros: MACROS,
    throwOnError: false,
    strict: "ignore",
    trust: (ctx) => ctx.command === "\\htmlClass" || ctx.command === "\\htmlData",
  };

  const tex = (src, display = false) => katex.renderToString(src, { ...OPTIONS, displayMode: display });

  // Render every .tex element under root (their text is the TeX source written by build.py).
  function render(root) {
    for (const el of root.querySelectorAll(".tex:not([data-done])")) {
      el.dataset.src = el.textContent;
      el.innerHTML = tex(el.textContent, el.classList.contains("tex-display"));
      el.dataset.done = "1";
    }
    fit(root);
  }

  // The parts of a formula written side by side: its top-level \qquad separators, none inside braces or environments.
  function parts(src) {
    const out = [];
    let depth = 0, from = 0;
    for (let i = 0; i < src.length; i++) {
      const c = src[i];
      if (c === "\\") {
        if (src.startsWith("\\begin{", i)) depth++;
        else if (src.startsWith("\\end{", i)) depth--;
        else if (!depth && src.startsWith("\\qquad", i) && !/[a-zA-Z]/.test(src[i + 6] || "")) { out.push(src.slice(from, i)); from = i + 6; }
        i++; // skip the escaped character, so \{ and \} do not count as braces
      } else if (c === "{") depth++;
      else if (c === "}") depth--;
    }
    out.push(src.slice(from));
    return out.map((t) => t.trim()).filter(Boolean);
  }

  // A display formula a little wider than its column shrinks to fit, down to 80% of its size. One written as parts
  // side by side stacks them instead when even that is too wide; anything still too wide scrolls.
  function fit(root) {
    for (const el of root.querySelectorAll(".tex-display[data-done]")) {
      el.style.fontSize = "";
      if (el.dataset.stacked) { el.innerHTML = tex(el.dataset.src, true); delete el.dataset.stacked; }
      const room = el.clientWidth;
      let need = el.scrollWidth;
      if (!room || need <= room + 1) continue; // hidden, or it fits
      const split = parts(el.dataset.src || "");
      if (need * 0.8 > room && split.length > 1) {
        el.innerHTML = tex(`\\begin{gathered} ${split.join(" \\\\[3pt] ")} \\end{gathered}`, true);
        el.dataset.stacked = "1";
        need = el.scrollWidth;
        if (need <= room + 1) continue;
      }
      const size = parseFloat(getComputedStyle(el).fontSize);
      el.style.fontSize = `${(size * Math.max(0.8, (room - 2) / need)).toFixed(2)}px`;
    }
  }
  // Fit again when the column changes width, and when a KaTeX font arrives (they load as first used).
  let resizing = 0;
  const refit = () => { clearTimeout(resizing); resizing = setTimeout(() => fit(document.body), 150); };
  addEventListener("resize", refit);
  document.fonts?.addEventListener?.("loadingdone", refit);

  RL.math = { tex, render, fit };
})(globalThis.RL = globalThis.RL || {});
