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
      el.innerHTML = tex(el.textContent, el.classList.contains("tex-display"));
      el.dataset.done = "1";
    }
  }

  RL.math = { tex, render };
})(globalThis.RL = globalThis.RL || {});
