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

  // A chain of relations, a = b = c, broken into lines that line up at the signs. Only top-level signs count (none
  // inside braces, brackets or environments); with fewer than two there is nothing to break, and null comes back.
  const RELATIONS = ["\\Longleftrightarrow", "\\Longrightarrow", "\\iff", "\\implies", "\\approx", "\\le", "\\ge", "="];
  function chain(src) {
    const tags = src.match(/\\tag\{[^}]*\}/g) || [], body = src.replace(/\\tag\{[^}]*\}/g, "");
    const cuts = [];
    let depth = 0;
    for (let i = 0; i < body.length; i++) {
      const c = body[i];
      if (c === "\\") {
        if (body.startsWith("\\begin{", i) || body[i + 1] === "{") depth++;
        else if (body.startsWith("\\end{", i) || body[i + 1] === "}") depth--;
        else if (!depth) {
          const rel = RELATIONS.find((r) => r !== "=" && body.startsWith(r, i) && !/[a-zA-Z]/.test(body[i + r.length] || ""));
          if (rel) { cuts.push(i); i += rel.length - 1; continue; }
        }
        i++;
      } else if (c === "{" || c === "(" || c === "[") depth++; // a sign inside brackets, as in E[G | S = s], stays
      else if (c === "}" || c === ")" || c === "]") depth--;
      else if (c === "=" && !depth) cuts.push(i);
    }
    if (cuts.length < 2) return null;
    const lines = cuts.map((at, k) => body.slice(at, cuts[k + 1] ?? body.length).trim());
    const lead = body.slice(0, cuts[0]).trim();
    // the spacing commands that sat around a sign (\quad, \qquad) would only push the next line out
    const clean = (t) => t.replace(/^(\\q?quad\s*)+|(\s*\\q?quad)+$/g, "");
    return `\\begin{aligned} ${clean(lead)} &${lines.map(clean).join(" \\\\ &")} \\end{aligned} ${tags.join(" ")}`;
  }

  // Shrink a formula wider than its box, down to a floor (a share of its size). Its width does not quite follow the
  // font (KaTeX keeps some lengths fixed), so it is measured again after each step. True when it fits.
  function shrink(el, floor) {
    const room = el.clientWidth, size = parseFloat(getComputedStyle(el).fontSize);
    for (let scale = 1, i = 0; i < 4; i++) {
      const need = el.scrollWidth;
      if (need <= room + 1) return true;
      if (scale <= floor) return false;
      scale = Math.max(floor, (scale * (room - 2)) / need);
      el.style.fontSize = `${(size * scale).toFixed(2)}px`;
    }
    return el.scrollWidth <= room + 1;
  }

  // The room between a numbered formula and its number: KaTeX centers the formula and pins the number to the right
  // edge, so a long formula runs into it.
  function tagGap(el) {
    const tag = el.querySelector(".katex-html > .tag");
    if (!tag) return Infinity;
    let right = -Infinity;
    for (const part of tag.parentElement.children) if (part !== tag) right = Math.max(right, part.getBoundingClientRect().right);
    return tag.getBoundingClientRect().left - right;
  }
  // A numbered formula keeps clear of its number: it may shrink a little, and if that is not enough the number goes
  // under it, at the right, and the formula keeps its size.
  function clearTag(el) {
    const GAP = 24;
    if (tagGap(el) >= GAP) return;
    const size = parseFloat(getComputedStyle(el).fontSize), was = el.style.fontSize;
    for (const scale of [0.95, 0.9]) {
      el.style.fontSize = `${(size * scale).toFixed(2)}px`;
      if (tagGap(el) >= GAP) return;
    }
    el.style.fontSize = was;
    el.classList.add("tag-below");
  }

  // A display formula a little wider than its column shrinks to fit, down to 80% of its size (65% on a phone). One
  // written as parts side by side stacks them instead when even that is too wide; anything still too wide scrolls.
  function fit(root) {
    for (const el of root.querySelectorAll(".tex-display[data-done]")) {
      el.style.fontSize = "";
      el.classList.remove("tag-below");
      if (el.dataset.stacked) { el.innerHTML = tex(el.dataset.src, true); delete el.dataset.stacked; }
      if (!el.clientWidth) continue; // hidden
      clearTag(el);
      if (el.scrollWidth <= el.clientWidth + 1) continue; // it fits
      const src = el.dataset.src || "", floor = innerWidth < 640 ? 0.65 : 0.8;
      if (shrink(el, floor)) continue;
      // Too wide even shrunk: parts written side by side go one under the other, and then a chain of relations
      // breaks at its signs, before anything is left to scroll. The number (\\tag) stays outside the stack.
      const tags = (src.match(/\\tag\{[^}]*\}/g) || []).join(" "), bare = src.replace(/\\tag\{[^}]*\}/g, ""), split = parts(bare);
      const chained = split.map((p) => chain(p) || p);
      for (const layout of [split.length > 1 && `\\begin{gathered} ${split.join(" \\\\[3pt] ")} \\end{gathered} ${tags}`,
        split.length > 1 && chained.some((p, i) => p !== split[i]) && `\\begin{gathered} ${chained.join(" \\\\[6pt] ")} \\end{gathered} ${tags}`,
        split.length < 2 && chained[0] !== split[0] && `${chained[0]} ${tags}`]) {
        if (!layout) continue;
        el.style.fontSize = "";
        el.innerHTML = tex(layout, true);
        el.dataset.stacked = "1";
        if (shrink(el, floor)) break;
      }
      if (el.dataset.stacked) clearTag(el);
    }
  }
  // Fit again when the column changes width, and when a KaTeX font arrives (they load as first used).
  let resizing = 0;
  const refit = () => { clearTimeout(resizing); resizing = setTimeout(() => fit(document.body), 150); };
  addEventListener("resize", refit);
  document.fonts?.addEventListener?.("loadingdone", refit);

  RL.math = { tex, render, fit, shrink };
})(globalThis.RL = globalThis.RL || {});
