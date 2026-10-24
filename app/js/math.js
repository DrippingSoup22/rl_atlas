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

  // In running text a formula may break after any sign, but never inside a conditional like p(s', r | s, a), which
  // reads as one thing: it goes in braces, which KaTeX does not break.
  const whole = (src) => src.replace(/(?<![_^A-Za-z\\])((?:\\[A-Za-z]+|[A-Za-z])(?:_\{(?:[^{}]|\{[^{}]*\})*\}|_\\[A-Za-z]+|_[A-Za-z0-9])?)\(([^()]*?\\mid[^()]*?)\)/g, "{$1($2)}");
  const tex = (src, display = false) => katex.renderToString(display ? src : whole(src), { ...OPTIONS, displayMode: display });

  // Render every .tex element under root (their text is the TeX source written by build.py).
  function render(root) {
    for (const el of root.querySelectorAll(".tex:not([data-done])")) {
      el.dataset.src = el.textContent;
      el.innerHTML = tex(el.textContent, el.classList.contains("tex-display"));
      el.dataset.done = "1";
      hold(el);
    }
    fit(root);
  }

  // A formula and the punctuation build.py keeps beside it (in a .nobr): the punctuation goes into the formula's
  // first or last box, which KaTeX never breaks, in the font of the text around it. The formula itself may then break
  // inside, after a sign, and a comma still never starts a line.
  function hold(el) {
    const nobr = el.parentElement, boxes = el.querySelectorAll(".katex-html > .katex-base");
    if (!nobr?.classList.contains("nobr") || !boxes.length) return;
    const look = getComputedStyle(nobr);
    const mark = (node) => {
      const span = document.createElement("span");
      span.className = "held";
      for (const k of ["fontFamily", "fontSize", "fontWeight", "fontStyle"]) span.style[k] = look[k];
      span.append(node);
      return span;
    };
    if (el.previousSibling?.nodeType === 3) boxes[0].prepend(mark(el.previousSibling));
    if (el.nextSibling?.nodeType === 3) boxes[boxes.length - 1].append(mark(el.nextSibling));
    nobr.classList.add("held");
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
  const RELATIONS = ["\\Longleftrightarrow", "\\Longrightarrow", "\\iff", "\\implies", "\\leftarrow", "\\approx", "\\le", "\\ge", "="];
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

  // A long formula broken at plus and minus signs after its first relation, into n lines that line up at that
  // relation, the later lines indented: the way a long equation is set on a narrow page. Signs inside braces,
  // environments or \left...\right stay; a sign inside a plain or \big bracket may break, as the brackets need not
  // close on the same line. The cuts nearest to even shares of the right side come first; null if there are none.
  const WRAP_RELATIONS = ["\\leftarrow", "\\approx", "\\propto", "\\le", "\\ge", "\\to", "="];
  function wraps(src, n) {
    if (/\\begin\{/.test(src)) return [];
    let depth = 0, rel = -1, after = 0;
    const cuts = [];
    for (let i = 0; i < src.length; i++) {
      const c = src[i];
      if (c === "\\") {
        const word = /^\\[a-zA-Z]+/.exec(src.slice(i))?.[0] || src.slice(i, i + 2);
        if (word === "\\left") depth++;
        else if (word === "\\right") depth--;
        else if (!depth && rel < 0 && WRAP_RELATIONS.includes(word)) { rel = i; after = i + word.length; }
        i += word.length - 1;
      } else if (c === "{") depth++;
      else if (c === "}") depth--;
      else if (!depth && rel < 0 && c === "=") { rel = i; after = i + 1; }
      else if (!depth && rel >= 0 && (c === "+" || c === "-")) {
        // a sign between two terms, not a leading minus: what comes before it ends a term
        const before = src.slice(after, i).replace(/\\[,;:! ]|\\q?quad/g, "").trimEnd();
        if (/[\w)\]}'|]$/.test(before)) cuts.push(i);
      }
    }
    if (rel < 0) return [];
    const lhs = src.slice(0, rel).trim(), sign = src.slice(rel, after), len = src.length - after;
    const layout = (at) => {
      const pieces = [src.slice(after, at[0]), ...at.map((c, k) => src.slice(c, at[k + 1] ?? src.length))].map((t) => t.trim());
      return `\\begin{aligned} ${lhs} &${sign} ${pieces[0]} \\\\ ${pieces.slice(1).map((p) => `&\\quad {}${p}`).join(" \\\\ ")} \\end{aligned}`;
    };
    // or the left side alone on the first line, and the rest under it (itself in two lines if at is given)
    const alone = (at) => {
      const rest = at ? `${src.slice(after, at).trim()} \\\\ &\\qquad {}${src.slice(at).trim()}` : src.slice(after).trim();
      return lhs ? `\\begin{aligned} &${lhs} \\\\ &\\quad ${sign} ${rest} \\end{aligned}` : null;
    };
    const middle = (k, m) => after + (len * k) / m;
    const near = (goal, from = -1) => cuts.filter((c) => c > from).sort((a, b) => Math.abs(a - goal) - Math.abs(b - goal));
    if (n === 2) return [...near(middle(1, 2)).slice(0, 4).map((c) => layout([c])), alone(null)].filter(Boolean);
    const at = [];
    for (let k = 1; k < n; k++) {
      const best = near(middle(k, n), at[at.length - 1])[0];
      if (best === undefined) break;
      at.push(best);
    }
    return [at.length === n - 1 && layout(at), cuts.length && alone(near(middle(1, 2))[0])].filter(Boolean);
  }

  // A long left side before a short result, such as a sum of numbers and its value: it breaks into lines at the
  // separators nested least deep in brackets. Those are the commas of a list if there are any there (n lines), and
  // then, with signs, each item of the list breaks again at its plus and minus signs (n lines an item); otherwise its
  // plus and minus signs (n lines). The result stays at the end of the last line. Empty if there are not enough
  // places to break.
  function leading(src, n, signs = false) {
    if (/\\begin\{/.test(src)) return [];
    let depth = 0, nest = 0, rel = -1;
    const cuts = [];
    for (let i = 0; i < src.length; i++) {
      const c = src[i];
      if (c === "\\") {
        const word = /^\\[a-zA-Z]+/.exec(src.slice(i))?.[0] || src.slice(i, i + 2);
        if (word === "\\left") depth++;
        else if (word === "\\right") depth--;
        else if (word === "\\{") nest++;
        else if (word === "\\}") nest--;
        else if (!depth && WRAP_RELATIONS.includes(word)) rel = i;
        i += word.length - 1;
      } else if (c === "{") depth++;
      else if (c === "}") depth--;
      else if (c === "(" || c === "[") nest++;
      else if (c === ")" || c === "]") nest--;
      else if (!depth && c === "=") rel = i;
      else if (!depth && c === ",") cuts.push({ at: i + 1, nest, comma: true });
      else if (!depth && (c === "+" || c === "-") && /[\w)\]}'|]$/.test(src.slice(0, i).replace(/\\[,;:! ]|\\q?quad/g, "").trimEnd())) cuts.push({ at: i, nest });
    }
    const before = cuts.filter((c) => c.at < rel);
    if (!before.length) return [];
    const least = Math.min(...before.map((c) => c.nest)), level = before.filter((c) => c.nest === least);
    const commas = level.filter((c) => c.comma).map((c) => c.at);
    // up to n - 1 places among spots, nearest to even shares of from..to, after the last place taken
    const at = [];
    const share = (spots, from, to, k) => {
      for (let j = 1; j < k; j++) {
        const goal = from + ((to - from) * j) / k, best = spots.filter((c) => c > (at[at.length - 1] ?? -1)).sort((x, y) => Math.abs(x - goal) - Math.abs(y - goal))[0];
        if (best !== undefined) at.push(best);
      }
    };
    if (commas.length && signs) {
      const bounds = [0, ...commas, rel];
      for (let k = 0; k + 1 < bounds.length; k++) {
        if (k) at.push(bounds[k]);
        const inner = before.filter((c) => !c.comma && c.at > bounds[k] && c.at < bounds[k + 1]);
        if (!inner.length) continue;
        const low = Math.min(...inner.map((c) => c.nest));
        share(inner.filter((c) => c.nest === low).map((c) => c.at), bounds[k], bounds[k + 1], n);
      }
      if (at.length <= commas.length) return []; // no item broke: the list alone did that already
    } else {
      const places = commas.length ? commas : level.map((c) => c.at);
      if (places.length < n - 1) return [];
      share(places, 0, rel, n);
      if (at.length < n - 1) return [];
    }
    // a new item of the list starts a little in, the rest of an item further in
    const pieces = [0, ...at].map((c, k) => src.slice(c, at[k] ?? src.length).replace(/^\s*(\\[,;: ]\s*)*/, "").trim());
    const lines = pieces.slice(1).map((p, k) => `&${commas.length && !commas.includes(at[k]) ? "\\qquad" : "\\quad"} {}${p}`);
    return [`\\begin{aligned} &${pieces[0]} \\\\ ${lines.join(" \\\\ ")} \\end{aligned}`];
  }

  // The rows of an aligned or gathered formula: its top-level \\ (none inside braces or inner environments).
  function rowsOf(body) {
    const out = [];
    let depth = 0, from = 0;
    for (let i = 0; i < body.length; i++) {
      const c = body[i];
      if (c === "\\") {
        if (body.startsWith("\\begin{", i)) depth++;
        else if (body.startsWith("\\end{", i)) depth--;
        else if (!depth && body[i + 1] === "\\") { out.push(body.slice(from, i)); from = i + 2; }
        i++;
      } else if (c === "{") depth++;
      else if (c === "}") depth--;
    }
    out.push(body.slice(from));
    return out.map((t) => t.trim().replace(/^\[[^\]]*\]\s*/, "")).filter(Boolean);
  }
  const inner = (t) => t.replace(/^\s*\\begin\{aligned\}/, "").replace(/\\end\{aligned\}[\s\S]*$/, "");
  // roughly how wide a row is: its source without the commands that print nothing
  const visible = (t) => t.replace(/\\(?:val|rew|pol|err|knob|mathbf|boldsymbol|text|mathrm|operatorname\*?|big|Big|bigg|Bigg|left|right|htmlClass\{[^}]*\}|htmlData\{[^}]*\})/g, "")
    .replace(/\\[a-zA-Z]+/g, "x").replace(/[{}\s&]/g, "").length;
  // An aligned or gathered formula (a chain is one) with rows still too wide: in an aligned one, each row breaks at
  // its own relations first; then the longest row, and then every long row, breaks at a sign.
  function realign(src) {
    const m = /^\s*\\begin\{(aligned|gathered)\}([\s\S]*)\\end\{\1\}\s*$/.exec(src);
    if (!m) return [];
    const [, env, body] = m, plain = (r) => r.replace("&", "");
    let rows = rowsOf(body);
    if (env === "aligned") rows = rows.flatMap((r) => { const c = chain(plain(r)); return c ? rowsOf(inner(c)) : [r]; });
    const wrap = (r) => { const w = wraps(plain(r), 2)[0]; return !w ? [r] : env === "aligned" ? rowsOf(inner(w)) : [w]; };
    const build = (rs) => `\\begin{${env}} ${rs.join(" \\\\ ")} \\end{${env}}`;
    const longest = Math.max(...rows.map(visible)), first = rows.findIndex((r) => visible(r) === longest);
    const out = [build(rows), build(rows.flatMap((r, i) => (i === first ? wrap(r) : [r]))), build(rows.flatMap((r) => (visible(r) > 0.6 * longest ? wrap(r) : [r])))];
    // last, a long left side alone on the first line, every relation under it
    const amp = rows[0].indexOf("&"), lead = env === "aligned" && amp > 0 ? rows[0].slice(0, amp).trim() : "";
    if (lead) out.push(build([`&${lead}`, ...rows.map((r) => `&\\quad ${r.slice(r.indexOf("&") + 1).trim()}`)]));
    return [...new Set(out)].filter((l) => l.replace(/\s+/g, "") !== src.replace(/\s+/g, ""));
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
    const tag = el.querySelector(".katex-html > .katex-tag");
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

  // How wide a rendered formula really is (KaTeX's box always takes the whole line): from its first piece to its last,
  // plus, for a numbered one, the number and its gap on both sides, since the formula is centered and the number is not.
  function inkWidth(el) {
    const html = el.querySelector(".katex-html");
    if (!html) return 0;
    let left = Infinity, right = -Infinity, tag = 0;
    for (const part of html.children) {
      const r = part.getBoundingClientRect();
      if (part.classList.contains("katex-tag")) tag = r.width;
      else { left = Math.min(left, r.left); right = Math.max(right, r.right); }
    }
    return right > left ? right - left + (tag ? 2 * (tag + 24) : 0) : 0;
  }

  // A formula with room to spare grows into it, up to the factor its place allows (--grow in the CSS: more on a card or
  // beside a story's picture than in running text) and never past 92% of its box. The formulas of one card section
  // share one size, the largest that suits them all; a section where one of them had to shrink keeps them all as set.
  function grow(root) {
    const groups = new Map();
    for (const el of root.querySelectorAll(".tex-display[data-done]")) {
      if (!el.clientWidth) continue;
      const cap = parseFloat(getComputedStyle(el).getPropertyValue("--grow")) || 1;
      if (cap <= 1) continue;
      const key = el.closest(".sec") || el;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push([el, cap]);
    }
    for (const members of groups.values()) {
      if (members.some(([el]) => el.dataset.stacked || el.style.fontSize || el.classList.contains("tag-below"))) continue;
      let scale = Infinity;
      for (const [el, cap] of members) {
        const ink = inkWidth(el);
        scale = Math.min(scale, cap, ink ? (el.clientWidth * 0.92) / ink : cap);
      }
      if (!(scale > 1.02)) continue;
      for (const [el] of members) el.style.fontSize = `${(parseFloat(getComputedStyle(el).fontSize) * scale).toFixed(2)}px`;
    }
  }

  // A display formula a little wider than its column shrinks to fit, down to 80% of its size (65% on a phone). One
  // written as parts side by side stacks them instead when even that is too wide; anything still too wide scrolls.
  // A formula with room to spare may grow (grow, above).
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
      const layouts = [split.length > 1 && `\\begin{gathered} ${split.join(" \\\\[3pt] ")} \\end{gathered} ${tags}`,
        split.length > 1 && chained.some((p, i) => p !== split[i]) && `\\begin{gathered} ${chained.join(" \\\\[6pt] ")} \\end{gathered} ${tags}`,
        split.length < 2 && chained[0] !== split[0] && `${chained[0]} ${tags}`];
      // still too wide: rows of a chain or of an aligned formula break further, and a long side breaks at its plus and
      // minus signs, into two lines and then three
      if (split.length < 2) {
        layouts.push(...realign(chained[0]).map((l) => `${l} ${tags}`));
        if (chained[0] === split[0]) for (const n of [2, 3]) layouts.push(...wraps(split[0], n).map((w) => `${w} ${tags}`));
        // a long left side before its result, a line of numbers for instance, breaks into as many lines as it needs
        if (chained[0] === split[0]) {
          for (let n = 2; n <= 8; n++) layouts.push(...leading(split[0], n).map((w) => `${w} ${tags}`));
          for (let n = 2; n <= 4; n++) layouts.push(...leading(split[0], n, true).map((w) => `${w} ${tags}`));
        }
      } else for (const n of [2, 3]) {
        const each = split.map((p) => (wraps(p, n)[0] || p));
        if (each.some((p, i) => p !== split[i])) layouts.push(`\\begin{gathered} ${each.join(" \\\\[6pt] ")} \\end{gathered} ${tags}`);
      }
      for (const layout of new Set(layouts)) {
        if (!layout) continue;
        el.style.fontSize = "";
        el.innerHTML = tex(layout, true);
        el.dataset.stacked = "1";
        if (shrink(el, floor)) break;
      }
      if (el.dataset.stacked) clearTag(el);
    }
    grow(root);
  }
  // Fit again when the column changes width, and when a KaTeX font arrives (they load as first used).
  let resizing = 0;
  const refit = () => { clearTimeout(resizing); resizing = setTimeout(() => fit(document.body), 150); };
  addEventListener("resize", refit);
  document.fonts?.addEventListener?.("loadingdone", refit);

  RL.math = { tex, render, fit, shrink, inkWidth };
})(globalThis.RL = globalThis.RL || {});
