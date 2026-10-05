"""Build RL Atlas from content/: check it, compile it to app/content.js, bundle rl_atlas.html.

    python build.py

Errors stop the build; warnings name content that is still incomplete.
Needs Python 3.11+ and nothing else.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTENT = ROOT / "content"
APP = ROOT / "app"
OUT = ROOT / "rl_atlas.html"

log = logging.getLogger("build")

# What a written entry must contain. Card sections are named by the ids of their "### Title {#id}" headings.
REQUIRED = {
    "algorithm": {
        "fields": ("summary", "lab", "sources", "story"),
        "card": ("idea", "update", "change", "backup", "pseudocode", "perks", "flaws", "knobs", "pitfalls", "check"),
    },
    "concept": {"fields": ("summary", "sources"), "card": ("idea",)},
}
FRONT_MATTER = {"summary", "change", "prereqs", "lab", "sources", "story"}
# The keys of a Lab preset (content/lab.toml); any other key is a knob, and what its charts may plot.
PRESET_KEYS = {"title", "env", "algorithms", "racers", "units", "seed", "runs", "charts", "measures", "film", "intro"}
CHARTS = {"return", "steps", "optimal", "left", "delta", "error", "optimal-error", "match", "greedy", "ve", "weights"}
MEASURES = {"error", "optimal-error", "match", "greedy", "ve"}
# Stations the math macros of app/js/math.js link to.
MACRO_TERMS = ("step-size", "discount", "epsilon-greedy", "lambda-return", "td-error")
# What the Textbook numbers, and how a \ref{label} to each one reads.
REF_TEXT = {
    "section": "§{}", "equation": "({})", "figure": "Figure {}", "algorithm": "Algorithm {}",
    "definition": "Definition {}", "theorem": "Theorem {}", "lemma": "Lemma {}", "example": "Example {}",
}
# ::: blocks that state something, as in a textbook. The first four are numbered.
STATEMENTS = {
    "definition": "Definition", "theorem": "Theorem", "lemma": "Lemma", "example": "Example",
    "remark": "Remark", "proof": "Proof",
}

FRONT = re.compile(r"\+\+\+\n(.*?)\n\+\+\+\n", re.S)
BLOCK = re.compile(r"^::: *([\w-]+) *(.*?)\n(.*?)^:::[ \t]*$", re.M | re.S)
MATH = re.compile(r"\$\$(.+?)\$\$|\$([^\n$]+?)\$", re.S)
TERM = re.compile(r"\[\[([\w-]+)(?:\|([^\]]+))?\]\]")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
DEMO = re.compile(r"\{\{([\w-]+)(?: +([\w-]+))?\}\}")
SLOT = re.compile("\x00(\\d+)\x00")
REF = re.compile("\x01([\\w-]+)\x01")


class Problems:
    def __init__(self) -> None:
        self.errors = 0
        self.warnings = 0

    def error(self, where: str, msg: str) -> None:
        self.errors += 1
        log.error("%s: %s", where, msg)

    def warn(self, where: str, msg: str) -> None:
        self.warnings += 1
        log.warning("%s: %s", where, msg)


class Markdown:
    """The Markdown subset the content uses, plus [[term]] links, (lab:preset) links, $math$ and {{demo}} lines.
    In a Textbook it also numbers sections, equations (\\label{..}), figures, algorithms and statements,
    and turns \\ref{label} into links to them."""

    def __init__(self, stations: dict, presets: dict, where: str, problems: Problems, textbook: bool = False) -> None:
        self.stations, self.presets, self.where, self.problems = stations, presets, where, problems
        self.textbook = textbook
        self.labels: dict[str, tuple[str, int]] = {}
        self.counts: dict[str, int] = {}

    # ---- numbering and references (Textbook only) ----
    def number(self, kind: str, label: str | None = None) -> int:
        n = self.counts[kind] = self.counts.get(kind, 0) + 1
        if label:
            if label in self.labels:
                self.problems.error(self.where, f"the label '{label}' is used twice")
            self.labels[label] = (kind, n)
        return n

    def resolve(self, text: str) -> str:
        """Turn \\ref placeholders into links. Runs after the whole Textbook is compiled, so references may look ahead."""

        def link(m: re.Match) -> str:
            label = m.group(1)
            if label not in self.labels:
                self.problems.error(self.where, f"\\ref{{{label}}}: no such label in the Textbook")
                return "??"
            kind, n = self.labels[label]
            return f'<a class="ref" href="#" data-ref="{label}" data-kind="{kind}">{REF_TEXT[kind].format(n)}</a>'

        return REF.sub(link, text)

    def _ref(self, m: re.Match) -> str:
        if not self.textbook:
            self.problems.error(self.where, f"\\ref{{{m.group(1)}}}: references work only in the Textbook")
            return ""
        return f"\x01{m.group(1)}\x01"

    # ---- inline ----
    def inline(self, text: str) -> str:
        text, slots = self._protect_math(text)
        return self._restore(self._inline(text), slots)

    def _inline(self, text: str) -> str:
        text = html.escape(text, quote=False)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
        text = re.sub(r"\\ref\{([\w-]+)\}", self._ref, text)
        text = TERM.sub(self._term, text)
        return LINK.sub(self._link, text)

    def _term(self, m: re.Match) -> str:
        sid, label = m.group(1), m.group(2)
        if sid not in self.stations:
            self.problems.error(self.where, f"[[{sid}]] is not a station on the map")
            return label or sid
        label = label or html.escape(self.stations[sid]["title"])
        return f'<a class="term" data-term="{sid}" href="#/e/{sid}">{label}</a>'

    def _link(self, m: re.Match) -> str:
        text, href = m.group(1), m.group(2)
        if href.startswith("lab:"):
            preset = href[4:]
            if preset not in self.presets:
                self.problems.error(self.where, f"lab preset '{preset}' is not in content/lab.toml")
            return f'<a class="lab-link" href="#/lab/{preset}">{text}</a>'
        if href.startswith(("http://", "https://")):
            return f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'
        if href.startswith("#/"):
            return f'<a href="{href}">{text}</a>'
        self.problems.error(self.where, f"unknown link target '{href}'")
        return text

    def _protect_math(self, text: str) -> tuple[str, list[str]]:
        slots: list[str] = []

        def keep(m: re.Match) -> str:
            if m.group(1) is not None:
                tex, anchor = m.group(1).strip(), ""
                if label := re.search(r"\\label\{([\w-]+)\}", tex):
                    if not self.textbook:
                        self.problems.error(self.where, "numbered equations (\\label) work only in the Textbook")
                    n = self.number("equation", label.group(1))
                    tex = f"{tex[: label.start()]}\\tag{{{n}}}{tex[label.end() :]}"
                    anchor = f' data-anchor="{label.group(1)}"'
                slots.append(f'<div class="tex tex-display"{anchor}>{html.escape(tex)}</div>')
            else:
                slots.append(f'<span class="tex">{html.escape(m.group(2))}</span>')
            return f"\x00{len(slots) - 1}\x00"

        return MATH.sub(keep, text), slots

    @staticmethod
    def _restore(text: str, slots: list[str]) -> str:
        return SLOT.sub(lambda m: slots[int(m.group(1))], text)

    # ---- blocks ----
    def blocks(self, text: str) -> str:
        """Paragraphs, #### headings, lists, tables, display math and {{demo}} lines, separated by blank lines."""
        text, slots = self._protect_math(text)
        out = []
        for chunk in re.split(r"\n\s*\n", text.strip()):
            lines = [line.rstrip() for line in chunk.strip().splitlines()]
            if not lines:
                continue
            first = lines[0]
            if first.startswith("#### "):
                out.append(f"<h4>{self._inline(first[5:])}</h4>")
            elif all(line.startswith("- ") for line in lines):
                out.append("<ul>" + "".join(f"<li>{self._inline(line[2:])}</li>" for line in lines) + "</ul>")
            elif all(re.match(r"\d+\. ", line) for line in lines):
                items = (re.sub(r"^\d+\. ", "", line) for line in lines)
                out.append("<ol>" + "".join(f"<li>{self._inline(item)}</li>" for item in items) + "</ol>")
            elif all(line.startswith("|") for line in lines):
                out.append(self._table(lines))
            elif len(lines) == 1 and (demo := DEMO.fullmatch(first)):
                out.append(self._demo(demo))
            elif len(lines) == 1 and (slot := SLOT.fullmatch(first)) and slots[int(slot.group(1))].startswith("<div"):
                out.append(first)
            else:
                out.append(f"<p>{self._inline(' '.join(lines))}</p>")
        return self._restore("\n".join(out), slots)

    @staticmethod
    def _demo(m: re.Match) -> str:
        arg = f' data-arg="{m.group(2)}"' if m.group(2) else ""
        return f'<div class="demo" data-demo="{m.group(1)}"{arg}></div>'

    def _table(self, lines: list[str]) -> str:
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
        if len(rows) < 2 or not all(re.fullmatch(r":?-+:?", c) for c in rows[1]):
            self.problems.error(self.where, "a table needs a header row and a | --- | row")
            return ""
        head = "".join(f"<th>{self._inline(c)}</th>" for c in rows[0])
        body = "".join("<tr>" + "".join(f"<td>{self._inline(c)}</td>" for c in row) + "</tr>" for row in rows[2:])
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def pseudocode(self, text: str) -> str:
        items = []
        for raw in text.splitlines():
            if not raw.strip():
                continue
            indent = (len(raw) - len(raw.lstrip(" "))) // 2
            line = raw.strip()
            lid = ""
            if m := re.search(r"\s*\{#([\w-]+)\}$", line):
                lid, line = m.group(1), line[: m.start()]
            attr = f' data-line="{lid}"' if lid else ""
            items.append(f'<li style="--indent:{indent}"{attr}>{self.inline(line)}</li>')
        return '<ol class="pseudo">' + "".join(items) + "</ol>"

    def question(self, text: str) -> str:
        parts = re.split(r"^---[ \t]*$", text, maxsplit=1, flags=re.M)
        if len(parts) != 2:
            self.problems.error(self.where, "a question needs its answer after a --- line")
            return ""
        q, a = (self.blocks(p) for p in parts)
        return (
            '<div class="question" tabindex="0" role="button" aria-label="Show the answer">'
            f'<div class="face q"><span class="tag">Question</span>{q}</div>'
            f'<div class="face a"><span class="tag">Answer</span>{a}</div></div>'
        )

    # ---- textbook blocks: "::: theorem {#label} Title", "::: figure {#label}", "::: algorithm {#label} Title" ----
    @staticmethod
    def _label(args: str) -> tuple[str | None, str]:
        if m := re.match(r"\{#([\w-]+)\}\s*", args):
            return m.group(1), args[m.end() :].strip()
        return None, args.strip()

    def statement(self, kind: str, args: str, body: str) -> str:
        label, title = self._label(args)
        if kind == "proof":
            head, title = title or "Proof", ""
        elif kind == "remark":
            head = "Remark"
        else:
            head = f"{STATEMENTS[kind]} {self.number(kind, label)}"
        lead = f'<span class="tb-lead"><b>{head}</b>{f" ({self.inline(title)})" if title else ""}.</span> '
        inner = self.blocks(body)
        inner = inner.replace("<p>", f"<p>{lead}", 1) if inner.startswith("<p>") else f"<p>{lead}</p>{inner}"
        anchor = f' data-anchor="{label}"' if label else ""
        return f'<div class="tb-block tb-{kind}"{anchor}>{inner}</div>'

    def figure(self, args: str, body: str) -> str:
        label, _ = self._label(args)
        first, _, caption = body.strip().partition("\n")
        demo = DEMO.fullmatch(first.strip())
        if not demo:
            self.problems.error(self.where, "a figure starts with a {{demo}} line, followed by its caption")
            return ""
        n = self.number("figure", label)
        anchor = f' data-anchor="{label}"' if label else ""
        return (f'<figure class="tb-fig"{anchor}>{self._demo(demo)}'
                f'<figcaption><b>Figure {n}.</b> {self.inline(" ".join(caption.split()))}</figcaption></figure>')

    def algorithm(self, args: str, body: str) -> str:
        label, title = self._label(args)
        n = self.number("algorithm", label)
        anchor = f' data-anchor="{label}"' if label else ""
        return (f'<figure class="tb-alg"{anchor}><figcaption><b>Algorithm {n}</b> {self.inline(title)}</figcaption>'
                f"{self.pseudocode(body)}</figure>")

    def rich(self, text: str) -> str:
        """Markdown with ::: blocks: pseudocode, question, analogy, and in a Textbook figure, algorithm and statements."""
        out, pos = [], 0
        for m in BLOCK.finditer(text):
            out.append(self.blocks(text[pos : m.start()]))
            name, args, body = m.group(1), m.group(2), m.group(3)
            if name == "pseudocode":
                out.append(self.pseudocode(body))
            elif name == "question":
                out.append(self.question(body))
            elif name == "analogy":
                out.append(f'<aside class="analogy">{self.blocks(body)}</aside>')
            elif self.textbook and name in STATEMENTS:
                out.append(self.statement(name, args, body))
            elif self.textbook and name == "figure":
                out.append(self.figure(args, body))
            elif self.textbook and name == "algorithm":
                out.append(self.algorithm(args, body))
            else:
                self.problems.error(self.where, f"unknown block '::: {name}'" + ("" if self.textbook else " (outside the Textbook)"))
            pos = m.end()
        out.append(self.blocks(text[pos:]))
        return "\n".join(part for part in out if part)


def split_headings(text: str, level: str) -> list[tuple[str, str]]:
    """[(heading, body)] for every heading of the given level ('##' or '###'); text before the first is dropped."""
    parts = re.split(rf"^{level} +(.+)$", text, flags=re.M)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts), 2)]


def heading_id(title: str) -> tuple[str, str]:
    if m := re.search(r"\s*\{#([\w-]+)\}$", title):
        return title[: m.start()], m.group(1)
    return title, re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def compile_entry(path: Path, site: dict, problems: Problems) -> dict | None:
    where = str(path.relative_to(ROOT))
    raw = path.read_text(encoding="utf-8")
    front = FRONT.match(raw)
    if not front:
        problems.error(where, "missing the +++ front matter")
        return None
    try:
        meta = tomllib.loads(front.group(1))
    except tomllib.TOMLDecodeError as exc:
        problems.error(where, f"front matter: {exc}")
        return None

    sid = path.stem
    station = site["stations"].get(sid)
    if not station:
        problems.error(where, f"'{sid}' is not a station on the map (content/map.toml)")
        return None
    if path.parent.name != station["folder"]:
        problems.warn(where, f"belongs in content/{station['folder']}/")

    md = Markdown(site["stations"], site["presets"], where, problems)
    need = REQUIRED[station["kind"]]
    for field in need["fields"]:
        if field not in meta:
            problems.warn(where, f"no '{field}' yet")
    for field in meta.keys() - FRONT_MATTER:
        note = " (parents and labels live in content/map.toml)" if field in ("parent", "labels") else ""
        problems.warn(where, f"unknown front matter field '{field}'{note}")
    for ref in meta.get("prereqs", []):
        if ref not in site["stations"]:
            problems.error(where, f"'{ref}' is not a station on the map")
    if station.get("parent") and not meta.get("change"):
        problems.warn(where, "has a parent but no 'change' saying what changed")
    if "lab" in meta and meta["lab"] not in site["presets"]:
        problems.error(where, f"lab preset '{meta['lab']}' is not in content/lab.toml")

    body = dict((title.lower(), text) for title, text in split_headings(raw[front.end() :], "##"))

    story = None
    if "story" in meta:
        steps = []
        for m in BLOCK.finditer(body.get("story", "")):
            if m.group(1) != "step":
                problems.error(where, f"the story only takes '::: step' blocks, not '::: {m.group(1)}'")
                continue
            try:
                state = tomllib.loads(f"s = {m.group(2) or '{}'}")["s"]
            except tomllib.TOMLDecodeError as exc:
                problems.error(where, f"story step {len(steps) + 1}: {exc}")
                continue
            steps.append({"state": state, "html": md.blocks(m.group(3))})
        if not steps:
            problems.warn(where, "has a [story] but no story steps")
        story = {"config": meta["story"], "steps": steps}

    textbook = None
    if "textbook" in body:
        tb = Markdown(site["stations"], site["presets"], where, problems, textbook=True)
        sections = []
        for title, text in split_headings(body["textbook"], "###"):
            title, sid = heading_id(title)
            sections.append({"id": sid, "n": tb.number("section", sid), "title": tb.inline(title), "html": tb.rich(text)})
        for section in sections:
            section["title"], section["html"] = tb.resolve(section["title"]), tb.resolve(section["html"])
        words = len(re.findall(r"[A-Za-z]+", MATH.sub(" ", body["textbook"])))
        textbook = {"minutes": max(1, round(words / 160)), "sections": sections}
    elif station["kind"] == "algorithm":
        problems.warn(where, "no '## Textbook' yet")

    card, pseudocode = [], ""
    for title, text in split_headings(body.get("card", ""), "###"):
        title, cid = heading_id(title)
        html_ = md.rich(text)
        card.append({"id": cid, "title": md.inline(title), "html": html_})
        if cid == "pseudocode":
            pseudocode = html_
    for cid in need["card"]:
        if cid not in {section["id"] for section in card}:
            problems.warn(where, f"the card has no '{cid}' section")

    return {
        "summary": meta.get("summary", ""),
        "change": meta.get("change", ""),
        "prereqs": meta.get("prereqs", []),
        "lab": meta.get("lab"),
        "sources": meta.get("sources", []),
        "story": story,
        "textbook": textbook,
        "card": card,
        "pseudocode": pseudocode,
    }


def compile_preset(pid: str, raw: dict, stations: dict, problems: Problems) -> dict:
    """A Lab preset in one shape: its racers (each an algorithm, a name and knobs of its own) and the shared knobs."""
    where = f"content/lab.toml [{pid}]"
    for key in ("title", "env"):
        if key not in raw:
            problems.error(where, f"no '{key}'")
    racers = []
    for r in raw.get("racers") or [{"algorithm": a} for a in raw.get("algorithms", [])]:
        algo = r.get("algorithm", "")
        if stations.get(algo, {}).get("kind") != "algorithm":
            problems.error(where, f"runs '{algo}', which is not an algorithm station")
            continue
        own = {k: v for k, v in r.items() if k not in ("algorithm", "name")}
        racers.append({"algorithm": algo, "name": r.get("name") or stations[algo]["title"], "params": own})
    if not racers:
        problems.error(where, "runs no algorithm")
    charts, measures = raw.get("charts", ["return"]), raw.get("measures", [])
    for chart in charts:
        if chart not in CHARTS:
            problems.error(where, f"cannot chart '{chart}' (one of: {', '.join(sorted(CHARTS))})")
        elif chart in MEASURES and chart not in measures:
            problems.error(where, f"charts '{chart}' but does not measure it (add it to measures)")
    for m in measures:
        if m not in MEASURES:
            problems.error(where, f"unknown measure '{m}'")
    return {
        "title": raw.get("title", pid), "env": raw.get("env", ""), "racers": racers,
        "units": raw.get("units", 500), "seed": raw.get("seed", 1), "runs": raw.get("runs", 1),
        "charts": charts, "measures": measures, "film": raw.get("film", []), "intro": raw.get("intro", ""),
        "params": {k: v for k, v in raw.items() if k not in PRESET_KEYS},
    }


def compile_content(problems: Problems) -> dict:
    atlas = tomllib.loads((CONTENT / "map.toml").read_text(encoding="utf-8"))
    presets = tomllib.loads((CONTENT / "lab.toml").read_text(encoding="utf-8"))
    notation = tomllib.loads((CONTENT / "notation.toml").read_text(encoding="utf-8"))

    stations, parts = {}, []
    for part in atlas["part"]:
        if part["line"] not in atlas["lines"]:
            problems.error("content/map.toml", f"part {part['n']} uses an unknown line '{part['line']}'")
        for st in part["stations"]:
            if st["id"] in stations:
                problems.error("content/map.toml", f"station '{st['id']}' appears twice")
            st.setdefault("kind", "concept")
            stations[st["id"]] = {**st, "part": part["n"], "folder": part["folder"]}
        parts.append({key: part[key] for key in ("n", "title", "line", "stations")})
    for label in atlas["labels"].values():
        if label["station"] not in stations:
            problems.error("content/map.toml", f"label '{label['text']}' points to unknown station '{label['station']}'")
    for sid, st in stations.items():
        if st.get("parent") and stations.get(st["parent"], {}).get("kind") != "algorithm":
            problems.error("content/map.toml", f"'{sid}' has parent '{st['parent']}', which is not an algorithm station")
        for label in st.get("labels", []):
            if label not in atlas["labels"]:
                problems.error("content/map.toml", f"'{sid}' has label '{label}', which is not in [labels]")
        if st["kind"] == "algorithm" and not st.get("labels"):
            problems.warn("content/map.toml", f"algorithm '{sid}' has no labels")
    for sid in atlas.get("unified", {}):
        if stations.get(sid, {}).get("kind") != "algorithm":
            problems.error("content/map.toml", f"[unified] places '{sid}', which is not an algorithm station")
    for sid in MACRO_TERMS:
        if sid not in stations:
            problems.error("app/js/math.js", f"a math macro links to unknown station '{sid}'")
    presets = {pid: compile_preset(pid, raw, stations, problems) for pid, raw in presets.items()}
    for sym in notation.get("symbol", []):
        if sym.get("station") and sym["station"] not in stations:
            problems.error("content/notation.toml", f"symbol {sym['tex']} points to unknown station '{sym['station']}'")

    site = {"stations": stations, "presets": presets, "labels": atlas["labels"]}
    entries = {}
    for path in sorted(CONTENT.glob("*/*.md")):
        entry = compile_entry(path, site, problems)
        if entry:
            entries[path.stem] = entry
    for pid, preset in presets.items():
        for algo in sorted({r["algorithm"] for r in preset["racers"]}):
            if not entries.get(algo, {}).get("pseudocode"):
                problems.warn("content/lab.toml", f"preset '{pid}' runs '{algo}', which has no written pseudocode yet")

    return {
        "rows": atlas["rows"],
        "rowTitles": atlas["row_titles"],
        "lines": atlas["lines"],
        "labels": atlas["labels"],
        "parts": parts,
        "unified": atlas.get("unified", {}),
        "entries": entries,
        "presets": presets,
        "notation": notation,
    }


def bundle() -> int:
    """Inline every stylesheet, script and font of app/index.html into one offline file."""
    page = (APP / "index.html").read_text(encoding="utf-8")

    def font(css_dir: Path):
        def data_uri(m: re.Match) -> str:
            data = base64.b64encode((css_dir / m.group(1)).read_bytes()).decode()
            return f"url(data:font/woff2;base64,{data})"

        return data_uri

    def style(m: re.Match) -> str:
        path = (APP / m.group(1)).resolve()
        css = re.sub(r"url\((fonts/[^)]+\.woff2)\)", font(path.parent), path.read_text(encoding="utf-8"))
        return "<style>\n" + css.replace("</style", "<\\/style") + "\n</style>"

    def script(m: re.Match) -> str:
        code = (APP / m.group(1)).resolve().read_text(encoding="utf-8")
        return "<script>\n" + code.replace("</script", "<\\/script") + "\n</script>"

    page = re.sub(r'<link rel="stylesheet" href="([^"]+)">', style, page)
    page = re.sub(r'<script src="([^"]+)"></script>', script, page)
    OUT.write_text(page, encoding="utf-8")
    return OUT.stat().st_size


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    problems = Problems()
    data = compile_content(problems)
    if problems.errors:
        log.error("%d error(s); nothing was written", problems.errors)
        return 1

    script = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (APP / "content.js").write_text(
        "// Generated by build.py from content/. Do not edit.\n"
        f"(globalThis.RL = globalThis.RL || {{}}).content = {script};\n",
        encoding="utf-8",
    )
    size = bundle()

    kinds = [st["kind"] for part in data["parts"] for st in part["stations"]]
    written = set(data["entries"])
    algos = sum(1 for part in data["parts"] for st in part["stations"] if st["kind"] == "algorithm" and st["id"] in written)
    log.info(
        "%d stations, %d written (%d of %d algorithms, %d of %d concepts), %d lab preset(s), %d warning(s)",
        len(kinds), len(written), algos, kinds.count("algorithm"),
        len(written) - algos, kinds.count("concept"), len(data["presets"]), problems.warnings,
    )
    log.info("wrote %s (%.1f MB)", OUT.name, size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
