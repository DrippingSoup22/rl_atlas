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
FRONT_MATTER = {"summary", "change", "prereqs", "lab", "sources", "story", "story_in"}
# The keys of a Lab preset (content/lab.toml); any other key is a knob, and what its charts may plot.
# Lab algorithms that teach a concept station, each with its station: DPG on the throw, offline RL on the cliff,
# imitation on the ice bridge, planning in a learned model among hidden cliffs. Their pseudocode is the station's.
RUNNABLE_CONCEPTS = {"dpg": "dpg", "offline-q": "offline-rl", "offline-bcq": "offline-rl", "bc": "imitation", "dagger": "imitation",
                     "model-planner": "model-based-deep"}
PRESET_KEYS = {"title", "env", "algorithms", "racers", "units", "runs", "charts", "measures", "film", "intro", "success", "sweep", "seeds", "home", "fixed"}
CHARTS = {"return", "steps", "optimal", "left", "delta", "error", "optimal-error", "match", "greedy", "ve", "weights",
          "policy-value", "right", "aim", "kl", "clipped", "deployed", "labels"}
# What a preset of recorded runs (racers that name a recording) can chart: from the recordings, nothing is recomputed.
RECORDED_CHARTS = {"return", "test", "steps", "loss", "td", "q", "eps", "kl", "clipped", "jitter", "jitter-train"}
RECORDINGS = CONTENT / "recordings"
DEEP_WORKER = ["dmath", "nn", "worlds", "dqn", "pg", "ac", "record", "worker"]  # lab/deep/, in load order: the Worker's text
MEASURES = {"error", "optimal-error", "match", "greedy", "ve", "policy-value", "right", "aim", "deployed"}
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

# The appendix: short explanations of the mathematics the atlas uses, off the map, reached by [[a:id]] links. Its
# groups, in the order its index lists them.
APPENDIX = CONTENT / "appendix"
APPENDIX_FRONT = {"title", "summary", "group", "sources"}
APPENDIX_GROUPS = ("Notation", "Probability", "Sums and series", "Linear algebra", "Analysis", "Calculus and optimization")

FRONT = re.compile(r"\+\+\+\n(.*?)\n\+\+\+\n", re.S)
BLOCK = re.compile(r"^::: *([\w-]+) *(.*?)\n(.*?)^:::[ \t]*$", re.M | re.S)
# Inline math takes along an opening bracket just before it and the punctuation just after it ("$x$," "($x$)" and
# "$n$-step"), so that a line never ends or begins between a formula and its punctuation.
MATH = re.compile(r"\$\$(.+?)\$\$|([(“‘]?)\$([^\n$]+?)\$((?:[.,;:!?)'’”…]|-\w+)*)", re.S)
TERM = re.compile(r"\[\[((?:a:)?[\w-]+)(?:\|([^\]]+))?\]\]")  # [[station]], [[station|label]], [[a:appendix-entry|label]]
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)([.,;:!?)]*)")
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

    appendix: dict = {}  # the appendix entries' front matter, by id, for [[a:id]] links
    used: dict = {}  # appendix id -> the files that link to it

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
        # an operator in prose ("old guess + step size × ...") stays on the line of the word before it
        text = re.sub(r" ([−×=+≈≤≥→←]) ", "\u00a0\\1 ", text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
        text = re.sub(r"\\ref\{([\w-]+)\}", self._ref, text)
        text = TERM.sub(self._term, text)
        return LINK.sub(self._link, text)

    def _term(self, m: re.Match) -> str:
        sid, label = m.group(1), m.group(2)
        if sid.startswith("a:"):
            aid = sid[2:]
            if aid not in self.appendix:
                self.problems.error(self.where, f"[[{sid}]]: no such appendix entry (content/appendix/{aid}.md)")
                return label or aid
            self.used.setdefault(aid, set()).add(self.where)
            label = label or html.escape(self.appendix[aid]["title"])
            return f'<a class="term appx" data-appendix="{aid}" href="#/a/{aid}">{label}</a>'
        if sid not in self.stations:
            self.problems.error(self.where, f"[[{sid}]] is not a station on the map")
            return label or sid
        label = label or html.escape(self.stations[sid]["title"])
        return f'<a class="term" data-term="{sid}" href="#/e/{sid}">{label}</a>'

    def _link(self, m: re.Match) -> str:
        text, href, after = m.group(1), m.group(2), m.group(3)
        if href.startswith("lab:"):
            preset = href[4:]
            if preset not in self.presets:
                self.problems.error(self.where, f"lab preset '{preset}' is not in content/lab.toml")
            link = f'<a class="lab-link" href="#/lab/{preset}">{text}</a>'
            # a lab link is a pill, after which a line can break: keep the punctuation that follows it on its line
            return f'<span class="nobr">{link}{after}</span>' if after else link
        if href.startswith(("http://", "https://")):
            return f'<a href="{href}" target="_blank" rel="noopener">{text}</a>{after}'
        if href.startswith("#/"):
            return f'<a href="{href}">{text}</a>{after}'
        self.problems.error(self.where, f"unknown link target '{href}'")
        return text + after

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
                before, tex, after = m.group(2), m.group(3), m.group(4)
                span = f'<span class="tex">{html.escape(tex)}</span>'
                slots.append(f'<span class="nobr">{html.escape(before)}{span}{html.escape(after)}</span>' if before or after else span)
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
    for ref in meta.get("prereqs", []) + meta.get("story_in", []):
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
        # an entry without a story of its own can point to the stories that show its idea at work
        "storyIn": meta.get("story_in", []),
        "lab": meta.get("lab"),
        "sources": meta.get("sources", []),
        "story": story,
        "textbook": textbook,
        "card": card,
        "pseudocode": pseudocode,
    }


def read_appendix(problems: Problems) -> dict:
    """The front matter of every appendix entry, read before anything is compiled, so that links can be checked."""
    metas = {}
    for path in sorted(APPENDIX.glob("*.md")):
        where = str(path.relative_to(ROOT))
        front = FRONT.match(path.read_text(encoding="utf-8"))
        if not front:
            problems.error(where, "missing the +++ front matter")
            continue
        try:
            meta = tomllib.loads(front.group(1))
        except tomllib.TOMLDecodeError as exc:
            problems.error(where, f"front matter: {exc}")
            continue
        for field in ("title", "summary", "group"):
            if not meta.get(field):
                problems.error(where, f"no '{field}'")
        for field in meta.keys() - APPENDIX_FRONT:
            problems.warn(where, f"unknown front matter field '{field}'")
        if meta.get("group") and meta["group"] not in APPENDIX_GROUPS:
            problems.error(where, f"group '{meta['group']}' is not one of: {', '.join(APPENDIX_GROUPS)}")
        metas[path.stem] = meta
    return metas


def compile_appendix(aid: str, meta: dict, site: dict, problems: Problems) -> dict:
    """An appendix entry: its text, written like a Textbook (numbered equations, definitions, theorems, examples), with
    an introduction before its first ### section."""
    path = APPENDIX / f"{aid}.md"
    where = str(path.relative_to(ROOT))
    raw = path.read_text(encoding="utf-8")
    body = raw[FRONT.match(raw).end():]
    tb = Markdown(site["stations"], site["presets"], where, problems, textbook=True)
    first = re.search(r"^### ", body, flags=re.M)
    intro = body[: first.start()] if first else body
    sections = [{"id": "intro", "title": "", "html": tb.rich(intro)}] if intro.strip() else []
    for title, text in split_headings(body, "###"):
        title, sid = heading_id(title)
        sections.append({"id": sid, "title": tb.inline(title), "html": tb.rich(text)})
    for section in sections:
        section["title"], section["html"] = tb.resolve(section["title"]), tb.resolve(section["html"])
    return {"title": meta["title"], "summary": meta["summary"], "group": meta["group"],
            "sources": meta.get("sources", []), "sections": sections}


def compile_preset(pid: str, raw: dict, stations: dict, recordings: dict, problems: Problems) -> dict:
    """A Lab preset in one shape: its racers (each an algorithm, a name and knobs of its own, or a recorded run) and the
    shared knobs."""
    where = f"content/lab.toml [{pid}]"
    for key in ("title", "env"):
        if key not in raw:
            problems.error(where, f"no '{key}'")
    racers = []
    for r in raw.get("racers") or [{"algorithm": a} for a in raw.get("algorithms", [])]:
        if "recording" in r:  # a run recorded offline by recorder/record.py: its algorithm is the recording's station
            rec = recordings.get(r["recording"])
            if not rec:
                problems.error(where, f"races recording '{r['recording']}', which is not in content/recordings/")
            elif rec["world"] != raw.get("env"):
                problems.error(where, f"recording '{r['recording']}' is on '{rec['world']}', not on '{raw.get('env')}'")
            else:
                racers.append({"recording": r["recording"], "algorithm": rec["station"], "name": r.get("name") or rec["title"], "params": {}})
            continue
        algo = r.get("algorithm", "")
        if stations.get(algo, {}).get("kind") != "algorithm" and algo not in RUNNABLE_CONCEPTS:
            problems.error(where, f"runs '{algo}', which is not an algorithm station")
            continue
        own = {k: v for k, v in r.items() if k not in ("algorithm", "name")}
        racers.append({"algorithm": algo, "name": r.get("name") or stations[RUNNABLE_CONCEPTS.get(algo, algo)]["title"], "params": own})
    if not racers:
        problems.error(where, "runs no algorithm")
    charts, measures = raw.get("charts", ["return"]), raw.get("measures", [])
    recorded = any("recording" in r for r in racers)
    if recorded and not all("recording" in r for r in racers):
        problems.error(where, "races recorded and live runs together")
    for chart in charts:
        if recorded:
            if chart not in RECORDED_CHARTS:
                problems.error(where, f"cannot chart '{chart}' from recordings (one of: {', '.join(sorted(RECORDED_CHARTS))})")
        elif chart not in CHARTS:
            problems.error(where, f"cannot chart '{chart}' (one of: {', '.join(sorted(CHARTS))})")
        elif chart in MEASURES and chart not in measures:
            problems.error(where, f"charts '{chart}' but does not measure it (add it to measures)")
    for m in measures:
        if m not in MEASURES:
            problems.error(where, f"unknown measure '{m}'")
    success = raw.get("success")
    if success is not None:  # what counts as a run that ended well, for the odds the Lab counts over many seeds
        units = raw.get("units", 500)
        metric, window = success.get("metric"), success.get("window")
        if set(success) - {"metric", "min", "max", "window", "text"}:
            problems.error(where, f"success has unknown keys: {', '.join(sorted(set(success) - {'metric', 'min', 'max', 'window', 'text'}))}")
        if recorded:  # only the curves every seed of a recording kept can be judged over seeds
            kept = {"return": "train", "test": "test", "q": "q", "jitter": "jitter", "jitter-train": "jitterTrain"}
            if metric not in kept:
                problems.error(where, f"success judges '{metric}', which recordings do not keep for every seed (one of: {', '.join(kept)})")
            elif not all(kept[metric] in c for r in racers if r.get("recording") in recordings for c in recordings[r["recording"]]["curves"]):
                problems.error(where, f"success judges '{metric}', but not every seed of its recordings kept it (record them again)")
        elif metric not in CHARTS:
            problems.error(where, f"success judges '{metric}', which no run records (one of: {', '.join(sorted(CHARTS))})")
        elif metric in MEASURES and metric not in measures:
            problems.error(where, f"success judges '{metric}' but the preset does not measure it (add it to measures)")
        if ("min" in success) == ("max" in success):
            problems.error(where, "success needs exactly one of min and max")
        if not success.get("text"):
            problems.error(where, "success needs a text: what the runs that end well do, as in \"how many <text>\"")
        if window is not None and not (isinstance(window, list) and len(window) == 2 and 1 <= window[0] <= window[1] <= units):
            problems.error(where, f"success window must be [first, last] units within 1 to {units}")
    sweep = raw.get("sweep")
    if sweep is not None and not (isinstance(sweep, dict) and all(isinstance(v, list) and len(v) > 1 and all(isinstance(x, (int, float)) for x in v) for v in sweep.values())):
        problems.error(where, "sweep must map knobs to lists of at least two numbers, as in sweep = { q0 = [-100.0, 0.0] }")
    fixed = raw.get("fixed", [])
    if not (isinstance(fixed, list) and all(isinstance(k, str) for k in fixed)):
        problems.error(where, 'fixed must list knobs, as in fixed = ["gamma"]: those that change nothing in this lab, kept off its bar')
    elif not any(r.get("recording") for r in raw.get("racers", [])) and (stray := [k for k in fixed if k not in raw]):
        problems.error(where, f"fixed names knobs the lab does not set: {', '.join(stray)}")
    if not isinstance(raw.get("seeds", False), bool):
        problems.error(where, "seeds must be true or false (true, for recorded runs: the charts open on the thin lines of every seed)")
    return {
        "title": raw.get("title", pid), "env": raw.get("env", ""), "racers": racers,
        "units": raw.get("units", 500), "seed": 1, "runs": raw.get("runs", 1),
        "charts": charts, "measures": measures, "film": raw.get("film", []), "intro": raw.get("intro", ""),
        "params": {k: v for k, v in raw.items() if k not in PRESET_KEYS},
        **({"success": success} if success is not None else {}),
        **({"sweep": sweep} if sweep is not None else {}),
        **({"seeds": True} if raw.get("seeds") is True else {}),
        **({"fixed": fixed} if fixed else {}),
        **({"home": raw["home"]} if raw.get("home") else {}),
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
    unified = atlas.get("unified", {})
    columns = [c["id"] for c in unified.get("columns", [])]
    rows = {r["id"]: r for r in unified.get("rows", [])}
    placed: dict[str, str] = {}
    for row in rows.values():
        if row.get("band"):
            if row["band"] not in atlas["lines"]:
                problems.error("content/map.toml", f"[unified] row '{row['id']}' spreads unknown line '{row['band']}'")
            for sid, st in stations.items():
                if st.get("line", parts[st["part"]]["line"]) == row["band"]:
                    placed.setdefault(sid, row["id"])
    for rid, cells in unified.get("cells", {}).items():
        if rid not in rows or rows[rid].get("band"):
            problems.error("content/map.toml", f"[unified.cells] fills '{rid}', which is not a row of cells")
        for cell in cells:
            span = cell["at"].split("-")
            if not 1 <= len(span) <= 2 or any(c not in columns for c in span) or columns.index(span[0]) > columns.index(span[-1]):
                problems.error("content/map.toml", f"[unified.cells] {rid}: '{cell['at']}' is not a column or a span of columns")
            for sid in cell["stations"]:
                if sid not in stations:
                    problems.error("content/map.toml", f"[unified.cells] {rid} places unknown station '{sid}'")
                elif sid in placed:
                    problems.error("content/map.toml", f"[unified] places '{sid}' twice ({placed[sid]} and {rid})")
                placed[sid] = rid
    for sid in stations:
        if sid not in placed:
            problems.error("content/map.toml", f"station '{sid}' has no place in [unified]")
    for sid in MACRO_TERMS:
        if sid not in stations:
            problems.error("app/js/math.js", f"a math macro links to unknown station '{sid}'")
    recordings = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(RECORDINGS.glob("*.json"))}
    for name, rec in recordings.items():
        missing = [k for k in ("world", "station", "learner", "snapshots", "curves", "grid", "block") if k not in rec]
        if missing:
            problems.error(f"content/recordings/{name}.json", f"lacks {', '.join(missing)}")
        elif rec["station"] not in stations:
            problems.error(f"content/recordings/{name}.json", f"follows unknown station '{rec['station']}'")
    presets = {pid: compile_preset(pid, raw, stations, recordings, problems) for pid, raw in presets.items()}
    for sym in notation.get("symbol", []):
        if sym.get("station") and sym["station"] not in stations:
            problems.error("content/notation.toml", f"symbol {sym['tex']} points to unknown station '{sym['station']}'")

    site = {"stations": stations, "presets": presets, "labels": atlas["labels"]}
    Markdown.appendix, Markdown.used = read_appendix(problems), {}
    entries = {}
    for path in sorted(CONTENT.glob("*/*.md")):
        if path.parent == APPENDIX:
            continue
        entry = compile_entry(path, site, problems)
        if entry:
            entries[path.stem] = entry
    appendix = {aid: compile_appendix(aid, meta, site, problems) for aid, meta in Markdown.appendix.items()
                if all(meta.get(f) for f in ("title", "summary", "group"))}
    # Where each entry is used: the stations that link to it, in reading order, and the appendix entries linked to or
    # from it.
    order = {sid: i for i, sid in enumerate(stations)}
    for aid, item in appendix.items():
        users = [ROOT / w for w in Markdown.used.get(aid, ())]
        mine = str((APPENDIX / f"{aid}.md").relative_to(ROOT))
        item["usedIn"] = sorted({u.stem for u in users if u.parent != APPENDIX and u.stem in stations}, key=order.get)
        item["related"] = sorted({u.stem for u in users if u.parent == APPENDIX and u.stem != aid}
                                 | {other for other, ws in Markdown.used.items() if mine in ws and other != aid})
        if not item["usedIn"] and not item["related"]:
            problems.warn(f"content/appendix/{aid}.md", "nothing links to this entry")
    for eid, entry in entries.items():
        for ref in entry["storyIn"]:
            if not entries.get(ref, {}).get("story"):
                problems.error(f"content/{eid}.md", f"story_in names '{ref}', which has no story")
        # a story's runs: the recordings they play must exist, and every step must name runs the story has
        if not entry["story"]:
            continue
        runs = entry["story"]["config"].get("runs", {})
        for name, run in runs.items():
            if isinstance(run, dict) and run.get("recording") and run["recording"] not in recordings:
                problems.error(f"content/{eid}.md", f"story run '{name}' plays recording '{run['recording']}', which is not in content/recordings/")
        for k, step in enumerate(entry["story"]["steps"], 1):
            st = step["state"]
            for name in [st.get("run")] + list(st.get("curves", [])):
                if name and name not in runs:
                    problems.error(f"content/{eid}.md", f"story step {k} names run '{name}', which [story.runs] lacks")
            if st.get("curves") and all(isinstance(runs.get(n), dict) and runs[n].get("recording") for n in st["curves"]) and st.get("metric", "optimal") not in RECORDED_CHARTS:
                problems.error(f"content/{eid}.md", f"story step {k} charts '{st.get('metric', 'optimal')}' from recordings (one of: {', '.join(sorted(RECORDED_CHARTS))})")
    for pid, preset in presets.items():
        for algo in sorted({r["algorithm"] for r in preset["racers"]}):
            if not entries.get(RUNNABLE_CONCEPTS.get(algo, algo), {}).get("pseudocode"):
                problems.warn("content/lab.toml", f"preset '{pid}' runs '{algo}', which has no written pseudocode yet")

    return {
        "rows": atlas["rows"],
        "rowTitles": atlas["row_titles"],
        "lines": atlas["lines"],
        "labels": atlas["labels"],
        "parts": parts,
        "unified": atlas.get("unified", {}),
        "entries": entries,
        "appendix": appendix,
        "appendixGroups": list(APPENDIX_GROUPS),
        "presets": presets,
        "notation": notation,
    }, recordings


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
    data, recordings = compile_content(problems)
    if problems.errors:
        log.error("%d error(s); nothing was written", problems.errors)
        return 1

    script = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (APP / "content.js").write_text(
        "// Generated by build.py from content/. Do not edit.\n"
        f"(globalThis.RL = globalThis.RL || {{}}).content = {script};\n",
        encoding="utf-8",
    )
    # each recording's sweeps (recorder/record.py --sweep): its knobs over several values, training curves only
    sweeps = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted((RECORDINGS / "sweeps").glob("*.json"))}
    (APP / "recordings.js").write_text(
        "// Generated by build.py from content/recordings/, which recorder/record.py writes. Do not edit.\n"
        f"(globalThis.RL = globalThis.RL || {{}}).recordings = {json.dumps(recordings, separators=(',', ':'))};\n"
        f"globalThis.RL.sweeps = {json.dumps(sweeps, separators=(',', ':'))};\n",
        encoding="utf-8",
    )
    # the averages the story charts show, worked out offline (tools/story-curves.js); a chart missing there is averaged live
    curves_file = CONTENT / "story-curves.json"
    curves = json.loads(curves_file.read_text(encoding="utf-8")) if curves_file.exists() else {}
    (APP / "story-curves.js").write_text(
        "// Generated by build.py from content/story-curves.json, which tools/story-curves.js writes. Do not edit.\n"
        f"(globalThis.RL = globalThis.RL || {{}}).storyCurves = {json.dumps(curves, separators=(',', ':'))};\n",
        encoding="utf-8",
    )
    # the deep trainer (lab/deep/) as one text, from which the page starts its Web Worker
    worker = "\n".join((ROOT / "lab" / "deep" / f"{name}.js").read_text(encoding="utf-8") for name in DEEP_WORKER)
    (APP / "deep-worker.js").write_text(
        "// Generated by build.py from lab/deep/. Do not edit.\n"
        f"(globalThis.RL = globalThis.RL || {{}}).deepWorker = {json.dumps(worker)};\n",
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
