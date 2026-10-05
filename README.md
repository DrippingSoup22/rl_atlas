# RL Atlas

A visual, interactive guide to reinforcement learning, from the agent–environment loop to PPO and SAC.
Local and offline: everything is in one file, `rl_atlas.html`.

## Use it

Open `rl_atlas.html` in a browser. The home screen is a map: each station is one idea, bright
stations are written, faded ones are planned. It has three views (the metro map in reading order, the
family tree of algorithms, and Sutton & Barto's unified view of tabular methods) and filters by label,
in a side panel opened by **Views & filters** at the top left (or `V`). Drag to move and Ctrl + scroll
to zoom; the map always stays in view.
Each algorithm can be read three ways, in this order, and then watched:

- **Story**: the idea. Scroll the text, and the picture follows it step by step.
- **Textbook**: the theory, like a book chapter: definitions, derivations, theorems, figures and references.
- **Card**: the summary: the formula, pseudocode, perks, flaws, knobs and questions on one sheet.
- **Lab**: watch it learn, one pseudocode line at a time or 50 episodes per second.

Hover any colored symbol to light up every symbol of its kind; hover any underlined term for a short
explanation. `Ctrl K` searches the whole atlas.

## Build and check

```
python build.py
node --test
```

`build.py` (Python 3.11+, standard library only) checks `content/`, writes `app/content.js` and bundles
`rl_atlas.html`. It stops on broken links and lists what is still missing. `node --test` (Node 22+)
checks the Lab against the textbook results it teaches. After a build, `app/index.html` also opens
directly, which is handier while editing.

## Folders

| Folder | What is in it |
| --- | --- |
| `content/` | `map.toml` (every station in reading order, with each algorithm's parent and labels), `lab.toml` (Lab presets), `notation.toml` (symbols page), and one Markdown file per written entry |
| `app/` | the page: `index.html`, `css/` and `js/` (shell, map, pages, stories and their scenes, textbook, diagrams, figures, demos, math) |
| `lab/` | worlds, algorithms, runs and dynamic programming (no DOM, also used by the tests), and their views |
| `vendor/` | KaTeX 0.19 (MIT license) |
| `tests/` | `lab.test.js` |

## Writing an entry

An entry is `content/<part folder>/<id>.md`, where `<id>` is a station in `map.toml`. It starts with TOML
front matter between `+++` lines: `summary`, `change` (what changed from the parent), `prereqs`, `lab`, `sources`,
and for a story a `[story]` table: its scene (`grid`, `loop`, `timeline`, `mdp`, or one of the Lab's views: `bandit`,
`chain`, `cards`, `graph`), the scene's settings, and a formula whose pieces are wrapped in `\step{n}{…}`. Scenes on
a Lab view, and grid stories that replay dynamic programming or Monte Carlo, name their runs in a `[story.runs]`
table (`name = { algorithm, <knobs> }`); a step then picks a run and a moment (`run`, `at`), can replay some units
(`play`, `pace`) and can chart runs averaged over many seeds (`curves`, `metric`).
[q-learning.md](content/05-temporal-difference/q-learning.md) is a complete algorithm,
[bellman.md](content/01-problem/bellman.md) a complete concept, and [epsilon-greedy.md](content/02-bandits/epsilon-greedy.md)
a story on Lab runs.

The body has up to three parts: a `## Story` made of `::: step {…}` blocks (the braces say what the picture
shows), a `## Textbook` and a `## Card`, both made of `###` sections. On top of plain Markdown you can use:

- `[[station]]` or `[[station|text]]` for links that explain themselves on hover, and `[text](lab:preset)` for Lab links;
- `$…$` and `$$…$$` math with the color macros `\val \rew \pol \err` and `\alp \gam \eps \lam \del`;
- `{{demo arg}}` for a diagram, figure or demo: `backup` (bandit, mc, mc-q, td0, sarsa, q-learning, expected-sarsa,
  double-q, v-pi, q-pi, v-star, q-star), `gridworld` (random, optimal), `cliff-paths`, `cliff-curves`, `cliff-alpha`,
  `loop`, `mdp-graph`, `discount`, `be-the-agent`, `testbed`, `sample-average`, `step-weights`, `step-sizes`,
  `bandit-curves` (epsilon, optimistic, ucb, gradient, drift), `bandit-study`, `dp-sweeps`, `frozen` (pi, optimal),
  `dp-race`, `gpi`, `blackjack-values`, `blackjack-policy`, `blackjack-match`, `frozen-mc`, `is-blackjack`,
  `is-infinite`, `random-walk` (values, error, batch), `max-bias`. A backup label breaks into lines at `\n`, and
  a display formula too wide for its column shrinks a little, then stacks the parts written side by side with `\qquad`;
- `::: pseudocode` (end a line with `{#id}` to link it to the Lab's step-by-step mode), `::: question` (answer after `---`) and `::: analogy`.

The Textbook numbers its sections, figures and statements, and any display equation with a `\label{name}`;
`\ref{name}` links to them (§2, (3), Figure 1, Theorem 1). Its blocks are `::: definition`, `theorem`, `lemma`,
`example` (all written `::: kind {#name} Title`), `remark`, `proof` (`::: proof Proof idea` renames it),
`::: figure {#name}` (a `{{demo}}` line, then the caption) and `::: algorithm {#name} Title` (pseudocode lines).
