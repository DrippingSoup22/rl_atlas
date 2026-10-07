# RL Atlas

A visual, interactive guide to reinforcement learning, from the agent–environment loop to PPO and SAC, and on to
where the field goes next: 95 stations in 14 parts, each read as a story, a textbook chapter and a card.
Local and offline: everything is in one file, `rl_atlas.html`.

## Use it

Open `rl_atlas.html` in a browser. The home screen is a map: each station is one idea, and the
stations you have read fill with color. It has three views (the metro map in reading order, a tree per
line of what builds on what, and Sutton & Barto's unified view with every algorithm placed) and filters by label,
in a side panel opened by **Views & filters** at the top left (or `V`). Drag to move and Ctrl + scroll
to zoom; the map always stays in view.
Each algorithm can be read three ways, in this order, and then watched:

- **Story**: the idea. Scroll the text, and the picture follows it step by step.
- **Textbook**: the theory, like a book chapter: definitions, derivations, theorems, figures and references.
- **Card**: the summary: the formula, pseudocode, perks, flaws, knobs and questions on one sheet.
- **Lab**: watch it learn, one pseudocode line at a time or 50 episodes per second, and see how often its
  settings end well over many runs, and how those odds move with each knob. The world and the display options sit
  in a drawer opened by **World & display** in the play bar (Esc closes it).

Hover any colored symbol to light up every symbol of its kind; hover any underlined term for a short
explanation. `Ctrl K` searches the whole atlas.

## Build and check

```
python build.py
node --test
```

`build.py` (Python 3.11+, standard library only) checks `content/`, writes `app/content.js` and `app/recordings.js`,
and bundles `rl_atlas.html`. It stops on broken links and lists what is still missing. `node --test` (Node 22+)
checks the Lab against the textbook results it teaches. After a build, `app/index.html` also opens
directly, which is handier while editing.

The three generated files are committed with their sources, so build before committing. `main` is always the guide
as it stands, and every commit on it builds with no warnings and passes the tests. Work is committed straight to
`main` in such steps. A branch, when one is needed, is short-lived: it ends merged into `main` and deleted, or just
deleted.

## Folders

| Folder | What is in it |
| --- | --- |
| `content/` | `map.toml` (every station in reading order, with each algorithm's parent and labels), `lab.toml` (Lab presets), `notation.toml` (symbols page), one Markdown file per written entry, and `recordings/` (what the recorder wrote) |
| `app/` | the page: `index.html`, `css/` and `js/` (shell, map, pages, stories and their scenes, textbook, diagrams, figures, demos, math), and the generated `content.js` and `recordings.js` |
| `lab/` | worlds, features, algorithms, runs (computed live, or played back from recordings) and dynamic programming (no DOM, also used by the tests), and their views; `lab/deep/` is the trainer of the networks (deterministic math, so a seed gives the same run in Node and in any browser), which the page also runs in a Web Worker for seeds and settings off the recordings |
| `vendor/` | KaTeX 0.19 (MIT license) |
| `tests/` | `lab.test.js` |
| `recorder/` | the recorders of the runs with neural networks. `node recorder/deep.js [name …]` trains with `lab/deep/`, one thread per core, resumable (`recorder/.cache`), and `--sweep` makes the sweeps. The older NumPy recorder (NumPy 2.4 and Gymnasium 1.4) stays as a cross-check: `python recorder/record.py [name …]` trains 20 seeds and writes `content/recordings/<name>.json` (every seed's training and test returns, and one seed's snapshots), and `--sweep` writes `content/recordings/sweeps/<name>.json` (each knob's values over many seeds); `build.py` bundles both. `RECORDINGS_OUT` sends them elsewhere, to compare before replacing |

## Writing an entry

An entry is `content/<part folder>/<id>.md`, where `<id>` is a station in `map.toml`. It starts with TOML
front matter between `+++` lines: `summary`, `change` (what changed from the parent), `prereqs`, `lab`, `sources`,
`story_in` (for an entry without a story of its own: the stations whose stories show its idea at work),
and for a story a `[story]` table: its scene (`grid`, `loop`, `timeline`, `mdp`, or one of the Lab's views: `bandit`,
`chain`, `cards`, `graph`, `line`, `car`, `star`, `corridor`, `throw`, `cartpole`, `pendulum`), the scene's settings, and a formula whose pieces are wrapped in `\step{n}{…}`. Scenes on
a Lab view, and grid stories that replay dynamic programming or Monte Carlo, name their runs in a `[story.runs]`
table (`name = { algorithm, <knobs> }`, or `name = { recording }` for a run trained offline); a step then picks a run and a moment (`run`, `at`), can replay some units
(`play`, `pace`), or only the first few updates of one (`updates`; with `instant`, without animation, to jump to a moment inside a unit), or hold a few moments of the run in turn (`checkpoints`, a list of units
or a count for evenly spaced ones; `hold`, in milliseconds), and can chart runs averaged over many seeds (`curves`,
`metric`, `domain` to fix the range, `log` for a log axis, `ref = [value, "label"]` for a reference line and `title` for a title of its own). Grid steps can paint a batch's advantages on the move triangles (`advantages`),
and a `[story.numbers]` table names lines of numbers that steps show under the formula (`numbers = "name"`).
A story runs on a seed picked because it proves its point clearly; how often settings succeed over many seeds is
the Lab's job. Every number in a story comes from its runs.
[q-learning.md](content/05-temporal-difference/q-learning.md) is a complete algorithm,
[bellman.md](content/01-problem/bellman.md) a complete concept, and [epsilon-greedy.md](content/02-bandits/epsilon-greedy.md)
a story on Lab runs.

The body has up to three parts: a `## Story` made of `::: step {…}` blocks (the braces say what the picture
shows), a `## Textbook` and a `## Card`, both made of `###` sections. On top of plain Markdown you can use:

- `[[station]]` or `[[station|text]]` for links that explain themselves on hover, and `[text](lab:preset)` for Lab links;
- `$…$` and `$$…$$` math with the color macros `\val \rew \pol \err` and `\alp \gam \eps \lam \del`;
- `{{demo arg}}` for a diagram, figure or demo: `backup` (bandit, mc, mc-q, td0, sarsa, q-learning, expected-sarsa,
  double-q, ddpg, td3, sac, n-step-td, n-step-sarsa, lambda, lambda-q, v-pi, q-pi, v-star, q-star, reinforce, baseline, actor-critic, a2c, gae), `gridworld` (random, optimal), `cliff-paths`, `cliff-curves`, `cliff-alpha`,
  `loop`, `mdp-graph` (robot), `discount`, `be-the-agent`, `testbed`, `sample-average`, `step-weights`, `step-sizes`,
  `bandit-curves` (epsilon, optimistic, ucb, gradient, drift), `bandit-study`, `dp-sweeps`, `frozen` (pi, optimal),
  `dp-race`, `gpi`, `blackjack-values`, `blackjack-policy`, `blackjack-match`, `frozen-mc`, `is-blackjack`,
  `is-infinite`, `random-walk` (values, error, batch), `max-bias`, `n-step-study`, `lambda-study` (offline, online),
  `lambda-weights`, `trace-shapes`, `n-step-paths` (lambda), `dyna-architecture`, `dyna-curves`, `dyna-midway`,
  `changing-maze` (blocking, shortcut), `expected-vs-sample`, `sweeping-curves`, `touch-tiles`, `feature-shapes`,
  `coarse-widths`, `walk-fit` (mc), `walk-alpha`, `walk-n-study`, `basis-study`, `tiling-study`, `car-surfaces`, `car-curves` (n),
  `baird-weights`, `corridor-values`, `softmax-play`, `gaussian-play`, `pg-estimates`, `reinforce-alpha`, `baseline-curves`,
  `critic-speed`, `entropy-study`, `gae-weights`, `gae-study`, `trust-region` (clip), `ppo-clip`. A backup label breaks into lines at `\n`, and
  a display formula too wide for its column shrinks a little, then stacks the parts written side by side with `\qquad`;
- `::: pseudocode` (end a line with `{#id}` to link it to the Lab's step-by-step mode), `::: question` (answer after `---`) and `::: analogy`.

The Textbook numbers its sections, figures and statements, and any display equation with a `\label{name}`;
`\ref{name}` links to them (§2, (3), Figure 1, Theorem 1). Its blocks are `::: definition`, `theorem`, `lemma`,
`example` (all written `::: kind {#name} Title`), `remark`, `proof` (`::: proof Proof idea` renames it),
`::: figure {#name}` (a `{{demo}}` line, then the caption) and `::: algorithm {#name} Title` (pseudocode lines).

## License

[MIT](LICENSE). KaTeX, in `vendor/katex/`, keeps its own MIT license.
