+++
summary = "A map of reinforcement learning: one idea per station, three ways to read each one, and a Lab to watch it learn."
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., the main source of the atlas", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, for the deep methods", url = "https://spinningup.openai.com/" },
]
+++

## Card

### Idea

The atlas is a map of reinforcement learning, from the agent–environment loop to PPO and SAC, and on to where the field goes next. Every **station** is one idea: a concept such as the [[discount|discount factor]], or an algorithm such as [[q-learning]]. Stations come in order along **lines**, and each one builds on the ones before it on its line, so the best way to read is from the top left, part by part. You can also jump anywhere: every underlined term explains itself when you hover it.

::: analogy
A metro map of a city you are about to explore. You can ride the lines in order, or get off wherever something catches your eye; the map always shows where you are.
:::

### The map {#map}

- Stations you have visited fill with color, so the map shows how far you have come.
- **Metro** shows the curriculum in reading order: the foundations, then methods that keep a table, then methods that scale.
- **Tree** gives each line its own tree, read from the top: every station grows from the one it builds on, and an algorithm from the one it changes. Most algorithms are their parent plus **one change**. Hover a station to light up the path that leads to it.
- **Unified** places every algorithm by how far its updates look ahead and whether they sample or average.
- **Lines** in the panel count what you have read; point at one to light up its stations, click it to bring them into view.
- **Show only** dims every algorithm without the chosen labels, such as *off-policy* or *model-based*.

### Reading an entry {#reading}

Each algorithm is read three ways, in this order:

| | |
| --- | --- |
| **Story** | the idea: scroll the text, and the picture follows it step by step |
| **Textbook** | the theory, like a book chapter: definitions, derivations, theorems, figures, references |
| **Card** | the summary: formula, pseudocode, perks, flaws, knobs and questions on one sheet |

Then the **Lab** shows it learning: one pseudocode line at a time, or fifty episodes a second. It also counts how often its settings work over many seeds, and sweeps any knob to show where it works and where it fails. The deep methods' Labs play back networks trained offline, 20 seeds each. Concepts use the views that help them.

### Colors {#colors}

The same quantity always has the same color, in formulas and in pictures:

| | |
| --- | --- |
| value | $\val{V(s)},\ \val{Q(s,a)}$ |
| reward and return | $\rew{R},\ \rew{G}$ |
| policy | $\pol{\pi(a \mid s)}$ |
| surprise | $\del,\ \err{A}$ |
| knobs, in ink | $\alp,\ \gam,\ \eps,\ \lam$ |

Hover any colored symbol and every symbol of its kind lights up. The [Symbols page](#/symbols) lists them all.

### Where to start {#start}

- New to reinforcement learning: play [[be-the-agent]] for two minutes, then read Part 1 in order, from [[agent-environment]] to [[reward-design]].
- Know the basics: go straight to [[sarsa]] and [[q-learning]], and race them in the Lab.
- Training deep agents: read [[neural-networks]] and [[dqn]], and keep the Toolbox close: the [[hyperparameters|hyperparameter guide]], [[seeds]] and [[debugging|My agent doesn't learn]].
- Looking for something specific: press `Ctrl K` and type its name.

### Shortcuts {#shortcuts}

- `Ctrl K` or `/` searches the whole atlas.
- In a Textbook, hover a reference such as (3) to read it in place, and click it to jump there.
- In the Lab, `Space` plays and pauses, `→` takes one step.
- The sun or moon at the top right switches between light and dark.
