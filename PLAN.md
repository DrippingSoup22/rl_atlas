# RL Atlas: the plan

## Goal

A guide for someone who knows a little about reinforcement learning and wants a clear, complete view of
it. The algorithms are the main course; everything that works alongside them (γ, bias vs variance, GAE,
exploration…) is explained too. It must be complete, clean, fun to use and appealing at first sight:
you learn by watching things happen, not only by reading.

## How it teaches

- Almost every update is *old guess + step size × (target − old guess)*; algorithms differ mainly in the target.
- Each algorithm is its parent plus one change; perks and flaws follow from that change.
- Every algorithm is read three ways, in this order: **Story** (get the idea), **Textbook** (the theory, like a book
  chapter: complete, dry, section by section, building on the stations before it on its line), **Card** (the summary,
  for a quick look). Then the **Lab** shows it learning. Concepts use the same views where they help.
- A story shows a very good run: a seed picked because it proves the point clearly.
- The Lab shows the odds. Everything here is stochastic, so each experiment says how often its settings succeed over
  many seeds, and a sweep shows how that rate moves with a knob: where the good settings are, and what a bad one
  costs. Bad settings are run on purpose, to be understood, not as warnings.
- The tabular / function-approximation split is a visible line on the map ("Learning with tables" / "Learning at scale").
- Sources: Sutton & Barto (2018), OpenAI Spinning Up and the original papers, in our own words.

The full curriculum, 95 stations in 14 parts, lives in [content/map.toml](content/map.toml).

## Look and feel

- One color per quantity, everywhere: value blue, reward gold, policy violet, surprise (δ, advantage) red; knobs stay in ink.
  The palette was checked with a color validator in both themes (marks and text separately).
- Values on tiles use a diverging scale: orange below zero, gray at zero, blue above.
- Motion explains cause: zoom into what you click, formulas that build themselves, values that ripple back
  from the goal, sparks where an update happens, an agent that hops, falls and respawns.
- Light and dark themes; reduced motion follows the operating system setting.

## The Lab

- Tabular and linear methods, and policy gradients (softmax and Gaussian policies, from REINFORCE to PPO), run live in the browser:
  a whole run is computed at once (milliseconds), then played like a video and replayable line by line.
- Neural-network methods (DQN, A2C, PPO, TRPO, DDPG, TD3, SAC) will be **recorded runs** made by a standalone Python
  recorder (NumPy and Gymnasium; none of RL_lib or Centipede) and played back with the same player,
  with a few pre-recorded variants per knob. The networks are small (two hidden layers of 64 units), so they are
  written in NumPy, gradients by hand: no PyTorch to install, and every recording reproducible bit for bit from its seed.
- Worlds are picked so that values and policies stay pictures: grids, random walks, Blackjack, Mountain Car,
  Pendulum, the short corridor, the throw, CartPole with value and probability bars.

## Technology (decided)

| Part | Choice | Why |
| --- | --- | --- |
| The guide | one offline HTML file, plain JavaScript, SVG and canvas | animation, hover-linking and scrubbing need to run in the browser at 60 fps; opens with a double-click |
| Math | KaTeX, vendored | standard, fast, offline, supports the color and link macros |
| 3D (M4) | a small canvas surface renderer of our own | one surface of a few hundred tiles needs no library; keeps the file small and offline |
| Content | Markdown + TOML front matter, built by `build.py` | readable and editable; Python standard library only |
| Tests | `node --test`, one file | checks the Lab against the book's results |
| Deep RL runs (M6) | Python recorder → data files played in the page | training is heavy and offline; playback is light |

Not a Python app (Streamlit and similar): they redraw on the server for every interaction, so the
animations and hover effects this guide relies on would not be smooth.

## Milestones

- [x] **M0 · Look and feel**: the shell, the metro map, SARSA and Q-learning (story, card), the cliff-walking race in the Lab.
  After review: the Textbook view, with live figures computed by the Lab.
- [x] **M1 · Foundations**: Parts 0–1 (16 entries; every Part 1 concept has a Textbook, seven have a Story) with
  the "be the agent" game, the discount slider, the loop, timeline and MDP scenes, and the gridworld and robot computed
  exactly; the map's family-tree and unified views and its filters. Parents and labels now live in `map.toml`.
- [x] **M2 · Tabular core**: Bandits, DP, Monte Carlo, the rest of TD; Lab worlds (gridworld, Frozen Lake, random walk, Blackjack), filmstrip, seeds, sandbox.
  Parts 2–5 are written: 25 new entries, each with a Textbook and a Card, 18 with a Story whose numbers come from
  Lab runs. The book's figures are recomputed live by the Lab (the testbed, Figure 4.1, policy iteration on Frozen
  Lake, the Blackjack maps, importance sampling, the random walk, step sizes on the cliff, maximization bias), and
  a final pass retold the passages that still followed the book's wording too closely.
- [x] **M3 · Traces and planning**: n-step, TD(λ) with glowing traces, Dyna.
  Parts 6–7 are written: 9 new entries (n-step TD and SARSA, the λ-return, TD(λ), SARSA(λ), models and planning,
  Dyna-Q, Dyna-Q+, prioritized sweeping), each with a Textbook and a Card, 7 with a Story on Lab runs. The Lab runs
  them all, with replay checked by the tests: eligibility traces glow and fade on the tiles, the n-step window shows as
  a bracket, planning updates as dashed sparks (in queue order for prioritized sweeping), fog hides what a model has
  never seen, and maze walls move mid-run. Eight new Lab presets; new worlds: the 19-state random walk, the Dyna,
  blocking and shortcut mazes. Figures recomputed live: Sutton & Barto's 7.2, 7.4, 8.2–8.5, 8.7, 12.3 and 12.6.
- [x] **M4 · Function approximation**: features, the "touch a tile" demo, Mountain Car in 3D, the deadly triad.
  Part 8 is written: 8 entries (why tables break, features, value error, gradient Monte Carlo, semi-gradient TD and
  SARSA, generalization, the deadly triad), each with a Textbook and a Card, 7 with a Story on Lab runs. The Lab learns
  with features (groups, tile coding, polynomials, Fourier cosines, tables, or a world's own) built from scaled
  coordinates; new worlds: the 1000-state walk with its exact values and visit shares, Mountain Car, Baird's
  counterexample. New views: the walk as one value line with the footprint of each update, Mountain Car with a
  cost-to-go landscape you can turn (or a map from above), Baird's star with its weights on a log scale. Five new
  presets. Figures recomputed live: Sutton & Barto's 9.1, 9.2, 9.5, 9.8, 9.10, 10.1–10.3 and 11.2, plus touch a tile.
- [x] **M5 · Policy gradients, live**: gradient bandit, short corridor, REINFORCE, actor–critic, A2C; GAE and PPO demos.
  Part 10 is written: 11 entries (why learn a policy, softmax and Gaussian policies, the policy gradient theorem,
  REINFORCE, REINFORCE with baseline, actor–critic, A2C and A3C, the entropy bonus, GAE, TRPO, PPO), each with a Textbook
  and a Card, 10 with a Story on Lab runs. The Lab learns policies directly: softmax and Gaussian policies over its
  features; REINFORCE with and without a baseline; actor–critic with eligibility traces; and the batch methods A2C, TRPO
  (closed-form natural gradient and line search) and PPO, with GAE advantages and an entropy bonus, played by several
  workers at once, with replay checked by the tests. New worlds: the short corridor, its landscape worked out exactly; the
  throw, a continuous action; two gems. New views: the corridor above its value landscape, the throw with its bell curve;
  grids show workers, TD errors and a batch's advantages. Figures computed exactly or live: Sutton & Barto's Example 13.1
  and Figures 13.1 and 13.2, single-episode gradient estimates, softmax and Gaussian scores to play with, the critic's
  step size, the entropy bonus, GAE's weights and λ, and TRPO's trust region and PPO's clip on the corridor. Stories
  gained chart legends and partial replays.
- [ ] **M6 · Odds and recorded runs**: every experiment shows how often its settings succeed; the recorder; DQN, A2C,
  TRPO and PPO on CartPole and Pendulum. Mountain Car is dropped from the deep runs: with random exploration the car
  almost never reaches the flag, so a network has nothing to learn from.
  Next: a sweep of each deep algorithm's main knob over 5 seeds before recording it, then the Lab's player, the
  remaining recordings and Part 9.
  Done: Mountain Car without optimism (a Lab, linked from Part 8: with γ = 0.99 no run ends up getting out when the values
  start at or below −100, the value of paying 1 forever; just above it the push to explore is feeble, and from −80 up every
  run gets out); the odds in the Lab (success rules on 26 presets, calibrated over 30 to 100 seeds; a tally of the runs that end
  well; a sweep of any knob, charting the share of runs that end well and their score against its values); fixes
  from profiling every preset over many seeds (Frozen Lake's shown seed, two intros, the odds of the failures the
  PPO, TRPO and REINFORCE stories show); the recorder (`recorder/`: NumPy networks checked against numerical gradients, DQN with switchable
  replay, target network, double, dueling and prioritized replay; A2C, PPO and TRPO with exact Fisher-vector products),
  tuned settings (DQN balances CartPole with a squared loss and lr 5·10⁻⁴; A2C, PPO and TRPO balance it within the
  first 20–40% of their training; PPO swings Pendulum up), the first two recordings, their decoder and a CartPole view.
- [ ] **M7 · The rest**: continuous control, toolbox, where next, quizzes everywhere, polish.
  Done: Parts 11–13 written (95 of 95 stations); every story reviewed for calm visuals and concept first (checkpoints
  instead of fast replays); new stories for DPG (a deterministic aim on the throw, with a Lab), on/off-policy,
  exploration and bias–variance; deep recordings over 20 seeds, and their sweeps for DQN, PPO on Pendulum and TRPO.
  Still open:
  - Sweeps over 20 seeds still to fold in (`recorder/record.py --sweep`, then the texts that quote them): A2C and PPO on
    CartPole (a2c.md, the toolbox table), DDPG, TD3 and SAC (their odds paragraphs; DDPG's reward scale in
    normalization.md), and DQN's Huber knob (dqn.md says the squared loss did better).
  - Stories still to write, each on a run that shows the idea: experience replay and the target network (from the
    recorded runs: the no-replay run learns and forgets; without a target the values pass 100, the most a state can be
    worth, within 5,000 steps), offline RL (Q-learning from a fixed log prefers the moves the log never tried),
    imitation (behavior cloning drifts off the expert's path after a slip; DAgger asks the expert there), normalization
    (the same A2C with rewards in other units), model-based deep RL (a planner exploiting a learned model's blind spots).
  - No story, on purpose: which-algorithm (a decision guide, not a run), pg-theorem (a derivation; REINFORCE and the
    baseline show it at work), multi-agent and RLHF (they need worlds with several learners or a learned reward, beyond
    the guide's), MCTS unless a tree view is built, debugging until a fair bug-hunt run is found.
