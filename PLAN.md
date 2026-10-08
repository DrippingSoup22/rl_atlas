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
- Motion explains cause: a page opens out of the station you click, formulas that build themselves, values that ripple back
  from the goal, sparks where an update happens, an agent that hops, falls and respawns.
- Light and dark themes; reduced motion follows the operating system setting.

## The Lab

- Tabular and linear methods, and policy gradients (softmax and Gaussian policies, from REINFORCE to PPO), run live in the browser:
  a whole run is computed at once (milliseconds), then played like a video and replayable line by line.
- Neural-network methods (DQN, A2C, PPO, TRPO, DDPG, TD3, SAC) are **recorded runs** made offline by the page's own
  trainer (`lab/deep/`, run in Node by `recorder/deep.js`; none of RL_lib or Centipede) and played back with the same
  player, each over 20 seeds, with a sweep of its main knobs that shows how the odds of success move with each. The
  networks are small (two hidden layers of 64 units by default), with gradients by hand and math every engine rounds
  alike: no library to install, every recording reproducible bit for bit from its seed, and any other seed or setting
  trainable in the page on request.
- Worlds are picked so that values and policies stay pictures: grids, random walks, Blackjack, Mountain Car,
  Pendulum, the short corridor, the throw, CartPole with value and probability bars.
- Every racer is on screen at once, at the largest size that fits: each arrangement (how many across) is laid out and
  measured. The panes stay under the header, in sight with the knobs, when they are wide enough to read (400 px) or
  not much narrower than they could be; otherwise they take the whole window and playing scrolls them into view. The
  algorithm column (pseudocode and "This step") folds into a tab at the edge, and the panes grow into its room.

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
- [x] **M6 · Odds and recorded runs**: every experiment shows how often its settings succeed; the recorder; DQN, A2C,
  TRPO and PPO on CartPole and Pendulum. Mountain Car is dropped from the deep runs: with random exploration the car
  almost never reaches the flag, so a network has nothing to learn from.
  Done: Mountain Car without optimism (a Lab, linked from Part 8: with γ = 0.99 no run ends up getting out when the values
  start at or below −100, the value of paying 1 forever; just above it the push to explore is feeble, and from −80 up every
  run gets out); the odds in the Lab (success rules on 26 presets, calibrated over 30 to 100 seeds; a tally of the runs that end
  well; a sweep of any knob, charting the share of runs that end well and their score against its values); fixes
  from profiling every preset over many seeds (Frozen Lake's shown seed, two intros, the odds of the failures the
  PPO, TRPO and REINFORCE stories show); the recorder (`recorder/`: NumPy networks checked against numerical gradients, DQN with switchable
  replay, target network, double, dueling and prioritized replay; A2C, PPO and TRPO with exact Fisher-vector products),
  tuned settings (DQN balances CartPole with a squared loss and lr 5·10⁻⁴; A2C, PPO and TRPO balance it within the
  first 20–40% of their training; PPO swings Pendulum up), the first two recordings, their decoder and a CartPole view.
- [x] **M7 · The rest**: continuous control, toolbox, where next, quizzes everywhere, polish.
  Done: Parts 11–13 written (95 of 95 stations); every story reviewed for calm visuals and concept first (checkpoints
  instead of fast replays); new stories for DPG (a deterministic aim on the throw, with a Lab), on/off-policy,
  exploration and bias–variance; deep recordings over 20 seeds, and their sweeps for DQN, PPO on Pendulum and TRPO.
  Then: stories for experience replay, the target network, offline RL, imitation, normalization, model-based deep RL
  and MCTS (UCT on a tic-tac-toe position, with a tree view and the odds of flat Monte Carlo against the tree); sweeps over
  10 to 20 seeds per value for every recorded algorithm, folded into the texts that quote them (A2C and PPO on CartPole,
  DQN's Huber knob, DDPG's τ, noise and reward scale, TD3's delay and target noise, SAC's α and τ); a third quiz question
  wherever an entry had two; a one-by-one review of every textbook against Sutton & Barto and newer sources; a review
  of every story and card: stories play at human pace (a move every 300–400 ms, long episodes shown by their first moves
  and their end), wordy steps cut to the idea and one telling number, and every step's picture checked.
  The map's other views completed: the tree gives every line its own tree of what builds on what, stacked in reading
  order, and the unified view places all 37 algorithms (bandits in a strip of their own); the panel became a view switch,
  a legend that counts what you have read and lights up a line, and the filters. The hand shows only while dragging; a
  click opens the page out of the station instead of flying the camera in.
  Decided:
  - No story, on purpose: which-algorithm (a decision guide, not a run), pg-theorem (a derivation; REINFORCE and the
    baseline show it at work), multi-agent and RLHF (they need worlds with several learners or a learned reward, beyond
    the guide's), debugging until a fair bug-hunt run is found.

- [x] **M8 · Honest runs**: stories show a run chosen offline that sells the idea; the Lab lets anyone test settings
  and seeds. One phase at a time, each reviewed with screenshots before the next. The deep presets (recorded on
  CartPole and Pendulum) wait until the live ones work; they then adopt the same method.

  **Principles**
  - *Stories sell the idea, with runs chosen offline.* For every story run, a script searches knobs and seeds
    offline, over many seeds, for a good representative: the idea shows cleanly, the run is good but not a freak, and
    the knobs are ones that work in general. The choice is then pinned: same seed and knobs give the same run on every
    machine, and a test stores a fingerprint of each pinned run, so a code change that alters one fails loudly instead
    of quietly changing the story. Nothing is justified in the story. One faint line at its end names the seed and
    settings ("Runs: seed 58, ε = 0.1, 1,000 pulls. Enter them in the Lab to replay."), so anyone can check.
  - *The Lab measures; it does not sell.* Two things decide a run: the settings and the luck (the seed). The Lab keeps them
    apart.
    - A **bench** of 20 seeds (1–20, the same for every setting) runs live in the browser whenever a knob changes. It
      gives the odds and the spread.
    - The run played is, by default, the bench's **typical** seed for those settings (the median). Good settings show a
      good typical run, bad settings a bad one.
    - Anyone can instead type a seed or roll the dice. It plays, and the Lab shows where it falls among the bench.
  - Races share their seed, so the comparison is fair: the typical seed of a race is the one closest to every racer's
    median. Odds are counted on shared seeds ("UCB wins on 17 of 20").
  - *Charts stay clean.* No bundle of thin lines. Over time: the played run as a line, the bench as a soft band (the
    middle half, the median as a faint line). At the end: a **seed strip** per racer, the 20 final scores as dots on one
    row, the success threshold as a mark that splits them into the odds, and the played seed ringed.

  **Phase 1 · The bench (lab engine, no visible change yet)** (done)
  - The bench: seeds 1–20 per racer, run in the background in small slices (as the odds already are). Results are cached
    by settings, so going back to a setting is instant.
  - Typical seed: the median of the success score (or of the preset's main measure when it has no success rule). For a
    race, it is the seed nearest every racer's median rank.
  - Rank of any seed against the bench; paired wins between racers; the band (quartiles and median per unit).
  - Tests: typical seeds, ranks and bands are deterministic.

  **Phase 2 · Seed and odds in the Lab** (done)
  - The seed control: *Typical* (the default), a box to type any seed, and the dice to roll one.
  - Each racer's pane header: its odds on the bench, and where the played seed falls ("better than 14 of 20").
  - Charts: the played run, the bench band, the seed strip (see Principles). "This run" alone stays as an option.
  - The sweep keeps its form (every value on the same bench) and marks the played settings.
  - The preset `seed` keys go. Preset intros that quote numbers from one run are checked and reworded.

  **Phase 3 · A layout made for comparing, and playback speeds that each mean something** (done)
  Today the racers are stacked full-width, so two worlds never fit on one screen, the transport bar covers the second,
  and the charts and odds sit far below. There are 45 live presets: 6 with one racer, 25 with two, 13 with three, 1 with
  four.
  - Every racer on screen at once, whatever their number: compact panes side by side (two or three across, four as a
    2 × 2), sized to fit the window. One shared play bar drives them all, at the same moment.
  - The charts directly under the panes, in the racers' colors; the filmstrip as a row per racer.
  - The side column: pseudocode with a tab per racer and the lines where they differ marked; the "Show" options shared.
  - One racer: the same frame, with the pane at full width.
  - Speeds, a ladder where each rung shows what can be seen at that pace:
    - *Line by line* (about 2 lines a second): every line of the pseudocode, with its numbers.
    - *Step by step* (about 3 steps a second): every move and its update.
    - *Fast* (about 10 steps a second) and *Faster* (about 40): the agent runs, the trail and the values change as it
      goes; the numbers of each update are no longer shown.
    - *1 episode a second*: each frame is a whole episode, its path drawn at once, the values as they stand after it.
    - *10 and 50 episodes a second*: single episodes can no longer be read, so the view shows what they add up to: the
      tiles shaded by how often the recent episodes visited them (a path has no median; where the agent goes most often
      does), and the path the greedy policy would take now.
    - The same ladder for the other unit kinds (pulls, sweeps, rounds), each with its own sensible rates.
  - Prototype on the cliff race (grid view) and review it. Then every other view kind: arms, cards, chain and line, car
    and surface, star, corridor, throw, graph.

  **Phase 4 · Stories** (done)
  - `tools/story-runs.js` checks every live story run (142 of them) against seeds 1–40: where its seed falls, and whether
    the story's runs keep the order of their medians. Most were representative. Re-pinned: on-policy MC control (its
    run reached 78% where the typical one reaches about 45%; now seed 96, numbers updated), and the feature and
    generalization stories (a rough seed for every run; now seeds 6 and 34). The runs that fail on purpose (PPO without
    the clip, TRPO with δ = 0.2, the thrown REINFORCE run, a gem worth 100) already say how rare they are.
  - `tests/stories.test.js` keeps a fingerprint of every pinned run (`tests/story-runs.json`); it is a developer check
    only and shows nowhere in the guide. A faint line at the end of each story names its seeds.
  - The story charts' averages (up to 100 runs a curve; the REINFORCE-with-baseline chart alone took 100 s) are worked
    out offline by `tools/story-curves.js` with the page's own runs and seeds, kept in `content/story-curves.json`,
    and shown at once; `tests/stories.test.js` replays each curve's first run against the fingerprint kept for it, and
    fails when a chart has none (it would be averaged live, in slices, as before). The runs a story steps through are
    still made in the page, in slices of 10 ms that stop between two events of a unit (`lab.simulateJob`), while the
    reader is on the first steps, and then the units the steps draw at rest (the one before each moment shown: a PPO
    round with its ten passes takes tens of milliseconds to replay); a step that comes before they are ready says so
    and shows them when they are.

  **Later (decided separately)**
  - The deep presets: the same bench-and-typical method on recorded grids (each value's seeds trained offline), how much
    of the knob space to record, and the flat Pendulum sweeps (see `recorder/RUNS-REVIEW.md`). Their story runs are
    re-picked from the recorded seeds then.
  - The Pendulum and CartPole stories play their recorded test episodes instead of stills (code only). *Done*: a step
    after some training plays the test episode of the network it shows, in a loop, at the world's own pace.

- [x] **M9 · Worlds for every algorithm** (closed with release 1.0.0): a shared set of worlds that every algorithm can
  be tried on, where its kind allows; for each algorithm, a study of which world shows it best; then the deep Lab on top
  (`recorder/DEEP-LAB.md`). Classic algorithms first. What it set out to do and was left out is listed at its end.

  **The shared worlds.** All of them run in the page: Gymnasium's equations ported exactly, or, where Gymnasium's
  physics can't be ported (Box2D, MuJoCo), a world of our own in the same spirit, named as such.
  - *Discrete actions, discrete states* (tables can hold them): Cliff Walking; Frozen Lake 4 × 4; Frozen Lake 8 × 8
    (new); Taxi (new, 500 states); the windy gridworld (new, Sutton & Barto's Example 6.5); the Dyna maze; Blackjack.
  - *Discrete actions, continuous or large states* (approximation only): CartPole (now live, not only recorded);
    Mountain Car; Acrobot (new); Catch (new, bsuite's tiny game: a falling ball, a paddle, 10 × 5 cells, 250 states).
  - *Continuous actions* (approximation only): Pendulum.
  - The chapter worlds built to make one point (the short corridor, two gems, the ice bridge, the hidden cliffs, the
    max-bias pair, Baird's star, the random walks) stay in their chapters; they are not part of the shared set.

  **Who may run where.**
  - Tabular methods (Monte Carlo, TD, SARSA, Q-learning, n-step, λ, Dyna, prioritized sweeping): discrete states and
    discrete actions only. Dynamic programming also needs the world's model written out: the grid worlds and Taxi
    (Blackjack keeps its exact solution).
  - Value methods with approximation (linear SARSA and TD, DQN and its extensions): any states, discrete actions only:
    choosing the best action is a max over a list.
  - Policy-gradient and actor–critic methods (REINFORCE, actor–critic, A2C, GAE, TRPO, PPO): any world; a softmax over
    discrete actions, a Gaussian over continuous ones.
  - DDPG, TD3, SAC: continuous actions only.
  - The Lab lists its worlds in a settings drawer (**World & display**, in the play bar), with the view's display
    options: the shared set, grouped (tables, approximation, continuous actions), showing only the worlds the chosen
    algorithm may run. Monte Carlo never lists a continuous-action world. The side column keeps only what follows the
    simulation: the pseudocode, its line lit in the racer's color at every step, and "This step".

  **Every world trains live.** The continuous ones included: their physics are a few lines, and their networks are
  small. Training runs in a Web Worker, so the animation never waits for the weights.

  **The network.** Two hidden layers by default, with depth (1–3), width (16–128) and activation as knobs: small, as the
  networks of most control work still are, robots included.
  - *Smoothness first.* Before the knobs open, every network shape is timed on every world: updates per second, and so
    the fastest playback it can keep. Shapes that would stall the page at a speed are not offered at that speed (or at
    all); the fast rates of the ladder show what training actually manages ("training: 1,800 steps a second").

  **Steps.**
  1. Port the new worlds; CartPole and Acrobot live; the compatibility table. *Frozen Lake 8 × 8, the windy gridworld,
     Taxi, and CartPole and Acrobot live done* (`lab/envs/control.js`, checked step for step against Gymnasium's own
     trajectories; tile coding, 8 tilings of 6 cells per coordinate; new Acrobot view, the CartPole view drawing live
     snapshots too). Linear SARSA and the policy-gradient methods run on them from the World panel. *Catch* (bsuite's
     falling ball, 250 states: tables, DP and policy methods) added with its own view.
  2. The world study for the classic algorithms: each on every world it may run, 20 seeds, sensible settings; a world
     is a good showcase when good settings succeed on most seeds, learning takes long enough to watch, the chapter's
     knob visibly matters, and the hero beats the foil on most seeds. Written up like the runs review; presets and
     stories move where a world shows the idea better. *First round done* (`recorder/WORLD-STUDY.md`): the settings
     each algorithm starts from in each world (3 seeds per candidate), episode caps, Frozen Lake judged by its chance of
     reaching the gem, prediction on the grids with a near-optimal policy and episodes starting anywhere; pairings that
     rarely end well are marked "hard" in the World panel, not hidden. The second round is left for after 1.0 (below).
  3. The scenario picker in the Lab: any allowed world for any algorithm, with its bench. *Done for the discrete
     worlds so far* (`lab/worlds.js`): a "World" panel in the Lab's side lists the shared worlds every racer may run,
     grouped, with the lab's own world marked; a click switches the world, starting from that world's profile (units,
     discount, charts, odds rule), with a bench of its own. Recorded labs and the sandbox keep their world.
     *Lab review after it* (every lab × every world it offers, in the browser): Monte Carlo and the other prediction
     methods now get every world whose values can be worked out; a racer's step size keeps its ratio to the lab's; the
     views handle every kind of learner (action values, a policy with or without a critic, a fixed policy); playback
     never runs more than one move per tick, so Monte Carlo and REINFORCE walk their episodes with the pseudocode in step;
     long runs train in slices with their progress shown, the page staying responsive.
  4. The deep Lab (`recorder/DEEP-LAB.md`), on the same worlds. *Done*: the trainer in
     JavaScript (`lab/deep/`: deterministic math, so a seed is the same run in Node and in any browser); every recording
     and sweep made with it (`recorder/deep.js`): the DQN family on CartPole, network depth, width and activation among
     its knobs, A2C, PPO and TRPO on CartPole, and PPO, DDPG, TD3 and SAC on Pendulum; the Lab training other seeds and
     settings of those recordings in a Web Worker, with the odds from the recording, its sweep, or a bench trained in
     the page on request; and a strip under each recorded world showing how often each hidden unit fires. A test keeps
     the trainer from drifting off the recordings (`tests/recordings.test.js`), and every lab offers only knobs that
     change its runs (`lab/knobs.js`). What the long runs cost, and the rules for the next: `recorder/DEEP-LAB.md`.
  5. The map keeps its order, which follows Sutton & Barto. A new station goes only where a topic needs its own, and
     then between the two stations where it naturally fits.

  **Left out, on purpose.**
  - Learning from pixels (a MinAtar-style Breakout behind a small convolutional encoder): convolutional networks belong
    to a course on neural networks, not to this guide. The DQN chapter tells the Atari story in words.
  - More continuous worlds of our own (continuous Mountain Car, CartPole pushed with any force, a torque-driven Acrobot,
    a reaching arm, a pushed puck): Pendulum already shows what continuous actions change, and each new world would
    need its recordings, sweeps and checks.

  **After 1.0** (not scheduled)
  - The world study's second round: the 20-seed showcase choice per chapter, and moving presets or stories where
    another world shows the idea better.
  - A check on real browsers and devices, which the headless checks here cannot replace (`docs/browser-check.md`, with
    `tools/browser-check.js` for what can be measured). *First round done* on a Windows laptop in Chrome, Brave and
    Edge; release 1.0.2 fixes what it found. Still to do: Firefox, Safari and a real phone.

- [x] **M10 · Release 1.0.0**: a design pass over the stories, the animations and the Lab's layout, then the release.
  The Lab's panes take the largest size that fits for any number and shape of racers, and the algorithm column folds
  into a tab at the edge; playback holds its rate at every speed without forced layouts; stories follow the scroll
  position frame by frame, prepare their runs and the moments their steps draw in slices that stop between two events
  of a unit, and show their charts at once (averaged offline, `tools/story-curves.js`). The README got pictures; a
  workflow (`.github/workflows/release.yml`) publishes `rl_atlas.html` as the download of each version tag. 1.0.1 fixes
  formulas that stopped a few pixels short of fitting their column (they now shrink until they fit). 1.0.2 fixes what
  the first check in real browsers found: hover cards kept inside the window, story formulas stacked in narrow
  windows, story runs prepared while the reader scrolls, labs at a few rounds a second without freezes, and the phone's
  header and symbols page.
