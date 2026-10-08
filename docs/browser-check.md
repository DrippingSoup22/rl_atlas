# Browser check: RL Atlas in real browsers

A brief for an agent running on the owner's computer (Chrome and Brave are installed there). Read it all before
starting.

## Your task

Check how **RL Atlas** behaves in the browsers on this computer, measure what can be measured, try by hand what cannot,
and write a report of what works and what does not, with evidence. **Do not change the code, commit or push.** The
report is the deliverable: the owner takes it back to the cloud session that develops the atlas, where the fixes are
made. Only report a problem you have seen happen. Mark a guess as a guess.

RL Atlas is a guide to reinforcement learning in one HTML file (about 8 MB) that works offline. Readers download
`rl_atlas.html` and open it from disk (a `file://` address). It has a **map** of 95 stations; for each one a **story**
(the text scrolls and an animated picture follows it), a **textbook** chapter and a one-page **card**; and a **Lab**
where algorithms learn live, side by side. Every check so far ran in headless Chromium in the cloud, without a GPU or
a real screen. What matters most now:

1. **Smoothness on a real screen:** stories follow the scroll without stutter, and labs keep their frame rate at every
   speed.
2. **Layout:** nothing cut off, overlapping or wider than the window, at common window sizes and zoom levels.
3. **Things only a real browser shows:** the Web Worker that trains networks when started from a local file, progress
   kept in `localStorage` for a `file://` page, and Brave's Shields.
4. **No errors** in the console, and no network requests: the page must work offline.

## Setup

1. Get the repository: `git clone https://github.com/DrippingSoup22/rl_atlas.git`, or `git pull` an existing clone,
   then `cd rl_atlas`.
2. Test the file readers download. Download
   <https://github.com/DrippingSoup22/rl_atlas/releases/latest/download/rl_atlas.html> into a new folder `downloads/`
   inside the clone. Check it is the same file as the clone's `rl_atlas.html` by comparing their SHA-256: on Windows
   `Get-FileHash <file> -Algorithm SHA256`, on macOS `shasum -a 256 <file>`, on Linux `sha256sum <file>`. Release 1.0.1
   is `20219cba9cfbeafc47a0718911d41acd78f8be43f98daae3936323acb8f4dc50`. If a newer release is out, its hash is on
   its release page. If the two files differ, say so and test the downloaded one.
3. Install Node 18 or newer (22 recommended). In the clone, run `npm install --no-save playwright-core`.
   playwright-core drives the browsers already installed and downloads none. `node_modules/` and
   `browser-check-results/` are ignored by git.

## Part A: the automated run

```
node tools/browser-check.js --browser chrome
node tools/browser-check.js --browser brave
```

Add `--file downloads/rl_atlas.html` if you test the downloaded copy. If Brave is not found, pass
`--exe "<path to brave.exe or Brave Browser>"`. Edge runs with `--browser edge`. Firefox and Safari cannot be driven by
playwright-core, so check them by hand (Part B) if they are installed.

A full run takes 25 to 35 minutes; `--quick` takes about 8. It opens the browser in a maximized window and drives
it there, at the screen's own pixel ratio. **Keep that window in front and uncovered, and do not use the mouse or
keyboard meanwhile.** A browser slows the frames and timers of a window it takes for hidden: Chrome on Windows does so
for a window covered by another app, down to one frame a second. The report flags any measure taken while the page
was hidden; run those again. Close heavy apps, and keep a laptop plugged in. A run:

- measures the display's refresh rate on the idle map;
- scrolls 15 stories from top to bottom with the mouse wheel, once slowly and once in a fast flick. For each pass it
  records frames per second, the worst gap between frames, long tasks (over 50 ms), how long "Working out this
  run…" showed, and whether the last step was reached;
- plays 7 labs at each of their speeds for 5 s each, measuring the same, then opens and closes the **World & display**
  drawer and folds and unfolds the pseudocode column;
- trains a seed the recordings do not have (PPO on CartPole, seed 77) in the page's Web Worker. It checks that the
  training starts and ends and that the page stays free meanwhile;
- switches to the dark theme and takes screenshots;
- reloads after visiting a station and checks it is still marked read;
- looks for formulas wider than their column on every page, in a 1024×768 window, where the columns are narrowest;
- screenshots the map, a story, a textbook chapter, a card, a lab, a deep lab and the symbols page at 1920×1080,
  1440×900, 1366×768, 1280×800 and 1024×768, and in an emulated phone (390×844). It checks that no page is wider than
  its window. A full run also opens every page on the phone for that check;
- records console errors, and any request to the network.

It writes `browser-check-results/<browser>/report.md`, `report.json` and the screenshots. The report's first section,
**To look at**, lists every measure outside its bounds. **Look at the screenshots yourself**: the script cannot see
overlapping text, a panel cut off, a blank picture or unreadable colors.

## Part B: by hand

Use your browser tools (Claude in Chrome or computer use), or ask the owner for the steps you cannot do. In Chrome, an
extension can open `file://` pages only if **Allow access to file URLs** is on in its settings. Do each step in Chrome
and in Brave. Note the browser, page and window size of anything wrong, and take a screenshot.

- **Opening:** double-click the downloaded file. The map should show within about 2 seconds. Turn the network off and
  reload: it should still work.
- **Map:** drag to move, Ctrl + scroll to zoom (pinch on a trackpad), and use the zoom and fit buttons. Open
  **Views & filters** (or press `V`), switch between its three views (they animate) and try a filter. A station
  grows under the cursor; a click opens its page; coming back, the page shrinks into its station.
- **Stories:** scroll with the real wheel or trackpad, slowly and fast, through Q-learning, PPO, A2C, MCTS (in "Where
  next"), DQN and SAC. The picture should follow the text without stutter. Charts should appear at once, formula pieces
  light up step by step, and no formula should have a scrollbar or a cut-off end. "Working out this run…" may show for a
  moment after a fast jump, but not for long.
- **Textbook and card:** hovering an underlined term shows a short explanation. Hovering a colored symbol lights up
  every symbol of its kind. Math and figures render.
- **Lab** (open "SARSA vs Q-learning on the cliff"): play at each speed. With **Change world** or **World & display**,
  move the experiment to another world; Esc closes the drawer. Fold the pseudocode with the arrow at its top. Roll a
  seed with the dice. Run a sweep: it shows "Running N of M runs…" and can take a few minutes (about 3 on a laptop).
  Then click a dot in the odds to play that seed.
- **Deep lab** ("DQN balances a pole"): it plays at once. Change a knob (for example units per layer). The run should
  train in the page with its progress shown, and the page should stay responsive meanwhile. The new run is then ready:
  it waits for Play, like a recorded one.
- **Everywhere:** `Ctrl K` opens search; type "sarsa", use the arrow keys and Enter. Check the **Symbols** page. Switch
  the theme with the button at the top right. Then put the operating system in dark mode and reload: the page should
  follow it unless switched by hand. Tab through the controls: focus should be visible. Esc closes drawers and search.
- **Zoom and scaling:** set browser zoom to 80%, 125% and 150%. On Windows, also try display scaling at 125% and 150%
  if allowed. Nothing should overlap or be cut off. Put every setting back afterwards.
- **Progress:** read a few stations, close the browser, open the file again. They should still be filled on the map.
  With Brave, also note whether Shields or its settings for clearing site data change this.
- **Brave:** with Shields up (the default), everything should work as in Chrome. Note any difference.
- **Other browsers, if installed:** do a short pass in Edge, Firefox and Safari: the map, a story, a lab and a deep lab
  training. In Firefox and Safari, check especially whether the deep lab trains and whether progress is kept.
- **A phone (optional):** in the folder holding the file, run `python -m http.server 8000`. Open
  `http://<this computer's local IP>:8000/rl_atlas.html` on a phone on the same Wi-Fi. Scroll a story with touch,
  play a lab, open the drawer, and take screenshots. Stop the server afterwards.

## Known and expected (not bugs)

- After a lab opens, its odds fill in within some seconds: 20 seeds run in the background.
- A deep lab plays its recorded seeds at once. Any other seed or setting trains first, for about 10 seconds to a few
  minutes, with progress under the racer's name; the trained run then waits for Play.
- A lab whose seeds were still running when you pressed Play keeps playing seed 1; once they are done, its seed button
  offers "▸ Typical: seed N".
- The cloud checks found at most one long task (50 to 60 ms) in two of the 79 stories, and slow frames that seemed
  due to software rendering. A real GPU should do better, and confirming that is a goal of this check.
- The page saves progress under the `localStorage` key `rl-atlas:v1`. Clear it, or use a fresh browser profile, to
  start as a new reader.

## The report

Write `browser-check-results/REPORT.md` and give it to the owner, with the screenshot folders (zip them if you can):

1. **Environment:** OS and version, CPU, screen size and scaling, and each browser's version.
2. **Summary:** the problems found, the most serious first. For each one: what happens, where (browser, page, window
   size, zoom), the steps to reproduce it, what was expected, and the evidence (screenshot file, console text,
   numbers).
3. **Automated results:** each browser's `report.md`, in full or as a link to its file.
4. **Checklist:** the Part B items, each marked passed, failed or partly, per browser, with a short note.
5. **Anything else noticed:** confusing wording, hard-to-find controls, slow spots.

Rules: change no code, settings or files outside `downloads/` and `browser-check-results/`, and put every browser or
system setting you change back afterwards. Visit no site except GitHub to download the file. If something blocks you
(a browser not found, a permission prompt), note it and go on with the rest.
