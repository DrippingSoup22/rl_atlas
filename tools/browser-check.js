// The atlas in a real browser, the one a reader has: node tools/browser-check.js --browser chrome
// It opens rl_atlas.html from disk, as a reader does, in a browser installed on this computer, with its window and its
// GPU, and measures what the headless checks of the cloud sessions cannot: frames and long tasks while stories scroll
// and labs play, a deep lab training in its Web Worker, formulas wider than their column, pages wider than the window,
// progress kept across a reload, requests to the network (there should be none) and console errors. It writes
// report.md, report.json and screenshots to --out. docs/browser-check.md says what to check by hand besides.
// Needs Node 18+ and playwright-core, which drives an installed browser and downloads none:
//   npm install --no-save playwright-core
//   node tools/browser-check.js --browser chrome     (or brave, edge; --exe <path> for another Chromium-based browser)
// Options: --file <html> (default: rl_atlas.html here), --out <dir> (default: browser-check-results/<browser>),
// --quick (fewer stories and labs; formulas only where they were too wide before 1.0.1), --headless.
// Keep the window visible while it runs: a hidden or minimized window slows its frames and timers down.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
let chromium;
try { ({ chromium } = require("playwright-core")); } catch { ({ chromium } = require("playwright")); }

const arg = (name, value) => { const i = process.argv.indexOf(`--${name}`); return i < 0 ? value : process.argv[i + 1]; };
const flag = (name) => process.argv.includes(`--${name}`);
const BROWSER = arg("browser", arg("exe") ? path.basename(arg("exe")).replace(/\..*$/, "") : "chrome"), QUICK = flag("quick");
const FILE = path.resolve(arg("file", path.join(__dirname, "..", "rl_atlas.html")));
const OUT = path.resolve(arg("out", path.join(__dirname, "..", "browser-check-results", BROWSER)));
const URL = pathToFileURL(FILE).href;

const STORIES = QUICK ? ["q-learning", "mcts", "a2c", "ppo", "dqn"]
  : ["q-learning", "expected-sarsa", "dyna-q", "deadly-triad", "reinforce", "baseline", "a2c", "gae", "trpo", "ppo", "dqn", "sac", "mcts", "bias-variance", "imitation"];
const LABS = QUICK ? ["cliff-race", "ppo-maze", "dqn-cartpole"] : ["cliff-race", "dyna-maze", "mountain-car", "a2c-maze", "ppo-maze", "dqn-cartpole", "ppo-pendulum"];
const FORMULAS = ["mcts", "imitation", "prioritized-sweeping", "mc-vs-td", "value-error", "pg-theorem"]; // too wide before 1.0.1
const SIZES = [[1920, 1080], [1440, 900], [1366, 768], [1280, 800], [1024, 768]];
const PAGES = [["map", "#/map"], ["story", "#/e/mcts/story", 5], ["textbook", "#/e/td0/textbook"], ["card", "#/e/q-learning/card"], ["lab", "#/lab/cliff-race"], ["deep-lab", "#/lab/dqn-cartpole"]];

function launchOptions() {
  const headless = flag("headless"), exe = arg("exe");
  if (exe) return { executablePath: exe, headless };
  if (BROWSER === "chrome") return { channel: "chrome", headless };
  if (BROWSER === "edge") return { channel: "msedge", headless };
  if (BROWSER === "brave") {
    const places = {
      win32: [process.env.PROGRAMFILES, process.env["PROGRAMFILES(X86)"], process.env.LOCALAPPDATA].filter(Boolean)
        .map((dir) => path.join(dir, "BraveSoftware", "Brave-Browser", "Application", "brave.exe")),
      darwin: ["/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"],
      linux: ["/usr/bin/brave-browser", "/usr/bin/brave", "/snap/bin/brave", "/opt/brave.com/brave/brave"],
    }[process.platform] || [];
    const found = [process.env.BRAVE_PATH, ...places].find((p) => p && fs.existsSync(p));
    if (!found) throw new Error("Brave was not found: pass --exe <path to Brave> or set BRAVE_PATH");
    return { executablePath: found, headless };
  }
  throw new Error(`unknown browser "${BROWSER}": use chrome, brave or edge, or --exe <path>`);
}

// In the page: long tasks, frames (the worst gap between two), and how long a story said "Working out this run…"
function probe() {
  const s = (window.__check = { on: false, long: [], frames: 0, worst: 0, last: 0, busy: 0, t0: 0 });
  try { new PerformanceObserver((list) => { if (s.on) for (const e of list.getEntries()) s.long.push(Math.round(e.duration)); }).observe({ type: "longtask" }); } catch {}
  const tick = (t) => { if (s.on) { s.frames++; if (s.last) s.worst = Math.max(s.worst, t - s.last); s.last = t; } requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  setInterval(() => { if (s.on && /^Working out/.test(document.querySelector(".scene-note")?.textContent || "")) s.busy += 0.1; }, 100);
}
const start = (page) => page.evaluate(() => Object.assign(window.__check, { on: true, long: [], frames: 0, worst: 0, last: 0, busy: 0, t0: performance.now() }));
const stop = (page) => page.evaluate(() => {
  const s = window.__check, secs = (performance.now() - s.t0) / 1000;
  s.on = false;
  return { secs: +secs.toFixed(1), fps: Math.round(s.frames / secs), worst: Math.round(s.worst), long: s.long.length, longest: Math.max(0, ...s.long), busy: +s.busy.toFixed(1) };
});
const toStep = (page, k) => page.evaluate((k) => { const el = document.querySelectorAll(".story-step")[k]; if (el) scrollTo(0, el.getBoundingClientRect().top + scrollY - innerHeight * 0.3); }, k);

const R = { errors: [], external: [], stories: [], labs: [], lab: [], training: null, formulas: [], layout: [], theme: null, kept: null };
let where = "start";
function watch(page, tag = "") {
  page.on("console", (m) => { if (m.type() === "error") R.errors.push(`${tag}${where}: ${m.text()}`); });
  page.on("pageerror", (e) => R.errors.push(`${tag}${where}: ${e.message}`));
  page.on("request", (r) => { if (!/^(file|data|blob|about|chrome-extension):/.test(r.url())) R.external.push(`${where}: ${r.url()}`); });
}
const shot = (page, name) => page.screenshot({ path: path.join(OUT, `${name}.png`) });

// A story scrolled from top to bottom with the mouse wheel: slowly, as when reading, or in a fast flick
async function story(page, id, mode) {
  where = `story ${id} (${mode})`;
  await page.goto(`${URL}#/e/${id}/story`);
  await page.waitForSelector(".story-step");
  await page.evaluate(() => scrollTo(0, 0));
  await page.waitForTimeout(1500);
  await page.mouse.move(320, 450);
  await start(page);
  const [dy, every] = mode === "slow" ? [80, 110] : [300, 25];
  for (let i = 0, last = -1, still = 0; i < 3000 && still < 8; i++) {
    await page.mouse.wheel(0, dy);
    await page.waitForTimeout(every);
    const y = await page.evaluate(() => Math.round(scrollY));
    still = y === last ? still + 1 : 0;
    last = y;
  }
  await page.waitForTimeout(1500);
  const m = await stop(page);
  const end = await page.evaluate(() => { const s = document.querySelectorAll(".story-step"); return !!s.length && s[s.length - 1].classList.contains("active"); });
  return { id, mode, ...m, end };
}

// A lab played at each of its speeds, then its drawer and its pseudocode column
async function lab(page, id) {
  where = `lab ${id}`;
  await page.goto(`${URL}#/lab/${id}`);
  await page.waitForSelector(".speed");
  await page.waitForTimeout(2500);
  const speeds = await page.$$eval(".speed option", (o) => o.map((x) => [x.value, x.textContent.trim()]));
  for (const [value, label] of speeds) {
    await page.selectOption(".speed", value);
    await start(page);
    await page.click(".play");
    await page.waitForTimeout(QUICK ? 3000 : 5000);
    await page.click(".play");
    R.labs.push({ lab: id, speed: label, ...(await stop(page)) });
  }
  await shot(page, `lab-${id}`);
  const state = () => page.evaluate(() => { const l = document.querySelector(".lab"); return `${l?.dataset.drawer}/${l?.dataset.side}`; });
  const steps = [await state()];
  await page.getByRole("button", { name: /World & display/ }).click();
  await page.waitForTimeout(700);
  steps.push(await state());
  await shot(page, `lab-${id}-drawer`);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(500);
  steps.push(await state());
  await page.click(".side-fold");
  await page.waitForTimeout(700);
  steps.push(await state());
  await shot(page, `lab-${id}-folded`);
  await page.click(".side-open");
  await page.waitForTimeout(500);
  steps.push(await state());
  R.lab.push({ lab: id, steps: steps.join(" → "), ok: steps.join() === "closed/open,open/open,closed/open,closed/folded,closed/open" });
}

// A seed the recording does not have trains in the page's Web Worker: does it start, end, and leave the page free?
async function training(page) {
  where = "deep training";
  await page.goto(`${URL}#/lab/ppo-cartpole`);
  await page.waitForSelector(".seed-in");
  await page.waitForTimeout(2500);
  await page.fill(".seed-in", "77");
  await start(page);
  const t0 = Date.now();
  await page.press(".seed-in", "Enter");
  const began = await page.waitForSelector(".training", { timeout: 15000 }).then(() => true, () => false);
  const done = began && (await page.waitForFunction(() => !document.querySelector(".training"), null, { timeout: 300000, polling: 1000 }).then(() => true, () => false));
  const m = await stop(page);
  await page.waitForTimeout(1500);
  await shot(page, "deep-trained");
  R.training = { preset: "ppo-cartpole", seed: 77, began, done, took: Math.round((Date.now() - t0) / 1000), ...m };
}

// Formulas still wider than their column once fitted: a story's at its widest step, the textbook's and the card's
async function formulas(page, ids) {
  for (const id of ids) {
    where = `formulas ${id}`;
    const steps = await page.evaluate((id) => (RL.content.entries[id]?.story?.steps || []).map((s) => s.state.formula || 0), id);
    if (steps.some((n) => n > 0)) {
      await page.goto(`${URL}#/e/${id}/story`);
      await page.waitForTimeout(1200);
      await toStep(page, steps.indexOf(Math.max(...steps)));
      await page.waitForTimeout(1300);
      const r = await page.evaluate(() => { const s = document.querySelector(".f-sym"); return s && [s.clientWidth, s.scrollWidth]; });
      if (r && r[1] > r[0] + 1) R.formulas.push(`${id} story: ${r[1]}px in ${r[0]}px`);
    }
    for (const tab of ["textbook", "card"]) {
      await page.goto(`${URL}#/e/${id}/${tab}`);
      await page.waitForTimeout(700);
      const rs = await page.evaluate(() => [...document.querySelectorAll(".tex-display[data-done]")]
        .filter((e) => e.clientWidth && e.scrollWidth > e.clientWidth + 1).map((e) => [e.clientWidth, e.scrollWidth, (e.dataset.src || "").slice(0, 40)]));
      for (const [room, need, src] of rs) R.formulas.push(`${id} ${tab}: ${need}px in ${room}px · ${src}`);
    }
  }
}

// Each page at each window size: a screenshot, and whether the page is wider than the window
async function layouts(page, sizes, tag) {
  for (const [w, h] of sizes) {
    await page.setViewportSize({ width: w, height: h });
    for (const [name, hash, step] of PAGES) {
      where = `layout ${tag} ${w}×${h} ${name}`;
      await page.goto(URL + hash);
      await page.waitForTimeout(1800);
      if (step !== undefined) { await toStep(page, step); await page.waitForTimeout(2500); }
      const wide = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      if (wide > 1) R.layout.push(`${name} at ${w}×${h}${tag ? ` (${tag})` : ""}: ${wide}px wider than the window`);
      await shot(page, `${tag || "desktop"}-${w}x${h}-${name}`);
    }
  }
}

function report(env) {
  const hz = env.hz, look = [];
  const row = (cells) => `| ${cells.join(" | ")} |`;
  const lines = [`# Browser check: ${env.browser} ${env.version}`, "",
    `${env.os} · ${env.cpu} · screen ${env.screen} at ${env.dpr}× · the display refreshes about ${hz} times a second · ${env.when}`,
    `File: ${FILE}${QUICK ? " · quick run" : ""}`, ""];
  for (const s of R.stories) if (s.long > 2 || s.worst > 250 || !s.end || s.busy > 2) look.push(`story ${s.id} (${s.mode}): ${s.long} long tasks (longest ${s.longest} ms), worst frame ${s.worst} ms${s.end ? "" : ", the last step was not reached"}${s.busy > 2 ? `, "Working out this run…" for ${s.busy} s` : ""}`);
  for (const l of R.labs) if (l.fps < 0.85 * hz || l.long > 1) look.push(`lab ${l.lab} at "${l.speed}": ${l.fps} frames/s, ${l.long} long tasks (longest ${l.longest} ms)`);
  for (const l of R.lab) if (!l.ok) look.push(`lab ${l.lab}: drawer and pseudocode went ${l.steps}`);
  const t = R.training;
  if (t && (!t.done || t.long > 2)) look.push(`deep training: ${t.began ? (t.done ? `done in ${t.took} s` : "did not finish in 5 minutes") : "did not start"}, ${t.long} long tasks (longest ${t.longest} ms)`);
  if (R.kept === false) look.push("progress: a station read before a reload was no longer marked read");
  if (R.theme && R.theme !== "dark") look.push(`theme: the switch left data-theme="${R.theme}"`);
  look.push(...R.formulas.map((f) => `formula too wide: ${f}`), ...R.layout.map((l) => `layout: ${l}`));
  lines.push("## To look at", "", ...(look.length ? look.map((l) => `- ${l}`) : ["Nothing: every measure is within its bounds."]), "");
  lines.push(`## Console errors (${R.errors.length})`, "", ...([...new Set(R.errors)].map((e) => `- ${e}`)), "");
  lines.push(`## Requests to the network (${R.external.length}; there should be none)`, "", ...([...new Set(R.external)].slice(0, 20).map((e) => `- ${e}`)), "");
  lines.push("## Stories, scrolled with the mouse wheel", "", "Bounds: at most 2 long tasks, no frame over 250 ms, the last step reached, \"Working out this run…\" at most 2 s.", "",
    row(["story", "scroll", "seconds", "frames/s", "worst frame (ms)", "long tasks", "longest (ms)", "working out (s)", "last step"]), row(Array(9).fill("---")),
    ...R.stories.map((s) => row([s.id, s.mode, s.secs, s.fps, s.worst, s.long, s.longest, s.busy, s.end ? "yes" : "**no**"])), "");
  lines.push("## Labs, played at each speed", "", `Bounds: at least ${Math.round(0.85 * hz)} frames/s (85% of the display's rate), at most 1 long task.`, "",
    row(["lab", "speed", "seconds", "frames/s", "worst frame (ms)", "long tasks", "longest (ms)"]), row(Array(7).fill("---")),
    ...R.labs.map((l) => row([l.lab, l.speed, l.secs, l.fps, l.worst, l.long, l.longest])), "",
    ...R.lab.map((l) => `- ${l.lab}: drawer/pseudocode ${l.steps}${l.ok ? "" : " (expected closed/open → open/open → closed/open → closed/folded → closed/open)"}`), "");
  if (t) lines.push("## A deep lab training in its Web Worker", "", `${t.preset}, seed ${t.seed}: ${t.began ? "started" : "**did not start**"}, ${t.done ? `done in ${t.took} s` : "**not done**"}; meanwhile ${t.fps} frames/s, worst frame ${t.worst} ms, ${t.long} long tasks (longest ${t.longest} ms).`, "");
  lines.push("## Formulas wider than their column", "", ...(R.formulas.length ? R.formulas.map((f) => `- ${f}`) : ["None."]), "");
  lines.push("## Pages wider than the window", "", ...(R.layout.length ? R.layout.map((l) => `- ${l}`) : ["None."]), "");
  lines.push("## Other", "", `- Progress kept across a reload: ${R.kept ? "yes" : "**no**"}`, `- Theme after the switch: ${R.theme}`, "", `Screenshots: ${OUT}`, "");
  return lines.join("\n");
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  if (!fs.existsSync(FILE)) throw new Error(`no such file: ${FILE}`);
  const browser = await chromium.launch(launchOptions());
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript(probe);
  const page = await context.newPage();
  watch(page);
  await page.goto(`${URL}#/map`);
  await page.waitForSelector(".metro .st", { timeout: 30000 });
  await page.waitForTimeout(1500);
  await start(page);
  await page.waitForTimeout(2000);
  const idle = await stop(page);
  const env = {
    browser: BROWSER, version: browser.version(), os: `${process.platform} ${os.release()}`, cpu: `${os.cpus()[0]?.model} × ${os.cpus().length}`,
    when: new Date().toISOString().slice(0, 16).replace("T", " "), hz: idle.fps,
    ...(await page.evaluate(() => ({ screen: `${screen.width}×${screen.height}`, dpr: devicePixelRatio, ua: navigator.userAgent }))),
  };
  console.log(`${env.browser} ${env.version}, display at ${env.hz} Hz`);

  for (const id of STORIES) {
    if (!(await page.evaluate((id) => !!RL.content.entries[id]?.story, id))) continue;
    for (const mode of ["slow", "fast"]) R.stories.push(await story(page, id, mode));
    console.log(`story ${id}: done`);
  }
  for (const id of LABS) { await lab(page, id); console.log(`lab ${id}: done`); }
  await training(page);
  console.log(`deep training: ${R.training.done ? `${R.training.took} s` : "not done"}`);

  where = "theme";
  await page.goto(`${URL}#/map`);
  await page.waitForTimeout(1000);
  await page.click(".icon-btn.theme");
  await page.waitForTimeout(800);
  R.theme = await page.evaluate(() => document.documentElement.dataset.theme || "(none)");
  for (const [name, hash, step] of PAGES) {
    await page.goto(URL + hash);
    await page.waitForTimeout(1800);
    if (step !== undefined) { await toStep(page, step); await page.waitForTimeout(2500); }
    await shot(page, `dark-${name}`);
  }
  await page.goto(`${URL}#/map`);
  await page.waitForTimeout(800);
  await page.click(".icon-btn.theme");

  where = "progress";
  await page.goto(`${URL}#/e/td0/story`);
  await page.waitForTimeout(1500);
  await page.reload();
  await page.waitForTimeout(2000);
  R.kept = await page.evaluate(() => { try { return JSON.parse(localStorage.getItem("rl-atlas:v1") || "{}").visited?.td0 === 1; } catch { return false; } });

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${URL}#/map`);
  await page.waitForTimeout(1000);
  await formulas(page, QUICK ? FORMULAS : await page.evaluate(() => Object.keys(RL.content.entries)));
  console.log(`formulas: ${R.formulas.length} too wide`);
  await layouts(page, SIZES, "");
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true });
  const p2 = await phone.newPage();
  watch(p2, "phone · ");
  await layouts(p2, [[390, 844]], "phone");
  console.log(`layouts: ${R.layout.length} pages wider than the window`);

  await browser.close();
  fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify({ env, ...R }, null, 2));
  fs.writeFileSync(path.join(OUT, "report.md"), report(env));
  console.log(`report: ${path.join(OUT, "report.md")}`);
})().catch((e) => { console.error(e); process.exit(1); });
