/* The knobs a lab can offer, and which of them a lab shows. No DOM: the tests check with the same rule that every
   knob a lab shows changes its runs (one that changes nothing there is fixed instead: it keeps its value, off the bar). */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  // The knobs a preset can show as sliders. alpha = 0 means sample averages (1/n), where an algorithm allows it.
  // sweep: the values a sweep of the odds tries for a slider (a knob with choices tries its choices).
  lab.KNOBS = {
    alpha: { sym: "α", name: "step size", min: 0.01, max: 1, step: 0.01, sweep: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] },
    epsilon: { sym: "ε", name: "exploration", min: 0, max: 0.5, step: 0.01, sweep: [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5] },
    gamma: { sym: "γ", name: "discount", min: 0.5, max: 1, step: 0.01, sweep: [0.5, 0.7, 0.9, 0.95, 0.99, 1] },
    c: { sym: "c", name: "confidence", min: 0, max: 5, step: 0.1, sweep: [0, 0.5, 1, 2, 3, 5] },
    q0: { sym: "Q₁", name: "first estimate", min: -2, max: 10, step: 0.5, sweep: [-2, 0, 1, 2, 5, 10] },
    theta: { sym: "θ", name: "tolerance", choices: [0.1, 0.01, 0.001, 0.0001] },
    n: { sym: "n", name: "steps ahead", choices: [1, 2, 3, 4, 8, 16, 32, 64] },
    lambda: { sym: "λ", name: "trace decay", min: 0, max: 1, step: 0.01, sweep: [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1] },
    planning: { sym: "n", name: "planning steps", choices: [0, 1, 5, 10, 20, 50, 100] },
    kappa: { sym: "κ", name: "exploration bonus", choices: [0, 0.0001, 0.001, 0.01] },
    alphaW: { sym: "αw", name: "critic step size", min: 0.01, max: 1, step: 0.01, sweep: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] },
    beta: { sym: "β", name: "entropy bonus", choices: [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] },
    workers: { sym: "N", name: "workers", choices: [1, 2, 4, 8, 16] },
    epochs: { sym: "K", name: "passes over each batch", choices: [1, 2, 4, 10, 20] },
    clip: { sym: "ε", name: "clip range (0: no clip)", choices: [0, 0.1, 0.2, 0.3, 0.5] },
    delta: { sym: "δ", name: "trust region (KL)", choices: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2] },
    noise: { sym: "σ", name: "exploration noise (degrees)", min: 0, max: 30, step: 1, sweep: [0, 1, 2, 5, 10, 20, 30] },
    logEpisodes: { sym: "E", name: "episodes in the log", choices: [1, 2, 5, 10, 20, 50, 100, 200] },
    rewardScale: { sym: "c", name: "rewards × c (other units)", choices: [0.01, 0.1, 1, 10, 100] },
    normalize: { sym: "Â", name: "normalize advantages (1: yes)", choices: [0, 1] },
    doubt: { sym: "d", name: "doubt: extra cost of an unseen tile", min: 0, max: 3, step: 0.1, sweep: [0, 0.2, 0.4, 0.6, 0.8, 1, 2] },
  };
  // The settings an algorithm reads in its first units on a world (a name, or a function that makes it): a knob it
  // never reads turns nothing for it.
  lab.reads = function (world, algorithm, params) {
    const reads = new Set();
    lab.simulate({ world, algorithm, params, units: 2, seed: 1, snapshots: false, reads });
    return reads;
  };

  // The knobs on a lab's bar: its settings (params) that a racer reads (reads[i], when given) and does not set for
  // itself, minus the fixed ones. In another world (away), the step sizes racers set themselves are shown too: the
  // knob is the reference their ratios apply to (lab.carryRacer).
  lab.knobsShown = (params, racers, { fixed = [], away = false, reads = null } = {}) =>
    Object.keys(lab.KNOBS).filter((k) => k in params && !fixed?.includes(k) && racers.some((r, i) => (!reads || reads[i].has(k)) && (!(k in r.params) || (away && (k === "alpha" || k === "alphaW")))));
})(globalThis.RL = globalThis.RL || {});
