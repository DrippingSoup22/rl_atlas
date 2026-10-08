/* The numbers around a control world's drawing (CartPole, Acrobot, Pendulum), in HTML so that they keep their size
   however narrow the pane: a caption over the drawing (the step and the state), rows of bars under it (a value or a
   probability per action, or a torque drawn from the middle), and a message that pops over the drawing when an
   episode ends. In a narrow pane each bar takes a line of its own, under its name and number (app/css/lab.css). */
(function (RL) {
  "use strict";
  const pct = (v) => `${(100 * Math.max(0, Math.min(1, v))).toFixed(2)}%`;
  RL.readouts = {
    // The caption line, and the box the drawing goes in (which the pops are placed over).
    frame(pane) {
      const caption = pane.appendChild(RL.h('<div class="ctl-cap"></div>'));
      return { caption, draw: pane.appendChild(RL.h('<div class="track-draw"></div>')) };
    },
    // One row of bars per name, in a grid under the drawing (the network strip joins it). A row's bar runs from the
    // left (set(share)) or, centered, from the middle (span(from, to), and a band, in shares of the bar's width).
    bars(pane, names, { centered = false } = {}) {
      const grid = pane.appendChild(RL.h(`<div class="ctl-grid${centered ? " centered" : ""}"></div>`));
      const rows = names.map((name) => {
        const row = grid.appendChild(RL.h(`<div class="ctl-row"><span class="bar-name"></span><span class="hbar">${centered ? '<i class="band"></i>' : ""}<i class="fill"></i>${centered ? '<i class="zero"></i>' : ""}</span><span class="bar-num"></span></div>`));
        row.querySelector(".bar-name").textContent = name;
        const fill = row.querySelector(".fill"), band = row.querySelector(".band"), num = row.querySelector(".bar-num");
        const place = (e, a, b) => { e.style.left = pct(Math.min(a, b)); e.style.width = pct(Math.abs(b - a)); };
        return {
          row,
          set(share) { fill.style.width = pct(share); },
          span(a, b) { place(fill, a, b); },
          band(a, b) { place(band, a, b); },
          num(text) { num.textContent = text; },
          chosen(on) { row.classList.toggle("chosen", on); },
        };
      });
      return { grid, rows };
    },
    // A message over the drawing at height y (a share of the drawing, from its top), rising as it fades.
    pop(draw, text, y = 0.2) {
      const t = draw.appendChild(RL.h('<div class="ctl-pop"></div>'));
      t.textContent = text;
      t.style.top = pct(y);
      t.animate([{ opacity: 0, transform: "translate(-50%, 0)" }, { opacity: 1, offset: 0.15, transform: "translate(-50%, 0)" }, { opacity: 0, transform: "translate(-50%, -12px)" }], { duration: 1400, easing: "ease-out" }).onfinish = () => t.remove();
    },
  };
})(globalThis.RL = globalThis.RL || {});
