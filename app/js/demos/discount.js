/* Demo "discount": a slider for γ, and how much a reward counts by how far away it is. */
(function (RL) {
  "use strict";

  // ---- the discount factor: how much a reward counts, by how far away it is ----
  RL.demos.discount = function (host) {
    const K = 40, X0 = 26, DX = 15, BASE = 120, HMAX = 100;
    let bars = "";
    for (let k = 0; k < K; k++) bars += `<rect class="db" x="${X0 + k * DX - 5}" width="10" rx="2"/>`;
    for (const k of [0, 10, 20, 30]) bars += `<text class="tick" x="${X0 + k * DX}" y="${BASE + 16}" text-anchor="middle">${k}</text>`;
    host.innerHTML = `<div class="disc">
      <label class="disc-knob"><span class="sym">γ</span><input type="range" min="0" max="0.99" step="0.01" value="0.9" aria-label="Discount factor γ"><output></output></label>
      <svg class="disc-bars" viewBox="0 0 640 150" role="img" aria-label="How much a reward counts, by how many steps away it is">
        <line class="axis" x1="${X0 - 10}" x2="${X0 + (K - 1) * DX + 10}" y1="${BASE}" y2="${BASE}"/>${bars}
        <line class="hz"/><text class="hz-name" y="16"></text>
        <text class="tick" x="${X0 + (K - 1) * DX + 12}" y="${BASE + 16}" text-anchor="end">steps away</text>
      </svg>
      <ul class="disc-facts"></ul>
    </div>`;
    const input = host.querySelector("input"), out = host.querySelector("output"), svg = host.querySelector("svg"), facts = host.querySelector(".disc-facts");
    const rects = svg.querySelectorAll(".db"), hz = svg.querySelector(".hz"), hzName = svg.querySelector(".hz-name");
    const fmt = (v, d = 2) => String(+v.toFixed(d));
    function draw() {
      const g = +input.value, horizon = 1 / (1 - g), later = 10 * g ** 20;
      out.textContent = g.toFixed(2);
      rects.forEach((r, k) => {
        const h = g ** k * HMAX;
        r.style.y = `${BASE - h}px`;
        r.style.height = `${Math.max(h, 0.001)}px`;
      });
      const hx = X0 + Math.min(horizon - 1, K - 1) * DX;
      hz.setAttribute("x1", hx);
      hz.setAttribute("x2", hx);
      hz.setAttribute("y1", 22);
      hz.setAttribute("y2", BASE);
      hzName.setAttribute("x", Math.min(hx, 560));
      hzName.setAttribute("text-anchor", hx > 560 ? "end" : "middle");
      hzName.textContent = `horizon ≈ ${fmt(horizon, horizon < 10 ? 1 : 0)} steps`;
      facts.innerHTML = `
        <li>A reward 10 steps away counts γ<sup>10</sup> = <b>${(g ** 10).toFixed(2)}</b> of its size.</li>
        <li>All the weights add up to 1/(1 − γ) = <b>${fmt(horizon, 1)}</b>: roughly, the agent looks that many steps ahead.</li>
        <li>+1 now, or +10 in 20 steps? Seen from now, the +10 is worth 10γ<sup>20</sup> = <b>${later.toFixed(2)}</b>, so the agent
          <b>${later > 1 ? "waits for the +10" : "takes the +1 now"}</b>.</li>`;
    }
    input.addEventListener("input", draw);
    draw();
  };
})(globalThis.RL = globalThis.RL || {});
