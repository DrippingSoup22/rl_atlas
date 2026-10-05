/* Cards view: Blackjack, shared by the Lab and the stories. On the left the table: the dealer's cards (one face
   down until the player sticks) and the player's. On the right the 200 states as two maps, with and without a
   usable ace: player sum 12–21 up the side, the dealer's face-up card along the bottom, colored by value or by
   the action the policy takes. The state being played is outlined. */
(function (RL) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const C = 21, ML = 30, MB = 24, MT = 4;
  const SUITS = ["♠", "♥", "♦", "♣"];
  const rank = (c) => (c === 1 ? "A" : String(c));
  const fmt = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  class CardsView {
    constructor(host, env, options = {}) {
      this.env = env;
      this.o = { tiles: "policy", numbers: false, ...options };
      this.d = { V: new Float64Array(env.nS) };
      this.root = RL.h(`<div class="cardsview">
        <div class="table">
          <div class="hand dealer"><span class="who">Dealer</span><div class="cards"></div><span class="sum"></span></div>
          <div class="hand player"><span class="who">Player</span><div class="cards"></div><span class="sum"></span></div>
          <div class="verdict"></div>
        </div>
        <div class="maps"></div>
      </div>`);
      host.appendChild(this.root);
      this.table = this.root.querySelector(".table");
      const maps = this.root.querySelector(".maps");
      this.cells = [];
      for (const usable of [true, false]) {
        const fig = RL.h(`<figure class="bj-map"><figcaption>${usable ? "Usable ace" : "No usable ace"}</figcaption></figure>`);
        const svg = el("svg", { viewBox: `0 0 ${ML + 10 * C + 2} ${MT + 10 * C + MB}`, role: "img", "aria-label": `${usable ? "With" : "Without"} a usable ace` }, fig);
        for (let sum = 21; sum >= 12; sum--) {
          const yy = MT + (21 - sum) * C;
          el("text", { class: "tick", x: ML - 5, y: yy + C / 2 + 4, "text-anchor": "end" }, svg).textContent = sum;
          for (let d = 1; d <= 10; d++) {
            const s = env.encode(sum, d, usable), g = el("g", { class: "cell", transform: `translate(${ML + (d - 1) * C} ${yy})` }, svg);
            g.bg = el("rect", { x: 0.75, y: 0.75, width: C - 1.5, height: C - 1.5, rx: 3 }, g);
            g.txt = el("text", { x: C / 2, y: C / 2 + 4 }, g);
            g.dataset.s = s;
            this.cells[s] = g;
          }
        }
        for (let d = 1; d <= 10; d++) el("text", { class: "tick", x: ML + (d - 0.5) * C, y: MT + 10 * C + 14, "text-anchor": "middle" }, svg).textContent = rank(d);
        svg.addEventListener("pointermove", (e) => this._hover(e));
        svg.addEventListener("pointerleave", () => RL.tip.hide());
        maps.appendChild(fig);
      }
      this.setOptions({});
    }

    setOptions(o) {
      Object.assign(this.o, o);
      this.root.dataset.tiles = this._mode();
      this._paint();
    }
    _mode() { return this.d.Q ? this.o.tiles : "v"; }

    show(d) {
      this.d = d;
      this.root.dataset.tiles = this._mode();
      this._paint();
    }

    _paint() {
      const d = this.d, mode = this._mode(), { STICK, HIT } = this.env;
      for (let s = 0; s < 200; s++) {
        const g = this.cells[s];
        if (!g) continue;
        const v = d.V ? d.V[s] : Math.max(d.Q[s * 2 + STICK], d.Q[s * 2 + HIT]);
        if (mode === "policy") {
          const hit = d.Q[s * 2 + HIT] > d.Q[s * 2 + STICK], tie = d.Q[s * 2 + HIT] === d.Q[s * 2 + STICK];
          g.bg.style.fill = tie ? "" : hit ? "color-mix(in oklab, var(--pol) 72%, var(--surface))" : "color-mix(in oklab, var(--pol) 10%, var(--surface))";
          g.txt.textContent = tie ? "" : hit ? "H" : "S";
          g.classList.toggle("hit", hit && !tie);
        } else {
          g.bg.style.fill = RL.GridView.valueColor(v, 1);
          g.txt.textContent = this.o.numbers ? fmt(v, 1).replace("0.", ".") : "";
          g.classList.remove("hit");
        }
      }
    }

    // ---- the table ----
    _cards(who, cards, hidden = 0) {
      const box = this.table.querySelector(`.${who} .cards`);
      box.innerHTML = cards.map((c, i) => `<span class="pcard${i >= cards.length - hidden ? " down" : ""}${c === 1 ? " ace" : ""}" data-suit="${SUITS[(i + (who === "dealer" ? 1 : 0)) % 4]}">${i >= cards.length - hidden ? "" : rank(c)}</span>`).join("");
    }
    _sum(who, text) { this.table.querySelector(`.${who} .sum`).textContent = text; }
    _verdict(text, kind = "") {
      const v = this.table.querySelector(".verdict");
      v.textContent = text;
      v.className = `verdict ${kind}`;
    }
    _player(s) {
      const h = this.env.hand;
      if (h?.player) this._cards("player", h.player);
      else this.table.querySelector(".player .cards").innerHTML = '<span class="pcard blank">?</span>';
      if (s !== this.env.TERMINAL) {
        const { sum, usable } = this.env.decode(s);
        this._sum("player", `${sum}${usable ? ", usable ace" : ""}`);
      }
    }
    mark(s) {
      for (const g of this.root.querySelectorAll(".cell.focus")) g.classList.remove("focus");
      if (s >= 0) this.cells[s]?.classList.add("focus");
    }
    spark(s) {
      const g = this.cells[s];
      if (!g) return;
      g.classList.remove("sparked");
      g.getBoundingClientRect();
      g.classList.add("sparked");
    }

    event(ev, { line = false } = {}) {
      const env = this.env, h = env.hand;
      switch (ev.type) {
        case "start":
          this._cards("dealer", h.dealer.slice(0, 2), 1);
          this._sum("dealer", `shows ${rank(h.dealer[0])}`);
          this._player(ev.s);
          this._verdict(h.player ? "" : "exploring start: a random state");
          this.mark(ev.s);
          return 350;
        case "choose":
          this._verdict(ev.a === env.HIT ? "hit" : "stick", "act");
          return 0;
        case "move":
          if (ev.a === env.HIT) {
            if (h.player) this._cards("player", h.player);
            if (ev.bust) { this._sum("player", "over 21"); this._verdict("bust: −1", "lose"); this.mark(-1); return 600; }
            this._player(ev.s2);
            this.mark(ev.s2);
            return 250;
          }
          this._cards("dealer", h.dealer);
          this._sum("dealer", ev.dealer > 21 ? "over 21" : String(ev.dealer));
          this._verdict(ev.r > 0 ? (ev.dealer > 21 ? "dealer busts: +1" : "you win: +1") : ev.r < 0 ? "dealer wins: −1" : "a draw: 0", ev.r > 0 ? "win" : ev.r < 0 ? "lose" : "");
          this.mark(-1);
          return 700;
        case "update": this.spark(ev.s); return 0;
        case "return": this.mark(ev.s); return line ? 200 : 0;
        case "skip": return 0;
        case "cut": this._verdict("not the greedy action: stop", "lose"); this.mark(ev.s); return 400;
        default: return 0;
      }
    }

    rest(events) {
      this.mark(-1);
      const h = this.env.hand;
      if (!events.length || !h) { this._cards("dealer", []); this._cards("player", []); this._sum("dealer", ""); this._sum("player", ""); this._verdict("waiting for the first hand"); return; }
      const start = events.find((e) => e.type === "start"), last = [...events].reverse().find((e) => e.type === "move");
      const stuck = last && last.a === this.env.STICK;
      this._cards("dealer", stuck ? h.dealer : h.dealer.slice(0, 2), stuck ? 0 : 1);
      this._sum("dealer", stuck ? (last.dealer > 21 ? "over 21" : String(last.dealer)) : `shows ${rank(h.dealer[0])}`);
      if (h.player) this._cards("player", h.player);
      if (last?.bust) this._sum("player", "over 21");
      else this._player(last ? last.s2 !== this.env.TERMINAL ? last.s2 : last.s : start.s);
      if (last) this._verdict(last.r > 0 ? "won: +1" : last.r < 0 ? (last.bust ? "bust: −1" : "lost: −1") : "a draw: 0", last.r > 0 ? "win" : last.r < 0 ? "lose" : "");
    }

    _hover(e) {
      const g = e.target.closest?.(".cell");
      if (!g) { RL.tip.hide(); return; }
      const s = +g.dataset.s, { sum, dealer, usable } = this.env.decode(s), d = this.d;
      const head = `<div class="head">Player ${sum}${usable ? " with a usable ace" : ""}, dealer shows ${rank(dealer)}</div>`;
      const row = (v, text) => `<div class="row"><b>${v}</b><span>${text}</span></div>`;
      const body = d.Q ? row(fmt(d.Q[s * 2 + this.env.STICK]), "stick: Q") + row(fmt(d.Q[s * 2 + this.env.HIT]), "hit: Q") + (d.N ? row(d.N[s * 2] + d.N[s * 2 + 1], "visits") : "")
        : row(fmt(d.V[s]), "V, the value of this state") + (d.N ? row(d.N[s], "visits") : "");
      RL.tip.show(e.clientX, e.clientY, head + body);
    }

    destroy() { RL.tip.hide(); this.root.remove(); }

    static options(env, displays) {
      const hasQ = displays.some((d) => d.Q);
      return [
        ...(hasQ ? [{ key: "tiles", type: "seg", label: "Color the states by", value: "policy", choices: [["policy", "Policy"], ["v", "Values"]] }] : []),
        { key: "numbers", type: "check", label: "Values as numbers", value: false },
      ];
    }

    static legend(env) {
      return '<div class="scale"><span>lose</span><i></i><span>0</span><i class="up"></i><span>win</span></div><p class="arms-legend"><i class="lg-hit"></i>H: hit · <i class="lg-stick"></i>S: stick</p>';
    }

    static thumb(env, d) {
      const S = 5.5;
      let g = "";
      [true, false].forEach((usable, k) => {
        const ox = k * (10 * S + 8);
        for (let sum = 21; sum >= 12; sum--) {
          for (let dl = 1; dl <= 10; dl++) {
            const s = env.encode(sum, dl, usable);
            let fill;
            if (d.Q) {
              const hit = d.Q[s * 2 + 1] > d.Q[s * 2], tie = d.Q[s * 2 + 1] === d.Q[s * 2];
              fill = tie ? "var(--surface-3)" : hit ? "color-mix(in oklab, var(--pol) 72%, var(--surface))" : "color-mix(in oklab, var(--pol) 10%, var(--surface))";
            } else fill = RL.GridView.valueColor(d.V[s], 1);
            g += `<rect x="${ox + (dl - 1) * S}" y="${(21 - sum) * S}" width="${S - 0.6}" height="${S - 0.6}" style="fill:${fill}"/>`;
          }
        }
      });
      return `<svg class="thumb-bj" viewBox="0 0 ${20 * S + 8} ${10 * S}">${g}</svg>`;
    }
  }

  RL.CardsView = CardsView;
  (RL.labViews = RL.labViews || {}).blackjack = CardsView;
})(globalThis.RL = globalThis.RL || {});
