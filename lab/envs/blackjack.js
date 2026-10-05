/* Blackjack as in Sutton & Barto (Example 5.1). Cards come from an infinite deck: 1 (ace) to 9, and 10 for ten and the
   face cards. The player sees their own sum (12–21: below 12 they always hit), the dealer's face-up card and whether
   they hold a usable ace (one that counts 11 without going over 21): 200 states. Hitting draws a card; going over 21
   loses. Sticking hands the turn to the dealer, who hits until 17 or more. Win +1, draw 0, lose −1.
   A hand dealt as a natural (an ace and a ten) is decided at once, with no decision to make, so it is dealt again. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});
  const STICK = 0, HIT = 1, TERMINAL = 200;
  const PROB = (c) => (c === 10 ? 4 / 13 : 1 / 13); // the chance of each card value

  const encode = (sum, dealer, usable) => (sum - 12) * 20 + (dealer - 1) * 2 + (usable ? 1 : 0);
  const decode = (s) => ({ sum: 12 + Math.floor(s / 20), dealer: 1 + Math.floor((s % 20) / 2), usable: s % 2 === 1 });
  // A hand is its cards added with aces as 1, and whether it holds an ace: one ace may then count 11.
  const total = (raw, ace) => (ace && raw + 10 <= 21 ? raw + 10 : raw);

  lab.blackjack = function () {
    const both = [STICK, HIT], none = [];
    const draw = (rng) => Math.min(10, 1 + rng.int(13));
    const env = {
      name: "blackjack", kind: "blackjack", title: "Blackjack", nS: 201, nA: 2, TERMINAL, STICK, HIT, valueRange: 1, unitName: "hand",
      encode, decode,
      hand: null, // the cards of the episode being played (the dealer's hidden card included)
      starts: Array.from({ length: 200 }, (_, s) => s),
      acts: (s) => (s === TERMINAL ? none : both),
      terminal: (s) => s === TERMINAL,
      describe(s, a) {
        if (s === TERMINAL) return "end of the hand";
        const { sum, dealer, usable } = decode(s);
        return `player ${sum}${usable ? " with a usable ace" : ""}, dealer shows ${dealer === 1 ? "an ace" : dealer}${a >= 0 ? `, ${a === HIT ? "hitting" : "sticking"}` : ""}`;
      },
      // Deal a new hand, or (exploring starts) start from the state s0 with a fresh hidden card for the dealer.
      reset(rng, s0) {
        let h;
        if (s0 !== undefined) {
          const { sum, dealer, usable } = decode(s0);
          h = { player: null, raw: usable ? sum - 10 : sum, ace: usable, dealer: [dealer, draw(rng)] };
        } else {
          do {
            const p = [draw(rng), draw(rng)];
            h = { player: p, raw: p[0] + p[1], ace: p.includes(1), dealer: [draw(rng), draw(rng)] };
          } while (total(h.raw, h.ace) === 21);
          while (total(h.raw, h.ace) < 12) { const c = draw(rng); h.player.push(c); h.raw += c; h.ace ||= c === 1; }
        }
        env.hand = h;
        return encode(total(h.raw, h.ace), h.dealer[0], h.ace && h.raw + 10 <= 21);
      },
      step(s, a, rng) {
        const h = env.hand;
        if (a === HIT) {
          const c = draw(rng);
          h.player?.push(c);
          h.raw += c;
          h.ace ||= c === 1;
          const sum = total(h.raw, h.ace);
          if (sum > 21) return { s2: TERMINAL, r: -1, card: c, bust: true };
          return { s2: encode(sum, h.dealer[0], h.ace && h.raw + 10 <= 21), r: 0, card: c };
        }
        let raw = h.dealer[0] + h.dealer[1], ace = h.dealer.includes(1);
        while (total(raw, ace) < 17) { const c = draw(rng); h.dealer.push(c); raw += c; ace ||= c === 1; }
        const mine = total(h.raw, h.ace), theirs = total(raw, ace);
        return { s2: TERMINAL, r: theirs > 21 || mine > theirs ? 1 : mine === theirs ? 0 : -1, dealer: theirs };
      },
      // Named policies for prediction: "stick-20" sticks only on 20 or 21 (Sutton & Barto's example policy).
      policy(name) {
        const P = new Float64Array(201 * 2);
        for (let s = 0; s < 200; s++) {
          const stick = name === "stick-20" ? decode(s).sum >= 20 : 0.5;
          P[s * 2 + STICK] = +stick;
          P[s * 2 + HIT] = 1 - stick;
        }
        return P;
      },
    };
    return env;
  };

  // Exact values, from the rules alone. The dealer's final sum depends only on their face-up card, so its chances
  // are computed once; then every state follows from the ones a hit can reach (hard hands first, high sums first).
  // policy: an nS × nA table to evaluate, or nothing for the optimal values. Returns { V, Q, policy }.
  lab.blackjackExact = function (policy = null) {
    const finals = []; // finals[d][f]: chance the dealer ends on 17..21 (f = 0..4) or busts (f = 5), showing d
    for (let d = 1; d <= 10; d++) {
      const memo = new Map();
      const from = (raw, ace) => {
        const key = raw * 2 + (ace ? 1 : 0);
        if (memo.has(key)) return memo.get(key);
        const t = total(raw, ace), out = new Float64Array(6);
        if (t > 21) out[5] = 1;
        else if (t >= 17) out[t - 17] = 1;
        else for (let c = 1; c <= 10; c++) from(raw + c, ace || c === 1).forEach((p, f) => { out[f] += PROB(c) * p; });
        memo.set(key, out);
        return out;
      };
      finals[d] = from(d, d === 1);
    }
    const stickValue = (sum, d) => {
      const f = finals[d];
      let v = f[5];
      for (let k = 0; k < 5; k++) v += f[k] * Math.sign(sum - (17 + k));
      return v;
    };
    const V = new Float64Array(201), Q = new Float64Array(201 * 2), P = new Float64Array(201 * 2);
    for (const usable of [false, true]) {
      for (let sum = 21; sum >= 12; sum--) {
        for (let d = 1; d <= 10; d++) {
          const s = encode(sum, d, usable), raw = usable ? sum - 10 : sum;
          let hit = 0;
          for (let c = 1; c <= 10; c++) {
            const r2 = raw + c, a2 = usable || c === 1, t = total(r2, a2);
            hit += PROB(c) * (t > 21 ? -1 : V[encode(t, d, a2 && r2 + 10 <= 21)]);
          }
          const stick = stickValue(sum, d);
          Q[s * 2 + STICK] = stick;
          Q[s * 2 + HIT] = hit;
          const pStick = policy ? policy[s * 2 + STICK] : +(stick >= hit);
          P[s * 2 + STICK] = pStick;
          P[s * 2 + HIT] = 1 - pStick;
          V[s] = pStick * stick + (1 - pStick) * hit;
        }
      }
    }
    return { V, Q, policy: P };
  };

  lab.worlds.blackjack = () => lab.blackjack();
})(globalThis.RL = globalThis.RL || {});
