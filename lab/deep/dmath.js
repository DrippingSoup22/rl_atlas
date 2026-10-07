/* The deep trainer's math: exp, log, tanh, sin, cos and pow from + − × ÷ alone (and the square root, which every engine
   rounds exactly), so every browser computes them bit for bit alike. Math's own versions may differ between engines by
   an ulp, and one ulp is enough to make two training runs drift apart: with these, a seed is the same run everywhere,
   in Node offline and in any browser's Worker. Accurate to a few ulps. */
(function (RL) {
  "use strict";
  const LN2_HI = 0.6931471803691238, LN2_LO = 1.9082149292705877e-10, INV_LN2 = 1.4426950408889634;
  const f64 = new Float64Array(1), u32 = new Uint32Array(f64.buffer); // little-endian: u32[1] holds the exponent
  const pow2 = (k) => { u32[0] = 0; u32[1] = (k + 1023) << 20; return f64[0]; };
  const round = (x) => (x >= 0 ? Math.floor(x + 0.5) : -Math.floor(0.5 - x));
  function exp(x) {
    if (x > 709) return Infinity;
    if (x < -745) return 0;
    const k = round(x * INV_LN2), r = x - k * LN2_HI - k * LN2_LO;
    let p = 1, t = 1;
    for (let n = 1; n <= 13; n++) { t = (t * r) / n; p += t; }
    return k < -1021 ? p * pow2(k + 60) * pow2(-60) : p * pow2(k);
  }
  function expm1(x) {
    if (x > -0.5 && x < 0.5) { let p = 0, t = 1; for (let n = 1; n <= 17; n++) { t = (t * x) / n; p += t; } return p; }
    return exp(x) - 1;
  }
  function log(x) {
    if (!(x > 0)) return x === 0 ? -Infinity : NaN;
    if (x === Infinity) return x;
    let e = 0;
    if (x < 2.2250738585072014e-308) { x *= 18014398509481984; e = -54; } // subnormals: scale by 2^54 first
    f64[0] = x; e += ((u32[1] >>> 20) & 0x7ff) - 1023; u32[1] = (u32[1] & 0x800fffff) | 0x3ff00000;
    let m = f64[0];
    if (m > 1.4142135623730951) { m /= 2; e++; }
    const s = (m - 1) / (m + 1), s2 = s * s;
    let p = 0, t = s;
    for (let n = 1; n <= 23; n += 2) { p += t / n; t *= s2; }
    return 2 * p + e * LN2_LO + e * LN2_HI;
  }
  function tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    const e = expm1(2 * x);
    return e / (e + 2);
  }
  // sin and cos: reduce to [−π/4, π/4] by multiples of π/2 (Cody–Waite, three parts of π/2), then a polynomial.
  const P1 = 1.5707963267341256, P2 = 6.077100506506192e-11, P3 = 2.0222662487959506e-21, INV_P = 0.6366197723675814;
  function reduce(x) { const k = round(x * INV_P); return [x - k * P1 - k * P2 - k * P3, ((k % 4) + 4) % 4]; }
  function sinP(r) { const r2 = r * r; let p = 0, t = r; for (let n = 1; n <= 23; n += 2) { p += t; t = (-t * r2) / ((n + 1) * (n + 2)); } return p; }
  function cosP(r) { const r2 = r * r; let p = 0, t = 1; for (let n = 0; n <= 22; n += 2) { p += t; t = (-t * r2) / ((n + 1) * (n + 2)); } return p; }
  function sin(x) { const [r, q] = reduce(x); return q === 0 ? sinP(r) : q === 1 ? cosP(r) : q === 2 ? -sinP(r) : -cosP(r); }
  function cos(x) { const [r, q] = reduce(x); return q === 0 ? cosP(r) : q === 1 ? -sinP(r) : q === 2 ? -cosP(r) : sinP(r); }
  const pow = (x, y) => (y === 0 ? 1 : x === 0 ? 0 : exp(y * log(x))); // x > 0 (or 0)
  RL.dmath = { exp, expm1, log, tanh, sin, cos, pow };
})(globalThis.RL = globalThis.RL || {});
