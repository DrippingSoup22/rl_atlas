// Run from rl_atlas/ with: node --test
// The recordings of content/recordings/ took hours to train with lab/deep/, and the Lab retrains any of their seeds bit
// for bit. A change to the trainer that would make it disagree with them fails here: each recording's shown seed is
// trained for its first block, and the network's outputs then, at full precision, must hash to the sum kept in
// recordings.first-block.json. (A block is too short for a tiny change to show in the recordings' rounded numbers, but
// over a whole run it grows until it does.) The sweeps come from the same code.
//
// After a deliberate change to the trainer, record again (recorder/deep.js) and then renew the sums with
//   RENEW=1 node --test tests/recordings.test.js
// which first checks that the trainer gives each recording's own snapshot again.
const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const deep = require("../lab/deep/node.js");

const DIR = path.join(__dirname, "../content/recordings"), SUMS = path.join(__dirname, "recordings.first-block.json");
const renew = !!process.env.RENEW, sums = fs.existsSync(SUMS) ? JSON.parse(fs.readFileSync(SUMS, "utf8")) : {};
const STOP = Symbol("one block is enough");
const plain = (x) => JSON.parse(JSON.stringify(x, (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v)));
// the run's first block: its training return, its test, and the snapshot after it (raw: unrounded, unpacked)
function firstBlock(rec, raw) {
  const spec = { world: rec.world, learner: rec.learner, steps: rec.steps, block: rec.block, cfg: rec.config }; // as the Lab's (lab/recorded.js)
  let got = null;
  try {
    deep.run(spec, rec.shown, { snapshots: true, raw, onBlock: (k, info) => { got = info; throw STOP; } });
  } catch (e) {
    if (e !== STOP) throw e;
  }
  return got;
}

for (const file of fs.readdirSync(DIR).filter((f) => f.endsWith(".json")).sort()) {
  const rec = JSON.parse(fs.readFileSync(path.join(DIR, file), "utf8"));
  if (rec.trainer !== "lab/deep") continue;
  test(`${rec.name}: the first block of seed ${rec.shown} trains as recorded`, () => {
    const got = firstBlock(rec, true), curve = rec.curves.find((c) => c.seed === rec.shown);
    assert.equal(got.train, curve.train[0], "the first block's training return");
    assert.equal(got.test, curve.test[0], "the first test episode");
    const sum = crypto.createHash("sha1").update(JSON.stringify(plain(got.shot))).digest("hex");
    if (!renew) return assert.equal(sum, sums[rec.name], "the network after the first block (renew only after recording again)");
    assert.deepEqual(plain(firstBlock(rec, false).shot), rec.snapshots[1], "the recording's own snapshot after the first block");
    sums[rec.name] = sum;
  });
}
if (renew) test.after(() => fs.writeFileSync(SUMS, JSON.stringify(sums, null, 2) + "\n"));
