/* The deep trainer in a Web Worker: the page sends {id, spec, seed, snapshots}; the Worker trains that run (with a
   snapshot after every block unless snapshots is false: a bench seed needs only its curves), posting each block as it
   ends ({id, block, train, test, shot, secs}) and then the whole run ({id, run}). Numbers stay
   arrays (raw), copied across, never packed. The page stops a run by terminating the Worker. build.py bundles this
   file after the trainer's own (app/deep-worker.js), and the page starts the Worker from that text. */
(function (RL) {
  "use strict";
  self.onmessage = ({ data: { id, spec, seed, snapshots = true } }) => {
    const t0 = Date.now();
    const run = RL.deep.run(spec, seed, { snapshots, raw: true, onBlock: (block, info) => self.postMessage({ id, block, ...info, secs: (Date.now() - t0) / 1000 }) });
    self.postMessage({ id, run });
  };
})(globalThis.RL = globalThis.RL || {});
