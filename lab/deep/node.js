// The deep trainer for Node (recorder/deep.js, tests), in the order the page's Worker loads it. Returns RL.deep.
for (const file of ["dmath", "nn", "worlds", "dqn", "pg", "ac", "record"]) require(`./${file}.js`);
module.exports = globalThis.RL.deep;
