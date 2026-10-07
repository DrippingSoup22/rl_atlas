// The Lab's engine for Node (tests and tools), in the order app/index.html loads it. Returns RL.lab.
const FILES = ["core", "envs/grid", "envs/bandit", "envs/chain", "envs/blackjack", "envs/taxi", "envs/mdp", "envs/approx", "envs/policy", "dp", "run", "measures", "features",
  "policies", "agents/td", "agents/mc", "agents/dp", "agents/bandit", "agents/traces", "agents/planning", "agents/linear", "agents/policy", "agents/offline", "agents/model",
  "envs/control", "envs/deep", "recorded", "mcts", "bench", "worlds"];
for (const file of FILES) require(`./${file}.js`);
module.exports = globalThis.RL.lab;
