/* A world of the recorded runs: Gymnasium's Pendulum-v1, as the Lab draws and describes it. Its physics ran offline,
   in recorder/record.py: the page only plays back what the recorder saw. (CartPole's recordings use the Lab's own
   CartPole, envs/control.js, which also trains live.) */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  // ---- Pendulum: swing a weak motor's pendulum up and hold it upright ----
  // A state is (θ, θ̇), with θ = 0 upright; the action is a torque in [−2, 2], too weak to lift the pendulum straight up:
  // it must swing. Every step pays −(θ² + 0.1 θ̇² + 0.001 u²), so 0 is the best possible; episodes last 200 steps.
  lab.pendulum = function () {
    return {
      name: "pendulum", key: "pendulum", kind: "pendulum", title: "Pendulum", continuousActions: { lo: -2, hi: 2, unit: 1 },
      maxSteps: 200, valueRange: 1000, recorded: true,
      describe: (s, a) => (a !== undefined ? `torque ${a < 0 ? "−" : "+"}${Math.abs(a).toFixed(2)}` : "the pendulum"),
    };
  };
  lab.worlds.pendulum = () => lab.pendulum();
})(globalThis.RL = globalThis.RL || {});
