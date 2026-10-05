/* Worlds of the recorded runs: Gymnasium's CartPole-v1 and Pendulum-v1, as the Lab draws and describes them. Their
   physics ran offline, in recorder/record.py (Mountain Car's recordings use the Lab's own Mountain Car world): the page
   only plays back what the recorder saw. */
(function (RL) {
  "use strict";
  const lab = (RL.lab = RL.lab || {});

  // ---- CartPole: push a cart left or right to keep a pole balanced on it ----
  // A state is (x, ẋ, θ, θ̇): the cart's position and speed, the pole's angle from upright and its angular speed. An
  // episode ends when the pole leans more than 12° or the cart leaves the track (|x| > 2.4), or after 500 steps; every
  // step it stays up pays +1.
  lab.cartpole = function () {
    return {
      name: "cartpole", key: "cartpole", kind: "cartpole", title: "CartPole", nA: 2, actionNames: ["push left", "push right"],
      xMax: 2.4, thetaMax: (12 * Math.PI) / 180, poleLength: 1.0, maxSteps: 500, valueRange: 100, recorded: true,
      describe: (s, a) => (a >= 0 ? (a ? "pushing right" : "pushing left") : "the cart and pole"),
    };
  };
  lab.worlds.cartpole = () => lab.cartpole();

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
