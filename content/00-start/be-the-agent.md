+++
summary = "Play an agent for two minutes: press buttons, collect rewards, and find out afterwards what you were really doing."
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §1.1 and §2.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Strens (2000), A Bayesian framework for reinforcement learning, Proceedings of the 17th International Conference on Machine Learning (the chain problem)" },
]
+++

## Card

### Idea

Before any theory, be an agent yourself. You are somewhere, shown only as a symbol. You can press one of two buttons, and after each press you get a number and find yourself somewhere, perhaps somewhere else. Nobody tells you what the buttons do. Try to make the numbers add up to as much as possible: three episodes of ten presses each.

{{be-the-agent}}

### What you were doing {#doing}

| In the game | In reinforcement learning |
| --- | --- |
| the symbol you saw | the **state** ([[state-action-reward]]) |
| the two buttons | the **actions** |
| the number after each press | the **reward** |
| ten presses | an **episode** ([[episodes]]) |
| the sum of the numbers | the **return** ([[return]]) |
| your rule for pressing | your **policy** ([[policy]]) |
| your sense that a room was “good” | its **value** ([[value-functions]]) |
| trying the other button “to see” | **exploration** ([[explore-exploit]]) |

### What made it hard {#hard}

- **No instructions.** The numbers graded your presses but never said which press would have been better.
- **Delayed reward.** Walking toward ★ paid nothing for several presses before it paid +10 at every press.
- **Chance.** One press in ten did the opposite of usual, so the same press could turn out differently.
- **Little time.** With ten presses, exploring costs a large share of the episode.

### How an agent learns it {#learn}

An RL agent faces exactly this situation, without even the symbols' shapes to hint at anything. It learns which button is better in each room from the rewards that follow, and, more importantly, from the value of the room it lands in. That second part is what lets it discover that pressing forward, which pays nothing at first, is worth far more than the +2 of going back. [[q-learning]] learns this from experience alone.

### Check yourself {#check}

::: question
Judged only by the next reward, which button looks better at the start? And in the long run?
---
At the start, going back: it pays +2 at once, forward pays 0. In the long run, forward: it leads to the room that pays +10 at every press.
:::

::: question
Why does a policy that only looks at the next reward get stuck at about 2 per press?
---
It always takes the +2 and never walks far enough forward to discover the +10 room.
:::

::: question
One press in ten did the opposite of usual. Why is one episode a poor judge of a button?
---
The same press can turn out differently, so a single outcome may be the unlucky one. To know what a button does on average you have to try it several times; an agent keeps its estimates as averages over many presses for the same reason.
:::
