# Q-learning Maze Agent

This project implements a simple Q-learning agent whose goal is to reach a treasure in a maze environment. The maze is generated each time randomly and has a fixed size of 20×20. It contains walls, traps, and costly paths that affect the agent’s reward and behavior. The agent learns by interacting with the environment and updating a table of Q-values for each cell and action. The learning process can be visualized, allowing observation of how Q-values change over time and how the agent gradually improves its navigation strategy.

---

## Key Features

- Randomly generated maze with walls and obstacles  
- Traps and costly paths with negative rewards  
- Treasure goal with a positive reward  
- Q-learning with ε-greedy exploration  
- Visualization of agent movement and Q-value updates  
- Informed A* search used as an optimal baseline for evaluation  

---

## Algorithms Used

- **Q-learning** for learning an action policy through interaction

```
    def update(self, s, a, r, ns):
        self.Q[s][a] = (
            (1 - LEARNING_RATE) * self.Q[s][a]
            + LEARNING_RATE * (r + DISCOUNT * np.max(self.Q[ns]))
        )
```

- **ε-greedy strategy** for balancing exploration and exploitation. Epsilon dynamically changes based on episodes, in early episodes the agent randomly explores, and in later episodes, the agent exploits.

  ```
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05
eps = 1.0

for ep in range(1, EPISODES + 1):
    ...
    a = agent.act(s, eps)
    ...
    eps = max(MIN_EPSILON, eps * EPSILON_DECAY)

  ```
  
- **A\* search** as an informed search algorithm to compute an optimal reference path  

---
