import numpy as np
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(QLearningAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=200000):
        s, _ = self.env.reset()

        for i in range(n_timesteps):
            # TODO 1.3: Implement Q-learning training loop
            # You will have to call self.update_Q(...) at every step
            # Do not forget to reset the environment if you receive a 'terminated' signal
            a = self.action(s)  # Select action following the epsilon-greedy policy
            s_, r, terminated, _, _ = self.env.step(a)  # Observe the next state and the reward
            self.update_Q(s, a, r, s_)  # You will have to call self.update_Q(...) at every step
            s = s_  # Update state
            if terminated:  # Do not forget to reset the environment if you receive a 'terminated' signal
                s, _ = self.env.reset()
            pass

    def update_Q(self, s, a, r, s_):
        # TODO 1.3: Implement Q-learning update
        self.Q[*s, a] = self.Q[*s, a] + self.lr * (r + self.g * np.max(self.Q[*s_]) - self.Q[*s, a])
