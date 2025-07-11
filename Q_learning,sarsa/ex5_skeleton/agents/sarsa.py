from .base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=200000):
        s, _ = self.env.reset()
        a = self.action(s)  # Epsilon-greedy action for initial state

        for i in range(n_timesteps):
            # TODO 1.2: Implement SARSA training loop
            # You will have to call self.update_Q(...) at every step
            # Do not forget to reset the environment and update the action if you receive a 'terminated' signal

            s_,r, terminated, _, _ = self.env.step(a)
            a_ = self.action(s_)
            self.update_Q(s,a,r,s_,a_)
            s,a = s_,a_
            if terminated :
                s,_ = self.env.reset()
                a = self.action(s)
            pass

    def update_Q(self, s, a, r, s_, a_):
        # TODO 1.2: Implement SARSA update
        self.Q[*s, a] =self.Q[*s, a] + self.lr*(r+self.g*self.q(*s_,a_) - self.Q[*s,a])
