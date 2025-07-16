import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from utils import episode_reward_plot, visualize_agent


def compute_returns(rewards, next_value, discount):
    """ Compute returns based on episode rewards.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, bootstrapped value otherwise.
    discount : float
        Discount factor.

    Returns
    -------
    list of float
        Episode returns.
    """

    returns_list = []
    next_value_cp = next_value
    for r in reversed(rewards):
        next_value_cp = r + discount * next_value_cp
        returns_list.append(next_value_cp)


    # TODO 1.3: Compute returns
    return returns_list[::-1]

"""
Transitionmemory store : it stores imformation exprienced from environment(s,a,s_,a_)
it is for learning step
 

"""
class TransitionMemory:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""


    def __init__(self, gamma):
        self.obs_list=[]
        self.action_list=[]
        self.reward_list=[]
        self.logprob_list=[]
        self.return_list = []
        self.gamma = gamma
        self.trg_start = 0


    def put(self, obs, action, reward, logprob):
        """Put a transition into the memory."""
        self.obs_list.append(obs)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.logprob_list.append(logprob)


    def get(self):
        """Get all stored transition attributes in the form of lists."""

        # TODO 1.2.c
        return self.obs_list,self.action_list,self.reward_list,self.logprob_list,self.return_list

    def clear(self):
        """Reset the transition memory."""
        self.obs_list=[]
        self.action_list=[]
        self.reward_list=[]
        self.logprob_list=[]
        self.return_list = []
        self.trg_start = 0

        # TODO 1.2.d

    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).
        
        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """

        # TODO 1.2.b
        # it is only for reward during current episode
        reward_traj = self.reward_list[self.trg_start:]
        return_traj = compute_returns(reward_traj, next_value, self.gamma)
        self.return_list.extend(return_traj)
        self.trg_start = len(self.reward_list)


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()

        # TODO 1.1: Set up actor network
        self.net = nn.Sequential(
            nn.Linear(num_observations,128),
            nn.ReLU(),
            nn.Linear(128,num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):

        # TODO 1.1: Forward pass
        return self.net(obs)


class VPG:
    """The vanilla policy gradient (VPG) approach."""

    def __init__(self, env, episodes_update=5, gamma=0.99, lr=0.01):
        """ Constructor.
        
        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        episodes_update : int
            Number episodes to collect for every optimization step.
        gamma : float, optional
            Discount factor.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')
        
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.memory = TransitionMemory(gamma)
        self.episodes_update = episodes_update

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)

    def learn(self, total_timesteps):
        """Train the VPG agent.
        
        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """

        # TODO 1.6.a:
        obs, _ = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        episodes_counter = 0

        for timestep in range(1, total_timesteps + 1):
            action,logprob = self.predict(obs,train_returns=True)
            obs_,reward,terminated,truncated,_ =self.env.step(action)
            self.memory.put(obs,action,reward,logprob)
            episode_rewards.append(reward)

            obs = obs_

            # TODO 1.6.b: Do one step, put into transition buffer, and store reward in episode_rewards for plotting

            if terminated or truncated:
                obs,_=self.env.reset()
                overall_rewards.append(sum(episode_rewards))
                episode_rewards=[]
                self.memory.finish_trajectory(0.0)

                # TODO 1.6.c: reset environment, finish trajectory
                episodes_counter += 1

                if episodes_counter == self.episodes_update:

                    # TODO 1.6.d: optimize the actor
                    _,_,_,logprob_list,return_list = self.memory.get()
                    loss = self.calc_actor_loss(logprob_list,return_list)
                    self.optim_actor.zero_grad()
                    loss.backward()
                    self.optim_actor.step()

                    # Clear memory
                    episodes_counter = 0
                    self.memory.clear()

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1, wait=timestep == total_timesteps)

    @staticmethod
    def calc_actor_loss(logprob_lst, return_lst):
        """Calculate actor "loss" for one batch of transitions."""

        # TODO 1.5: Compute loss
        return -(torch.Tensor(return_lst) * torch.stack(logprob_lst)).mean()

    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.
        
        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.

        """

        probs = self.actor_net(torch.Tensor(obs))
        policy = Categorical(probs=probs)
        action = policy.sample()
        logprob = policy.log_prob(action)

        # TODO 1.4

        if train_returns:

            # TODO 1.4: Return action, logprob
            return action.item(), logprob
        else:

            # TODO 1.4: Return action
            return action.item()


if __name__ == '__main__':
    env_id = "CartPole-v1"
    _env = gym.make(env_id)
    vpg = VPG(_env)
    vpg.learn(100000)

    # Visualize the agent
    visualize_agent(gym.make("CartPole-v1", render_mode='human'), vpg)
