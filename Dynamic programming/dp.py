import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy

def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    """
    # Init value function array
    V_new = V.copy()

    # TODO: Write your implementation here
    """ mdp : object
        v : value function given current state
        policy : (num_states,num_actions)
    """
    # loop over all states
    for s in range(mdp.num_states):
        value = 0
        # loop over all the possible actions
        for a in range(mdp.num_actions):
            action_prob = policy[s, a]
            # if there's no possibility, we don't need to consider to take the action
            if action_prob == 0:
                continue
            for (prob, s_next, reward, done) in mdp.P[s][a]:
                value += action_prob * prob * (reward + discount * V[s_next])
        V_new[s] = value

    
    return V_new

def policy_evaluation(mdp, policy, discount=0.99, theta=0.01):
    """ Computes full policy evaluation until convergence.
    Arguments: MDP, policy, discount factor, theta
    Returns: Value function of policy
    """
    # Init value function array
    V = init_value(mdp)
    loss = float("inf")

    # TODO: Write your implementation here
    while loss >= theta :
        V_new = V.copy()
        for s in range(mdp.num_states):
            value = 0
            # loop over all the possible actions
            for a in range(mdp.num_actions):
                action_prob = policy[s, a]
                # if there's no possibility, we don't need to consider to take the action
                if action_prob == 0:
                    continue
                for (prob, s_next, reward, done) in mdp.P[s][a]:
                    value += action_prob * prob * (reward + discount * V[s_next])
            V_new[s] = value
        loss = np.sum(np.abs(V_new - V))
        V = V_new

    return V

def policy_improvement(mdp, V, discount=0.99):
    """ Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # Initialize a policy array in which to save the greed policy 
    policy = np.zeros_like(random_policy(mdp))

    # TODO: Write your implementation here
    for s in range(mdp.num_states):
        next_reward = np.zeros(mdp.num_actions)

        for a in range(mdp.num_actions) :
            sum = 0
            for(prob,s_next,reward,done) in mdp.P[s][a] :
                sum += prob*(reward + discount*V[s_next])
            next_reward[a] = sum

        best_action = np.argmax(next_reward)
        policy[s, :] = 0
        policy[s,best_action] = 1.0



    return policy


def policy_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the policy iteration (PI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """

    # Start from random policy
    policy = random_policy(mdp)
    # This is only here for the skeleton to run.
    V = init_value(mdp)

    # TODO: Write your implementation here

    while True :
        V = policy_evaluation(mdp,policy,discount,theta)

        new_policy = policy_improvement(mdp,V,discount)

        if np.array_equal(policy,new_policy) :
            break
        else : policy = new_policy


    return V, policy

def value_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here
    while True:
        delta = 0
        for s in range(mdp.num_states):
            v = V[s]
            action_values = np.zeros(mdp.num_actions)

            for a in range(mdp.num_actions):
                total = 0
                for (prob, s_next, reward, done) in mdp.P[s][a]:
                    total += prob * (reward + discount * V[s_next])
                action_values[a] = total

            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break
    # Get the greedy policy w.r.t the calculated value function
    policy = policy_improvement(mdp, V)
    
    return V, policy


if __name__ == "__main__":
    # Create the MDP
    mdp = GridworldMDP([6, 6])
    discount = 0.99
    theta = 0.01

    # Print the gridworld to the terminal
    print('---------')
    print('GridWorld')
    print('---------')
    mdp.render()

    # Create a random policy
    V = init_value(mdp)
    policy = random_policy(mdp)
    # Do one step of policy evaluation and print
    print('----------------------------------------------')
    print('One step of policy evaluation (random policy):')
    print('----------------------------------------------')
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    print_value(V, mdp)

    # Do a full (random) policy evaluation and print
    print('---------------------------------------')
    print('Full policy evaluation (random policy):')
    print('---------------------------------------')
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print_value(V, mdp)

    # Do one step of policy improvement and print
    # "Policy improvement" basically means "Take greedy action w.r.t given a given value function"
    print('-------------------')
    print('Policy improvement:')
    print('-------------------')
    policy = policy_improvement(mdp, V, discount=discount)
    print_deterministic_policy(policy, mdp)

    # Do a full PI and print
    print('-----------------')
    print('Policy iteration:')
    print('-----------------')
    V, policy = policy_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration')
    print('---------------')
    V, policy = value_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)