### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [0, nS-1] and actions in [0, nA-1],
        P[state][action] is a tuple of the form
        (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.ones(nS)
    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    diff = tol
    #print(f'policy = {policy}')
    while (diff >= tol):
        new_value_func = np.zeros(nS)
        for state in range(nS):
            action = policy[state]
            length = len(P[state][action])
            curr_valu = np.zeros(length)
            for i in range(length):
                prob = P[state][action][i][0]
                nexS = P[state][action][i][1]
                rewd = P[state][action][i][2]
                trmnl = P[state][action][i][3]
                future_valu = 0
                if trmnl == False:
                    future_valu = value_function[nexS]
                curr_valu[i] = prob * (rewd + gamma * future_valu)
                
            sum_valu = curr_valu.sum()
            new_value_func[state] = sum_valu
            #print(f'curr_valu = {curr_valu}')
            #print(f'sum_valu = {sum_valu}')
        #print(f'new_value_func = {new_value_func}')
        #print(f'value_function = {value_function}')
        #compute differential for current iteration
        value_diff = abs(new_value_func - value_function)
        diff = value_diff.max()
        #print(f'diff = {diff}')
        value_function = new_value_func
    
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    for state in range(nS):
        curr_valu = np.zeros(nA)
        for action in range(nA):
            length = len(P[state][action])
            for i in range(length):
                prob = P[state][action][i][0]
                nexS = P[state][action][i][1]
                rewd = P[state][action][i][2]
                trmnl = P[state][action][i][3]
                future_valu = 0
                if trmnl == False:
                    future_valu = value_from_policy[nexS]
                curr_valu[action] += rewd + prob * (gamma * future_valu)
        best_act = curr_valu.argmax()
        new_policy[state] = best_act
        #print(f'curr_valu = {curr_valu}')
        #print(f'best_act = {best_act}')
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    diff = tol
    while (diff >= tol):
        inner_diff = tol
        inner_value = np.zeros(nS)
        #print(f'policy = {policy}')
        while (inner_diff >= tol):
            inner_new_value = policy_evaluation(P, nS, nA, policy, gamma=gamma, tol=tol)
            inner_value_diff = abs(inner_new_value - inner_value)
            #print(f'inner_value_diff = {inner_value_diff}')
            inner_diff = inner_value_diff.max()
            inner_value = inner_new_value
        new_value_func = inner_value
        policy = policy_improvement(P, nS, nA, new_value_func, policy, gamma=gamma)
        value_diff = abs(new_value_func - value_function)
        diff = value_diff.max()
        value_function = new_value_func
        #print(f'value_function = {value_function}')
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    diff = tol 
    while (diff >= tol):
        new_value_func = np.zeros(nS)
        for state in range(nS):
            curr_valu = np.zeros(nA)
            for action in range(nA):
                length = len(P[state][action])
                #curr_valu = np.zeros(length)
                for i in range(length):
                    prob = P[state][action][i][0]
                    nexS = P[state][action][i][1]
                    rewd = P[state][action][i][2]
                    trmnl = P[state][action][i][3]
                    future_valu = 0
                    if trmnl == False:
                        future_valu = value_function[nexS]
                    curr_valu[action] += prob * (rewd + gamma * future_valu)
            max_valu = curr_valu.max()
            best_act = curr_valu.argmax()
            new_value_func[state] = max_valu
            policy[state] = best_act
            #print(f'curr_valu = {curr_valu}')
            #print(f'max_valu = {max_valu}')
        #print(f'new_value_func = {new_value_func}')
        #print(f'value_function = {value_function}')
        #compute differential for current iteration
        value_diff = abs(new_value_func - value_function)
        diff = value_diff.max()
        #print(f'diff = {diff}')
        value_function = new_value_func
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function, policy

def render_single(env, policy, max_steps=100, show_rendering=True):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        if show_rendering:
            env.render()
        time.sleep(0.0025) #original 0.25
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    if show_rendering:
        env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

def evaluate(env, policy, max_steps=100, max_episodes=32):
    """
    This function does not need to be modified,
    evaluates your policy over multiple episodes.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_rewards = []
    dones = []
    for _ in range(max_episodes):
        episode_reward = 0
        ob = env.reset()
        for t in range(max_steps):
            a = policy[ob]
            ob, rew, done, _ = env.step(a)
            episode_reward += rew
            if done:
                break

        episode_rewards.append(episode_reward)
        dones.append(done)

    episode_rewards = np.array(episode_rewards).mean()
    success = np.array(dones).mean()

    print(f"> Average reward over {max_episodes} episodes:\t\t\t {episode_rewards}")
    print(f"> Percentage of episodes goal reached:\t\t\t {success * 100:.0f}%")
