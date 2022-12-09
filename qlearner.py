import gym
import numpy as np
import math, random

class Qlearner:

    def __init__(self, epsilon, learning_rate, gamma, env):
        self.name = "Qlearner"
        self.e = epsilon
        self.a = learning_rate
        self.g = gamma
        self.env = env
        self.initializeQtable(self.env)

    def initializeQtable(self, env):
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def e_greedy(self, state, epsilon):
        return np.argmax(self.Q[state]) if random.uniform(0,1) > epsilon else self.env.action_space.sample()

    def greedy(self, state):
        return np.max(self.Q[state])

    def train(self, num_episodes, max_steps):
        for i in range(num_episodes):
            if (i  % 1000 == 0): print(i)
            state = self.env.reset()
            done = False
            for step in range(max_steps):
                epsilon = 0.001 + self.e * np.exp(-0.0005*i)
                action = self.e_greedy(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                self.Q[state][action] = self.Q[state][action] + self.a * (reward + self.g * np.max(self.Q[next_state]) - self.Q[state][action])
                if done:
                    break
                state = next_state

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    if seed:
      state = env.reset(seed=seed[episode])
    else:
      state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    
    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = np.argmax(Q[state][:])
      new_state, reward, done, info = env.step(action)
      total_rewards_ep += reward
        
      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    agent = Qlearner(epsilon=0.9, learning_rate=0.7, gamma=0.99, env=env)
    agent.train(10000, 100)
    print(agent.Q)

    print(evaluate_agent(env, 100, 100, agent.Q, 0))
    

main()