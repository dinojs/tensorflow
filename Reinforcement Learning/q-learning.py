# The goal of the agent is to navigate a frozen lake and find the Goal without falling through the ice the
# Using the Q-learning algorithm
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')  # environment
RENDER = False  # Visualise training
STATES = env.observation_space.n  # n of states
ACTIONS = env.action_space.n  # up, down, right, left
Q = np.zeros((STATES, ACTIONS))  # Q table (matrix) with all 0 values - Random actions at the beginning of the training

EPISODES = 5000  # n of times running the env
MAX_STEPS = 100  # max n of steps allowed for each run of env. Could be stuck in a loop.

LEARNING_RATE = 0.81  # High - each update will introduce a large change to the current state-action value, Low - subtle
GAMMA = 0.96  # Discount factor - how much importance is put on the current and future reward
epsilon = 0.9  # start with a 90% chance of picking a random action

rewards = []
for episode in range(EPISODES):
    state = env.reset()
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()

        # Randomly picking a valid action
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # take random action
        else:
            action = np.argmax(Q[state, :])  # use Q table to pick the action with MAX value

        next_state, reward, done, _ = env.step(action)  # takes the action and returns values about it

        # Q-learning algorithm - Using the current Q-Table to find the best action.
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001  # Increasing the chance of using the trained Q table
            break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards) / len(rewards)}:")

#%% lot the training progress and see how the agent improved over n of episodes
def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i+100]))  # from i to i+100

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()



