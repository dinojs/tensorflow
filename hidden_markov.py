# Hidden Markov Model - Predict the temperature on each day given the following information.

# 1) Cold days are encoded by a 0 and hot days are encoded by a 1.
# 2) The first day in the sequence has an 80% chance of being cold.
# 3) A cold day has a 30% chance of being followed by a hot day.
# 4) A hot day has a 20% chance of being followed by a cold day.
# 5) On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and 25 and 20 on a hot day.

import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])  # 80% chance of being cold, 20% of being hot
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])  # 3) & 4)
# The loc argument represents the mean and the scale is the standard deviation
observation_distribution = tfd.Normal(loc=[0., 25.], scale=[5., 20.])  # cold(mean = 0, SD = 5), hot(mean = 25, SD = 20)

#%% Create model
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)  # How many days you want to predict

#%% Prediction
mean = model.mean()
# Need to evaluate part of the graph from within a session to see the value of this tensor (Tensorflow doc)
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
# Temperature = [ 4.9999995  9.999999  12.499999  13.75      14.375001  14.687501 14.84375  ]