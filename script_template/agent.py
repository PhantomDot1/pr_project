import numpy as np
import tensorflow as tf
import model


class DQN:
    def __init__(self, num_actions, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = model(num_actions)
        self.experience = { 's': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        # accepts single state (as a 2d input) or batch of states, runs a forward pass and returns the model results
        # (logits for actions)
        # TODO: double check if these checks are needed in our final model
        return self.model(np.atleast_2d(inputs.astype('float32')))

    @tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0

        ids = np.random.randInt(low=0, high=len(self.experience['s']), size=self.batch_size)

        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        # balance exploration and exploitation with epsilon and random choice
        # implementation of epsilon-greedy behaviour
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            # TODO: check if these input shape checks are needed!
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)

        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainaible_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
