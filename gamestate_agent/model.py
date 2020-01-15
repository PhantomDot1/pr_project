import tensorflow as tf


class AgentModel(tf.keras.Model):
    def __init__(self, num_actions, hidden_units, num_states):
        super(AgentModel, self).__init__()

        # input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape=())

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation="linear", kernel_initializer='RandomNormal')

    @tf.function
    def call(self, input_shape):
        # Build the network from the selected layers with the correct layer shape
        z = self.input_layer(input_shape)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output
