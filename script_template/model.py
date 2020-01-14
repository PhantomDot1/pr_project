import tensorflow as tf


class AgentModel(tf.keras.model):
    def __init__(self, num_actions):
        super(AgentModel, self).__init__()

        # input layer
        # TODO: fix the input shape to suit the onscreen pixels or gamestate variables
        self.input_layer = tf.keras.layers.InputLayer(input_shape=())

        self.hidden_layers = []
        # TODO: add hidden layers to list for network

        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation="linear", kernel_initializer='RandomNormal')

    @tf.function
    def call(self, input_shape):
        # Build the network from the selected layers with the correct layer shape
        z = self.input_layer(input_shape)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output
