import tensorflow as tf


class AgentModel(tf.keras.Model):
    def __init__(self, num_actions, hidden_units, num_states):
        super(AgentModel, self).__init__()

        # input layer
        # TODO: check if input layer still works for cnn
        self.input_layer = tf.keras.layers.InputLayer()

        self.hidden_layers = []
        # hidden_units = [10, 10, 10]
        # for i in hidden_units:
        #         self.hidden_layers.append(
        #             tf.keras.layers.Dense(i, activation='relu'))

        conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(33, 40), strides=(4, 4), activation='relu')#(normalized)
        self.hidden_layers.append(conv_1)

        # TODO: add hidden layers:
        # TODO: Conv layers
        # TODO: flatten?
        # TODO: LSTM layer?

        self.output_layer = tf.keras.layers.Dense(num_actions, activation="linear", kernel_initializer='RandomNormal')

    @tf.function
    def call(self, input_shape):
        # Build the network from the selected layers with the correct layer shape
        z = self.input_layer(input_shape)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output
