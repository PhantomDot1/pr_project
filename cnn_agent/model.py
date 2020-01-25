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

        self.hidden_layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu'))
        self.hidden_layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        self.hidden_layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        self.hidden_layers.append(tf.keras.layers.Flatten())
        self.hidden_layers.append(tf.keras.layers.Dense(512, activation="relu"))
        # TODO: add hidden layers:
        # TODO: Conv layers
        # TODO: flatten?
        # TODO: LSTM layer?

        self.output_layer = tf.keras.layers.Dense(num_actions, activation="linear")

    @tf.function
    def call(self, input_shape):
        # Build the network from the selected layers with the correct layer shape
        z = self.input_layer(input_shape)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output
