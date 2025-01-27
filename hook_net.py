import tensorflow as tf
from tensorflow.keras import layers, Model


class HookNet:
    def __init__(self, input_shape, num_skip_connections=4):
        self.input_shape = input_shape
        self.num_skip_connections = num_skip_connections
        self.model = self.build_model()

    def conv_block(self, x, filters, kernel_size=3, padding="same", strides=1):
        """Creates a convolutional block: Conv -> BatchNorm -> Activation"""
        x = layers.Conv1D(filters, kernel_size, padding=padding, strides=strides)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def contracting_block(self, x, filters):
        """Creates a contracting block with vertical skip connections."""
        # skip = self.conv_block(x, filters)
        skip = self.conv_block(x, filters)
        x = layers.MaxPooling1D(pool_size=2)(skip)
        return x, skip

    def expanding_block(self, x, skip, filters, use_transpose_conv=True):
        """Creates an expanding block with transposed convolution and skip connections."""
        if use_transpose_conv:
            x = layers.Conv1DTranspose(
                filters, kernel_size=2, strides=2, padding="same"
            )(x)
        else:
            x = layers.Conv1DTranspose(
                filters, kernel_size=1, strides=1, padding="same"
            )(x)

        scale_down_by = skip.shape[1] // x.shape[1]

        skip = layers.MaxPooling1D(pool_size=scale_down_by)(skip)

        x = layers.Concatenate()([x, skip])
        x = self.conv_block(x, filters)
        return x

    def build_model(self):
        """Builds the Hook-Net architecture."""
        inputs = layers.Input(shape=self.input_shape)

        # Contracting path
        skips = []
        x = inputs
        filters = 24
        for i in range(self.num_skip_connections):
            x, skip = self.contracting_block(x, (filters * (i + 1)))
            skips.append(skip)

        # Bottleneck
        x = self.conv_block(x, self.num_skip_connections * 2 * filters)

        # Expanding path
        for i in range(self.num_skip_connections - 1, -1, -1):
            use_transpose_conv = i % 2 == 1
            x = self.expanding_block(
                x, skips[i], filters * (i + 1), use_transpose_conv=use_transpose_conv
            )

        # Output layer
        outputs = layers.Conv1D(1, kernel_size=1, activation="sigmoid")(x)

        model = Model(inputs, outputs, name="HookNet")
        return model

    def compile(
        self, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        self.model.summary()


# Define input shape (16384 samples of audio)
input_shape = (16384, 1)
hook_net = HookNet(input_shape, num_skip_connections=13)
hook_net.compile()
hook_net.summary()
