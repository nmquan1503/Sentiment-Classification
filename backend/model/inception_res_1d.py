import tensorflow as tf
from keras import layers, initializers

class InceptionRes1D(layers.Layer):
    def __init__(
        self, 
        branch_filters: list[int], 
        kernel_sizes: list[int], 
        output_dim: int
    ):
        super().__init__()
        assert len(branch_filters) == len(kernel_sizes), 'filters and kernel_sizes must have same length'
        self.output_dim = output_dim
        self.input_norm = layers.LayerNormalization()
        self.branches = []
        for f, k in zip(branch_filters, kernel_sizes):
            self.branches.append(
                layers.Conv1D(
                    filters=f, 
                    kernel_size=k, 
                    padding='same', 
                    activation=None, 
                    kernel_initializer=initializers.HeNormal()
                )
            )
        self.conv1x1 = layers.Conv1D(
            filters=output_dim, 
            kernel_size=1, 
            padding='same', 
            activation=None, 
            kernel_initializer=initializers.HeNormal()
        )

        self.res_conv = None
        self.out_norm = layers.LayerNormalization()

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim != self.output_dim:
            self.res_conv = layers.Conv1D(self.output_dim, 1, padding='same', activation=None)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.input_norm(x)
        branch_outputs = []
        for conv in self.branches:
            out = conv(x)
            out = tf.nn.gelu(out)
            branch_outputs.append(out)

        x_concat = tf.concat(branch_outputs, axis=-1)

        out = self.conv1x1(x_concat)
        
        residual = self.res_conv(x) if self.res_conv else x
            
        return tf.nn.gelu(self.out_norm(residual + out))