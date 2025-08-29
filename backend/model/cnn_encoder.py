import tensorflow as tf
from keras import layers, initializers, models
from inception_res_1d import InceptionRes1D

class CNNEncoder(layers.Layer):
    def __init__(
        self,
        inception_configs: list[dict],
        pooling_configs: list[dict],
        res_output_dims: list[int],
        fc_configs: list[dict]
    ):
        super(CNNEncoder, self).__init__()
        self.layers = []
        for inception_config, pooling_config, output_dim in zip(inception_configs, pooling_configs, res_output_dims):
            self.layers.append(InceptionRes1D(
                branch_filters=inception_config['branch_filters'],
                kernel_sizes=inception_config['kernel_sizes'],
                output_dim = output_dim
            ))
            if pooling_config['type'] == 'max':
                self.layers.append(layers.MaxPooling1D(
                    pool_size=pooling_config['size'],
                    strides=pooling_config['strides'],
                    padding=pooling_config['padding']
                ))
            elif pooling_config['type'] == 'global_max':
                self.layers.append(layers.GlobalMaxPooling1D())
            elif pooling_config['type'] == 'average':
                self.layers.append(layers.AveragePooling1D(
                    pool_size=pooling_config['size'],
                    strides=pooling_config['strides']
                ))
            elif pooling_config['type'] == 'global_average':
                self.layers.append(layers.GlobalAveragePooling1D())
        self.fcs = []
        for config in fc_configs:
            self.fcs.append(
                layers.Dense(
                    config['dim'], 
                    activation=config['activation'], 
                    kernel_initializer=initializers.HeNormal()
                )
            )
            if 'dropout' in config:
                self.fcs.append(
                    layers.Dropout(config['dropout'])
                )
        self.fcs = models.Sequential(self.fcs)

    def call(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, InceptionRes1D):
                x = layer(x, training=training)
            else:
                x = layer(x)
        pattern = self.fcs(x)    
        return pattern