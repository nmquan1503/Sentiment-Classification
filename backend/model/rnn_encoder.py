import tensorflow as tf
from keras import layers, models

class RNNEncoder(layers.Layer):
    def __init__(
        self,
        lstm_hidden_dim,
        fc_configs
    ):
        super().__init__()
        self.bilstm = layers.Bidirectional(
            layers.LSTM(lstm_hidden_dim, return_sequences=True, dropout=0.3)
        )
        self.lstm_post_norm = layers.LayerNormalization()
        self.attn_fc = layers.Dense(1, activation=None)
        self.fcs = []
        for config in fc_configs:
            self.fcs.append(layers.Dense(config['dim'], activation=config['activation']))
            if 'dropout' in config:
                self.fcs.append(layers.Dropout(config['dropout']))
        self.fcs = models.Sequential(self.fcs)
    def call(self, x, mask):
        hiddens = self.bilstm(x, mask=mask)
        hiddens = self.lstm_post_norm(hiddens)
        
        scores = self.attn_fc(hiddens)
        scores = tf.squeeze(scores, axis=-1)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.reduce_sum(hiddens * tf.expand_dims(weights, -1), axis=-2)
        
        context = self.fcs(context)
        return hiddens, context