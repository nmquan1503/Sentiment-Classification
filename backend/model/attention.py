import tensorflow as tf
from keras import layers

class Attention(layers.Layer):
    def __init__(self, attention_dim):
        super().__init__()
        self.dense = layers.Dense(attention_dim, activation='tanh')
        self.v = layers.Dense(1, activation=None)

    def call(self, hiddens, pattern, mask):
        """
        Args:
            hiddens: [B, S, 2H]
            pattern: [B, P]
        Returns:
            context: [B, 2H]
            attn_weights: [B, S, 1]
        """
        
        S = tf.shape(hiddens)[1]
        
        # [B, P] -> [B, 1, P] -> [B, S, P]
        exp_pattern = tf.tile(tf.expand_dims(pattern, 1), [1, S, 1])

        # [B, S, 2H] cat [B, S, P] -> [B, S, 2H + P]
        score_input = tf.concat([hiddens, exp_pattern], axis=-1)

        # [B, S, 2H + P] -> [B, S, A] -> [B, S, 1]
        score = self.v(self.dense(score_input))

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, -1)
            score += (1.0 - mask) * -1e9
        
        attn_weights = tf.nn.softmax(score, axis=1)

        # [B, S, 1] * [B, S, 2H] -> [B, S, 2H] -> [B, 2H]
        context = tf.reduce_sum(attn_weights * hiddens, axis=1)

        return context