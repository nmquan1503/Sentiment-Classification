import tensorflow as tf
from keras import models, layers
from attention import Attention
from cnn_encoder import CNNEncoder
from rnn_encoder import RNNEncoder

class HybridModel(models.Model):
    def __init__(
        self,
        vocab,
        cnn_inception_configs,
        cnn_pooling_configs,
        cnn_output_dims,
        cnn_fc_configs,
        lstm_hidden_dim,
        rnn_fc_configs,
        attention_dim,
        fc_out_configs,
        output_dim,
        trainable_embeddings=False,
    ):
        super().__init__()
        self.vocab = vocab

        self.embedding = layers.Embedding(
            input_dim=len(vocab.word_index),
            output_dim=vocab.embedding_dim,
            weights=[vocab.embedding_matrix],
            trainable=trainable_embeddings,
            mask_zero=False
        )

        self.embedding_adapter = layers.Dense(vocab.embedding_dim, activation='gelu')
        
        self.cnn_encoder = CNNEncoder(
            inception_configs=cnn_inception_configs,
            pooling_configs=cnn_pooling_configs,
            res_output_dims=cnn_output_dims,
            fc_configs=cnn_fc_configs
        )
        
        self.rnn_encoder = RNNEncoder(
            lstm_hidden_dim=lstm_hidden_dim,
            fc_configs=rnn_fc_configs
        )
        
        self.attention = Attention(attention_dim=attention_dim)

        self.rnn_context_norm = layers.LayerNormalization()
        self.cnn_context_norm = layers.LayerNormalization()
        self.pattern_norm = layers.LayerNormalization()

        self.fcs = []
        for config in fc_out_configs:
            self.fcs.append(layers.Dense(config['dim'], activation=config['activation']))
            if 'dropout' in config:
                self.fcs.append(layers.Dropout(config['dropout']))
        self.fcs.append(layers.Dense(output_dim, activation='softmax'))
        self.fcs = models.Sequential(self.fcs)

    def call(self, x, training=False):
        """
        Args:
            x: [B, S]
        Returns:
            out: [B, 3]
        """
        # [B, S] -> [B, S, D]
        embedded_x = self.embedding(x)
        embedded_x = self.embedding_adapter(embedded_x)

        mask = tf.not_equal(x, self.vocab.pad_id)
        
        # [B, S, 2H]
        rnn_hiddens, rnn_context = self.rnn_encoder(embedded_x, mask)
        rnn_context = self.rnn_context_norm(rnn_context)

        # [B, S, D] -> [B, P]
        pattern = self.cnn_encoder(rnn_hiddens, training=training)
        pattern = self.pattern_norm(pattern)

        # context: [B, 2H]
        cnn_context = self.attention(rnn_hiddens, pattern, mask)
        cnn_context = self.cnn_context_norm(cnn_context)

        out = tf.concat([cnn_context, rnn_context], axis=-1)

        out = self.fcs(out)

        return out