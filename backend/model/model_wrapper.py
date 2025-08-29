from bs4 import BeautifulSoup
import re
from underthesea import word_tokenize
from cleaner import Cleaner
from vocab import Vocab
from tokenizer import Tokenizer
from label_encoder import LabelEncoder
import tensorflow as tf
from hybrid_model import HybridModel
import numpy as np
import json

class ModelWrapper:
    def __init__(
        self,
        max_length_sent: int,
        model_config_path: str,
        model_weights_path: str,
        vocab_path: str,
        labels_path: str,
        batch_size: int
    ):
        self.cleaner = Cleaner()
        self.vocab = Vocab(vocab_path)
        self.tokenizer = Tokenizer(self.vocab, max_length_sent)
        self.label_encoder = LabelEncoder(labels_path)
        self.batch_size = batch_size
        self._load_model(model_config_path, model_weights_path)
    
    def predict(self, inputs: list[str]):
        results = []
        for i in range(0, len(inputs), self.batch_size):
            batch_inputs = inputs[i:i + self.batch_size]
            cleaned_inputs = [self.cleaner(input) for input in batch_inputs]
            tokenized_inputs = self.tokenizer.encode_batch(cleaned_inputs)
            input_tensors = tf.convert_to_tensor(tokenized_inputs, dtype=tf.int32)
            probs = self.model(input_tensors, training=False)
            probs = probs.numpy()
            for prob in probs:
                idx = int(prob.argmax())
                label = self.label_encoder.decode(idx)
                prob_dict = {name: float(prob[i]) for i, name in self.label_encoder.id2label.items()}
                results.append({
                    'label': label,
                    'probs': prob_dict
                })
        return results
    
    def _load_model(self, model_config_path, model_weights_path):
        with open(model_config_path) as f:
            model_config = json.load(f)

        model = HybridModel(
            vocab=self.vocab,
            lstm_hidden_dim=model_config['lstm_hidden_dim'],
            rnn_fc_configs=model_config['rnn_fc_configs'],
            cnn_inception_configs=model_config['cnn_inception_configs'],
            cnn_pooling_configs=model_config['cnn_pooling_configs'],
            cnn_output_dims=model_config['cnn_output_dims'],
            cnn_fc_configs=model_config['cnn_fc_configs'],
            attention_dim=model_config['attention_dim'],
            fc_out_configs=model_config['fc_out_configs'],
            output_dim=model_config['output_dim'],
            trainable_embeddings=model_config['trainable_embeddings']
        )

        dummy_input = tf.zeros((1, self.tokenizer.max_len), dtype=tf.int32)
        _ = model(dummy_input)

        model.load_weights(model_weights_path)

        self.model = model