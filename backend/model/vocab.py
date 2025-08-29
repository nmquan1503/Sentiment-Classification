import numpy as np

class Vocab:
    def __init__(self, w2v_file_path: str):
        self.word_index = {}
        self.embedding_matrix = []
        self.embedding_dim = None
        self.load_word2vec(w2v_file_path)
        self.embedding_matrix = np.array(self.embedding_matrix, dtype='float32')
        self.unk_id = self.word_index['<unk>']
        self.pad_id = self.word_index['<pad>']

    def load_word2vec(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip().split()
            if (len(first_line) == 2 and first_line[0].isdigit()):
                self.embedding_dim = int(first_line[1])
            else:
                f.seek(0)
            for line in f:
                values = line.strip().split()
                if len(values) < self.embedding_dim + 1:
                    continue
                word = '_'.join(values[: - self.embedding_dim]).lower()
                if word in self.word_index:
                    continue
                try:
                    vector = np.asarray(values[- self.embedding_dim :], dtype='float32')
                except ValueError:
                    print(f'Error line: {line.strip()}')
                    continue
                self.word_index[word] = len(self.word_index)
                self.embedding_matrix.append(vector)

        self.word_index['<unk>'] = len(self.word_index)
        self.embedding_matrix.append(np.random.uniform(-0.05, 0.05, self.embedding_dim))

        self.word_index['<pad>'] = len(self.word_index)
        self.embedding_matrix.append(np.zeros(self.embedding_dim))

        for word in ['happy', 'love', 'sad', 'angry', 'surprise', 'thinking', 'neutral']:
            if f'<{word}>' not in self.word_index:
                dot_id = self.word_index.get('.')
                word_id = self.word_index.get(word)
                if dot_id is None or word_id is None:
                    continue
                dot_vector = self.embedding_matrix[dot_id]
                word_vector = self.embedding_matrix[word_id]
                vector = (dot_vector + word_vector) / 2

                self.word_index[f'<{word}>'] = len(self.word_index)
                self.embedding_matrix.append(vector)

    def get_index(self, word: str) -> int:
        return self.word_index.get(word, self.unk_id)