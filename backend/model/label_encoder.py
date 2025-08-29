import json

class LabelEncoder:
    def __init__(self, labels_path: str):
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        self.label2id = {}
        self.id2label = {}
        for label in labels:
            self.label2id[label] = len(self.label2id)
            self.id2label[len(self.id2label)] = label

    def encode_batch(self, labels: list[str]):
        return [self.label2id[label] for label in labels]

    def decode(self, id: int):
        return self.id2label(id)

    def decode_batch(self, ids: list[int]):
        return [self.id2label[id] for id in ids]