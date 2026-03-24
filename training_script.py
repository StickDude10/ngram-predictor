import re
import pickle
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.context_counts = Counter()

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.split()
    
    def train_from_file(self, file_path):
        self.ngrams = defaultdict(Counter)
        self.context_counts = Counter()

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                tokens = self.preprocess(line)

                for j in range(len(tokens) - self.n + 1):
                    context = tuple(tokens[j:j+self.n-1])
                    word = tokens[j+self.n-1]

                    self.ngrams[context][word] += 1
                    self.context_counts[context] += 1

                if i % 1000 == 0:
                    print(f"Processed {i} lines...")

        print("Total contexts learned:", len(self.ngrams))

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.ngrams, self.context_counts), f)


# 🔥 TRAIN
model = NGramModel(3)

model.train_from_file("dataset.txt")
model.save()

print("Training complete & model saved!")
