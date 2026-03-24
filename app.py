import os
import re
import pickle
import requests
from collections import defaultdict, Counter
from flask import Flask, render_template, request, jsonify

DATASET_URL = "https://limewire.com/d/j7beL#dKTbYNCOOW"

app = Flask(__name__)

def download_dataset():
    if not os.path.exists("dataset.txt"):
        print("Downloading dataset...")

        URL = DATASET_URL
        session = requests.Session()

        response = session.get(URL, stream=True)

        with open("dataset.txt", "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

        print("Download complete.")

class NGramModel:
    def __init__(self, n=3):
        self.n = n

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.ngrams, self.context_counts = pickle.load(f)

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()
    
    def train_from_file(self, file_path, limit=None):
        from collections import defaultdict, Counter

        self.ngrams = defaultdict(Counter)
        self.context_counts = Counter()

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()   # 🔥 read FULL text

        tokens = self.preprocess(text)

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            word = tokens[i+self.n-1]

            self.ngrams[context][word] += 1
            self.context_counts[context] += 1

            if i % 10000 == 0:
                print(f"Processed {i} tokens...")
        print("Total contexts learned:", len(self.ngrams))

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.ngrams, self.context_counts), f)

    def predict(self, text, top_k=5):
        tokens = self.preprocess(text)

        if len(tokens) < self.n-1:
            return []

        context = tuple(tokens[-(self.n-1):])

        if context not in self.ngrams:
            return []

        total = self.context_counts[context]
        predictions = self.ngrams[context].most_common(top_k)

        return [f"{word} ({count/total:.2f})" for word, count in predictions]
    

# Load model
model = NGramModel(3)

# 🔥 FORCE retrain
download_dataset()
with open("dataset.txt", "r", encoding="utf-8", errors="ignore") as f:
    sample = f.read(500)
    print("DATA SAMPLE:", sample[:200])
print("Training model...")
model.train_from_file("dataset.txt", limit=20000)
model.save("model.pkl")

# if os.path.exists("model.pkl"):
#     model.load("model.pkl")
# else:
#     download_dataset()
#     print("Training model...")
#     model.train_from_file("dataset.txt", limit=20000)  # limit for speed
#     model.save("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    predictions = model.predict(text)
    return jsonify(predictions)


if __name__ == "__main__":
    app.run(debug=True)