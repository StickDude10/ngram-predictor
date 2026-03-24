import os
import re
import pickle
import requests
from flask import Flask, render_template, request, jsonify

DATASET_URL = "https://drive.google.com/uc?id=1SHZfN7G9WbPdapie-vUEbygIZH7Q9n4f"

app = Flask(__name__)

def download_dataset():
        if not os.path.exists("dataset.txt"):
            print("Downloading dataset...")
            r = requests.get(DATASET_URL)
            with open("dataset.txt", "wb") as f:
                f.write(r.content)
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

if os.path.exists("model.pkl"):
    model.load("model.pkl")
else:
    download_dataset()
    print("Training model...")
    model.train_from_file("dataset.txt", limit=20000)  # limit for speed
    model.save("model.pkl")

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