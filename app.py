import os
import re
import pickle
from collections import defaultdict, Counter
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

class NGramModel:
    def __init__(self, n=3):
        self.n = n

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.ngrams, self.context_counts = pickle.load(f)
    
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
    print("Loading trained model...")
    model.load("model.pkl")
else:
    print("Model not found! Please train locally.")

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