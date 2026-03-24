# N-gram Next Word Predictor

A Flask web app that predicts the next word using a pre-trained trigram model trained on the Children Stories Text Corpus, with fast and consistent performance.

## Live Demo
👉 https://ngram-app.onrender.com/

## Features
- Next word prediction using trigram (N=3)
- Pre-trained model for quick responses
- Eager loading for consistent performance
- Simple web interface

## Tech Stack
- Python
- Flask
- HTML/CSS

## Dataset
- Children Stories Text Corpus  
  https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus

## Notes
- The model is loaded at startup (no lazy loading) to ensure stable performance.
- Predictions are based on frequency patterns from the training dataset.

---

Feel free to explore and provide feedback!
