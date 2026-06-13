import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("word_vectorizer.pkl")

text = ["I feel very sad and hopeless"]

X = vectorizer.transform(text)

prediction = model.predict(X)

print("Prediction:", prediction)