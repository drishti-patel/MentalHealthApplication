from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("word_vectorizer.pkl")

print("Model Type:", type(model))
print("Model Classes:", model.classes_)

id_to_label = {
    0: "Anxiety",
    1: "Bipolar Disorder",
    2: "Depression",
    3: "Normal behavior",
    4: "Personality Disorder",
    5: "Stress",
    6: "Suicidal tendencies"
}

# Keyword safety layer
MENTAL_HEALTH_KEYWORDS = {
    "Anxiety": [
        "anxiety", "anxious", "panic", "panic attack",
        "nervous", "overthinking", "worried", "fear",
        "heart racing", "restless"
    ],

    "Bipolar Disorder": [
        "mania", "manic", "mood swings",
        "high energy", "racing thoughts",
        "impulsive", "extreme mood",
        "suddenly energetic"
    ],

    "Depression": [
        "depressed", "hopeless", "worthless",
        "empty", "sad all the time",
        "nothing matters", "feel numb",
        "crying", "lost interest",
        "can't get out of bed"
    ],

    "Normal behavior": [
        "happy", "good day", "doing well",
        "feeling great", "everything is fine",
        "normal day", "all good"
    ],

    "Personality Disorder": [
        "fear of abandonment",
        "unstable relationships",
        "identity issues",
        "personality disorder",
        "emotional instability",
        "extreme reactions"
    ],

    "Stress": [
        "stress", "stressed",
        "overwhelmed", "burnout",
        "too much pressure",
        "deadline", "exhausted",
        "workload","tensed","tension"
    ],

    "Suicidal tendencies": [
        "suicide", "suicidal",
        "kill myself", "want to die",
        "end my life", "don't want to live",
        "self harm", "harm myself",
        "take my life",
        "life is not worth living",
        "ending it all"
    ]
}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def keyword_prediction(text):
    text = text.lower()

    scores = {}

    for label, keywords in MENTAL_HEALTH_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[label] = score

    best_label = max(scores, key=scores.get)

    if scores[best_label] > 0:
        return best_label

    return None


def predict_text(text):
    text_clean = text.lower().strip()

    # Model prediction
    text_vec = vectorizer.transform([text_clean])
    pred = model.predict(text_vec)[0]

    model_prediction = id_to_label.get(
        int(pred),
        f"Unknown Label {pred}"
    )

    # Keyword prediction
    keyword_result = keyword_prediction(text_clean)

    final_prediction = model_prediction

    # Override only when keyword match exists
    if keyword_result:
        final_prediction = keyword_result

    print("\n====================")
    print("Input:", text)
    print("Raw Prediction:", pred)
    print("Model Prediction:", model_prediction)
    print("Keyword Prediction:", keyword_result)
    print("Final Prediction:", final_prediction)
    print("====================\n")

    return final_prediction


@app.route("/test")
def test():
    return jsonify({
        "message": "Backend is working"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data.get("text", "")

    if not text.strip():
        return jsonify({
            "error": "Text input is required"
        }), 400

    result = predict_text(text)

    suggestions = {
        "Anxiety":
            "Try deep breathing exercises, mindfulness techniques, or talking with someone you trust.",

        "Bipolar Disorder":
            "Consider consulting a qualified mental health professional for proper guidance and support.",

        "Depression":
            "Connecting with supportive friends, family, or a mental health professional may help.",

        "Normal behavior":
            "Keep maintaining your mental well-being through healthy habits and self-care.",

        "Personality Disorder":
            "A mental health professional can help assess symptoms and provide appropriate support.",

        "Stress":
            "Take breaks, practice relaxation techniques, exercise regularly, and maintain a healthy routine.",

        "Suicidal tendencies":
            "If you may be at risk of harming yourself, seek immediate help from emergency services, a crisis helpline, or a trusted person."
    }

    return jsonify({
        "prediction": result,
        "suggestion": suggestions.get(
            result,
            "Please take care of your mental well-being."
        )
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )