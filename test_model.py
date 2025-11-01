import joblib
import re
import string

# Load model and vectorizer
model = joblib.load('chatbot/animal_disease_model.pkl')
vectorizer = joblib.load('chatbot/tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def predict_disease(symptom_text):
    clean = clean_text(symptom_text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    try:
        proba = model.predict_proba(vec)[0]
        confidence = round(max(proba) * 100, 2)
    except AttributeError:
        confidence = 85.0
    return {'disease': prediction, 'confidence': confidence}

# Test with different inputs
test_inputs = [
    "fever, loss of appetite, coughing",
    "diarrhea, lethargy",
    "skin lesions, blisters",
    "the animal have decayed food",
    "it does not have appetite",
    "bleeding"
]

for inp in test_inputs:
    result = predict_disease(inp)
    print(f"Input: '{inp}' -> Disease: {result['disease']}, Confidence: {result['confidence']}%")
