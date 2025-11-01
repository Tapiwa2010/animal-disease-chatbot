from django.shortcuts import render
from django.http import JsonResponse
import joblib
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Static recommendations for diseases
disease_recommendations = {
    'Bovine Respiratory Disease': 'Isolate the animal, ensure proper ventilation, and consult a veterinarian for antibiotic treatment.',
    'Foot and Mouth Disease': 'Quarantine affected animals, disinfect premises, and seek veterinary advice for vaccination.',
    'Lumpy Skin Disease': 'Vaccinate herd, control insect vectors, and isolate infected animals.',
    'Coccidiosis': 'Administer coccidiostats in feed or water, ensure clean environment, and consult a vet for supportive care.',
    'Mastitis': 'Apply warm compresses, milk out the affected quarter frequently, and consult a vet for appropriate antibiotics.',
    # Add more as needed
}

def predict_disease(symptom_text):
    try:
        model = joblib.load('chatbot/animal_disease_model.pkl')
        vectorizer = joblib.load('chatbot/tfidf_vectorizer.pkl')
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return {'disease': 'Error', 'confidence': 0.0, 'recommendations': 'Unable to load AI model. Please check server logs.'}

    clean = clean_text(symptom_text)
    print(f"DEBUG: Cleaned text: '{clean}'")  # Debug log
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    print(f"DEBUG: Model prediction: '{prediction}'")  # Debug log
    try:
        proba = model.predict_proba(vec)[0]
        confidence = round(max(proba) * 100, 2)
        print(f"DEBUG: Confidence: {confidence}%")  # Debug log
    except AttributeError:
        # If model doesn't support predict_proba, use a default confidence
        confidence = 85.0
        print("DEBUG: Using default confidence")  # Debug log

    # Check if confidence is below 50%, indicating unrecognized symptoms
    if confidence < 50.0:
        return {'disease': 'Unrecognized Symptoms', 'confidence': confidence, 'recommendations': 'I\'m sorry, I couldn\'t identify a matching disease based on the symptoms provided. Please consult a veterinarian for accurate diagnosis.'}

    recommendations = disease_recommendations.get(prediction, 'Consult a veterinarian for further advice.')
    print(f"DEBUG: Recommendations: '{recommendations}'")  # Debug log
    return {'disease': prediction, 'confidence': confidence, 'recommendations': recommendations}

def chatbot_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')
        response = predict_disease(user_message)
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse(response)
        else:
            context = {'user_message': user_message, 'bot_response': response}
            return render(request, 'chatbot.html', context)
    return render(request, 'chatbot.html')
