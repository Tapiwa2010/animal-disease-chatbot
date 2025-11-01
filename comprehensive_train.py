import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import re
import string

# Comprehensive training data with balanced examples for each disease
data = {
    'symptoms': [
        # Bovine Respiratory Disease (focus on respiratory symptoms)
        'fever, loss of appetite, coughing',
        'coughing, fever, respiratory issues',
        'loss of appetite, fever, coughing',
        'respiratory distress, fever, no appetite',
        'coughing, lethargy, loss of appetite',
        'fever, coughing',
        'loss of appetite, coughing',
        'fever, no appetite',
        'coughing, loss of appetite',
        'respiratory issues, fever',
        'nasal discharge, coughing',
        'difficulty breathing, fever',
        'pneumonia symptoms, coughing',
        'lung infection, loss of appetite',
        'breathing problems, lethargy',
        'cough, fever, appetite loss',
        'respiratory infection, no appetite',
        'coughing fits, fever',
        'nasal congestion, loss of appetite',
        'breathing difficulty, coughing',
        # Foot and Mouth Disease (focus on oral and foot symptoms)
        'diarrhea, lethargy',
        'fever, diarrhea, loss of appetite',
        'diarrhea, blisters',
        'lethargy, diarrhea',
        'fever, diarrhea',
        'blisters, diarrhea',
        'lethargy, fever',
        'diarrhea, no appetite',
        'mouth sores, fever',
        'foot lesions, diarrhea',
        'oral blisters, lethargy',
        'hoof ulcers, loss of appetite',
        'salivation, fever',
        'tongue blisters, diarrhea',
        'gum lesions, lethargy',
        'mouth ulcers, fever',
        'foot blisters, no appetite',
        'oral lesions, diarrhea',
        'hoof pain, lethargy',
        'saliva drooling, fever',
        # Lumpy Skin Disease (focus on skin symptoms)
        'skin lesions, blisters',
        'lethargy, skin lesions',
        'blisters, skin lesions',
        'skin lesions, lethargy',
        'blisters, fever',
        'skin lesions, fever',
        'lethargy, blisters',
        'skin nodules, loss of appetite',
        'lumps on skin, fever',
        'skin bumps, lethargy',
        'blistering skin, no appetite',
        'cutaneous lesions, fever',
        'skin swelling, lethargy',
        'nodules, blisters',
        'skin rash, loss of appetite',
        'lumpy skin, fever',
        'cutaneous nodules, lethargy',
        'skin eruptions, no appetite',
        'blisters on body, fever',
        'skin lesions, diarrhea',
        # Coccidiosis (focus on digestive symptoms)
        'bloody diarrhea, weight loss, dehydration',
        'bloody stool, lethargy, dehydration',
        'dehydration, diarrhea, weakness',
        'weight loss, bloody diarrhea',
        'lethargy, dehydration, stool issues',
        'bloody diarrhea, no energy',
        'diarrhea, dehydration, weight loss',
        'bloody feces, lethargy',
        'intestinal parasites, diarrhea',
        'coccidia infection, weight loss',
        'bloody poop, dehydration',
        'stool blood, lethargy',
        'dehydration, bloody diarrhea',
        'weight loss, stool issues',
        'lethargy, bloody feces',
        'diarrhea, weakness',
        'bloody stool, no appetite',
        'intestinal infection, dehydration',
        'coccidiosis symptoms, weight loss',
        'bloody diarrhea, lethargy',
        # Additional varied inputs
        'the animal have decayed food',  # Map to Foot and Mouth
        'it does not have appetite',  # Bovine
        'bleeding',  # Lumpy Skin
        'appetite loss',  # Bovine
        'loss of appetite',  # Bovine
        'no appetite, fever',  # Bovine
        'cough',  # Bovine
        'skin rash',  # Lumpy Skin
        'mouth sores',  # Foot and Mouth
        'bloody stool',  # Coccidiosis
    ],
    'disease': [
        # Bovine
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        # Foot and Mouth
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        'Foot and Mouth Disease',
        # Lumpy Skin
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        'Lumpy Skin Disease',
        # Coccidiosis
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        # Additional
        'Foot and Mouth Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Foot and Mouth Disease',
        'Coccidiosis',
    ]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

df['clean_symptoms'] = df['symptoms'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))  # Include bigrams for better matching
X = vectorizer.fit_transform(df['clean_symptoms'])
y = df['disease']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratify for balance

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save
joblib.dump(model, 'chatbot/animal_disease_model.pkl')
joblib.dump(vectorizer, 'chatbot/tfidf_vectorizer.pkl')

print("Comprehensive model retrained and saved.")
