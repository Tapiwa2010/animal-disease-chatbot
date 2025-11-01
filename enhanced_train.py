import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import re
import string

# Enhanced training data with more balanced examples, especially for "loss of appetite"
data = {
    'symptoms': [
        # Bovine Respiratory Disease - emphasize fever, coughing, loss of appetite
        'fever, loss of appetite, coughing',
        'coughing, fever, loss of appetite',
        'loss of appetite, fever, coughing',
        'respiratory issues, fever, no appetite',
        'coughing, lethargy, loss of appetite',
        'fever, coughing',
        'loss of appetite, coughing',
        'fever, no appetite',
        'coughing, loss of appetite',
        'respiratory distress, fever',
        # Foot and Mouth Disease - diarrhea, lethargy, blisters
        'diarrhea, lethargy',
        'fever, diarrhea, loss of appetite',
        'diarrhea, blisters',
        'lethargy, diarrhea',
        'fever, diarrhea',
        'blisters, diarrhea',
        'lethargy, fever',
        'diarrhea, no appetite',
        # Lumpy Skin Disease - skin lesions, blisters
        'skin lesions, blisters',
        'lethargy, skin lesions',
        'blisters, skin lesions',
        'skin lesions, lethargy',
        'blisters, fever',
        'skin lesions, fever',
        'lethargy, blisters',
        # Coccidiosis - bloody diarrhea, dehydration, weight loss (less "appetite" focus)
        'bloody diarrhea, weight loss, dehydration',
        'bloody stool, lethargy, dehydration',
        'dehydration, diarrhea, weakness',
        'weight loss, bloody diarrhea',
        'lethargy, dehydration, stool issues',
        'bloody diarrhea, no energy',
        # Additional varied
        'the animal have decayed food',  # Neutral, map to Foot and Mouth
        'it does not have appetite',  # Map to Bovine
        'bleeding',  # Map to Lumpy Skin
        'appetite loss',  # Explicit for Bovine
        'loss of appetite',  # Explicit for Bovine
        'no appetite, fever',  # Bovine
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
        # Foot and Mouth
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
        # Coccidiosis
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
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_symptoms'])
y = df['disease']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save
joblib.dump(model, 'chatbot/animal_disease_model.pkl')
joblib.dump(vectorizer, 'chatbot/tfidf_vectorizer.pkl')

print("Enhanced model retrained and saved.")
