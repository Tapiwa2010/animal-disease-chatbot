import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import re
import string

# Sample training data (expand this with more diverse examples)
data = {
    'symptoms': [
        'fever, loss of appetite, coughing',
        'diarrhea, lethargy',
        'skin lesions, blisters',
        'fever, diarrhea, loss of appetite',
        'coughing, fever, respiratory issues',
        'lethargy, diarrhea, blisters',
        'loss of appetite, fever',
        'skin lesions, lethargy',
        'coughing, diarrhea',
        'blisters, fever',
        'fever, coughing, loss of appetite',
        'diarrhea, skin lesions',
        'lethargy, coughing',
        'loss of appetite, blisters',
        'fever, lethargy',
        'coughing, skin lesions',
        'diarrhea, fever',
        'blisters, lethargy',
        'loss of appetite, coughing',
        'skin lesions, diarrhea',
        # Add more for Coccidiosis if needed, but balance others
        'bloody diarrhea, weight loss, dehydration',  # Coccidiosis
        'bloody stool, lethargy, appetite loss',  # Coccidiosis
        'dehydration, diarrhea, weakness',  # Coccidiosis
        'weight loss, bloody diarrhea',  # Coccidiosis
        'lethargy, dehydration, stool issues',  # Coccidiosis
    ],
    'disease': [
        'Bovine Respiratory Disease',
        'Foot and Mouth Disease',
        'Lumpy Skin Disease',
        'Foot and Mouth Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Foot and Mouth Disease',
        'Lumpy Skin Disease',
        'Bovine Respiratory Disease',
        'Lumpy Skin Disease',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
        'Coccidiosis',
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

print("Model retrained and saved.")
