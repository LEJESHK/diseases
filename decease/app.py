from flask import Flask, render_template, request
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from collections import Counter  # To find the most common prediction

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv('Testing.csv')
df.columns = df.columns.str.strip()

X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Train all models
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X, y)

svm_model = SVC(probability=True)
svm_model.fit(X, y)

random_forest_model = RandomForestClassifier()
random_forest_model.fit(X, y)

# Disease to specialist mapping
specialist_map = {
    "Fungal infection": "Dermatologist", "Drug Reaction": "Dermatologist", "Chicken pox": "Dermatologist", 
    "Acne": "Dermatologist", "Psoriasis": "Dermatologist", "Impetigo": "Dermatologist",
    "Migraine": "Neurologist", "Cervical spondylosis": "Neurologist", 
    "Paralysis (brain hemorrhage)": "Neurologist", "(vertigo) Paroxysmal Positional Vertigo": "Neurologist",
    "GERD": "Gastroenterologist", "Chronic cholestasis": "Gastroenterologist", 
    "Peptic ulcer disease": "Gastroenterologist", "Gastroenteritis": "Gastroenterologist",
    "Hepatitis A": "Gastroenterologist", "Hepatitis B": "Gastroenterologist", 
    "Hepatitis C": "Gastroenterologist", "Hepatitis D": "Gastroenterologist", 
    "Hepatitis E": "Gastroenterologist", "Alcoholic hepatitis": "Gastroenterologist",
    "Dimorphic hemorrhoids (piles)": "Gastroenterologist",
    "Allergy": "General Physician", "AIDS": "General Physician", "Diabetes": "General Physician",
    "Bronchial Asthma": "General Physician", "Hypertension": "General Physician", 
    "Jaundice": "General Physician", "Malaria": "General Physician", 
    "Dengue": "General Physician", "Typhoid": "General Physician", 
    "Tuberculosis": "General Physician", "Common Cold": "General Physician",
    "Pneumonia": "General Physician", "Heart attack": "General Physician", 
    "Varicose veins": "General Physician", "Hypothyroidism": "General Physician",
    "Hyperthyroidism": "General Physician", "Hypoglycemia": "General Physician", 
    "Osteoarthritis": "General Physician", "Arthritis": "General Physician", 
    "Urinary tract infection": "General Physician"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    final_prediction = None
    final_specialist = None
    symptoms_entered = ""

    if request.method == 'POST':
        symptoms_entered = request.form['symptoms']
        selected_symptoms = [s.strip().lower() for s in symptoms_entered.split(',')]

        input_data = {symptom: 0 for symptom in X.columns}
        for col in X.columns:
            if col.replace('_', ' ').lower() in selected_symptoms:
                input_data[col] = 1

        input_df = pd.DataFrame([input_data])

        # Get predictions from all three models
        nb_prediction = naive_bayes_model.predict(input_df)[0]
        svm_prediction = svm_model.predict(input_df)[0]
        rf_prediction = random_forest_model.predict(input_df)[0]

        # Calculate the most common prediction
        predictions = [nb_prediction, svm_prediction, rf_prediction]
        most_common_prediction = Counter(predictions).most_common(1)[0][0]

        # Get the corresponding specialist for the final prediction
        final_specialist = specialist_map.get(most_common_prediction, "General Physician")
        final_prediction = most_common_prediction

    all_symptoms = [col.replace('_', ' ') for col in X.columns]

    return render_template("index.html",
                           final_prediction=final_prediction,
                           final_specialist=final_specialist,
                           symptoms_entered=symptoms_entered,
                           symptoms=all_symptoms)

if __name__ == '__main__':
    app.run(debug=True)
