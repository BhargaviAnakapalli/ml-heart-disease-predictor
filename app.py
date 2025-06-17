from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load model and scaler
model, scaler = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            features = [float(request.form.get(field)) for field in [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]]
            features_array = np.array([features])
            features_scaled = scaler.transform(features_array)  # ✅ Scale input
            prediction = model.predict(features_scaled)[0]
            result = "🟢 No signs of heart disease detected. Stay healthy and continue regular checkups." if prediction == 0 else "🔴  Risk of Heart Disease Detected. Please consult a cardiologist for further evaluation."
        except:
            result = "❌ Error in input values."

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)










