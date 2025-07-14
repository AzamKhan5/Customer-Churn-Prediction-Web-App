from flask import Flask, request, render_template
import pickle as pk

app = Flask(__name__)

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))  # Pre-fitted scaler

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    CreditScore = int(request.form['CreditScore'])
    Gender = int(request.form['Gender'])
    Age = int(request.form['Age'])
    Tenure = int(request.form['Tenure'])
    Balance = float(request.form['Balance'])
    HasCrCard = int(request.form['HasCrCard'])
    IsActiveMember = int(request.form['IsActiveMember'])
    EstimatedSalary = float(request.form['EstimatedSalary'])
    Geography = request.form['Geography']
    NumOfProducts = int(request.form['NumOfProducts'])

    # One-hot encoding for Geography
    Geography_France = 1 if Geography == 'France' else 0
    Geography_Germany = 1 if Geography == 'Germany' else 0
    Geography_Spain = 1 if Geography == 'Spain' else 0

    # One-hot encoding for NumOfProducts
    NumOfProducts_1 = 1 if NumOfProducts == 1 else 0
    NumOfProducts_2 = 1 if NumOfProducts == 2 else 0
    NumOfProducts_3 = 1 if NumOfProducts == 3 else 0
    NumOfProducts_4 = 1 if NumOfProducts == 4 else 0

    # Combine all features
    input_features = [
        CreditScore, Gender, Age, Tenure, Balance,
        HasCrCard, IsActiveMember, EstimatedSalary,
        Geography_France, Geography_Germany, Geography_Spain,
        NumOfProducts_1, NumOfProducts_2, NumOfProducts_3, NumOfProducts_4
    ]

    # Scale and predict
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]

    return render_template('index.html', prediction=int(prediction))

if __name__ == '__main__':
    app.run(debug=True)
