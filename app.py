from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your model
model = joblib.load(r'model/housing.joblib')  # Use raw string or forward slashes

# Define the field names (excluding MEDV)
FIELD_NAMES = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'tpr']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and handle missing fields
        data = [float(request.form.get(f, 0)) for f in FIELD_NAMES]

        # Make prediction
        prediction = model.predict([data])[0]

        return f"The predicted median value of the house (MEDV) is: ${prediction:.2f}k"
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
