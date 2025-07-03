from flask import Flask, request, jsonify
import joblib

# Load model
model = joblib.load('iris_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "🌸 Iris Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    prediction = model.predict([features])
    return jsonify({'predicted_species': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
