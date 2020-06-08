import sys
import pickle

from flask import Flask, request, jsonify
from training import model_template, ml_models

sys.modules['ml_models'] = ml_models
sys.modules['model_template'] = model_template


app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    context = request.get_json()
    with open(context.get("model_path", "models/") + context["model"] + '.pkl', 'rb') as f:
        model = pickle.load(f)
    pred = model.predict(context)
    if isinstance(pred, list):
        return jsonify(pred)
    return pred


@app.route("/", methods=['GET'])
def health():
    return 'healthy', 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
