from flask import Flask, request, jsonify
from model_loader import load_bert_model
from model_utils import compare_skills
import torch

# 1️⃣ Load state_dict once (already saved from training)
state_dict = torch.load('bert_model.pt', map_location='cpu')

# Load BERT model (model_loader.py handles state_dict inside)
model = load_bert_model(model_path='bert_model.pt', device='cpu')


# 3️⃣ Flask app
app = Flask(__name__)


@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    employee_skills = data.get('resume', '')
    job_skills = data.get('job', '')

    if not employee_skills or not job_skills:
        return jsonify({'error': 'Both resume and job fields are required.'}), 400

    # Ensure model/tokenizer are used correctly
    score = compare_skills(employee_skills, job_skills, device='cpu')
    
    # Format score to 2 decimal places
    formatted_score = round(score, 2)

    return jsonify({'similarity_score': formatted_score})

if __name__ == "__main__":
    app.run(debug=True)
